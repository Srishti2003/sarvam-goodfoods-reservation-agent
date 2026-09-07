"""GoodFoods agentic loop: hand-written, no framework.

Repeatedly calls the model with the MCP tool schemas from mcp_server/server.py
(via agent/mcp_client.py), executes whichever tool(s) it picks, feeds the
result back into the conversation, and loops until the model returns a
final natural-language reply (no more tool calls).

Two distinct failure categories are handled differently -- this distinction
is the whole point of this module, so it's worth stating explicitly instead
of just in comments scattered through the code:

  1. Business-logic failures: a tool call COMPLETES NORMALLY (call_tool
     returns, doesn't raise) with a body like {"success": false, "error":
     "..."} -- e.g. no availability, capacity exceeded, double-booking, an
     off-grid slot, an unknown reservation id. This is valid data, not a
     failure of the call itself: it's appended to the conversation exactly
     like a successful result and handed back to the model on the next
     iteration to react to. There is no special-case code for this at all
     -- "returned normally, whatever the content" IS the definition of
     "not retried", by construction. Retrying a "this restaurant is full"
     response can't change the answer.

  2. Transport/execution failures: call_tool RAISES (dropped subprocess,
     malformed response, timeout, MCP-level error). These are retried up
     to `max_transport_retries` times with exponential backoff. If every
     attempt fails, escalate_to_human is called DIRECTLY IN CODE -- the
     model is never asked "what do you want to do about this failure",
     because a model can rationalize "just one more try" indefinitely.

A hard cap on tool-call iterations (`max_iterations`) is enforced
independently of both of the above: if the model keeps requesting tools
without ever converging on a final reply, the same forced-escalation path
fires once the cap is hit.

A third surface, distinct from both: llm_call itself can raise (a
provider-level failure -- rate limit, auth, network, timeout -- rather
than a call_tool failure). This is NOT retried, unlike call_tool failures
-- a rate limit's suggested wait is on the order of tens of minutes, which
is useless mid-conversation, and retrying against one is a good way to
make it worse. It goes straight to the same forced-escalation path on the
first failure. Confirmed live: before this was handled, a real
groq.RateLimitError propagated all the way out of run_turn() as a raw,
unhandled exception (a traceback with full local file paths rendered
directly in the Streamlit chat).
"""

import asyncio
import json
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession

DEFAULT_MAX_ITERATIONS = 8
DEFAULT_MAX_TRANSPORT_RETRIES = 3
DEFAULT_BACKOFF_BASE_SECONDS = 0.5
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"  # see README.md "Model resolution" -- Llama is gone from Groq

# This is ALL the model gets: this prompt plus each tool's own docstring
# (surfaced to it as the tool's JSON-schema "description" via
# mcp_tools_to_openai_schemas). There is no other routing layer -- every
# decision about which tool to call, or whether to just reply in plain
# language, is made by the model reading this and the schemas, never by
# string-matching the user's message in code.
DEFAULT_SYSTEM_PROMPT = """You are the GoodFoods reservation assistant, a conversational agent for a restaurant reservation platform. You help guests find restaurants, check whether a table is available, get personalized recommendations, and book, change, or cancel reservations -- entirely by calling the tools available to you. You have no knowledge of any restaurant, availability, or reservation beyond what a tool call returns to you in this conversation; never invent a restaurant name, id, time slot, or reservation id.

Restaurants take reservations only at fixed 2-hour seatings (11:00, 13:00, 15:00, 17:00, 19:00, 21:00) -- there is nothing in between, so never propose or accept a time at face value without checking it against a tool first.

Here is what each tool is for and when to reach for it:

- ping: an internal connectivity check with no conversational purpose. You will essentially never need this.

- search_restaurants: a hard filter over restaurants by cuisine, neighborhood, price tier, and/or minimum party size. Use this when the user has specific, known criteria and wants a list of matching places -- e.g. "Italian places downtown" or "somewhere that can seat 8". Every filter given must match exactly; there's no fuzzy matching, so an empty result means nothing satisfies ALL the given criteria at once, not that nothing exists at all.

- check_availability: looks up which of a specific restaurant's fixed slots are open on a given date for a given party size. Use this once you and the user have a specific restaurant in mind (from search_restaurants, recommend_restaurants, or the user naming one directly) and need to know when they can actually be seated. Always get the exact slot string from this tool's result before booking -- never construct one yourself.

- recommend_restaurants: a ranked suggestion tool that already excludes anything without availability for the requested date and party size. Use this instead of search_restaurants when the user wants suggestions or doesn't have a specific place in mind -- e.g. "somewhere romantic for an anniversary" or "what's good nearby for a big group tonight". It takes the same optional filters as search_restaurants plus a freeform `preference` for vibe/mood, and an `exclude_restaurant_id` you should use for a "that one's no good, what else?" follow-up instead of re-deriving filters from scratch.

- create_reservation: books a table. Only call this once you have a specific restaurant, an exact slot string from check_availability (or recommend_restaurants' availability-gated results), a party size, and the guest's name and phone number. If it comes back with {"success": false, ...}, that is normal information, not a system error -- explain the reason to the user in plain language (e.g. the slot filled up, or they already have a booking there at that time) and help them pick another option; do not call it again with the exact same arguments expecting a different result.

- modify_reservation: changes the party size and/or slot of an existing reservation, identified by its reservation_id. Use this when a guest wants to change a booking they already have rather than cancelling and rebooking. Like create_reservation, a {"success": false} result is information to relay and act on, not a failure to retry blindly.

- cancel_reservation: cancels an existing reservation by its reservation_id. Confirm with the user which reservation they mean if there's any ambiguity before cancelling.

- escalate_to_human: hands the conversation off to a staff member. Use this when a request is outside what the other tools can do -- complaints, refund or policy questions, special accommodation requests you can't resolve, or when you've tried the reasonable tool calls and are still stuck. This tool cannot fail and always returns a fixed confirmation message; relay it as-is and don't imply a human has already responded in real time.

General guidance: ask the user for any required detail you don't have (date, party size, phone number, which restaurant) rather than guessing it. When a tool call fails for business reasons, explain what happened and offer a next step instead of silently retrying. When you have nothing further to check or ask, reply in plain natural language with no further tool calls."""


@dataclass
class ToolCall:
    """One tool invocation the model has requested."""

    name: str
    arguments: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class LLMTurn:
    """One model turn: either tool_calls to execute, or a final `content`
    reply -- never both. An empty `tool_calls` list means "final reply"."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    content: str | None = None


@dataclass
class TurnResult:
    """Outcome of run_turn: the updated message list plus a summary."""

    messages: list[dict]
    final_reply: str
    escalated: bool = False
    reason: str | None = None
    iterations: int = 0


class TransportExhausted(Exception):
    """A tool call kept raising across every retry attempt."""

    def __init__(self, tool_name: str, attempts: int, last_error: Exception | None):
        self.tool_name = tool_name
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"{tool_name!r} failed after {attempts} attempt(s): {last_error!r}")


# ---------------------------------------------------------------------------
# Production wiring: MCP tool schemas -> OpenAI/Groq function-calling format,
# and a call_tool that executes through a live agent/mcp_client.py session.
# Tests never use these -- they inject a ScriptedLLM and a fake call_tool
# instead (see tests/test_orchestrator.py), which is what makes every path
# below provable without a running MCP server or a GROQ_API_KEY.
# ---------------------------------------------------------------------------


def mcp_tools_to_openai_schemas(tools) -> list[dict]:
    """Convert MCP Tool objects (from ClientSession.list_tools().tools) into
    the OpenAI/Groq chat.completions `tools=[...]` schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools
    ]


def make_mcp_call_tool(session: ClientSession) -> Callable[[str, dict], Awaitable[dict]]:
    """Build the production `call_tool` callable: executes one tool through
    a live MCP ClientSession and parses its JSON result. Every mcp_server
    tool returns a JSON object as its text content (see server.py), so a
    normal return here is always that parsed dict -- including
    {"success": false, ...} bodies, which are NOT errors at this layer.

    Anything raised out of this function (a dropped subprocess, a
    malformed/non-JSON response, an MCP-level error result, a timeout) is a
    transport/execution failure for run_turn's retry logic.
    """

    async def call_tool(name: str, arguments: dict) -> dict:
        result = await session.call_tool(name, arguments)
        if result.isError:
            raise RuntimeError(f"MCP call_tool({name!r}) returned an error result: {result.content!r}")
        if not result.content:
            raise RuntimeError(f"MCP call_tool({name!r}) returned no content")
        return json.loads(result.content[0].text)

    return call_tool


def make_groq_llm_call(
    client: Groq, model: str = DEFAULT_GROQ_MODEL
) -> Callable[[list[dict], list[dict]], LLMTurn]:
    """Build an llm_call compatible with run_turn, backed by an already-
    constructed `groq.Groq` client via its OpenAI-compatible
    chat.completions API (see README.md's "Model resolution"). Not
    exercised by tests -- those use ScriptedLLM so every path is provable
    without a GROQ_API_KEY. Prefer build_groq_llm_call() below unless you
    need to supply your own already-configured client.
    """

    def llm_call(messages: list[dict], tools: list[dict]) -> LLMTurn:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        if message.tool_calls:
            calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]
            return LLMTurn(tool_calls=calls, content=None)
        return LLMTurn(tool_calls=[], content=message.content or "")

    return llm_call


def build_groq_llm_call(model: str = DEFAULT_GROQ_MODEL) -> Callable[[list[dict], list[dict]], LLMTurn]:
    """Load GROQ_API_KEY from .env (via python-dotenv, falling back to
    whatever's already in the environment) and build a real llm_call
    backed by Groq's chat.completions API. Call this once per process --
    the returned callable is exactly what run_turn expects.

    Raises RuntimeError if no GROQ_API_KEY is found anywhere, so a missing
    key fails loudly at startup instead of surfacing as a confusing auth
    error deep inside the first agentic turn.
    """
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and fill it "
            "in, or export GROQ_API_KEY in your shell."
        )
    return make_groq_llm_call(Groq(api_key=api_key), model=model)


def new_conversation(user_message: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict]:
    """Build the initial messages list for a fresh conversation turn."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


# ---------------------------------------------------------------------------
# The agentic loop itself.
# ---------------------------------------------------------------------------


def _assistant_tool_call_message(tool_calls: list[ToolCall]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in tool_calls
        ],
    }


def _tool_result_message(tc: ToolCall, result: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tc.id,
        "name": tc.name,
        "content": json.dumps(result),
    }


async def _call_tool_with_retries(
    call_tool: Callable[[str, dict], Awaitable[dict]],
    name: str,
    arguments: dict,
    max_retries: int,
    backoff_base_seconds: float,
    sleep_fn: Callable[[float], Awaitable[None]],
) -> dict:
    """Execute one tool call, retrying ONLY if `call_tool` raises (a
    transport/execution failure). A call that returns normally -- including
    a {"success": false, ...} body -- is returned immediately on the first
    attempt; that path never enters the retry loop at all.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return await call_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001 -- any transport/execution failure is retryable
            last_error = exc
            if attempt < max_retries:
                delay = backoff_base_seconds * (2 ** (attempt - 1))
                if delay > 0:
                    await sleep_fn(delay)
    raise TransportExhausted(name, max_retries, last_error)


async def _force_escalation(
    call_tool: Callable[[str, dict], Awaitable[dict]],
    messages: list[dict],
    reason: str,
    max_retries: int,
    backoff_base_seconds: float,
    sleep_fn: Callable[[float], Awaitable[None]],
    iterations: int,
) -> TurnResult:
    """Call escalate_to_human directly in code -- bypassing the model
    entirely -- and use ITS returned message as the final reply. Asking the
    model to decide (or to phrase the final message) here would reopen
    exactly the risk this path exists to close off.
    """
    fallback_reply = (
        "We're having trouble completing this right now and our escalation "
        "path is also unreachable. Please try again shortly or contact us "
        "directly."
    )
    try:
        result = await _call_tool_with_retries(
            call_tool, "escalate_to_human", {"reason": reason}, max_retries, backoff_base_seconds, sleep_fn
        )
        final_reply = result.get("message") or fallback_reply
    except TransportExhausted:
        final_reply = fallback_reply

    messages.append({"role": "assistant", "content": final_reply})
    return TurnResult(messages=messages, final_reply=final_reply, escalated=True, reason=reason, iterations=iterations)


async def run_turn(
    messages: list[dict],
    llm_call: Callable[[list[dict], list[dict]], LLMTurn],
    call_tool: Callable[[str, dict], Awaitable[dict]],
    tool_schemas: list[dict],
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> TurnResult:
    """Run one agentic turn to completion: call the model, execute whatever
    tool(s) it requests, feed results back, and repeat until the model
    replies with no tool calls -- or until a hard limit forces escalation.

    `messages` is mutated in place (assistant/tool messages are appended as
    the loop runs) and returned as part of the result, so the caller can
    feed it straight into the next turn.

    An "iteration" is one round in which the model was consulted and asked
    for tool(s) -- NOT one per individual tool call within a round (a
    single round may request several tools; that still counts as one
    iteration). `max_iterations` bounds how many such rounds can happen
    before a final reply is required; hitting it forces escalation instead
    of consulting the model again.
    """
    iterations = 0
    while True:
        if iterations >= max_iterations:
            return await _force_escalation(
                call_tool,
                messages,
                reason=f"gave up after {max_iterations} tool-call iterations without a final reply",
                max_retries=max_transport_retries,
                backoff_base_seconds=backoff_base_seconds,
                sleep_fn=sleep_fn,
                iterations=iterations,
            )

        try:
            turn = llm_call(messages, tool_schemas)
        except Exception as exc:  # noqa: BLE001 -- any provider-level failure, not retried
            # Deliberately no retry here (unlike call_tool transport
            # failures): a rate limit's suggested wait is tens of minutes,
            # useless mid-conversation, and retrying against one just makes
            # it worse. Straight to the same forced-escalation path used
            # everywhere else, on the first failure.
            return await _force_escalation(
                call_tool,
                messages,
                reason=f"the model request itself failed: {exc!r}",
                max_retries=max_transport_retries,
                backoff_base_seconds=backoff_base_seconds,
                sleep_fn=sleep_fn,
                iterations=iterations,
            )

        if not turn.tool_calls:
            final_reply = turn.content or ""
            messages.append({"role": "assistant", "content": final_reply})
            return TurnResult(messages=messages, final_reply=final_reply, escalated=False, iterations=iterations)

        iterations += 1
        messages.append(_assistant_tool_call_message(turn.tool_calls))

        for tc in turn.tool_calls:
            try:
                result = await _call_tool_with_retries(
                    call_tool, tc.name, tc.arguments, max_transport_retries, backoff_base_seconds, sleep_fn
                )
            except TransportExhausted as exc:
                if tc.name == "escalate_to_human":
                    # Retries for escalate_to_human itself already ran inside
                    # _call_tool_with_retries above -- don't recurse into a
                    # second round of retries for it in _force_escalation.
                    fallback_reply = (
                        "We're having trouble reaching our escalation system "
                        "right now. Please try again shortly or contact us "
                        "directly."
                    )
                    messages.append({"role": "assistant", "content": fallback_reply})
                    return TurnResult(
                        messages=messages, final_reply=fallback_reply, escalated=True, reason=str(exc)
                    )
                return await _force_escalation(
                    call_tool,
                    messages,
                    reason=f"{tc.name} failed after {max_transport_retries} attempts: {exc.last_error!r}",
                    max_retries=max_transport_retries,
                    backoff_base_seconds=backoff_base_seconds,
                    sleep_fn=sleep_fn,
                    iterations=iterations,
                )
            messages.append(_tool_result_message(tc, result))
