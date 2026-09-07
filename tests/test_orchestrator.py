import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.orchestrator import LLMTurn, ToolCall, run_turn

# ---------------------------------------------------------------------------
# Test doubles. Both the fake LLM and the fake call_tool append to the SAME
# shared `log` list, so tests can assert on the actual interleaved sequence
# of "the model was consulted" / "this tool was attempted" events, not just
# on each side's own count.
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Returns one pre-scripted LLMTurn per call, in order."""

    def __init__(self, script: list[LLMTurn], log: list):
        self._script = list(script)
        self._log = log

    def __call__(self, messages: list[dict], tools: list[dict]) -> LLMTurn:
        self._log.append(("llm_call",))
        if not self._script:
            raise AssertionError("ScriptedLLM script exhausted -- test provided too few turns")
        return self._script.pop(0)


class RepeatingToolCallLLM:
    """Always asks for the same tool call again -- never converges on a
    final reply. Used to exercise the iteration cap."""

    def __init__(self, tool_name: str, arguments: dict, log: list):
        self._tool_name = tool_name
        self._arguments = arguments
        self._log = log

    def __call__(self, messages: list[dict], tools: list[dict]) -> LLMTurn:
        self._log.append(("llm_call",))
        return LLMTurn(tool_calls=[ToolCall(name=self._tool_name, arguments=self._arguments)])


class RaisingLLM:
    """Raises on its first call, simulating a provider-level failure (e.g.
    a real groq.RateLimitError) from the llm_call step itself -- distinct
    from a call_tool transport failure."""

    def __init__(self, exc: Exception, log: list):
        self._exc = exc
        self._log = log

    def __call__(self, messages: list[dict], tools: list[dict]) -> LLMTurn:
        self._log.append(("llm_call",))
        raise self._exc


def make_fake_call_tool(behaviors: dict, log: list):
    """behaviors: {tool_name: fn(attempt_number) -> dict | raises}."""
    counts: dict[str, int] = {}

    async def call_tool(name: str, arguments: dict) -> dict:
        counts[name] = counts.get(name, 0) + 1
        attempt = counts[name]
        log.append(("tool_call", name, attempt))
        return behaviors[name](attempt)

    return call_tool


def tool_call_entries(log, name=None):
    return [e for e in log if e[0] == "tool_call" and (name is None or e[1] == name)]


def llm_call_count(log):
    return sum(1 for e in log if e[0] == "llm_call")


async def instant_sleep(_seconds: float) -> None:
    """No-op stand-in for asyncio.sleep so backoff tests run instantly."""


# ---------------------------------------------------------------------------
# (a) Normal multi-tool-call resolution ending in a final reply.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_resolution_ends_in_final_reply():
    log: list = []

    script = [
        LLMTurn(tool_calls=[ToolCall(name="search_restaurants", arguments={"cuisine": "Italian"}, id="c1")]),
        LLMTurn(
            tool_calls=[
                ToolCall(
                    name="check_availability",
                    arguments={"restaurant_id": 42, "date": "2026-09-10", "party_size": 2},
                    id="c2",
                )
            ]
        ),
        LLMTurn(tool_calls=[], content="Found you a table for 2 at Test Bistro on 2026-09-10 at 19:00."),
    ]
    behaviors = {
        "search_restaurants": lambda attempt: {
            "success": True,
            "count": 1,
            "restaurants": [{"id": 42, "name": "Test Bistro", "cuisine": "Italian"}],
        },
        "check_availability": lambda attempt: {
            "success": True,
            "restaurant_id": 42,
            "slots": [{"slot": "2026-09-10 19:00", "time": "19:00", "available": True}],
        },
    }

    result = await run_turn(
        messages=[{"role": "user", "content": "Book me an Italian place for 2 tonight"}],
        llm_call=ScriptedLLM(script, log),
        call_tool=make_fake_call_tool(behaviors, log),
        tool_schemas=[],
        sleep_fn=instant_sleep,
    )

    assert log == [
        ("llm_call",),
        ("tool_call", "search_restaurants", 1),
        ("llm_call",),
        ("tool_call", "check_availability", 1),
        ("llm_call",),
    ]
    assert result.escalated is False
    assert result.iterations == 2
    assert result.final_reply == "Found you a table for 2 at Test Bistro on 2026-09-10 at 19:00."

    # a business-logic failure in the mix must NOT be retried and must NOT
    # short-circuit the loop -- it's just handed back to the model.
    assert result.messages[2]["role"] == "tool"
    assert '"success": true' in result.messages[2]["content"]


@pytest.mark.asyncio
async def test_business_logic_failure_is_passed_through_not_retried():
    """{"success": false, ...} from a tool that returned normally must
    reach the model as-is, on the first attempt, with no retry."""
    log: list = []

    script = [
        LLMTurn(tool_calls=[ToolCall(name="create_reservation", arguments={"restaurant_id": 1}, id="c1")]),
        LLMTurn(tool_calls=[], content="That slot's full -- want me to check another time?"),
    ]
    behaviors = {
        "create_reservation": lambda attempt: {"success": False, "error": "capacity exceeded"},
    }

    result = await run_turn(
        messages=[{"role": "user", "content": "book it"}],
        llm_call=ScriptedLLM(script, log),
        call_tool=make_fake_call_tool(behaviors, log),
        tool_schemas=[],
        sleep_fn=instant_sleep,
    )

    # exactly ONE attempt at the failing tool -- a business failure is not a
    # transport failure and must never trigger the retry loop.
    assert tool_call_entries(log, "create_reservation") == [("tool_call", "create_reservation", 1)]
    assert result.escalated is False
    assert result.final_reply == "That slot's full -- want me to check another time?"


# ---------------------------------------------------------------------------
# (b) Transport error: fails twice, succeeds on the 3rd attempt.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transport_error_retries_then_succeeds():
    log: list = []

    script = [
        LLMTurn(tool_calls=[ToolCall(name="check_availability", arguments={"restaurant_id": 7}, id="c1")]),
        LLMTurn(tool_calls=[], content="Got it, here are your options."),
    ]

    def flaky_check_availability(attempt: int) -> dict:
        if attempt < 3:
            raise ConnectionError(f"dropped subprocess (attempt {attempt})")
        return {"success": True, "slots": []}

    behaviors = {"check_availability": flaky_check_availability}

    result = await run_turn(
        messages=[{"role": "user", "content": "what's open?"}],
        llm_call=ScriptedLLM(script, log),
        call_tool=make_fake_call_tool(behaviors, log),
        tool_schemas=[],
        sleep_fn=instant_sleep,
    )

    assert tool_call_entries(log, "check_availability") == [
        ("tool_call", "check_availability", 1),
        ("tool_call", "check_availability", 2),
        ("tool_call", "check_availability", 3),
    ]
    # the model must NOT be re-consulted between retries -- only once before
    # the tool call, and once after it finally succeeds.
    assert llm_call_count(log) == 2
    assert tool_call_entries(log, "escalate_to_human") == []
    assert result.escalated is False
    assert result.final_reply == "Got it, here are your options."


# ---------------------------------------------------------------------------
# (c) Transport error: fails all 3 attempts -> forced escalate_to_human.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transport_error_exhausted_forces_escalation():
    log: list = []

    # Only one scripted turn: if the model were consulted again after the
    # transport failure, ScriptedLLM would raise "script exhausted" and the
    # test would fail -- proving escalation truly bypasses the model.
    script = [
        LLMTurn(tool_calls=[ToolCall(name="check_availability", arguments={"restaurant_id": 7}, id="c1")]),
    ]

    def always_fails(attempt: int) -> dict:
        raise ConnectionError(f"dropped subprocess (attempt {attempt})")

    behaviors = {
        "check_availability": always_fails,
        "escalate_to_human": lambda attempt: {
            "success": True,
            "message": "This has been escalated to a member of our team, who will follow up with you shortly.",
        },
    }

    result = await run_turn(
        messages=[{"role": "user", "content": "what's open?"}],
        llm_call=ScriptedLLM(script, log),
        call_tool=make_fake_call_tool(behaviors, log),
        tool_schemas=[],
        sleep_fn=instant_sleep,
    )

    assert tool_call_entries(log, "check_availability") == [
        ("tool_call", "check_availability", 1),
        ("tool_call", "check_availability", 2),
        ("tool_call", "check_availability", 3),
    ]
    assert tool_call_entries(log, "escalate_to_human") == [("tool_call", "escalate_to_human", 1)]
    # the model is consulted exactly once -- the decision to escalate is
    # made in code, never by asking the model "what do you want to do".
    assert llm_call_count(log) == 1
    assert result.escalated is True
    assert "check_availability" in result.reason
    assert result.final_reply == "This has been escalated to a member of our team, who will follow up with you shortly."


# ---------------------------------------------------------------------------
# llm_call itself raises (provider-level failure, e.g. a rate limit) ->
# straight to forced escalation, no retry -- distinct from (b)/(c) above,
# which are about call_tool failing, not the model request itself.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_call_failure_forces_escalation_without_retry():
    log: list = []

    llm = RaisingLLM(RuntimeError("rate limit exceeded, retry in 30m"), log)
    behaviors = {
        "escalate_to_human": lambda attempt: {
            "success": True,
            "message": "This has been escalated to a member of our team, who will follow up with you shortly.",
        },
    }

    result = await run_turn(
        messages=[{"role": "user", "content": "book me a table"}],
        llm_call=llm,
        call_tool=make_fake_call_tool(behaviors, log),
        tool_schemas=[],
        sleep_fn=instant_sleep,
    )

    # the model is consulted exactly once -- its own failure is not retried,
    # unlike a call_tool transport failure.
    assert llm_call_count(log) == 1
    assert tool_call_entries(log, "escalate_to_human") == [("tool_call", "escalate_to_human", 1)]
    assert result.escalated is True
    assert "rate limit exceeded" in result.reason
    assert result.final_reply == "This has been escalated to a member of our team, who will follow up with you shortly."

    # exact end-to-end ordering: one failed llm_call, straight to escalation.
    assert log == [
        ("llm_call",),
        ("tool_call", "escalate_to_human", 1),
    ]


# ---------------------------------------------------------------------------
# (d) Model never stops calling tools -> iteration cap forces escalation.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iteration_cap_forces_escalation():
    log: list = []
    max_iterations = 3

    llm = RepeatingToolCallLLM("ping", {}, log)
    behaviors = {
        "ping": lambda attempt: {"success": True, "message": "pong"},
        "escalate_to_human": lambda attempt: {
            "success": True,
            "message": "This has been escalated to a member of our team, who will follow up with you shortly.",
        },
    }

    result = await run_turn(
        messages=[{"role": "user", "content": "help"}],
        llm_call=llm,
        call_tool=make_fake_call_tool(behaviors, log),
        tool_schemas=[],
        max_iterations=max_iterations,
        sleep_fn=instant_sleep,
    )

    # the model is consulted exactly max_iterations times -- never a
    # (max_iterations + 1)-th time, since the cap check happens BEFORE the
    # next model call, not after.
    assert llm_call_count(log) == max_iterations
    assert tool_call_entries(log, "ping") == [
        ("tool_call", "ping", 1),
        ("tool_call", "ping", 2),
        ("tool_call", "ping", 3),
    ]
    assert tool_call_entries(log, "escalate_to_human") == [("tool_call", "escalate_to_human", 1)]
    assert result.escalated is True
    assert result.iterations == max_iterations
    assert str(max_iterations) in result.reason
    assert result.final_reply == "This has been escalated to a member of our team, who will follow up with you shortly."

    # exact end-to-end ordering: 3x(llm, ping) then the forced escalation.
    assert log == [
        ("llm_call",),
        ("tool_call", "ping", 1),
        ("llm_call",),
        ("tool_call", "ping", 2),
        ("llm_call",),
        ("tool_call", "ping", 3),
        ("tool_call", "escalate_to_human", 1),
    ]
