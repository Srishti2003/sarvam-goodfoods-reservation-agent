"""Streamlit chat UI for GoodFoods.

Every user message goes through agent.orchestrator.run_turn() exactly as
tests/manual_conversations.py drives it: the real build_groq_llm_call()
and a real call_tool wired to a live mcp_server/server.py subprocess via
agent/mcp_client.py. Nothing here decides which tool to call -- that's
entirely the model's job, same as everywhere else in this project.

Streamlit reruns this whole script top-to-bottom on every interaction, so
two different kinds of state need two different homes:
  - conversation history -> st.session_state (per browser session; each
    visitor gets their own conversation)
  - the MCP client/subprocess and the Groq llm_call -> st.cache_resource
    (process-wide; created once and reused by every rerun and every
    session, so we never reconnect the subprocess or reload the API key
    per message)
"""

import asyncio
import json
import sys
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.mcp_client import mcp_session
from agent.orchestrator import (
    DEFAULT_SYSTEM_PROMPT,
    build_groq_llm_call,
    make_mcp_call_tool,
    mcp_tools_to_openai_schemas,
    run_turn,
)


# ---------------------------------------------------------------------------
# Async plumbing: one persistent event loop in a background thread, so the
# MCP session's subprocess (and its internal anyio task group) lives for
# the whole app process, not just one Streamlit rerun -- and so this
# synchronous script can still drive async run_turn() calls on it.
# ---------------------------------------------------------------------------


class LoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


@dataclass
class MCPBackend:
    # `cm` and `session` are kept here, referenced, even though nothing
    # calls methods on them directly after setup -- mcp_session()'s context
    # manager owns the anyio task group driving the subprocess's
    # reader/writer tasks, entered manually below via __aenter__ with no
    # matching __aexit__. If this object were the only thing NOT holding a
    # reference to `cm`, it would be garbage-collected once _connect()
    # returns, tearing the streams down under the live session (this was
    # tried without `cm` stored here first, and did reproduce exactly that
    # as a live ClosedResourceError on the first real tool call).
    cm: object
    session: object
    call_tool: Callable[[str, dict], Awaitable[dict]]
    schemas: list[dict]


@st.cache_resource
def get_loop_thread() -> LoopThread:
    return LoopThread()


@st.cache_resource
def get_mcp_backend() -> MCPBackend:
    """Start mcp_server/server.py as a subprocess and hand back a ready
    call_tool + tool schemas. The session is entered manually (not via
    `async with`) and deliberately never exited -- it needs to outlive
    this function call and every future Streamlit rerun, for the life of
    the app process.
    """
    loop_thread = get_loop_thread()

    async def _connect():
        cm = mcp_session()
        session = await cm.__aenter__()
        tools = (await session.list_tools()).tools
        return MCPBackend(
            cm=cm,
            session=session,
            call_tool=make_mcp_call_tool(session),
            schemas=mcp_tools_to_openai_schemas(tools),
        )

    return loop_thread.run(_connect())


@st.cache_resource
def get_llm_call():
    return build_groq_llm_call()


def make_logging_call_tool(inner_call_tool, log: list):
    """Wraps call_tool to record every (name, arguments, result-or-error)
    triple for THIS turn, in order -- purely for display in the trace
    expander; changes nothing about what run_turn does."""

    async def call_tool(name: str, arguments: dict) -> dict:
        try:
            result = await inner_call_tool(name, arguments)
        except Exception as exc:
            log.append({"tool": name, "arguments": arguments, "error": repr(exc)})
            raise
        log.append({"tool": name, "arguments": arguments, "result": result})
        return result

    return call_tool


def render_trace(trace: list[dict]) -> None:
    with st.expander(f"🔧 Tool calls ({len(trace)})", expanded=False):
        if not trace:
            st.caption("No tool calls this turn.")
        for i, entry in enumerate(trace, 1):
            st.markdown(f"**{i}. `{entry['tool']}`**")
            st.json(entry["arguments"])
            if "result" in entry:
                st.json(entry["result"])
            else:
                st.code(entry["error"], language="text")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="GoodFoods", page_icon="🍽️")
st.title("GoodFoods")
st.caption("Restaurant reservations, end to end through a real MCP tool-calling agent.")

try:
    backend = get_mcp_backend()
    llm_call = get_llm_call()
except Exception as exc:
    st.error(f"Couldn't start up: {exc}")
    st.stop()

if "protocol_messages" not in st.session_state:
    st.session_state.protocol_messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
if "display_turns" not in st.session_state:
    st.session_state.display_turns = []

with st.sidebar:
    st.caption(f"{len(backend.schemas)} MCP tools connected")
    if st.button("New conversation"):
        st.session_state.protocol_messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        st.session_state.display_turns = []
        st.rerun()

for turn in st.session_state.display_turns:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        # The escalated/fallback path is rendered exactly like any other
        # reply -- from the user's side it's just what the agent said,
        # not a special error state.
        st.markdown(turn["reply"])
        render_trace(turn["trace"])

if prompt := st.chat_input("Book a table, check availability, ask for a recommendation..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.protocol_messages.append({"role": "user", "content": prompt})

    log: list = []
    logged_call_tool = make_logging_call_tool(backend.call_tool, log)

    with st.chat_message("assistant"):
        with st.spinner("Working on it..."):
            loop_thread = get_loop_thread()
            result = loop_thread.run(
                run_turn(st.session_state.protocol_messages, llm_call, logged_call_tool, backend.schemas)
            )
        st.session_state.protocol_messages = result.messages
        st.markdown(result.final_reply)
        render_trace(log)

    st.session_state.display_turns.append({"user": prompt, "reply": result.final_reply, "trace": log})
