"""Manual, real-API conversation trace for GoodFoods.

THROWAWAY SCRIPT -- deliberately NOT part of the pytest suite. pytest only
discovers `test_*.py` / `*_test.py`; this file matches neither pattern on
purpose, so it never runs in CI or on a normal `pytest` invocation. Running
it makes real calls to the Groq API and spends real tokens. Run it
explicitly:

    python tests/manual_conversations.py

It drives agent.orchestrator.run_turn() with the REAL build_groq_llm_call()
(actual Groq chat.completions calls, model=orchestrator.DEFAULT_GROQ_MODEL --
see README.md "Model resolution") and the REAL call_tool wired to a live
mcp_server/server.py subprocess via
agent/mcp_client.py -- NOT the M5 fake-LLM test doubles. Every tool
decision the model makes here is its own; nothing in this script or in
agent/orchestrator.py routes on keywords in the user's message.

For each of 5 scenarios it prints every (tool_name, arguments, result)
triple in the order it actually happened, plus the model's final reply.
Requires GROQ_API_KEY (see .env.example / README.md Setup).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil

from agent.mcp_client import mcp_session
from agent.orchestrator import (
    build_groq_llm_call,
    make_mcp_call_tool,
    mcp_tools_to_openai_schemas,
    new_conversation,
    run_turn,
)
from mcp_server.pos_store import PosStore
from mcp_server.seed_data import seed_store

SCENARIO_RESULTS: list[tuple[str, "object"]] = []


# ---------------------------------------------------------------------------
# Tool-call logging: wraps the REAL call_tool (from make_mcp_call_tool) so
# every (name, arguments, result-or-error) triple is recorded in the order
# it actually happened. This is observation only -- it changes nothing
# about what run_turn does.
# ---------------------------------------------------------------------------


def make_logging_call_tool(inner_call_tool, log: list):
    async def call_tool(name: str, arguments: dict) -> dict:
        try:
            result = await inner_call_tool(name, arguments)
        except Exception as exc:  # record the failed attempt, then let it propagate
            log.append({"tool": name, "arguments": arguments, "error": repr(exc)})
            raise
        log.append({"tool": name, "arguments": arguments, "result": result})
        return result

    return call_tool


def print_trace(label: str, log: list, result) -> None:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    if not log:
        print("(no tool calls)")
    for i, entry in enumerate(log, 1):
        print(f"\n[{i}] tool_name = {entry['tool']}")
        print(f"    arguments = {json.dumps(entry['arguments'])}")
        if "result" in entry:
            print(f"    result    = {json.dumps(entry['result'])}")
        else:
            print(f"    ERROR     = {entry['error']}")
    print(f"\nFINAL REPLY: {result.final_reply}")
    print(f"[escalated={result.escalated} iterations={result.iterations} reason={result.reason}]")
    SCENARIO_RESULTS.append((label, result))


# ---------------------------------------------------------------------------
# Scenario 5 support: actually kill the real MCP server subprocess (found
# via psutil, exactly as verified in M0) mid-conversation, rather than
# simulating a transport error. Nothing about run_turn's retry/escalation
# logic is faked here.
# ---------------------------------------------------------------------------


def _kill_mcp_server_subprocess() -> bool:
    me = psutil.Process(os.getpid())
    for child in me.children(recursive=True):
        try:
            cmdline = " ".join(child.cmdline())
        except psutil.Error:
            continue
        if "server.py" in cmdline:
            print(f"    [!] killing real MCP server subprocess pid={child.pid}")
            child.kill()
            try:
                child.wait(timeout=5)
            except psutil.TimeoutExpired:
                pass
            return True
    return False


def make_connection_killer_call_tool(inner_call_tool, kill_before_call: int):
    """Wraps a REAL call_tool. Right before the Nth invocation (counting
    every attempt, including retries), it actually kills the real MCP
    server subprocess, so that call and every retry of it genuinely fail
    against a dead process -- not a simulated exception. A generous
    per-attempt timeout guards against the call hanging instead of raising
    once the process is gone.
    """
    count = 0
    killed = False

    async def call_tool(name: str, arguments: dict) -> dict:
        nonlocal count, killed
        count += 1
        if count == kill_before_call and not killed:
            killed = True
            found = _kill_mcp_server_subprocess()
            print(f"    [!] subprocess kill {'succeeded' if found else 'FAILED (not found)'}")
            await asyncio.sleep(0.3)
        return await asyncio.wait_for(inner_call_tool(name, arguments), timeout=10)

    return call_tool


# ---------------------------------------------------------------------------
# Fixture setup for scenario 2: a restaurant fully booked on a specific
# date, set up directly through pos_store (not through the LLM -- this is
# scenario preparation, not part of the traced conversation).
# ---------------------------------------------------------------------------

FULLY_BOOKED_DATE = "2026-11-11"


def ensure_seeded_and_pick_fully_booked_restaurant() -> dict:
    store = PosStore()
    try:
        if not store.list_restaurants():
            print("Seeding goodfoods.db (empty) ...")
            seed_store(store)

        target = min(store.list_restaurants(), key=lambda r: r.capacity)
        already_full = all(
            not store.check_availability(target.id, f"{FULLY_BOOKED_DATE} {hour:02d}:00", 1)
            for hour in (11, 13, 15, 17, 19, 21)
        )
        if already_full:
            print(
                f"{target.name!r} (id={target.id}) is already fully booked on "
                f"{FULLY_BOOKED_DATE} from a previous run -- skipping fixture setup."
            )
        else:
            print(
                f"Fully booking {target.name!r} (id={target.id}, capacity={target.capacity}) "
                f"on every {FULLY_BOOKED_DATE} slot ..."
            )
            for hour in (11, 13, 15, 17, 19, 21):
                slot = f"{FULLY_BOOKED_DATE} {hour:02d}:00"
                store.create_reservation(
                    restaurant_id=target.id,
                    customer_name="Fixture Filler",
                    customer_phone=f"fixture-{hour}",
                    party_size=target.capacity,
                    slot=slot,
                )
        return {
            "id": target.id,
            "name": target.name,
            "capacity": target.capacity,
            "cuisine": target.cuisine,
            "location": target.location,
        }
    finally:
        store.close()


def ensure_seeded() -> None:
    store = PosStore()
    try:
        if not store.list_restaurants():
            print("Seeding goodfoods.db (empty) ...")
            seed_store(store)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


async def scenario_1_and_3_and_4_booking_modify_cancel(llm_call) -> None:
    """One continuous conversation: (1) happy-path booking, then (3) modify
    it to a different slot, then (4) cancel it. modify/cancel need the
    reservation_id from the booking step, which the model can only get by
    having seen it earlier in this SAME conversation's tool results -- so
    these three scenarios share one message history on purpose.
    """
    async with mcp_session() as session:
        tools = (await session.list_tools()).tools
        schemas = mcp_tools_to_openai_schemas(tools)
        real_call_tool = make_mcp_call_tool(session)

        messages = new_conversation(
            "Hi! I'd like to book a table for 2 people at an Italian restaurant "
            "on 2026-10-05, around 7pm. My name is Alex Doe and my phone number "
            "is 555-0100. I don't have a preference beyond that -- please just "
            "pick whichever Italian place has a good rating and is free at that "
            "time, and confirm the booking."
        )

        log: list = []
        result = await run_turn(messages, llm_call, make_logging_call_tool(real_call_tool, log), schemas)
        print_trace("SCENARIO 1: happy-path booking -> confirmed reservation", log, result)
        messages = result.messages

        messages.append(
            {"role": "user", "content": "Actually, can you move that reservation to 9pm instead?"}
        )
        log = []
        result = await run_turn(messages, llm_call, make_logging_call_tool(real_call_tool, log), schemas)
        print_trace("SCENARIO 3: modify existing booking to a different slot", log, result)
        messages = result.messages

        messages.append(
            {"role": "user", "content": "On second thought, please just cancel that reservation entirely."}
        )
        log = []
        result = await run_turn(messages, llm_call, make_logging_call_tool(real_call_tool, log), schemas)
        print_trace("SCENARIO 4: cancel the booking", log, result)


async def scenario_2_fully_booked_recommend_fallback(llm_call, fully_booked: dict) -> None:
    """Ask for a restaurant known (via direct pos_store setup above) to be
    fully booked on FULLY_BOOKED_DATE. Expect the model to discover it's
    unavailable and -- unprompted, per its own system-prompt guidance about
    recommend_restaurants' exclude_restaurant_id -- fall back to a real
    available alternative on its own.
    """
    async with mcp_session() as session:
        tools = (await session.list_tools()).tools
        schemas = mcp_tools_to_openai_schemas(tools)
        real_call_tool = make_mcp_call_tool(session)

        messages = new_conversation(
            f"I'd like to book a table at {fully_booked['name']} -- it's the "
            f"{fully_booked['cuisine']} place in {fully_booked['location']} -- for "
            f"{fully_booked['capacity']} people on {FULLY_BOOKED_DATE} around 7pm. My name "
            "is Jordan Lee, phone 555-0200. If they genuinely can't seat us at all that day, "
            "please suggest another good option nearby instead of just telling me no."
        )

        log: list = []
        result = await run_turn(messages, llm_call, make_logging_call_tool(real_call_tool, log), schemas)
        print_trace(
            f"SCENARIO 2: {fully_booked['name']!r} is fully booked on {FULLY_BOOKED_DATE} "
            "-> expect an unprompted recommend_restaurants(exclude_restaurant_id=...) fallback",
            log,
            result,
        )

        called_recommend_with_exclude = any(
            e.get("tool") == "recommend_restaurants"
            and e.get("arguments", {}).get("exclude_restaurant_id") == fully_booked["id"]
            for e in log
        )
        print(
            f"\n[check] recommend_restaurants called with exclude_restaurant_id="
            f"{fully_booked['id']}: {called_recommend_with_exclude}"
        )


async def scenario_5_broken_connection_mid_conversation(llm_call) -> None:
    """Kill the real MCP server subprocess mid-conversation (after at least
    one real tool call has already succeeded) and confirm run_turn's real
    retry-then-forced-escalation path fires with the real model in the
    loop -- not the M5 fake-LLM version of this test.
    """
    async with mcp_session() as session:
        tools = (await session.list_tools()).tools
        schemas = mcp_tools_to_openai_schemas(tools)
        real_call_tool = make_mcp_call_tool(session)
        killer_call_tool = make_connection_killer_call_tool(real_call_tool, kill_before_call=2)

        messages = new_conversation(
            "Please search for Mexican restaurants, then check 2026-09-05's "
            "availability for the first one you find, for a party of 4."
        )

        log: list = []
        result = await run_turn(messages, llm_call, make_logging_call_tool(killer_call_tool, log), schemas)
        print_trace(
            "SCENARIO 5: MCP server subprocess killed mid-conversation "
            "-> expect retry x3 then forced escalate_to_human",
            log,
            result,
        )


async def main() -> None:
    print("Ensuring goodfoods.db is seeded before any conversation starts ...")
    ensure_seeded()
    fully_booked = ensure_seeded_and_pick_fully_booked_restaurant()

    llm_call = build_groq_llm_call()

    await scenario_1_and_3_and_4_booking_modify_cancel(llm_call)
    await scenario_2_fully_booked_recommend_fallback(llm_call, fully_booked)
    await scenario_5_broken_connection_mid_conversation(llm_call)

    print(f"\n{'#' * 78}\nSUMMARY\n{'#' * 78}")
    for label, result in SCENARIO_RESULTS:
        print(f"- {label.splitlines()[0]}: escalated={result.escalated} iterations={result.iterations}")


if __name__ == "__main__":
    asyncio.run(main())
