# GoodFoods AI Reservation Agent

Restaurant-reservation conversational agent for a 50-100 location chain,
built on the official MCP Python SDK, a hand-written LLM tool-calling loop
(no agent framework), and a SQLite-backed reservation store.

## Demo Video

DEMO VIDEO: [`docs/demo/sarvam_assignment_demo.mp4`](docs/demo/sarvam_assignment_demo.mp4)

Walks through restaurant discovery/recommendation, a live booking, a
modification, and a cancellation against the running app, with the
tool-call trace shown for each step.

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent / Tool-Calling Flow](#agent--tool-calling-flow)
- [MCP Architecture](#mcp-architecture)
- [Model](#model)
- [Prompt Engineering Approach](#prompt-engineering-approach)
- [Tools](#tools)
- [Recommendation System](#recommendation-system)
- [Reservation Logic](#reservation-logic)
- [Error Handling](#error-handling)
- [Example Conversations](#example-conversations)
- [Business Strategy Summary](#business-strategy-summary)
- [Assumptions](#assumptions)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Layout](#layout)
- [Setup](#setup)
- [Testing](#testing)

## Overview

GoodFoods needs a conversational agent that can actually **do** something
about a reservation, not just answer questions about one. This project is
a chat agent that finds restaurants, checks live availability, and books,
modifies, or cancels reservations against a real (SQLite-backed) store —
end to end, through natural conversation.

**Why an agent, not a chatbot.** A chatbot answers questions from a
knowledge base. This system is *action-oriented*: it reads and writes
directly to the reservation store via tool calls, the same way a human
host would use a booking system, and the store — not the language model —
is the source of truth for what's actually booked. The LLM never states
availability or reservation details from memory; every fact it gives a
user came back from a tool call in that same conversation. That
distinction (retrieval vs. safe transactional execution — validation,
duplicate detection, rollback-safe failure) is also the core of this
project's business case; see
[Business Strategy Summary](#business-strategy-summary).

## Architecture

```mermaid
flowchart TD
    User(["User"]) --> UI["Streamlit UI<br/>frontend/streamlit_app.py"]
    UI --> Orch["Agent Orchestrator<br/>agent/orchestrator.py — run_turn()"]
    Orch -->|messages + tool schemas| LLM["Groq LLM<br/>openai/gpt-oss-20b"]
    LLM -->|tool_calls OR final reply| Orch
    Orch -->|call_tool(name, arguments)| Client["MCP Client<br/>agent/mcp_client.py — ClientSession"]
    Client <-->|stdio, separate OS subprocess| Server["MCP Server<br/>mcp_server/server.py — FastMCP"]
    Server --> Tools["8 MCP tools<br/>search / check / recommend / create /<br/>modify / cancel / escalate / ping"]
    Tools --> Store["PosStore<br/>mcp_server/pos_store.py"]
    Store <--> DB[("SQLite<br/>goodfoods.db")]
    Orch -->|final reply + tool trace| UI
    UI --> User
```

- **Streamlit UI** — captures the user's message, renders the agent's
  reply and a collapsible tool-call trace, holds per-browser-session
  conversation history.
- **Agent Orchestrator** — the hand-written agentic loop: calls the LLM,
  executes whatever tool(s) it asks for, feeds results back, repeats
  until a final reply or a hard limit forces human escalation.
- **Groq LLM** — the only component that decides *which* tool to call, if
  any; see [Prompt Engineering Approach](#prompt-engineering-approach).
- **MCP Client / Server** — the official MCP Python SDK over stdio; see
  [MCP Architecture](#mcp-architecture).
- **PosStore / SQLite** — the actual source of truth for restaurants and
  reservations, independent of any LLM or Python process's lifetime.

## Agent / Tool-Calling Flow

Exactly what happens for one user message, in order:

```
user message
  → LLM (sees full conversation + live MCP tool schemas)
  → LLM either replies directly, or returns tool_calls
      → tool_calls cross the MCP boundary: session.call_tool(name, arguments)
      → MCP server dispatches to the tool function → pos_store → SQLite
      → result ({"success": true/false, ...}) returned over stdio
  → result appended to the conversation as a `role: "tool"` message
  → LLM is called again with the updated conversation
  → repeat until the LLM returns a final reply with no tool calls
→ final response rendered to the user
```

**Intent is determined entirely by the LLM.** There is no keyword
matching, regex, or `if "book" in message.lower()` routing anywhere in
this path — grep the repo and there is none. The model's own reading of
`DEFAULT_SYSTEM_PROMPT` (in `agent/orchestrator.py`) plus each tool's own
JSON-schema description (surfaced live from `list_tools()`, never a
hand-maintained copy) is the *only* thing that decides which tool gets
called, or whether to just reply in plain language.

An "iteration" is one round where the model was consulted and asked for
tool(s) — a single round may request several tools; that still counts as
one iteration. `run_turn()` hard-caps this at 8 iterations
(`DEFAULT_MAX_ITERATIONS`); hitting it forces escalation rather than
consulting the model a 9th time.

## MCP Architecture

**MCP (Model Context Protocol)** is the open protocol this project uses to
expose reservation tools to the LLM in a standardized way, instead of a
bespoke function-calling wrapper.

- **MCP server** (`mcp_server/server.py`) — built with the official
  `mcp.server.fastmcp.FastMCP` class. Every `@mcp.tool()`-decorated
  function becomes a tool the server advertises, with its Python type
  hints and docstring becoming the tool's JSON schema and description
  automatically.
- **MCP client** (`agent/mcp_client.py`) — wraps `mcp.ClientSession` and
  `mcp.client.stdio.stdio_client`. `mcp_session()` spawns
  `mcp_server/server.py` as a **separate OS subprocess** (via
  `sys.executable`) and performs the MCP handshake (`session.initialize()`).
- **Transport: stdio.** The client and server are two separate processes
  talking JSON-RPC over the subprocess's stdin/stdout — not an in-process
  function call. This is real inter-process communication, not a mock:
  killing the subprocess (as `tests/manual_conversations.py`'s scenario 5
  actually does) produces a genuine `ClosedResourceError` on the next call.
- **Tool discovery** — the orchestrator never hand-maintains a tool list.
  On startup it calls the live `session.list_tools()` and converts the
  result to OpenAI/Groq function-calling schema via
  `mcp_tools_to_openai_schemas()`. If a tool is added, removed, or its
  docstring changes in `server.py`, the model sees the new definition on
  the next process start with no other code change.
- **Tool execution** — `make_mcp_call_tool()` builds the `call_tool`
  callable the orchestrator uses: it calls `session.call_tool(name,
  arguments)`, checks `result.isError`, and parses the tool's JSON text
  response. Every `mcp_server` tool returns `{"success": ...}` JSON as its
  content — never raises for a business failure — so a normal return here
  (including `{"success": false, ...}`) is not an error at this layer; see
  [Error Handling](#error-handling).

Verify this yourself: `python tests/roundtrip_check.py` starts the real
server subprocess, connects a real `ClientSession` over stdio, and prints
the live `list_tools()` result and a real `call_tool("ping")` round trip.

## Model

The spec asks for "llama-3.3-8b (or similar size small model)". Llama-3.3
was only ever released as a 70B model — there is no 8B variant. The
original resolution here was **Llama-3.1-8B-Instant** on **Groq's**
OpenAI-compatible `chat.completions` API, as the closest match (Meta's
small-size Llama model in the 3.x line, with Groq's inference speed suiting
a conversational agent that interleaves LLM turns with MCP tool calls).

**That model is gone.** As of 2026-09-04, `client.models.list()` against a
real Groq API key returns no Llama models at all, and requesting
`llama-3.1-8b-instant` fails with a live 404 (`model_not_found`) — this
isn't a guess or stale documentation, it's what the API says right now. We
use **`openai/gpt-oss-20b`** instead: it's the closest remaining "small-ish
open model" on Groq's current lineup, and was verified live (a real
`chat.completions.create` call with `tools=[...]`) to correctly emit
`tool_calls`, which is the one hard requirement for this project's agentic
loop. `agent/orchestrator.py`'s `DEFAULT_GROQ_MODEL` constant is the single
source of truth for the model id in code — if Groq's lineup changes again,
that's the only place to update, but verify tool-calling support the same
way before swapping it in blind.

| | |
|---|---|
| **Model** | `openai/gpt-oss-20b` |
| **Provider** | Groq |
| **API** | OpenAI-compatible `chat.completions.create`, via the `groq` Python SDK |
| **Why this model** | Closest available small/open model on Groq's current lineup to the spec's Llama-3.3-8B ask, with confirmed live tool-calling support and Groq's low-latency inference |
| **Why it changed** | The originally-planned Llama-3.1-8B-Instant was removed from Groq's lineup entirely (live 404) partway through this project |

## Prompt Engineering Approach

**Philosophy: the system prompt is the *only* routing layer.** There is no
code anywhere that classifies intent or matches keywords — everything the
model needs to decide what to do lives in `DEFAULT_SYSTEM_PROMPT` (in
`agent/orchestrator.py`) plus each tool's own docstring (surfaced as the
tool's JSON-schema description). If that prompt doesn't say it, the model
doesn't know it — deliberately, so the agent's behavior is auditable by
reading two things instead of hunting through business logic.

Key design principles actually encoded in the prompt and the loop:

- **Never invent domain facts.** The prompt states directly: *"You have no
  knowledge of any restaurant, availability, or reservation beyond what a
  tool call returns to you in this conversation; never invent a restaurant
  name, id, time slot, or reservation id."* Slot times are called out
  specifically — the model is told the fixed 2-hour grid up front so it
  never proposes an off-grid time "at face value" before checking a tool.
- **When to ask vs. when to act.** *"Ask the user for any required detail
  you don't have (date, party size, phone number, which restaurant) rather
  than guessing it."* Each tool's docstring states its own preconditions
  (e.g. `create_reservation` needs a specific restaurant, an exact slot
  from `check_availability`, party size, name, and phone) so the model
  knows exactly what's missing before it's allowed to call it.
- **Tool selection guidance lives in each docstring, not the system
  prompt.** The system prompt gives one or two sentences of *when* to
  reach for each tool (e.g. `search_restaurants` for hard, known filters
  vs. `recommend_restaurants` when the user wants suggestions or doesn't
  have a place in mind); the tool's own docstring carries the full
  argument-level contract. This keeps the system prompt short enough to
  stay legible while the detail lives where the model actually needs it —
  right next to the schema it's filling in.
- **A `{"success": false, ...}` result is information, not an error.**
  The prompt is explicit: *"if it comes back with `{"success": false,
  ...}`, that is normal information, not a system error — explain the
  reason to the user in plain language ... and help them pick another
  option; do not call it again with the exact same arguments expecting a
  different result."* This is what makes the fully-booked → alternative
  recommendation flow (see [Example Conversations](#example-conversations)
  #5 below) work without any special-case code.
- **Booking safety is enforced by the tools, not the prompt.** The model
  is *told* about the fixed slot grid and to "always get the exact slot
  string from `check_availability`," but the actual guarantee is
  structural: `create_reservation`/`modify_reservation` reject any slot
  string off the fixed grid, and `pos_store`'s capacity/double-booking
  guards run unconditionally regardless of what the model does. Prompting
  reduces *how often* the model needs to be corrected; the tool layer is
  what makes an unsafe booking actually impossible.
- **Human escalation is never the model's call.** `escalate_to_human` is
  in the model's toolbox for judgment calls (complaints, policy questions,
  requests nothing else can resolve) — but every *forced* escalation
  (transport failure exhausted after 3 retries, the LLM request itself
  failing, or the 8-iteration cap) is triggered directly in
  `agent/orchestrator.py` code, bypassing the model entirely. A model
  given the option to retry "just one more time" can rationalize that
  indefinitely; the hard limits exist precisely so that decision isn't
  left to it.
- **Multi-turn context** is maintained the simple way: the full
  OpenAI-shaped message list (system/user/assistant/tool roles) persists
  in `st.session_state` across the whole browser session and is passed
  back into `run_turn()` on every new message — so a follow-up like
  "actually, move that to 9pm" resolves against the reservation id the
  model already saw earlier in the same conversation, with no separate
  memory/context layer.

## Tools

All 8 tools exposed by `mcp_server/server.py`, exactly as named in code:

| Tool | Purpose | Read/Write | Example |
|---|---|---|---|
| `ping` | Liveness check; no conversational purpose | Read | `ping()` → `{"success": true, "message": "pong"}` |
| `search_restaurants` | Hard AND filter by cuisine/location/price/min. capacity | Read | `search_restaurants(cuisine="Italian", location="Indiranagar")` |
| `check_availability` | List a restaurant's 6 fixed slots on a date + which have room | Read | `check_availability(restaurant_id=46, date="2026-10-05", party_size=2)` |
| `recommend_restaurants` | Ranked, availability-gated suggestions from a weighted score | Read | `recommend_restaurants(date="2026-10-05", party_size=2, cuisine="Italian")` |
| `create_reservation` | Book a table (capacity + double-booking guarded) | Write | `create_reservation(restaurant_id=46, customer_name="Alex Doe", customer_phone="555-0100", party_size=2, slot="2026-10-05 19:00")` |
| `modify_reservation` | Change party size and/or slot of an existing reservation | Write | `modify_reservation(reservation_id=7, slot="2026-10-05 21:00")` |
| `cancel_reservation` | Cancel a reservation; frees seats + slot immediately | Write | `cancel_reservation(reservation_id=7)` |
| `escalate_to_human` | Hand off to staff for anything the other tools can't resolve | Write (log only — see [Limitations](#limitations)) | `escalate_to_human(reason="refund request for a no-show fee")` |

## Recommendation System

`recommend_restaurants` ranks candidates by a **weighted blend of five 0–1
signals**, summing to 1.0:

| Signal | Weight | What it measures |
|---|---|---|
| Cuisine match | 0.30 | Exact match to requested cuisine (strongest driver of satisfaction) |
| Vibe/preference match | 0.25 | Fraction of the restaurant's fixed-vocabulary tags found as substrings of freeform `preference` text |
| Rating | 0.20 | Seeded 1.0–5.0 rating, rescaled to 0–1; a fair tiebreaker, capped so it can't dominate a good match |
| Price fit | 0.15 | Graded distance between price tiers (not a hard cutoff, unlike `search_restaurants`) |
| Proximity | 0.10 | 1.0 for exact neighborhood match, 0.0 otherwise — no real geo-distance data exists (see [Assumptions](#assumptions)) |

**Availability is a hard gate applied *before* scoring, not a sixth
signal.** A restaurant with zero open slots for the requested date/party
size is filtered out entirely before any weight is computed — no score
can compensate for being fully booked.

**Why this isn't just "ask the LLM to pick a restaurant."** The model
never sees the restaurant list and invents a ranking; `recommend_restaurants`
computes the score deterministically in code from stored attributes
(cuisine, tags, rating, price tier, location) and only returns the
already-sorted, already-availability-filtered result. This keeps
recommendations reproducible, auditable (the full `score_breakdown` is
returned per candidate), and immune to the model hallucinating a
restaurant that can't actually seat the party.

## Reservation Logic

- **Availability** — `check_availability` reports each of a restaurant's 6
  fixed daily slots (11:00–21:00, every 2 hours) as available or not, for
  a given party size.
- **Capacity validation** — every create/modify re-checks that the
  restaurant's total confirmed seats booked at that slot, plus this
  party, doesn't exceed its capacity.
- **Duplicate-booking protection** — a create/modify is independently
  rejected if the *same* `customer_phone` already has a confirmed
  reservation at that restaurant for that exact slot — this fires even
  when capacity would allow it; it's catching accidental double-bookings
  by one customer, not restaurant fullness.
- **Create / modify / cancel** — all three run through `pos_store.py`'s
  guards; a cancelled reservation (`status = 'cancelled'`) counts toward
  neither capacity nor double-booking, so cancelling immediately frees
  both for a new booking.
- **Persistence** — a real SQLite file (`goodfoods.db`), not an
  in-memory structure — Streamlit reruns the whole script on every
  interaction, so anything held only in Python state would be lost or
  duplicated across reruns.

Both guards are pytest-covered (`tests/test_pos_store.py`): exact-capacity
fit, over-capacity rejection, same-slot double-booking rejection,
cancel-then-rebook, and a failed modification leaving the original
reservation untouched.

## Error Handling

`agent/orchestrator.py`'s `run_turn()` distinguishes three failure
categories, handled differently on purpose:

1. **Business errors** — a tool call *completes normally* with
   `{"success": false, "error": "..."}` (no availability, capacity
   exceeded, double-booking, off-grid slot, unknown id). This is valid
   data, not a failure of the call — it's handed back to the model like
   any other result, with no retry. Retrying "this restaurant is full"
   can't change the answer.
2. **Tool/transport failures** — a tool call *raises* (dropped subprocess,
   malformed response, MCP-level error). Retried up to 3 times
   (`DEFAULT_MAX_TRANSPORT_RETRIES`) with exponential backoff
   (`0.5s, 1s, 2s`). If every attempt fails, `escalate_to_human` is called
   **directly in code**.
3. **LLM/API failures** — the call to the model itself raises (rate limit,
   auth, network). **Not retried** — a rate limit's suggested wait is
   tens of minutes, useless mid-conversation — straight to forced
   escalation on the first failure. Confirmed live: before this was
   handled, a real `groq.RateLimitError` propagated out of `run_turn()` as
   a raw traceback rendered directly in the Streamlit chat.

A fourth, independent limit — **8 tool-call iterations** — forces the same
escalation path if the model never converges on a final reply.

**Escalation is always forced in code, never left to the model**, in
every one of the above cases — a model given the option to try "just once
more" can rationalize that indefinitely. If even `escalate_to_human`
itself fails after retries (a genuinely dead subprocess), the user gets a
fixed apologetic fallback message rather than a crash — observed live in
[Example Conversations](#example-conversations) #6 below.

All of this is pytest-covered against a scripted fake LLM
(`tests/test_orchestrator.py`) with no API key required, so every path —
retry-then-succeed, retry-exhausted, LLM failure, iteration cap — is
provable without spending real tokens.

## Example Conversations

All six examples below (from 5 real script scenarios — the first covers
both discovery and booking in one continuous conversation) are **real,
captured output** from a live run against the actual Groq API and a real
`mcp_server/server.py` subprocess (`python tests/manual_conversations.py`,
run 2026-09-07) — not hand-written examples. Some large JSON results are
trimmed for readability (noted inline); tool names, arguments, and the
model's actual final replies are unedited.

### 1. Restaurant discovery/recommendation → 2. Successful booking

One continuous conversation — a realistic flow where discovery leads
straight into booking:

> **User:** "Hi! I'd like to book a table for 2 people at an Italian
> restaurant on 2026-10-05, around 7pm. My name is Alex Doe and my phone
> number is 555-0100. I don't have a preference beyond that — please just
> pick whichever Italian place has a good rating and is free at that
> time, and confirm the booking."

```
[1] recommend_restaurants(cuisine="Italian", date="2026-10-05", party_size=2)
    → success: true, 5 recommendations, top result:
      {"restaurant": {"id": 46, "name": "The Bold Terrace", "cuisine": "Italian",
        "location": "Basavanagudi", "capacity": 13, "price_tier": "$$$",
        "rating": 4.7, "tags": ["lively", "outdoor-seating"]},
       "score": 0.985,
       "score_breakdown": {"cuisine_match": 1.0, "vibe_match": 1.0,
         "rating": 0.925, "price_fit": 1.0, "proximity": 1.0}}
      (4 more, same shape)

[2] check_availability(restaurant_id=46, date="2026-10-05", party_size=2)
    → success: true, all 6 fixed slots available, including "2026-10-05 19:00"

[3] create_reservation(restaurant_id=46, customer_name="Alex Doe",
      customer_phone="555-0100", party_size=2, slot="2026-10-05 19:00")
    → success: true, reservation id=7, status="confirmed"
```

**Final reply:**
> ✅ Your reservation is confirmed!
> • **Restaurant:** The Bold Terrace (Italian)
> • **Date & Time:** 2026-10-05 19:00
> • **Party Size:** 2
> • **Name:** Alex Doe
> • **Phone:** 555-0100
>
> Your reservation ID is **7**. If you need to make any changes or have
> questions, just let me know! Enjoy your dinner!

*(escalated=False, 3 iterations — the model picked the tool sequence
entirely on its own; nothing in the code told it to call `recommend_`
before `check_` before `create_`.)*

### 3. Modification

Same conversation continues:

> **User:** "Actually, can you move that reservation to 9pm instead?"

```
[1] modify_reservation(reservation_id=7, slot="2026-10-05 21:00")
    → success: true, reservation id=7, slot now "2026-10-05 21:00"
```

**Final reply:**
> ✅ Your reservation has been updated.
> • **Restaurant:** The Bold Terrace
> • **New Time:** 2026-10-05 21:00
> • **Party Size:** 2
>
> Reservation ID remains **7**. Let me know if there's anything else!

Note the model correctly resolved "that reservation" to `reservation_id=7`
purely from earlier conversation context — no re-lookup was needed.

### 4. Cancellation

Same conversation continues:

> **User:** "On second thought, please just cancel that reservation
> entirely."

```
[1] cancel_reservation(reservation_id=7)
    → success: true, reservation id=7, status now "cancelled"
```

**Final reply:**
> Your reservation (ID 7) for The Bold Terrace on 2026-10-05 at 21:00 has
> been cancelled. If you'd like to book something else, just let me know!

### 5. Edge case — restaurant unavailable, agent offers real alternatives

A restaurant (`Sadashivanagar Garden`, id=1) was pre-filled to be fully
booked on 2026-11-11 for this scenario.

> **User:** "I'd like to book a table at Sadashivanagar Garden — it's the
> Fusion place in Sadashivanagar — for 2 people on 2026-11-11 around 7pm.
> My name is Jordan Lee, phone 555-0200. If they genuinely can't seat us
> at all that day, please suggest another good option nearby instead of
> just telling me no."

```
[1] search_restaurants(cuisine="Fusion", location="Sadashivanagar", party_size=2)
    → success: true, found Sadashivanagar Garden (id=1)

[2] check_availability(restaurant_id=1, date="2026-11-11", party_size=2)
    → success: true, all 6 slots available=false — fully booked

[3] recommend_restaurants(date="2026-11-11", location="Sadashivanagar", party_size=2)
    → success: true, 5 recommendations, availability-gated (Sadashivanagar
      Garden itself is correctly absent — it has no open slots)
```

**Final reply:** the model presented a table of 5 real alternatives
(Seafood Kettle, The Hidden Kettle, Ivory Kitchen, and two more) with
cuisine/price/rating/tags/slot for each, and explicitly called out the
top two as being in the same neighborhood, then asked whether to book one.

*(Honest note: the model achieved this by omitting the `cuisine` filter
on its `recommend_restaurants` call rather than passing
`exclude_restaurant_id=1` as the system prompt suggests for this exact
case — it still worked correctly, since the availability gate removes the
fully-booked restaurant regardless of which path gets there, but it's
worth documenting the model's actual behavior rather than the idealized
expectation.)*

### 6. Edge case — tool/transport failure → forced human escalation

The real MCP server subprocess was killed mid-conversation (via `psutil`,
not simulated) right after its first successful tool call, on a fresh
request: *"Please search for Mexican restaurants, then check 2026-09-05's
availability for the first one you find, for a party of 4."*

```
[1] search_restaurants(cuisine="Mexican") → success: true, 4 results

  [!! subprocess killed here — real process, real kill !!]

[2] check_availability(...) → ERROR: ClosedResourceError()
[3] check_availability(...) → ERROR: ClosedResourceError()   (retry 2/3)
[4] check_availability(...) → ERROR: ClosedResourceError()   (retry 3/3, exhausted)
[5] escalate_to_human(reason="check_availability failed after 3 attempts: ...")
    → ERROR: ClosedResourceError()   (the escalation call itself also fails —
      the subprocess is genuinely dead)
```

**Final reply:** *"We're having trouble completing this right now and our
escalation path is also unreachable. Please try again shortly or contact
us directly."*

*(escalated=True — this is the documented last-resort fallback message in
`_force_escalation()`, which fires only when even the escalation tool call
can't get through. No traceback, no crash, no silent failure — the user
gets a plain-language message either way.)*

## Business Strategy Summary

Full detail: [`docs/GoodFoods_Business_Strategy.docx`](docs/GoodFoods_Business_Strategy.docx)
(Part 1 of this submission). Summary below.

**Problem.** GoodFoods loses bookings to busy phone lines, hosts lose
hours a week to manual reservation admin, and no-shows leave tables
empty — none of it measured today.

**Solution.** An action-oriented AI agent (not an FAQ bot) that reads and
writes directly to the reservation/POS system — book, modify, cancel,
confirm, remind — with the POS, not the LLM, as the source of truth.
Launch narrow: one channel, one language, six core actions, 1–2
locations; recommendations, WhatsApp, CRM, and upsell are explicitly
deferred to later phases, not cut features.

**KPIs** (baselines are labeled illustrative assumptions, to be replaced
with GoodFoods' real numbers in Phase 0 discovery): reservation
conversion rate (55–65% phone → 75–85% target), automation/self-service
rate (0% → 70–80%), no-show rate (15–20% industry baseline → 20–30%
relative reduction), staff hours saved (~10 hrs/week/location → ~65–75%
reduction), plus funnel completion, human handoff rate, cancellation
re-fill rate, off-peak utilization, upsell attach rate, repeat-booking
rate, and CSAT.

**ROI.** Three *mutually exclusive* incremental-reservation buckets
(missed-demand recovery, no-show recovery, waitlist/cancellation
recovery — deliberately non-overlapping populations) sum to an
illustrative ≈$27,085 net annual benefit per location, after labor
savings and operating cost. At a 10-location pilot, that's ≈$270,800/year
against a one-time shared implementation cost of $40K–$60K — payback in
roughly 1.8–2.7 months. A sensitivity check on the largest single
assumption (no-show recovery rate) shows the case still holds, just less
dramatically, if that number comes in lower — this is explicitly the
number to validate first, not assume.

**Rollout.** Phase 0 (discovery/integration readiness, 0 live locations)
→ Phase 1 (1–2 location MVP pilot) → Phase 2 (5–10 locations, genuine
A/B test vs. matched control locations) → Phase 3 (chain-wide) → Phase 4
(forecasting/optimization, continuous). Each phase has an explicit
go/no-go gate — this is not a big-bang launch.

**Competitive advantages.** (1) Action-oriented execution — safe
create/modify/cancel with validation and rollback, not just retrieval —
is materially harder to build than an FAQ layer and is where the labor
and revenue value actually sits. (2) A reusable, multi-tenant
orchestration core with a thin per-location adapter, so marginal cost per
new location falls sharply. (3) A closed feedback loop from real
operational outcomes (which recommendation gets accepted, which reminder
timing actually cuts no-shows) — this compounding data advantage, not the
architecture itself, is the actual moat: a competitor can copy the
engineering in a quarter, but not the accumulated operational data before
the system has run against real bookings.

**Vertical expansion.** The core (LLM intent detection, slot-filling,
tool-calling orchestration, notifications, analytics loop) is
domain-agnostic; what changes per vertical is the knowledge base,
availability/resource model, and integration/compliance layer — other
restaurant chains (near-zero marginal cost), hotels (date-range stays,
PMS integration), salons/spas (stylist-availability matching), healthcare
(much higher compliance bar), and other appointment-based businesses.

**Key stakeholders.** GoodFoods' ops/IT team (POS/reservation-system
integration, Phase 0 data readiness), front-desk/host staff (pilot design
input, daily users of the escalation/override path), diners (the actual
conversational users), and leadership/finance (owns the go/no-go
decisions at each rollout phase). **Potential customers** beyond
GoodFoods itself: the vertical-expansion table above doubles as a
customer list — other restaurant chains first (same platform, near-zero
marginal cost), then hotels, salons/spas, healthcare scheduling, and
other appointment-based businesses as separate go-to-market segments once
the core platform is proven.

## Assumptions

**Reservation slots are fixed seatings, not arbitrary timestamps.** Every
restaurant is open 11:00–22:00 daily and offers one seating every 2 hours
(11:00, 13:00, 15:00, 17:00, 19:00, 21:00 — six slots/day), matching a
typical dining duration, rather than open-ended start times. `pos_store.py`'s
capacity and double-booking guards match slots by exact string equality
(e.g. `"2026-09-05 19:00"`), with no awareness of dining duration or
real-time overlap; that's safe *only* because slot values are constrained
upstream to this fixed grid, so two bookings can never partially overlap
without landing on the same slot string. This is a deliberate MVP
simplification, not an oversight. Every restaurant currently shares the
same 11:00–22:00 window — there is no per-restaurant hours concept yet.

**`recommend_restaurants`'s "proximity" signal is same-neighborhood-or-not,
not real distance.** Restaurants only have a `location` (neighborhood)
string, not coordinates, so there is no way to say *how much* closer one
neighborhood is than another. Proximity is scored 1.0 for an exact
neighborhood match and 0.0 otherwise, and is deliberately the
lowest-weighted of the five ranking signals for that reason. `rating` and
vibe `tags` are synthetic: `seed_data.py` assigns each restaurant a
deterministic 3.0–5.0 rating and 1–3 tags from a fixed vocabulary —
there's no real review data behind them, only internally-consistent
seeded values to rank against.

**Neighborhoods are real Bangalore localities, restaurants are fully
synthetic.** `seed_data.py`'s 20 neighborhoods (Indiranagar, Koramangala,
HSR Layout, Whitefield, …) are real Bangalore localities, but every
restaurant "in" them is generated — a deterministic name/cuisine/
capacity/price/rating/tags, not scraped or sourced from anywhere.

**SQLite stands in for a production POS.** `goodfoods.db` is a single
local file, chosen because it needs zero infrastructure for a take-home
project and Streamlit's rerun-on-every-interaction model needs state that
survives outside Python process memory. A production deployment would
integrate with GoodFoods' actual reservation/POS system instead (see
Phase 0 discovery in the business strategy doc), not replace SQLite with
a bigger database — the integration target is the operational system of
record, not "a real database" for its own sake.

**All business-case figures (KPIs, ROI, baselines) are illustrative**,
explicitly labeled as such in the strategy doc, meant to be replaced with
GoodFoods' real operational data during Phase 0 discovery — not
GoodFoods-specific measurements.

## Limitations

What this prototype deliberately does **not** do:

- **No real POS/reservation-system integration.** SQLite is the store;
  there is no connector to any actual restaurant POS vendor.
- **`escalate_to_human` is a stub.** It logs server-side and returns a
  fixed confirmation message; no human is actually paged. A production
  version needs a real notification/paging integration.
- **Fixed slot grid only.** No arbitrary times, no per-restaurant hours,
  no dining-duration-aware overlap — see [Assumptions](#assumptions).
- **No waitlist auto-notify.** A fully-booked restaurant is filtered out
  of recommendations; nothing re-offers a slot when it's cancelled.
- **Single channel, single language.** Web (Streamlit) only, English
  only — no WhatsApp, voice, or i18n.
- **No CRM, loyalty, or persistent customer profiles.** `customer_phone`
  is only used as an identity key for the double-booking check within
  this store; there's no cross-visit history, preferences, or loyalty
  status.
- **No payment/deposit handling.** Large-party or private-dining flows
  that would need a deposit aren't modeled.
- **No automated CI eval suite against the live LLM.**
  `tests/manual_conversations.py` is deliberately excluded from pytest
  discovery because it spends real API tokens per run; there's no
  scheduled/CI job re-running it to catch model-behavior regressions.
- **Single-process, single-file database.** No connection pooling,
  horizontal scaling, or concurrent-writer story beyond SQLite's own
  locking — fine for a demo, not for multi-location production load.
- **No authentication.** Any Streamlit visitor can book/modify/cancel any
  reservation by id; there's no verification that the person cancelling
  reservation #7 is the same person who created it.

## Future Enhancements

**Phase 2** (per the rollout plan in the business strategy doc):
WhatsApp channel, waitlist auto-notify, upsell prompts (occasion
packages, pre-orders), CRM/loyalty foundation, regional-language
expansion (gated on measured per-language model accuracy, not a fixed
checklist).

**Phase 3**: chain-wide rollout across GoodFoods' full location set,
voice channel, cross-location demand routing (turn "fully booked" into a
sale at a sister location), loyalty program integration.

**Phase 4 / long-term**: demand forecasting and dynamic promotions,
deeper personalization from the accumulated operational-outcomes data
(the actual competitive moat — see
[Business Strategy Summary](#business-strategy-summary)), vertical
expansion of the same core platform into hotels, salons/spas, or
healthcare scheduling.

**Technical roadmap, independent of the business phasing**: real staff
paging behind `escalate_to_human`; per-restaurant operating hours instead
of one shared grid; an automated (budget-capped, scheduled) re-run of the
live-LLM conversation scenarios to catch model-behavior regressions
between Groq model updates; basic authentication/ownership checks on
modify/cancel.

## Layout

```
mcp_server/       FastMCP server (stdio transport)
  server.py       MCP tools: ping, search_restaurants, check_availability,
                   recommend_restaurants, create_reservation,
                   modify_reservation, cancel_reservation,
                   escalate_to_human (stub)
  pos_store.py    SQLite-backed reservation store (Restaurant/Reservation
                   dataclasses, capacity + double-booking guards)
  seed_data.py    Deterministic generator for 50-100 seed restaurants
                   across 20 real Bangalore neighborhoods; run as
                   `python -m mcp_server.seed_data` to seed goodfoods.db
                   (only if empty) and print a distribution summary
  cuisine_normalization.py  Synonym table (e.g. "North Indian" -> "Indian")
                   used by search_restaurants/recommend_restaurants
agent/
  mcp_client.py   Reusable stdio MCP client (starts server subprocess, handshakes)
  orchestrator.py Hand-written agentic loop (no framework): calls the model,
                   executes whichever tool(s) it picks via mcp_client.py,
                   feeds results back, loops to a final reply. Retries
                   transport failures 3x with backoff; never retries a
                   {"success": false} business result; hard-caps tool-call
                   iterations; force-escalates immediately (no retry) if
                   the request to the LLM itself fails -- every forced
                   case escalates directly in code, not via the model's
                   judgment.
                   build_groq_llm_call() reads GROQ_API_KEY from .env
                   (python-dotenv) and wires the real Groq
                   chat.completions API with mcp_tools_to_openai_schemas()
                   as `tools=`. Every tool-selection decision goes through
                   the model this way -- there is no keyword/string-match
                   routing anywhere in the message-handling path; see
                   DEFAULT_SYSTEM_PROMPT for the only context the model
                   gets about the domain and the 8 tools.
frontend/
  streamlit_app.py  Streamlit chat UI. st.session_state for per-visitor
                   conversation history; st.cache_resource keeps the MCP
                   subprocess/client and the Groq client alive across
                   every rerun. Each message goes through
                   orchestrator.run_turn(); a collapsed expander under
                   each reply shows the full tool-call trace (name,
                   arguments, result). Escalated/fallback replies render
                   as a plain chat message, never a special error banner.
tests/
  roundtrip_check.py    Manual: starts the server, calls
                         list_tools()/call_tool("ping") -- proves the MCP
                         round trip is a real subprocess over stdio
  manual_conversations.py  Manual: 5 scenarios against the real Groq API
                         and real MCP server (spends real tokens;
                         deliberately not pytest-discoverable) -- see
                         Example Conversations above for real captured output
  test_pos_store.py     pytest coverage for pos_store.py's guards
  test_orchestrator.py  pytest coverage for orchestrator.py's agentic
                         loop, against a scripted fake LLM (no API key
                         needed)
  test_cuisine_normalization.py  pytest coverage for the cuisine synonym
                         table
docs/
  GoodFoods_Business_Strategy.docx  Part 1 of this submission: full use-case
                         and business strategy document (this README's
                         Business Strategy Summary condenses it)
.env.example      GROQ_API_KEY=
```

## Setup

Requires **Python 3.10+** (developed and tested on 3.14.5).

```
pip install -r requirements.txt
copy .env.example .env   # macOS/Linux: cp .env.example .env
# then fill in GROQ_API_KEY in .env
```

Seed the database (idempotent — safe to re-run; skips if already seeded):

```
python -m mcp_server.seed_data
```

This seeds `goodfoods.db` with 80 restaurants (50-100 range) if it's
currently empty, then prints a distribution summary (cuisine/neighborhood/
price-tier breakdown) and asserts no neighborhood exceeds 15% of the total.

Verify the MCP round trip:

```
python tests/roundtrip_check.py
```

This starts `mcp_server/server.py` as a subprocess, connects a real MCP
`ClientSession` over stdio, and prints the result of `list_tools()` and
`call_tool("ping")`.

Run the app:

```
python -m streamlit run frontend/streamlit_app.py
```

The MCP server subprocess and the Groq client are created once and cached
for the app's lifetime — restart the process after changing any
`mcp_server/` code. If plain `streamlit run` resolves to a different
Python install on your machine, use `python -m streamlit run ...` to force
the correct interpreter.

## Testing

**Automated (pytest, no API key needed) — 24 tests, all verified passing:**

```
pytest tests/
```

- `test_pos_store.py` — capacity/double-booking guards against a real
  temp SQLite file (exact-capacity fit, over-capacity rejection, same-slot
  double-booking, cancel-then-rebook, failed-modification leaves state
  unchanged).
- `test_orchestrator.py` — the full agentic loop against a scripted fake
  LLM: normal multi-tool resolution, business-failure pass-through
  (no retry), transport-failure retry-then-succeed, transport-failure
  exhausted → forced escalation, LLM-call failure → forced escalation (no
  retry), iteration-cap → forced escalation. Each test asserts on the
  exact interleaved call sequence, not just outcomes.
- `test_cuisine_normalization.py` — the synonym table, including that an
  unrecognized cuisine is left unchanged (never guessed) and still trips
  `search_restaurants`'s own validation error.

**MCP round trip (manual, no API key needed):**

```
python tests/roundtrip_check.py
```

Verifies the real stdio subprocess, live `list_tools()`, and a real
`call_tool("ping")` round trip.

**Manual live-LLM conversations (spends real Groq API tokens):**

```
python tests/manual_conversations.py
```

Runs 5 scenarios against the real Groq API and a real MCP server
subprocess — including actually killing the subprocess mid-conversation
via `psutil` for the transport-failure scenario, not a simulation. This is
deliberately excluded from pytest discovery (see
[Limitations](#limitations) on CI coverage). All 5 scenarios in
[Example Conversations](#example-conversations) above are real,
unedited output from an actual run of this script.

**What's verified vs. what isn't:** all of the above were run and passed
as part of preparing this submission. What's *not* covered: load/concurrency
testing, and any automated regression suite for LLM behavior specifically
(the live-conversation script is manual and cost-gated, not scheduled).
