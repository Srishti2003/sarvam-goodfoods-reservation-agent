"""GoodFoods MCP server.

Exposes mcp_server/pos_store.py's reservation store as MCP tools over
stdio. Business-logic failures (bad filters, capacity/double-booking guard
failures, unknown ids, malformed input, etc.) are returned as
{"success": False, "error": "..."} rather than raised -- a raised
exception here would crash the MCP transport instead of giving the calling
LLM something it can react to and recover from.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# server.py is launched directly as a script (agent/mcp_client.py spawns it
# by path, not via `-m`), so sys.path[0] is this file's own directory --
# add the project root too so the `mcp_server.*` absolute imports below
# resolve the same way they do everywhere else in this codebase.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server.cuisine_normalization import normalize_cuisine
from mcp_server.pos_store import PosStore, ReservationError
from mcp_server.seed_data import CUISINES, NEIGHBORHOODS, PRICE_TIERS

logger = logging.getLogger("goodfoods.server")

mcp = FastMCP("goodfoods")
_store = PosStore()

# ---------------------------------------------------------------------------
# Fixed-seating slot grid (see README.md "Assumptions"): reservation slots
# are NOT arbitrary timestamps. Every restaurant is treated as open
# OPEN_HOUR-CLOSE_HOUR daily and offers one seating every SLOT_SPACING_HOURS.
# pos_store.py's capacity/double-booking guards match slots by exact string
# equality and rely on this grid to make real-time overlap impossible by
# construction (two bookings can't partially overlap without landing on the
# same slot string). Do NOT narrow SLOT_SPACING_HOURS, or otherwise let a
# slot string reach pos_store off this grid, without first making those
# guards interval-aware -- see README.md's "Assumptions" section.
OPEN_HOUR = 11
CLOSE_HOUR = 22
SLOT_SPACING_HOURS = 2
_SLOT_FORMAT = "%Y-%m-%d %H:%M"


def _slot_string(date: str, hour: int) -> str:
    return f"{date} {hour:02d}:00"


def _generate_daily_slots(date: str) -> list[str]:
    """All fixed-grid slot strings for one calendar date, open to close."""
    return [
        _slot_string(date, hour)
        for hour in range(OPEN_HOUR, CLOSE_HOUR + 1, SLOT_SPACING_HOURS)
    ]


def _is_on_slot_grid(slot: str) -> bool:
    try:
        parsed = datetime.strptime(slot, _SLOT_FORMAT)
    except ValueError:
        return False
    return (
        parsed.minute == 0
        and OPEN_HOUR <= parsed.hour <= CLOSE_HOUR
        and (parsed.hour - OPEN_HOUR) % SLOT_SPACING_HOURS == 0
    )


def _off_grid_error(slot: str) -> dict:
    return {
        "success": False,
        "error": (
            f"slot {slot!r} is not a valid fixed-grid slot. Restaurants only "
            f"take reservations between {OPEN_HOUR:02d}:00 and {CLOSE_HOUR:02d}:00, "
            f"one seating every {SLOT_SPACING_HOURS} hours. Call "
            "check_availability for the desired date and pass back one of "
            "the exact slot strings it returns -- do not construct one "
            "yourself."
        ),
    }


def _valid_date(date: str) -> bool:
    try:
        datetime.strptime(date, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _parse_tags(tags: str) -> list[str]:
    return [t for t in (tag.strip().lower() for tag in tags.split(",")) if t]


def _restaurant_to_dict(r) -> dict:
    return {
        "id": r.id,
        "name": r.name,
        "cuisine": r.cuisine,
        "location": r.location,
        "capacity": r.capacity,
        "price_tier": r.price_tier,
        "rating": r.rating,
        "tags": _parse_tags(r.tags),
    }


def _reservation_to_dict(res) -> dict:
    return {
        "id": res.id,
        "restaurant_id": res.restaurant_id,
        "customer_name": res.customer_name,
        "customer_phone": res.customer_phone,
        "party_size": res.party_size,
        "slot": res.slot,
        "status": res.status,
        "created_at": res.created_at,
    }


@mcp.tool()
def ping() -> dict:
    """Throwaway liveness check. Returns a fixed pong response.

    Returns:
        Always: {"success": true, "message": "pong"}. Same
        {"success": ...} shape as every other tool here, so a caller never
        needs a special case for this one. This tool cannot fail.
    """
    return {"success": True, "message": "pong"}


@mcp.tool()
def search_restaurants(
    cuisine: str | None = None,
    location: str | None = None,
    price_range: str | None = None,
    party_size: int | None = None,
) -> dict:
    """Search restaurants with a hard AND filter across whichever of the
    four optional criteria are given; any criterion left as None is not
    filtered on. There is no fuzzy or partial matching and no ranking --
    cuisine and location must match a restaurant's stored value exactly
    (case-insensitive). Zero results means no restaurant satisfies ALL
    given criteria at once, not that none exist for any single one.

    Args:
        cuisine: Exact cuisine name, case-insensitive. One of: Italian,
            Japanese, Indian, Mexican, Thai, Chinese, French,
            Mediterranean, American, Korean, Vietnamese, Spanish, Greek,
            Lebanese, Steakhouse, Seafood, Vegan, Fusion. You don't need
            to pre-translate common regional/sub-cuisine names yourself --
            a handful of known variants are normalized automatically
            before matching (e.g. "North Indian", "South Indian",
            "Punjabi", "Mughlai" -> "Indian"; "Cantonese", "Indo-Chinese"
            -> "Chinese"; "Neapolitan" -> "Italian"), so pass the user's
            own wording if it's one of these. Anything else must still
            match one of the categories above exactly; an unrecognized
            value returns a structured error listing these exact options
            (the set may grow over time -- trust that error's list over
            this docstring if they ever disagree).
        location: Exact neighborhood name, case-insensitive. One of:
            Indiranagar, Koramangala, HSR Layout, Whitefield, Jayanagar,
            MG Road, Domlur, JP Nagar, Malleswaram, Basavanagudi,
            Electronic City, Rajajinagar, BTM Layout, Yelahanka,
            Marathahalli, Sadashivanagar, Ulsoor, Banashankari,
            Frazer Town, Cox Town. An unrecognized value returns a
            structured error listing these exact options.
        price_range: One of the four price tiers, cheapest to priciest:
            "$", "$$", "$$$", "$$$$". Must match exactly.
        party_size: Minimum guests the restaurant must be able to seat at
            all, i.e. restaurant.capacity >= party_size. This is a
            capacity ceiling check only -- it does NOT check whether a
            specific date/time has room; call check_availability for that.

    Returns:
        On success: {"success": true, "count": <int>, "restaurants": [
            {"id": int, "name": str, "cuisine": str, "location": str,
             "capacity": int, "price_tier": str, "rating": float,
             "tags": [str, ...]}, ...]}. "rating" is 1.0-5.0; "tags" are
            lowercase vibe descriptors (see recommend_restaurants'
            docstring for the full tag vocabulary) and may be empty.
            An empty "restaurants" list is a normal, successful result.
        On an unrecognized cuisine/location/price_range, or a
        non-positive party_size: {"success": false, "error": "..."}.
    """
    if cuisine is not None:
        cuisine = normalize_cuisine(cuisine)
        match = next((c for c in CUISINES if c.lower() == cuisine.lower()), None)
        if match is None:
            return {
                "success": False,
                "error": f"unknown cuisine {cuisine!r}. Valid cuisines: {', '.join(CUISINES)}",
            }
        cuisine = match

    if location is not None:
        match = next((n for n in NEIGHBORHOODS if n.lower() == location.lower()), None)
        if match is None:
            return {
                "success": False,
                "error": f"unknown location {location!r}. Valid locations: {', '.join(NEIGHBORHOODS)}",
            }
        location = match

    if price_range is not None and price_range not in PRICE_TIERS:
        return {
            "success": False,
            "error": f"unknown price_range {price_range!r}. Valid price tiers: {', '.join(PRICE_TIERS)}",
        }

    if party_size is not None and party_size <= 0:
        return {"success": False, "error": "party_size must be positive"}

    results = [
        r
        for r in _store.list_restaurants()
        if (cuisine is None or r.cuisine == cuisine)
        and (location is None or r.location == location)
        and (price_range is None or r.price_tier == price_range)
        and (party_size is None or r.capacity >= party_size)
    ]
    return {
        "success": True,
        "count": len(results),
        "restaurants": [_restaurant_to_dict(r) for r in results],
    }


@mcp.tool()
def check_availability(restaurant_id: int, date: str, party_size: int) -> dict:
    """List every fixed seating slot a restaurant offers on one calendar
    date, and whether each currently has room for `party_size` guests.

    Restaurants do not take reservations at arbitrary times. Every
    restaurant is open 11:00-22:00 daily and offers one seating every 2
    hours, so the only bookable start times on any date are 11:00, 13:00,
    15:00, 17:00, 19:00, and 21:00 -- there is nothing in between (e.g.
    "12:30" is never a valid slot, and never will be even if it looks free
    by inspection). This tool is the only correct way to discover which
    slots are open; never construct a slot string yourself for
    create_reservation/modify_reservation -- always pass back the exact
    "slot" string this tool returned.

    Args:
        restaurant_id: id of a restaurant, from search_restaurants.
        date: Calendar date in "YYYY-MM-DD" format, e.g. "2026-09-05".
        party_size: Number of guests; determines each slot's "available"
            value (a slot with some free seats but fewer than party_size
            is reported as unavailable).

    Returns:
        On success: {"success": true, "restaurant_id": int,
            "restaurant_name": str, "date": str, "party_size": int,
            "slots": [
                {"slot": "2026-09-05 11:00", "time": "11:00",
                 "available": bool}, ...
            ]} with exactly 6 entries, one per fixed slot on that date.
            "slot" is the exact string to pass as the `slot` argument to
            create_reservation/modify_reservation.
        On failure (unknown restaurant_id, malformed date, or
        non-positive party_size): {"success": false, "error": "..."}.
    """
    if not _valid_date(date):
        return {"success": False, "error": f"invalid date {date!r}; expected YYYY-MM-DD"}
    if party_size <= 0:
        return {"success": False, "error": "party_size must be positive"}

    try:
        restaurant = _store.get_restaurant(restaurant_id)
    except ReservationError as e:
        return {"success": False, "error": str(e)}

    slots = [
        {
            "slot": slot,
            "time": slot.split(" ")[1],
            "available": _store.check_availability(restaurant_id, slot, party_size),
        }
        for slot in _generate_daily_slots(date)
    ]

    return {
        "success": True,
        "restaurant_id": restaurant.id,
        "restaurant_name": restaurant.name,
        "date": date,
        "party_size": party_size,
        "slots": slots,
    }


# ---------------------------------------------------------------------------
# recommend_restaurants scoring. Weights sum to 1.0 so the blended score is
# a clean 0-1 number; see recommend_restaurants' docstring for the
# per-signal rationale behind these specific values.
CUISINE_MATCH_WEIGHT = 0.30
VIBE_MATCH_WEIGHT = 0.25
RATING_WEIGHT = 0.20
PRICE_FIT_WEIGHT = 0.15
PROXIMITY_WEIGHT = 0.10

RATING_MIN = 1.0
RATING_MAX = 5.0


def _cuisine_score(restaurant, cuisine: str | None) -> float:
    if cuisine is None:
        return 1.0
    return 1.0 if restaurant.cuisine.lower() == cuisine.lower() else 0.0


def _price_fit_score(restaurant, price_range: str | None) -> float:
    if price_range is None:
        return 1.0
    max_distance = len(PRICE_TIERS) - 1
    distance = abs(PRICE_TIERS.index(restaurant.price_tier) - PRICE_TIERS.index(price_range))
    return 1.0 - distance / max_distance


def _proximity_score(restaurant, location: str | None) -> float:
    if location is None:
        return 1.0
    return 1.0 if restaurant.location.lower() == location.lower() else 0.0


def _rating_score(restaurant) -> float:
    clamped = max(RATING_MIN, min(RATING_MAX, restaurant.rating))
    return (clamped - RATING_MIN) / (RATING_MAX - RATING_MIN)


def _vibe_score(restaurant, preference: str | None) -> float:
    if preference is None:
        return 1.0
    tags = _parse_tags(restaurant.tags)
    if not tags:
        return 0.0
    preference_lower = preference.lower()
    matched = sum(1 for tag in tags if tag in preference_lower)
    return matched / len(tags)


@mcp.tool()
def recommend_restaurants(
    date: str,
    party_size: int,
    cuisine: str | None = None,
    location: str | None = None,
    price_range: str | None = None,
    preference: str | None = None,
    exclude_restaurant_id: int | None = None,
    limit: int = 5,
) -> dict:
    """Recommend restaurants for a date/party size, ranked by a weighted
    blend of five 0-1 signals. A restaurant is NEVER recommended unless
    check_availability confirms it has at least one open slot for
    `party_size` on `date` -- availability is a hard gate applied BEFORE
    scoring, not one more weighted signal, so no score can compensate for
    being fully booked.

    Weighted signals (weights sum to 1.0; each is a 0-1 score, higher is
    better fit):
      - cuisine match (0.30): 1.0 if `cuisine` is given and matches the
        restaurant's cuisine exactly (case-insensitive), else 0.0.
        Weighted highest because matching what someone is actually in the
        mood to eat is the single strongest driver of satisfaction.
      - preference/vibe match (0.25): fraction of the restaurant's vibe
        tags -- from the fixed vocabulary romantic, casual,
        family-friendly, lively, quiet, trendy, cozy, upscale,
        outdoor-seating, date-night -- that appear as substrings of the
        freeform `preference` text, case-insensitive. Weighted just under
        cuisine because occasion/mood (date night vs. family dinner) is
        often the deciding factor once cuisine is settled, but it's a
        noisier freeform-text signal so it sits slightly below cuisine's
        exact match.
      - rating (0.20): the restaurant's seeded 1.0-5.0 rating, linearly
        rescaled to 0-1. Always contributes, even with no stated
        preference, since general quality is a reasonable tiebreaker --
        but it's capped below cuisine/vibe so a merely "pretty good"
        restaurant can't outrank a well-matched one on rating alone.
      - price fit (0.15): 1.0 minus the normalized distance between the
        restaurant's price tier and `price_range` (tiers are $, $$, $$$,
        $$$$; one tier apart scores 0.67, two apart 0.33, three apart
        0.0). Graded rather than a hard cutoff because most people have
        some budget flexibility, unlike search_restaurants' hard filter.
      - proximity (0.10): 1.0 if the restaurant's neighborhood exactly
        matches `location`, else 0.0. Weighted lowest and binary because
        this system has no real geographic distance data (see README.md
        "Assumptions") -- it can only tell "same neighborhood" from
        "elsewhere", not how far elsewhere is, and diners are usually
        willing to travel for a good match anyway.

    Leaving `cuisine`, `location`, `price_range`, or `preference` as None
    gives that signal a neutral 1.0 for every candidate (it neither
    penalizes nor boosts anyone) -- an unspecified preference simply drops
    out of the ranking rather than being guessed at.

    Args:
        date: Calendar date to check availability for, "YYYY-MM-DD".
        party_size: Number of guests; must be positive. Used for the
            availability gate, and to drop restaurants whose total
            capacity is below party_size before scoring (they could never
            fit regardless of any single slot's bookings).
        cuisine: Optional exact cuisine to prefer (case-insensitive); see
            search_restaurants' docstring for the valid cuisine list and
            for the common regional/sub-cuisine variants (e.g. "North
            Indian", "Cantonese") normalized automatically before
            matching. An unrecognized value returns a structured error,
            same as search_restaurants.
        location: Optional exact neighborhood to prefer (case-insensitive);
            see search_restaurants' docstring for the valid list.
        price_range: Optional preferred price tier: "$", "$$", "$$$", or
            "$$$$".
        preference: Optional freeform description of desired vibe/mood,
            e.g. "somewhere quiet and romantic for an anniversary". Matched
            against each restaurant's vibe tags by substring, not an LLM
            call -- so phrasing close to the tag words themselves (see the
            vibe-match bullet above) matches best; unrelated phrasing
            simply scores 0 on this signal without erroring.
        exclude_restaurant_id: Optional id to omit from consideration --
            use this for a "that one's full/no good, what else?"
            follow-up instead of re-deriving filters from scratch.
        limit: Max recommendations to return, default 5. Must be positive.

    Returns:
        On success: {"success": true, "date": str, "party_size": int,
            "count": <int>, "recommendations": [
                {"restaurant": {"id": int, "name": str, "cuisine": str,
                     "location": str, "capacity": int, "price_tier": str,
                     "rating": float, "tags": [str, ...]},
                 "score": float,
                 "score_breakdown": {"cuisine_match": float,
                     "vibe_match": float, "rating": float,
                     "price_fit": float, "proximity": float}},
                ...
            ] sorted by "score" descending (0-1, higher is better), at
            most `limit` entries. An empty "recommendations" list is a
            normal, successful result meaning nothing had availability --
            not an error.
        On failure (malformed date, non-positive party_size/limit, or an
        unrecognized cuisine/location/price_range): {"success": false,
        "error": "..."}.
    """
    if not _valid_date(date):
        return {"success": False, "error": f"invalid date {date!r}; expected YYYY-MM-DD"}
    if party_size <= 0:
        return {"success": False, "error": "party_size must be positive"}
    if limit <= 0:
        return {"success": False, "error": "limit must be positive"}

    if cuisine is not None:
        cuisine = normalize_cuisine(cuisine)
        match = next((c for c in CUISINES if c.lower() == cuisine.lower()), None)
        if match is None:
            return {
                "success": False,
                "error": f"unknown cuisine {cuisine!r}. Valid cuisines: {', '.join(CUISINES)}",
            }
        cuisine = match

    if location is not None:
        match = next((n for n in NEIGHBORHOODS if n.lower() == location.lower()), None)
        if match is None:
            return {
                "success": False,
                "error": f"unknown location {location!r}. Valid locations: {', '.join(NEIGHBORHOODS)}",
            }
        location = match

    if price_range is not None and price_range not in PRICE_TIERS:
        return {
            "success": False,
            "error": f"unknown price_range {price_range!r}. Valid price tiers: {', '.join(PRICE_TIERS)}",
        }

    candidates = [
        r
        for r in _store.list_restaurants()
        if r.capacity >= party_size and r.id != exclude_restaurant_id
    ]

    scored = []
    for restaurant in candidates:
        availability = check_availability(restaurant.id, date, party_size)
        if not availability["success"] or not any(s["available"] for s in availability["slots"]):
            continue  # hard gate: never recommend a fully-booked restaurant

        breakdown = {
            "cuisine_match": _cuisine_score(restaurant, cuisine),
            "vibe_match": _vibe_score(restaurant, preference),
            "rating": _rating_score(restaurant),
            "price_fit": _price_fit_score(restaurant, price_range),
            "proximity": _proximity_score(restaurant, location),
        }
        score = (
            breakdown["cuisine_match"] * CUISINE_MATCH_WEIGHT
            + breakdown["vibe_match"] * VIBE_MATCH_WEIGHT
            + breakdown["rating"] * RATING_WEIGHT
            + breakdown["price_fit"] * PRICE_FIT_WEIGHT
            + breakdown["proximity"] * PROXIMITY_WEIGHT
        )
        scored.append((score, restaurant, breakdown))

    scored.sort(key=lambda item: item[0], reverse=True)

    return {
        "success": True,
        "date": date,
        "party_size": party_size,
        "count": min(len(scored), limit),
        "recommendations": [
            {
                "restaurant": _restaurant_to_dict(restaurant),
                "score": round(score, 4),
                "score_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
            }
            for score, restaurant, breakdown in scored[:limit]
        ],
    }


@mcp.tool()
def create_reservation(
    restaurant_id: int,
    customer_name: str,
    customer_phone: str,
    party_size: int,
    slot: str,
) -> dict:
    """Book a reservation. Always call check_availability first and pass
    back its exact "slot" string here -- this tool rejects any slot string
    not on the fixed 2-hour grid (11:00, 13:00, ..., 21:00) rather than
    silently rounding or accepting it.

    Two independent checks can reject the booking:
      - Capacity: the restaurant's total seats booked at this slot (summed
        across all existing confirmed reservations) must not exceed its
        capacity once this party is added.
      - Double-booking: this exact `customer_phone` must not already have
        a confirmed reservation at this restaurant for this exact slot.
        This fires even if capacity would allow it -- it exists to catch
        accidental duplicate bookings by the same customer, independent of
        how full the restaurant is.

    Args:
        restaurant_id: id of an existing restaurant.
        customer_name: Guest's display name, e.g. "Jane Doe".
        customer_phone: Guest's phone number; used as their identity for
            the double-booking check and for any future lookup/modify/
            cancel of this booking. Any non-empty string is accepted --
            no format is enforced -- but reuse the exact same string
            across calls for the same customer.
        party_size: Number of guests; must be a positive integer.
        slot: Exact slot string from check_availability's
            "slots[].slot", e.g. "2026-09-05 19:00".

    Returns:
        On success: {"success": true, "reservation": {"id": int,
            "restaurant_id": int, "customer_name": str,
            "customer_phone": str, "party_size": int, "slot": str,
            "status": "confirmed", "created_at": str}}.
        On failure (unknown restaurant, slot off the fixed grid,
        non-positive party_size, capacity exceeded, or double-booking):
        {"success": false, "error": "<human-readable reason>"}. Nothing
        is booked when this happens.
    """
    if not _is_on_slot_grid(slot):
        return _off_grid_error(slot)

    try:
        reservation = _store.create_reservation(
            restaurant_id=restaurant_id,
            customer_name=customer_name,
            customer_phone=customer_phone,
            party_size=party_size,
            slot=slot,
        )
    except (ReservationError, ValueError) as e:
        return {"success": False, "error": str(e)}

    return {"success": True, "reservation": _reservation_to_dict(reservation)}


@mcp.tool()
def modify_reservation(
    reservation_id: int,
    party_size: int | None = None,
    slot: str | None = None,
) -> dict:
    """Change the party size and/or slot of an existing confirmed
    reservation. Pass only the field(s) you want to change; leave the
    other as None to keep its current value.

    The same capacity and double-booking guards from create_reservation
    are re-checked against the new party_size/slot combination (this
    reservation's own current booking is excluded from that count) -- if
    either would fail, NOTHING is changed and the reservation keeps its
    original party_size and slot.

    Args:
        reservation_id: id of an existing, still-confirmed reservation
            (from create_reservation's or a prior lookup's result). A
            cancelled reservation cannot be modified.
        party_size: New guest count, if changing it. Must be positive.
        slot: New exact slot string, if changing it -- must be one of the
            fixed-grid slots check_availability returns for the desired
            date (e.g. "2026-09-05 19:00"); this tool rejects any slot
            string not on that grid, same as create_reservation.

    Returns:
        On success: {"success": true, "reservation": {...same shape as
            create_reservation's "reservation"...}} reflecting the new
            values.
        On failure (unknown reservation_id, reservation already
        cancelled, slot off the fixed grid, non-positive party_size,
        capacity exceeded, or double-booking): {"success": false,
        "error": "<human-readable reason>"}.
    """
    if slot is not None and not _is_on_slot_grid(slot):
        return _off_grid_error(slot)

    try:
        reservation = _store.modify_reservation(
            reservation_id, party_size=party_size, slot=slot
        )
    except (ReservationError, ValueError) as e:
        return {"success": False, "error": str(e)}

    return {"success": True, "reservation": _reservation_to_dict(reservation)}


@mcp.tool()
def cancel_reservation(reservation_id: int) -> dict:
    """Cancel an existing reservation. This immediately frees its seats
    (for capacity checks) and its customer_phone/slot combination (for
    double-booking checks) for future bookings. Cancelling an
    already-cancelled reservation is safe and simply returns its current
    (cancelled) state rather than erroring.

    Args:
        reservation_id: id of the reservation to cancel.

    Returns:
        On success: {"success": true, "reservation": {..., "status":
            "cancelled", ...}} (same shape as create_reservation's
            "reservation").
        On failure (unknown reservation_id): {"success": false,
        "error": "..."}.
    """
    try:
        reservation = _store.cancel_reservation(reservation_id)
    except ReservationError as e:
        return {"success": False, "error": str(e)}

    return {"success": True, "reservation": _reservation_to_dict(reservation)}


@mcp.tool()
def escalate_to_human(reason: str, context: str = "") -> dict:
    """Hand this conversation off to a human staff member. Call this when
    the request is something none of the other tools can resolve -- e.g.
    a complaint, a special accommodation request, a policy question, or
    repeated tool failures you can't work around. This is currently a
    stub: it only logs the escalation server-side and always returns the
    same fixed confirmation message; no human is actually paged yet, so
    do not imply to the user that one has been notified in real time --
    just relay the returned message as-is.

    Args:
        reason: Short summary of why you're escalating, e.g. "customer
            requesting a refund for a no-show fee".
        context: Optional extra detail a human would need later
            (conversation summary, restaurant/reservation ids involved,
            etc.). Defaults to empty.

    Returns:
        Always: {"success": true, "message": "<fixed confirmation
        message to relay to the user>"}. This tool does not fail.
    """
    logger.info("escalate_to_human reason=%r context=%r", reason, context)
    return {
        "success": True,
        "message": (
            "This has been escalated to a member of our team, who will "
            "follow up with you shortly."
        ),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
