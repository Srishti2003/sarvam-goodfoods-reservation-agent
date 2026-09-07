"""Cuisine synonym normalization.

A small, explicit lookup table mapping common cuisine variants/synonyms to
GoodFoods' canonical cuisine categories (mcp_server/seed_data.py's CUISINES
list) -- e.g. "North Indian" -> "Indian", "Cantonese" -> "Chinese". This
exists so the LLM doesn't have to memorize which of many real-world
cuisine names map to which of our stored categories; search_restaurants
and recommend_restaurants call normalize_cuisine() on their `cuisine`
argument before validating/filtering on it.

Deliberately a finite, explicit mapping, not fuzzy/similarity matching --
that's the whole point: it can only ever produce the mappings listed here,
never guess an unrelated cuisine into a wrong bucket. Anything not in this
table is returned unchanged, so the caller's own "unknown cuisine"
validation still rejects it exactly as before -- normalization never masks
a genuinely unknown value.
"""

from mcp_server.seed_data import CUISINES

# lowercase variant/synonym phrase -> canonical cuisine. Every value here
# must be a member of mcp_server.seed_data.CUISINES. Add new entries here,
# not broader matching logic -- see the module docstring for why.
CUISINE_SYNONYMS: dict[str, str] = {
    "north indian": "Indian",
    "south indian": "Indian",
    "punjabi": "Indian",
    "mughlai": "Indian",
    "indo-chinese": "Chinese",
    "indo chinese": "Chinese",
    "cantonese": "Chinese",
    "neapolitan": "Italian",
}

assert all(canonical in CUISINES for canonical in CUISINE_SYNONYMS.values()), (
    "CUISINE_SYNONYMS maps to a cuisine that isn't in seed_data.CUISINES"
)


def normalize_cuisine(raw: str) -> str:
    """Map `raw` to its canonical cuisine category, case-insensitively.

    - If `raw` already matches a canonical cuisine (in any case), returns
      the canonically-cased form.
    - Else if `raw` matches a known synonym/variant, returns the canonical
      cuisine it maps to.
    - Otherwise returns `raw` completely unchanged -- this function never
      invents a mapping; an unrecognized value is left for the caller's
      own validation to reject with a clear error.
    """
    key = raw.strip().lower()

    for canonical in CUISINES:
        if canonical.lower() == key:
            return canonical

    return CUISINE_SYNONYMS.get(key, raw)
