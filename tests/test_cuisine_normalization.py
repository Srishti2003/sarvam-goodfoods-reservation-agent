import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server.cuisine_normalization import normalize_cuisine
from mcp_server.seed_data import CUISINES
from mcp_server.server import search_restaurants


def test_north_indian_normalizes_to_indian():
    assert normalize_cuisine("North Indian") == "Indian"


def test_north_indian_lowercase_normalizes_to_indian():
    assert normalize_cuisine("north indian") == "Indian"


def test_south_indian_normalizes_to_indian():
    assert normalize_cuisine("South Indian") == "Indian"


def test_punjabi_normalizes_to_indian():
    assert normalize_cuisine("Punjabi") == "Indian"


def test_mughlai_normalizes_to_indian():
    assert normalize_cuisine("Mughlai") == "Indian"


def test_indo_chinese_normalizes_to_chinese():
    assert normalize_cuisine("Indo-Chinese") == "Chinese"


def test_cantonese_normalizes_to_chinese():
    assert normalize_cuisine("Cantonese") == "Chinese"


def test_neapolitan_normalizes_to_italian():
    assert normalize_cuisine("Neapolitan") == "Italian"


def test_synonym_matching_is_case_insensitive():
    assert normalize_cuisine("PUNJABI") == "Indian"
    assert normalize_cuisine("cAnToNeSe") == "Chinese"


def test_already_canonical_value_is_returned_as_canonical():
    assert normalize_cuisine("Indian") == "Indian"
    assert normalize_cuisine("indian") == "Indian"  # canonical match is also case-insensitive


def test_unknown_cuisine_is_returned_unchanged_not_guessed():
    # No fuzzy/similarity matching: an unrelated or made-up cuisine must
    # come back completely unchanged, never mapped to something close.
    assert normalize_cuisine("Klingon") == "Klingon"
    assert normalize_cuisine("Ethiopian") == "Ethiopian"


def test_synonym_targets_are_real_canonical_categories():
    # Sanity check on the table itself: every canonical cuisine we map a
    # synonym to must actually be a stored category (storage is unchanged
    # by this feature -- see requirement 1).
    for canonical in ("Indian", "Chinese", "Italian"):
        assert canonical in CUISINES


def test_search_restaurants_unknown_cuisine_still_returns_clear_validation_error():
    """Normalization must never mask a genuinely unknown cuisine -- the
    tool layer's existing validation error must still fire, unchanged."""
    result = search_restaurants(cuisine="Klingon")
    assert result["success"] is False
    assert "unknown cuisine" in result["error"]
    assert "Klingon" in result["error"]
