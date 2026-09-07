"""Deterministic seed data for GoodFoods restaurants.

Generates a fixed, reproducible set of restaurants spanning many
neighborhoods, cuisines, price tiers, and capacities, for populating
mcp_server/pos_store.py's SQLite-backed restaurant table in local dev/demo.

Run directly (`python -m mcp_server.seed_data`) to seed goodfoods.db (only
if it's currently empty -- safe to re-run) and print a distribution
summary of what's in it.
"""

import random
from collections import Counter
from dataclasses import dataclass

SEED = 20260903  # fixed for reproducibility; change deliberately, not casually
NUM_RESTAURANTS = 80  # within the requested 50-100 range

NEIGHBORHOODS = [
    "Indiranagar", "Koramangala", "HSR Layout", "Whitefield", "Jayanagar",
    "MG Road", "Domlur", "JP Nagar", "Malleswaram", "Basavanagudi",
    "Electronic City", "Rajajinagar", "BTM Layout", "Yelahanka",
    "Marathahalli", "Sadashivanagar", "Ulsoor", "Banashankari",
    "Frazer Town", "Cox Town",
]  # real Bangalore localities; 20 (>= the requested 15)

CUISINES = [
    "Italian", "Japanese", "Indian", "Mexican", "Thai", "Chinese",
    "French", "Mediterranean", "American", "Korean", "Vietnamese",
    "Spanish", "Greek", "Lebanese", "Steakhouse", "Seafood",
    "Vegan", "Fusion",
]  # 18 cuisines (>= the requested 10)

PRICE_TIERS = ["$", "$$", "$$$", "$$$$"]  # all 4 tiers

# (min_capacity, max_capacity, weight) -- spans small (2-4) to large (10+)
# groups; weight controls how often a bucket is picked.
CAPACITY_BUCKETS = [
    (2, 4, 0.35),    # small: couples / small groups
    (5, 9, 0.40),    # medium: typical parties
    (10, 20, 0.25),  # large: big groups / events
]

# Global-style vocabulary (Western-template names) -- one half of
# Bangalore's actual restaurant naming mix, see INDIAN_NAME_WORDS below
# for the other half.
NAME_ADJECTIVES = [
    "Golden", "Silver", "Rustic", "Urban", "Copper", "Velvet", "Amber",
    "Emerald", "Crimson", "Ivory", "Sapphire", "Wild", "Humble", "Bold",
    "Cozy", "Hidden", "Royal", "Lucky", "Midnight", "Sunny",
]

NAME_NOUNS = [
    "Table", "Spoon", "Fork", "Kettle", "Hearth", "Garden", "Bistro",
    "Kitchen", "Pantry", "Cellar", "Grill", "Oven", "Plate", "Vine",
    "Barrel", "Lantern", "Terrace", "Corner", "House", "Room",
]

# Indian-style vocabulary -- combined with NAME_NOUNS above (and with each
# other) to produce names that read as plausible for Bangalore's dining
# scene, not just uniformly Western template names. INDIAN_NAME_WORDS mixes
# Hindi/Kannada-flavored words (Kannada being Bangalore's local language)
# with common South Indian eatery-name patterns; these are original,
# generic-sounding words, not copies of specific real restaurant brands.
INDIAN_NAME_WORDS = [
    "Swagath", "Meghana", "Nandhini", "Vasantha", "Shanthi", "Karavalli",
    "Malnad", "Coorg", "Udupi", "Annapurna", "Namma", "Sagar",
    "Vidyarthi", "Empire", "Ganesh", "Lakshmi", "Adiga", "Anand",
    "Sri Krishna", "Mysore",
]

INDIAN_NAME_NOUNS = [
    "Bhavan", "Tiffin Room", "Mess", "Darbar", "Dhaba", "Refreshments",
    "Adda", "Vihar", "Delicacies", "Grand", "Restaurant", "Corner House",
]

MAX_NEIGHBORHOOD_SHARE = 0.15

# Vibe tags for recommend_restaurants' preference-match signal (see
# server.py). Fixed vocabulary rather than freeform so matching against a
# user's preference text stays a simple, predictable substring check.
VIBE_TAGS = [
    "romantic", "casual", "family-friendly", "lively", "quiet", "trendy",
    "cozy", "upscale", "outdoor-seating", "date-night",
]
TAGS_PER_RESTAURANT = (1, 3)  # min, max tags assigned per restaurant

RATING_RANGE = (3.0, 5.0)  # seeded restaurants skew decent-to-great, not 1-5 uniform


@dataclass
class SeedRestaurant:
    name: str
    cuisine: str
    location: str
    capacity: int
    price_tier: str
    rating: float
    tags: str


def _balanced_neighborhoods(rng: random.Random, count: int) -> list[str]:
    """Assign neighborhoods so no single one can exceed ~15% of `count`.

    Rather than relying on random.choice() luck to stay under the cap, split
    `count` evenly across all neighborhoods up front (remainder spread
    across a random subset) and then shuffle -- this makes the 15% cap hold
    by construction, not by chance.
    """
    base, remainder = divmod(count, len(NEIGHBORHOODS))
    counts = {n: base for n in NEIGHBORHOODS}
    for n in rng.sample(NEIGHBORHOODS, remainder):
        counts[n] += 1
    assignments = [n for n, c in counts.items() for _ in range(c)]
    rng.shuffle(assignments)
    return assignments


def _random_capacity(rng: random.Random) -> int:
    lo, hi, _weight = rng.choices(
        CAPACITY_BUCKETS, weights=[w for _, _, w in CAPACITY_BUCKETS]
    )[0]
    return rng.randint(lo, hi)


_LEADING_ARTICLES = ("the ", "a ", "an ")


def _uniqueness_key(name: str) -> str:
    """Normalize a name for duplicate detection: strip a leading article
    and lowercase. Without this, "Copper Cellar" and "The Copper Cellar"
    compare as different strings and both get generated as separate
    restaurants -- a real bug this project shipped with once already.
    """
    key = name.strip().lower()
    for article in _LEADING_ARTICLES:
        if key.startswith(article):
            return key[len(article):]
    return key


def _unique_name(rng: random.Random, cuisine: str, neighborhood: str, used_keys: set[str]) -> str:
    templates = [
        # Global-style
        lambda: f"{rng.choice(NAME_ADJECTIVES)} {rng.choice(NAME_NOUNS)}",
        lambda: f"The {rng.choice(NAME_ADJECTIVES)} {rng.choice(NAME_NOUNS)}",
        lambda: f"{neighborhood} {rng.choice(NAME_NOUNS)}",
        lambda: f"{cuisine} {rng.choice(NAME_NOUNS)}",
        # Indian-style
        lambda: f"{rng.choice(INDIAN_NAME_WORDS)} {rng.choice(INDIAN_NAME_NOUNS)}",
        lambda: f"{rng.choice(INDIAN_NAME_WORDS)} {rng.choice(NAME_NOUNS)}",
        lambda: f"{neighborhood} {rng.choice(INDIAN_NAME_NOUNS)}",
        lambda: f"New {rng.choice(INDIAN_NAME_WORDS)} {rng.choice(INDIAN_NAME_NOUNS)}",
    ]
    for _ in range(200):  # generous retry budget; collisions are rare
        name = rng.choice(templates)()
        key = _uniqueness_key(name)
        if key not in used_keys:
            used_keys.add(key)
            return name
    raise RuntimeError("exhausted retries generating a unique restaurant name")


def generate_restaurants(count: int = NUM_RESTAURANTS, seed: int = SEED) -> list[SeedRestaurant]:
    """Deterministically generate `count` restaurants for a given `seed`."""
    if not 50 <= count <= 100:
        raise ValueError("count must be within the 50-100 range")

    rng = random.Random(seed)
    neighborhoods = _balanced_neighborhoods(rng, count)
    used_name_keys: set[str] = set()

    restaurants = []
    for neighborhood in neighborhoods:
        cuisine = rng.choice(CUISINES)
        num_tags = rng.randint(*TAGS_PER_RESTAURANT)
        restaurants.append(
            SeedRestaurant(
                name=_unique_name(rng, cuisine, neighborhood, used_name_keys),
                cuisine=cuisine,
                location=neighborhood,
                capacity=_random_capacity(rng),
                price_tier=rng.choice(PRICE_TIERS),
                rating=round(rng.uniform(*RATING_RANGE), 1),
                tags=",".join(sorted(rng.sample(VIBE_TAGS, k=num_tags))),
            )
        )
    return restaurants


def seed_store(store, count: int = NUM_RESTAURANTS, seed: int = SEED) -> list:
    """Populate `store` (a PosStore) with the generated restaurants.

    Intended for a fresh store: re-running against an already-seeded store
    will raise on the UNIQUE name constraint rather than silently duplicating.
    """
    return [
        store.add_restaurant(
            name=r.name,
            cuisine=r.cuisine,
            location=r.location,
            capacity=r.capacity,
            price_tier=r.price_tier,
            rating=r.rating,
            tags=r.tags,
        )
        for r in generate_restaurants(count, seed)
    ]


def _print_summary(restaurants: list[SeedRestaurant]) -> None:
    total = len(restaurants)
    by_cuisine = Counter(r.cuisine for r in restaurants)
    by_location = Counter(r.location for r in restaurants)
    by_price = Counter(r.price_tier for r in restaurants)

    print(f"Generated {total} restaurants (seed={SEED})\n")

    print("By cuisine:")
    for cuisine, count in sorted(by_cuisine.items(), key=lambda kv: -kv[1]):
        print(f"  {cuisine:<15} {count:>3}  ({count / total:.1%})")

    print("\nBy neighborhood:")
    for location, count in sorted(by_location.items(), key=lambda kv: -kv[1]):
        print(f"  {location:<20} {count:>3}  ({count / total:.1%})")

    print("\nBy price tier:")
    for tier in PRICE_TIERS:
        count = by_price.get(tier, 0)
        print(f"  {tier:<5} {count:>3}  ({count / total:.1%})")

    max_location, max_count = max(by_location.items(), key=lambda kv: kv[1])
    max_share = max_count / total
    print(f"\nLargest neighborhood share: {max_location} at {max_share:.1%}")

    assert max_share <= MAX_NEIGHBORHOOD_SHARE, (
        f"{max_location} has {max_share:.1%} of restaurants, "
        f"exceeding the {MAX_NEIGHBORHOOD_SHARE:.0%} cap"
    )
    print(f"OK: no neighborhood exceeds {MAX_NEIGHBORHOOD_SHARE:.0%}")


if __name__ == "__main__":
    from mcp_server.pos_store import PosStore

    _store = PosStore()
    try:
        if _store.list_restaurants():
            print(f"goodfoods.db already has {len(_store.list_restaurants())} restaurants -- not re-seeding.\n")
        else:
            print(f"goodfoods.db is empty -- seeding {NUM_RESTAURANTS} restaurants ...\n")
            seed_store(_store)
    finally:
        _store.close()

    _print_summary(generate_restaurants())
