import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server.pos_store import (
    CapacityExceededError,
    DoubleBookingError,
    PosStore,
)

SLOT_A = "2026-09-05 19:00"
SLOT_B = "2026-09-05 20:00"


@pytest.fixture
def store(tmp_path):
    s = PosStore(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def restaurant(store):
    return store.add_restaurant(
        name="Test Bistro", cuisine="French", location="Downtown", capacity=4
    )


def test_exact_capacity_fit(store, restaurant):
    reservation = store.create_reservation(
        restaurant_id=restaurant.id,
        customer_name="Alice",
        customer_phone="111",
        party_size=restaurant.capacity,
        slot=SLOT_A,
    )

    assert reservation.party_size == restaurant.capacity
    assert reservation.status == "confirmed"
    # capacity is now exactly used up: even one more seat should not fit
    assert store.check_availability(restaurant.id, SLOT_A, 1) is False


def test_over_capacity_rejection(store, restaurant):
    store.create_reservation(
        restaurant_id=restaurant.id,
        customer_name="Alice",
        customer_phone="111",
        party_size=3,
        slot=SLOT_A,
    )

    with pytest.raises(CapacityExceededError):
        store.create_reservation(
            restaurant_id=restaurant.id,
            customer_name="Bob",
            customer_phone="222",
            party_size=2,  # 3 + 2 = 5 > capacity 4
            slot=SLOT_A,
        )

    # the failed attempt must not have been persisted
    assert store._booked_seats(restaurant.id, SLOT_A) == 3


def test_double_booking_same_slot_rejected(store, restaurant):
    store.create_reservation(
        restaurant_id=restaurant.id,
        customer_name="Alice",
        customer_phone="111",
        party_size=1,
        slot=SLOT_A,
    )

    # same customer, same restaurant, same slot again -- rejected even
    # though capacity would technically allow it (1 + 1 <= 4)
    with pytest.raises(DoubleBookingError):
        store.create_reservation(
            restaurant_id=restaurant.id,
            customer_name="Alice",
            customer_phone="111",
            party_size=1,
            slot=SLOT_A,
        )

    assert store._booked_seats(restaurant.id, SLOT_A) == 1


def test_cancel_then_rebook_same_slot(store, restaurant):
    first = store.create_reservation(
        restaurant_id=restaurant.id,
        customer_name="Alice",
        customer_phone="111",
        party_size=restaurant.capacity,
        slot=SLOT_A,
    )

    store.cancel_reservation(first.id)
    assert store.get_reservation(first.id).status == "cancelled"

    # slot should be fully free again since the cancelled reservation no
    # longer counts towards capacity or double-booking
    second = store.create_reservation(
        restaurant_id=restaurant.id,
        customer_name="Bob",
        customer_phone="222",
        party_size=restaurant.capacity,
        slot=SLOT_A,
    )

    assert second.status == "confirmed"
    assert store._booked_seats(restaurant.id, SLOT_A) == restaurant.capacity


def test_modify_into_conflicting_slot_rejected(store, restaurant):
    store.create_reservation(
        restaurant_id=restaurant.id,
        customer_name="Alice",
        customer_phone="111",
        party_size=restaurant.capacity,
        slot=SLOT_A,
    )
    reservation_b = store.create_reservation(
        restaurant_id=restaurant.id,
        customer_name="Bob",
        customer_phone="222",
        party_size=restaurant.capacity,
        slot=SLOT_B,
    )

    # moving B into A's already-full slot must be rejected
    with pytest.raises(CapacityExceededError):
        store.modify_reservation(reservation_b.id, slot=SLOT_A)

    # reservation must be unchanged after the failed modification
    unchanged = store.get_reservation(reservation_b.id)
    assert unchanged.slot == SLOT_B
    assert unchanged.party_size == restaurant.capacity
