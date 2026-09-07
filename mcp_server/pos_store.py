"""SQLite-backed POS store for restaurants and reservations.

This is the source of truth for reservation state. It's backed by a real
SQLite file (not an in-memory dict) because Streamlit reruns the whole
script on every interaction — anything held only in Python module/session
state would be lost or duplicated across reruns, whereas a SQLite file
persists independently of the Python process.

No MCP wrapping here — this module is plain Python that the MCP tool
layer (server.py) will call into once tools are added.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "goodfoods.db"


class ReservationError(Exception):
    """Base class for all reservation-store failures."""


class RestaurantNotFoundError(ReservationError):
    pass


class ReservationNotFoundError(ReservationError):
    pass


class CapacityExceededError(ReservationError):
    pass


class DoubleBookingError(ReservationError):
    pass


@dataclass
class Restaurant:
    id: int
    name: str
    cuisine: str
    location: str
    capacity: int
    price_tier: str
    rating: float
    tags: str  # comma-separated vibe tags, e.g. "romantic,quiet"


@dataclass
class Reservation:
    id: int
    restaurant_id: int
    customer_name: str
    customer_phone: str
    party_size: int
    slot: str
    status: str
    created_at: str


class PosStore:
    """SQLite-backed store for restaurants and reservations.

    `slot` is an opaque caller-formatted string identifying a booking
    window (e.g. "2026-09-05 19:00") — this layer does no date parsing or
    slot-granularity logic, it just guards capacity and duplicate bookings
    per (restaurant_id, slot).
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS restaurants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                cuisine TEXT NOT NULL,
                location TEXT NOT NULL,
                capacity INTEGER NOT NULL CHECK (capacity > 0),
                price_tier TEXT NOT NULL DEFAULT '$$'
                    CHECK (price_tier IN ('$', '$$', '$$$', '$$$$')),
                rating REAL NOT NULL DEFAULT 4.0 CHECK (rating BETWEEN 1.0 AND 5.0),
                tags TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS reservations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                restaurant_id INTEGER NOT NULL REFERENCES restaurants(id),
                customer_name TEXT NOT NULL,
                customer_phone TEXT NOT NULL,
                party_size INTEGER NOT NULL CHECK (party_size > 0),
                slot TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'confirmed',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_reservations_restaurant_slot
                ON reservations (restaurant_id, slot, status);
            """
        )
        self._conn.commit()

    # ---- restaurants ----

    def add_restaurant(
        self,
        name: str,
        cuisine: str,
        location: str,
        capacity: int,
        price_tier: str = "$$",
        rating: float = 4.0,
        tags: str = "",
    ) -> Restaurant:
        cur = self._conn.execute(
            "INSERT INTO restaurants (name, cuisine, location, capacity, price_tier, rating, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, cuisine, location, capacity, price_tier, rating, tags),
        )
        self._conn.commit()
        return self.get_restaurant(cur.lastrowid)

    def list_restaurants(self) -> list[Restaurant]:
        rows = self._conn.execute("SELECT * FROM restaurants ORDER BY id").fetchall()
        return [self._row_to_restaurant(row) for row in rows]

    def get_restaurant(self, restaurant_id: int) -> Restaurant:
        row = self._conn.execute(
            "SELECT * FROM restaurants WHERE id = ?", (restaurant_id,)
        ).fetchone()
        if row is None:
            raise RestaurantNotFoundError(f"no restaurant with id={restaurant_id}")
        return self._row_to_restaurant(row)

    # ---- reservations ----

    def check_availability(self, restaurant_id: int, slot: str, party_size: int) -> bool:
        """True if `party_size` more guests fit in `slot` without exceeding capacity."""
        restaurant = self.get_restaurant(restaurant_id)
        booked = self._booked_seats(restaurant_id, slot)
        return booked + party_size <= restaurant.capacity

    def create_reservation(
        self,
        restaurant_id: int,
        customer_name: str,
        customer_phone: str,
        party_size: int,
        slot: str,
    ) -> Reservation:
        if party_size <= 0:
            raise ValueError("party_size must be positive")

        restaurant = self.get_restaurant(restaurant_id)  # raises if missing

        if self._has_existing_booking(restaurant_id, slot, customer_phone):
            raise DoubleBookingError(
                f"{customer_phone} already has a confirmed reservation at "
                f"restaurant {restaurant_id} for slot {slot!r}"
            )

        booked = self._booked_seats(restaurant_id, slot)
        if booked + party_size > restaurant.capacity:
            raise CapacityExceededError(
                f"restaurant {restaurant_id} capacity {restaurant.capacity} exceeded for "
                f"slot {slot!r}: {booked} already booked + {party_size} requested"
            )

        created_at = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """INSERT INTO reservations
               (restaurant_id, customer_name, customer_phone, party_size, slot, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'confirmed', ?)""",
            (restaurant_id, customer_name, customer_phone, party_size, slot, created_at),
        )
        self._conn.commit()
        return self.get_reservation(cur.lastrowid)

    def modify_reservation(
        self,
        reservation_id: int,
        *,
        party_size: int | None = None,
        slot: str | None = None,
    ) -> Reservation:
        existing = self.get_reservation(reservation_id)  # raises if missing
        if existing.status != "confirmed":
            raise ReservationError(
                f"reservation {reservation_id} is {existing.status!r} and cannot be modified"
            )

        new_party_size = party_size if party_size is not None else existing.party_size
        new_slot = slot if slot is not None else existing.slot
        if new_party_size <= 0:
            raise ValueError("party_size must be positive")

        restaurant = self.get_restaurant(existing.restaurant_id)

        if self._has_existing_booking(
            existing.restaurant_id, new_slot, existing.customer_phone, exclude_id=reservation_id
        ):
            raise DoubleBookingError(
                f"{existing.customer_phone} already has a confirmed reservation at "
                f"restaurant {existing.restaurant_id} for slot {new_slot!r}"
            )

        booked = self._booked_seats(existing.restaurant_id, new_slot, exclude_id=reservation_id)
        if booked + new_party_size > restaurant.capacity:
            raise CapacityExceededError(
                f"restaurant {existing.restaurant_id} capacity {restaurant.capacity} exceeded "
                f"for slot {new_slot!r}: {booked} already booked + {new_party_size} requested"
            )

        self._conn.execute(
            "UPDATE reservations SET party_size = ?, slot = ? WHERE id = ?",
            (new_party_size, new_slot, reservation_id),
        )
        self._conn.commit()
        return self.get_reservation(reservation_id)

    def cancel_reservation(self, reservation_id: int) -> Reservation:
        self.get_reservation(reservation_id)  # raises if missing
        self._conn.execute(
            "UPDATE reservations SET status = 'cancelled' WHERE id = ?", (reservation_id,)
        )
        self._conn.commit()
        return self.get_reservation(reservation_id)

    def get_reservation(self, reservation_id: int) -> Reservation:
        row = self._conn.execute(
            "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
        ).fetchone()
        if row is None:
            raise ReservationNotFoundError(f"no reservation with id={reservation_id}")
        return self._row_to_reservation(row)

    def list_reservations(self, restaurant_id: int | None = None) -> list[Reservation]:
        if restaurant_id is None:
            rows = self._conn.execute("SELECT * FROM reservations ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM reservations WHERE restaurant_id = ? ORDER BY id",
                (restaurant_id,),
            ).fetchall()
        return [self._row_to_reservation(row) for row in rows]

    # ---- internal helpers ----

    def _booked_seats(self, restaurant_id: int, slot: str, exclude_id: int | None = None) -> int:
        query = (
            "SELECT COALESCE(SUM(party_size), 0) AS total FROM reservations "
            "WHERE restaurant_id = ? AND slot = ? AND status = 'confirmed'"
        )
        params: list = [restaurant_id, slot]
        if exclude_id is not None:
            query += " AND id != ?"
            params.append(exclude_id)
        row = self._conn.execute(query, params).fetchone()
        return row["total"]

    def _has_existing_booking(
        self,
        restaurant_id: int,
        slot: str,
        customer_phone: str,
        exclude_id: int | None = None,
    ) -> bool:
        query = (
            "SELECT 1 FROM reservations WHERE restaurant_id = ? AND slot = ? "
            "AND customer_phone = ? AND status = 'confirmed'"
        )
        params: list = [restaurant_id, slot, customer_phone]
        if exclude_id is not None:
            query += " AND id != ?"
            params.append(exclude_id)
        return self._conn.execute(query, params).fetchone() is not None

    @staticmethod
    def _row_to_restaurant(row: sqlite3.Row) -> Restaurant:
        return Restaurant(
            id=row["id"],
            name=row["name"],
            cuisine=row["cuisine"],
            location=row["location"],
            capacity=row["capacity"],
            price_tier=row["price_tier"],
            rating=row["rating"],
            tags=row["tags"],
        )

    @staticmethod
    def _row_to_reservation(row: sqlite3.Row) -> Reservation:
        return Reservation(
            id=row["id"],
            restaurant_id=row["restaurant_id"],
            customer_name=row["customer_name"],
            customer_phone=row["customer_phone"],
            party_size=row["party_size"],
            slot=row["slot"],
            status=row["status"],
            created_at=row["created_at"],
        )
