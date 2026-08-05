"""Tests for src/features/panel.py.

Primary fixture is one station over one week (2025-01-06 Mon .. 2025-01-12
Sun) with hand-picked trip timestamps, so every departure/arrival count and
every dense-grid boundary can be verified against a number computed by hand
rather than re-derived from the module under test. A second station with no
capacity/zone match exercises the "keep with null capacity" decision end to
end.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.panel import (  # noqa: E402
    add_calendar_features,
    build_panel,
    run_assertions,
)


def _trip(start_station: str, started_at: dt.datetime, end_station: str, ended_at: dt.datetime) -> dict:
    return {
        "start_station_id": start_station,
        "started_at": started_at,
        "end_station_id": end_station,
        "ended_at": ended_at,
    }


# Station "A": active Mon 2025-01-06 08:00 .. Sun 2025-01-12 23:45 (its
# first departure interval to its last arrival interval).
#   - Two departures land in the 08:00 interval (08:03, 08:10).
#   - One departure lands in the 08:15 interval (08:20).
#   - One arrival lands in the 08:00 interval (08:05, a different trip's end).
#   - One departure lands in Tue 17:45 (17:47).
#   - One arrival lands in the final interval, Sun 23:45 (23:50) -- this is
#     station A's last observed activity, fixing last_interval.
# Station "B": a single trip, and NOT present in the stations fixture below
# -- exercises the null-capacity/zone keep decision.
TRIPS = [
    _trip("A", dt.datetime(2025, 1, 6, 8, 3), "X", dt.datetime(2025, 1, 6, 8, 30)),
    _trip("A", dt.datetime(2025, 1, 6, 8, 10), "X", dt.datetime(2025, 1, 6, 8, 40)),
    _trip("A", dt.datetime(2025, 1, 6, 8, 20), "X", dt.datetime(2025, 1, 6, 8, 50)),
    _trip("X", dt.datetime(2025, 1, 6, 7, 50), "A", dt.datetime(2025, 1, 6, 8, 5)),
    _trip("A", dt.datetime(2025, 1, 7, 17, 47), "X", dt.datetime(2025, 1, 7, 18, 10)),
    _trip("X", dt.datetime(2025, 1, 12, 23, 20), "A", dt.datetime(2025, 1, 12, 23, 50)),
    _trip("B", dt.datetime(2025, 1, 8, 12, 1), "X", dt.datetime(2025, 1, 8, 12, 20)),
]

# Station "X" is a sink for the other end of each trip above so departures
# and arrivals both stay internally consistent; it's not asserted on.
STATIONS = pl.DataFrame(
    {
        "station_id": ["A", "X"],
        "name": ["Station A", "Station X"],
        "lat": [40.73, 40.75],
        "lng": [-73.99, -73.98],
        "capacity": [25, 30],
        "zone_h3": ["882a100d27fffff", "882a100d2ffffff"],
        "zone_agglomerative": ["agglom_0", "agglom_1"],
    }
)


def _weather_fixture() -> pl.DataFrame:
    hours = pl.datetime_range(
        dt.datetime(2025, 1, 6), dt.datetime(2025, 1, 13), interval="1h", eager=True, closed="both"
    )
    n = len(hours)
    return pl.DataFrame(
        {
            "timestamp": hours,
            "temp_c": [5.0] * n,
            "precip_mm": [0.0] * n,
            "wind_kph": [10.0] * n,
            "humidity_pct": [50] * n,
        }
    )


@pytest.fixture
def trips_lf() -> pl.LazyFrame:
    return pl.DataFrame(TRIPS).lazy()


@pytest.fixture
def panel_and_bounds(trips_lf: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    return build_panel(trips_lf, STATIONS, _weather_fixture())


def _station_a(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.filter(pl.col("station_id") == "A").sort("interval_start")


def test_departures_and_arrivals_hand_computed(panel_and_bounds) -> None:
    panel, _ = panel_and_bounds
    a = _station_a(panel)

    def count_at(interval_start: dt.datetime, col: str) -> int:
        row = a.filter(pl.col("interval_start") == interval_start)
        assert row.height == 1, f"expected exactly one row at {interval_start}"
        return row[col][0]

    assert count_at(dt.datetime(2025, 1, 6, 8, 0), "departures") == 2
    assert count_at(dt.datetime(2025, 1, 6, 8, 0), "arrivals") == 1
    assert count_at(dt.datetime(2025, 1, 6, 8, 15), "departures") == 1
    assert count_at(dt.datetime(2025, 1, 6, 8, 15), "arrivals") == 0
    assert count_at(dt.datetime(2025, 1, 7, 17, 45), "departures") == 1
    assert count_at(dt.datetime(2025, 1, 12, 23, 45), "arrivals") == 1
    # An interior interval with genuinely zero activity must still be a
    # present, zero-filled row -- not absent (that's the whole point of dense).
    assert count_at(dt.datetime(2025, 1, 6, 9, 0), "departures") == 0
    assert count_at(dt.datetime(2025, 1, 6, 9, 0), "arrivals") == 0
    assert count_at(dt.datetime(2025, 1, 8, 12, 0), "departures") == 0


def test_dense_grid_bounds_and_row_count(panel_and_bounds) -> None:
    panel, _ = panel_and_bounds
    a = _station_a(panel)

    expected_first = dt.datetime(2025, 1, 6, 8, 0)
    expected_last = dt.datetime(2025, 1, 12, 23, 45)
    assert a["interval_start"].min() == expected_first
    assert a["interval_start"].max() == expected_last

    # Independent row-count arithmetic (not calling any panel.py code):
    # inclusive count of 15-min steps between first and last.
    total_minutes = int((expected_last - expected_first).total_seconds() // 60)
    expected_rows = total_minutes // 15 + 1
    assert a.height == expected_rows


def test_no_rows_before_opening_or_after_closing(panel_and_bounds) -> None:
    panel, _ = panel_and_bounds
    a = _station_a(panel)
    assert a.filter(pl.col("interval_start") < dt.datetime(2025, 1, 6, 8, 0)).height == 0
    assert a.filter(pl.col("interval_start") > dt.datetime(2025, 1, 12, 23, 45)).height == 0


def test_trip_count_invariant(trips_lf: pl.LazyFrame, panel_and_bounds) -> None:
    panel, bounds = panel_and_bounds
    n_trips = len(TRIPS)
    assert panel["departures"].sum() == n_trips
    assert panel["arrivals"].sum() == n_trips
    # run_assertions must also pass cleanly on this fixture.
    run_assertions(panel, trips_lf, bounds)


def test_unmatched_station_keeps_rows_with_null_capacity(panel_and_bounds) -> None:
    panel, _ = panel_and_bounds
    b = panel.filter(pl.col("station_id") == "B")
    assert b.height > 0
    assert b["departures"].sum() == 1
    assert b["capacity"].null_count() == b.height
    assert b["zone_h3"].null_count() == b.height
    assert b["zone_agg"].null_count() == b.height


def test_calendar_features_known_values() -> None:
    panel = pl.DataFrame(
        {
            "interval_start": [
                dt.datetime(2025, 1, 6, 8, 0),  # Monday
                dt.datetime(2025, 1, 12, 23, 45),  # Sunday
                dt.datetime(2025, 1, 1, 0, 0),  # New Year's Day -- holiday
            ]
        }
    )
    out = add_calendar_features(panel)

    row0 = out.row(0, named=True)
    assert row0["dow"] == 0  # Monday == 0
    assert row0["hour"] == 8
    assert row0["hour_of_week"] == 8  # 0 * 24 + 8
    assert row0["month"] == 1
    assert row0["is_holiday"] is False

    row1 = out.row(1, named=True)
    assert row1["dow"] == 6  # Sunday == 6
    assert row1["hour"] == 23
    assert row1["hour_of_week"] == 6 * 24 + 23

    row2 = out.row(2, named=True)
    assert row2["is_holiday"] is True


def test_weather_joined(panel_and_bounds) -> None:
    panel, _ = panel_and_bounds
    assert panel["temp_c"].null_count() == 0
    assert (panel["temp_c"] == 5.0).all()
