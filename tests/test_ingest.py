"""Synthetic-fixture tests for src/ingest/trips.py, one fixture per schema era.

Values are hand-picked to exercise every filter/assertion in isolation:
one row that must survive, and one row per drop reason. Expected survivor
counts are asserted explicitly rather than just "fewer rows than before" --
if a filter starts dropping the wrong rows, this must fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest.trips import (  # noqa: E402
    apply_filters,
    build_raw_lazyframe,
    detect_era,
    normalize_era_a,
    normalize_era_b,
    run_assertions,
)

ERA_A_HEADER = (
    "tripduration,starttime,stoptime,start station id,start station name,"
    "start station latitude,start station longitude,end station id,"
    "end station name,end station latitude,end station longitude,bikeid,"
    "usertype,birth year,gender"
)

# Row A1: normal trip, different stations, 5 min -- must survive.
# Row A2: 30s trip -- dropped by the <60s filter.
# Row A3: self-loop, 90s -- dropped by the self-loop-<2min filter.
# Row A4: self-loop, 200s (>=2min) -- must survive (not a short self-loop).
ERA_A_ROWS = [
    "300,2019-09-01 00:00:00.0000,2019-09-01 00:05:00.0000,100,Start St,"
    "40.7,-73.9,200,End St,40.8,-74.0,1,Subscriber,1980,1",
    "30,2019-09-01 01:00:00.0000,2019-09-01 01:00:30.0000,100,Start St,"
    "40.7,-73.9,200,End St,40.8,-74.0,2,Customer,1990,0",
    "90,2019-09-01 02:00:00.0000,2019-09-01 02:01:30.0000,100,Start St,"
    "40.7,-73.9,100,Start St,40.7,-73.9,3,Subscriber,1985,1",
    "200,2019-09-01 03:00:00.0000,2019-09-01 03:03:20.0000,100,Start St,"
    "40.7,-73.9,100,Start St,40.7,-73.9,4,Customer,1995,2",
]

ERA_B_HEADER = (
    "ride_id,rideable_type,started_at,ended_at,start_station_name,"
    "start_station_id,end_station_name,end_station_id,start_lat,start_lng,"
    "end_lat,end_lng,member_casual"
)

# Row B1: normal trip -- must survive.
# Row B2: null end_station_id/name (dockless drop) -- dropped by null-station filter.
# Row B3: 45s trip -- dropped by the <60s filter.
ERA_B_ROWS = [
    "R1,classic_bike,2021-03-01 00:00:00.000,2021-03-01 00:10:00.000,Start St,"
    "100.01,End St,200.02,40.7,-73.9,40.8,-74.0,member",
    "R2,electric_bike,2021-03-01 01:00:00.000,2021-03-01 01:05:00.000,Start St,"
    "100.01,,,40.7,-73.9,,,casual",
    "R3,classic_bike,2021-03-01 02:00:00.000,2021-03-01 02:00:45.000,Start St,"
    "100.01,End St,200.02,40.7,-73.9,40.8,-74.0,member",
]


@pytest.fixture
def era_a_csv(tmp_path: Path) -> Path:
    p = tmp_path / "201909-citibike-tripdata.csv"
    p.write_text(ERA_A_HEADER + "\n" + "\n".join(ERA_A_ROWS) + "\n")
    return p


@pytest.fixture
def era_b_csv(tmp_path: Path) -> Path:
    p = tmp_path / "202103-citibike-tripdata.csv"
    p.write_text(ERA_B_HEADER + "\n" + "\n".join(ERA_B_ROWS) + "\n")
    return p


def test_detect_era_a(era_a_csv: Path) -> None:
    assert detect_era(era_a_csv) == "A"


def test_detect_era_b(era_b_csv: Path) -> None:
    assert detect_era(era_b_csv) == "B"


def test_detect_era_unrecognized(tmp_path: Path) -> None:
    p = tmp_path / "mystery.csv"
    p.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="match neither Era A nor Era B"):
        detect_era(p)


def test_normalize_era_a_maps_to_target_columns(era_a_csv: Path) -> None:
    df = normalize_era_a(era_a_csv).collect()
    assert set(df.columns) == {
        "ride_id",
        "started_at",
        "ended_at",
        "start_station_id",
        "end_station_id",
        "start_lat",
        "start_lng",
        "end_lat",
        "end_lng",
        "rideable_type",
        "member_casual",
    }
    assert df["rideable_type"].is_null().all()  # not recorded pre-transition
    assert df["member_casual"].to_list() == ["member", "casual", "member", "casual"]
    assert df["start_station_id"].dtype == pl.Utf8
    assert df["start_station_id"].to_list() == ["100", "100", "100", "100"]


def test_normalize_era_b_maps_to_target_columns(era_b_csv: Path) -> None:
    df = normalize_era_b(era_b_csv).collect()
    assert df["start_station_id"].dtype == pl.Utf8
    assert df["start_station_id"].to_list() == ["100.01", "100.01", "100.01"]
    assert df["ride_id"].to_list() == ["R1", "R2", "R3"]


def test_filters_drop_expected_rows_era_a(era_a_csv: Path) -> None:
    lf = build_raw_lazyframe([era_a_csv])
    filtered = apply_filters(lf).collect()
    # Only row A1 (normal) and A4 (self-loop >= 2min) survive.
    assert filtered.height == 2
    assert set(filtered["ended_at"].dt.strftime("%H:%M:%S").to_list()) == {
        "00:05:00",
        "03:03:20",
    }


def test_filters_drop_expected_rows_era_b(era_b_csv: Path) -> None:
    lf = build_raw_lazyframe([era_b_csv])
    filtered = apply_filters(lf).collect()
    # Only row B1 survives: B2 has null end station, B3 is a 45s trip.
    assert filtered.height == 1
    assert filtered["ride_id"].to_list() == ["R1"]


def test_assertions_pass_on_filtered_data(era_a_csv: Path, era_b_csv: Path) -> None:
    lf = build_raw_lazyframe([era_a_csv, era_b_csv])
    filtered = apply_filters(lf)
    run_assertions(filtered)  # must not raise


def test_mixed_era_files_concat_to_stable_schema(era_a_csv: Path, era_b_csv: Path) -> None:
    lf = build_raw_lazyframe([era_a_csv, era_b_csv])
    df = apply_filters(lf).collect()
    assert df.height == 3  # 2 from era A + 1 from era B
    assert df["started_at"].dtype == pl.Datetime
    assert df["ended_at"].dtype == pl.Datetime
