"""Tests for src/features/zones.py: known lat/lng -> known cell/cluster.

H3 fixture values were generated directly via `h3.latlng_to_cell` for a
fixed NYC coordinate and are asserted as a regression fixture (catches an
accidental resolution swap or a lat/lng argument-order bug), backed up by
property checks that don't depend on the library's internal hashing:
a point ~50m away must land in the same res-8 cell (edge length ~460m),
a point ~5km away must not.

The agglomerative fixtures use synthetic stations at known distances
(~50m within a group, kilometers between groups) so the expected cluster
assignment is known analytically, independent of any real station data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.zones import (  # noqa: E402
    assign_agglomerative_zones,
    assign_h3_zones,
    zone_diameters_m,
)

# Union Square-ish point. Ground truth generated via:
#   h3.latlng_to_cell(40.735863, -73.991084, 8) == "882a100d27fffff"
KNOWN_LAT, KNOWN_LNG = 40.735863, -73.991084
KNOWN_CELL_RES8 = "882a100d27fffff"

# ~1 degree latitude == 111,320m; ~1 degree longitude at this latitude ==
# 111,320 * cos(40.7 deg) == ~84,440m. Used to derive offsets of known
# real-world distance without depending on any geodesy library in the test.
_M_PER_DEG_LAT = 111_320
_M_PER_DEG_LNG = 84_440

AGGLOMERATIVE_THRESHOLD_M = 700.0


def _offset(lat: float, lng: float, north_m: float = 0.0, east_m: float = 0.0) -> tuple[float, float]:
    return lat + north_m / _M_PER_DEG_LAT, lng + east_m / _M_PER_DEG_LNG


def _station_df(rows: list[tuple[str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "station_id": [r[0] for r in rows],
            "name": [r[0] for r in rows],
            "lat": [r[1] for r in rows],
            "lng": [r[2] for r in rows],
            "capacity": [20 for _ in rows],
        }
    )


def test_h3_known_point_maps_to_known_cell() -> None:
    df = _station_df([("s1", KNOWN_LAT, KNOWN_LNG)])
    out = assign_h3_zones(df, resolution=8)
    assert out["zone_h3"].to_list() == [KNOWN_CELL_RES8]


def test_h3_nearby_point_shares_cell() -> None:
    # ~50m away -- well inside a res-8 cell's ~460m edge length.
    lat2, lng2 = _offset(KNOWN_LAT, KNOWN_LNG, north_m=50)
    df = _station_df([("s1", KNOWN_LAT, KNOWN_LNG), ("s2", lat2, lng2)])
    out = assign_h3_zones(df, resolution=8)
    assert out["zone_h3"].to_list()[0] == out["zone_h3"].to_list()[1]


def test_h3_distant_point_different_cell() -> None:
    # ~5km away -- far outside any single res-8 cell.
    lat2, lng2 = _offset(KNOWN_LAT, KNOWN_LNG, north_m=5_000)
    df = _station_df([("s1", KNOWN_LAT, KNOWN_LNG), ("s2", lat2, lng2)])
    out = assign_h3_zones(df, resolution=8)
    cells = out["zone_h3"].to_list()
    assert cells[0] != cells[1]


def test_agglomerative_groups_nearby_stations_into_one_zone() -> None:
    # Three stations mutually within ~50-70m of each other -- all inside
    # the 700m distance threshold, so they must land in the same zone.
    base_lat, base_lng = 40.7300, -73.9900
    a1 = (base_lat, base_lng)
    a2 = _offset(base_lat, base_lng, north_m=50)
    a3 = _offset(base_lat, base_lng, east_m=50)
    df = _station_df([("a1", *a1), ("a2", *a2), ("a3", *a3)])

    out = assign_agglomerative_zones(df, distance_threshold_m=AGGLOMERATIVE_THRESHOLD_M)
    zones = out["zone_agglomerative"].to_list()
    assert len(set(zones)) == 1


def test_agglomerative_separates_distant_clusters() -> None:
    # Two clusters of 3 stations each, ~2km apart -- far outside the 700m
    # threshold, so they must land in different zones even though each
    # cluster is internally tight.
    a_base = (40.7300, -73.9900)
    b_base = _offset(*a_base, north_m=2_000)

    a_rows = [
        ("a1", *a_base),
        ("a2", *_offset(*a_base, north_m=50)),
        ("a3", *_offset(*a_base, east_m=50)),
    ]
    b_rows = [
        ("b1", *b_base),
        ("b2", *_offset(*b_base, north_m=50)),
        ("b3", *_offset(*b_base, east_m=50)),
    ]
    df = _station_df(a_rows + b_rows)

    out = assign_agglomerative_zones(df, distance_threshold_m=AGGLOMERATIVE_THRESHOLD_M)
    zones = out["zone_agglomerative"].to_list()
    a_zones = set(zones[:3])
    b_zones = set(zones[3:])
    assert len(a_zones) == 1
    assert len(b_zones) == 1
    assert a_zones != b_zones


def test_agglomerative_isolated_station_gets_singleton_zone() -> None:
    # One station far (~10km) from everything else -- with no unmerged
    # concept of "noise" in agglomerative clustering, it must simply end up
    # alone in its own zone rather than forced into the nearest cluster.
    a_base = (40.7300, -73.9900)
    lone = _offset(*a_base, north_m=10_000)

    a_rows = [
        ("a1", *a_base),
        ("a2", *_offset(*a_base, north_m=50)),
        ("a3", *_offset(*a_base, east_m=50)),
    ]
    df = _station_df(a_rows + [("lone", *lone)])

    out = assign_agglomerative_zones(df, distance_threshold_m=AGGLOMERATIVE_THRESHOLD_M)
    zones = out["zone_agglomerative"].to_list()

    lone_zone = zones[3]
    assert lone_zone not in set(zones[:3])
    assert zones.count(lone_zone) == 1


def test_agglomerative_no_zone_exceeds_distance_threshold() -> None:
    # Stress test for the DBSCAN-chaining bug this replaced: a dense,
    # contiguous cloud of 300 stations spaced 50-300m apart (the kind of
    # layout that made fixed-radius DBSCAN chain across an entire borough).
    # Complete linkage must never let a zone's internal diameter exceed the
    # distance_threshold, regardless of how contiguous the input is.
    rng = np.random.default_rng(42)
    n = 300
    # Points laid out on a jittered grid so neighbors are consistently
    # close (50-300m spacing) across a ~5km square -- deliberately
    # contiguous, no natural gaps.
    grid_side = int(np.ceil(np.sqrt(n)))
    xs, ys = [], []
    for i in range(n):
        row, col = divmod(i, grid_side)
        xs.append(row * 150.0 + rng.uniform(-50, 50))
        ys.append(col * 150.0 + rng.uniform(-50, 50))

    lats = [40.70 + x / _M_PER_DEG_LAT for x in xs]
    lngs = [-73.95 + y / _M_PER_DEG_LNG for y in ys]
    df = _station_df([(f"s{i}", lats[i], lngs[i]) for i in range(n)])

    out = assign_agglomerative_zones(df, distance_threshold_m=AGGLOMERATIVE_THRESHOLD_M)
    diam = zone_diameters_m(out, "zone_agglomerative")

    assert diam.height > 1  # sanity: it didn't just collapse to one zone
    assert diam["diameter_m"].max() <= AGGLOMERATIVE_THRESHOLD_M + 1e-6
