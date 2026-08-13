"""Synthetic-fixture tests for src/models/od_shares.py and
src/sim/simulator.py -- Phase 7 forward simulator (SPEC.md §4, RUNBOOK
Phase 7). Each test constructs a small scenario where the correct answer is
known analytically, per CLAUDE.md's "every model function gets a test with
a synthetic fixture" rule. None of these touch real data on disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import od_shares  # noqa: E402
from sim import simulator  # noqa: E402

DAYPART_EDGES = [7, 10, 16, 19, 22]


# ---------------------------------------------------------------------------
# OD backoff-hierarchy tier selection (src/models/od_shares.py)
# ---------------------------------------------------------------------------


def test_build_cell_tier_backs_off_by_threshold():
    """S1@hour10 has 25 trips of its own (>= threshold 10) -> station_hour.
    S2@hour10 has only 3 of its own but its zone (Z2) has 15 at that
    hour-of-week (>= threshold) -> zone_hour. S3@hour10 has neither its own
    nor its zone's hour-of-week total above threshold, but its zone's
    DAYPART total is -> zone_daypart. S4@hour10 has nothing anywhere above
    threshold -> global, the always-available last resort."""
    params = od_shares.ODShareParams(min_trips_per_cell=10, daypart_hour_edges=DAYPART_EDGES)
    daypart_10 = 2  # hour 10 falls in bucket index 2 (10 is the start of the 3rd edge-bounded bucket) on a weekday

    counts = {
        "station_hour": pl.DataFrame(
            {
                "start_station_id": ["S1", "S2", "S3", "S4"],
                "hour_of_week": [10, 10, 10, 10],
                "end_station_id": ["D1", "D1", "D1", "D1"],
                "n": [25, 3, 2, 1],
            }
        ),
        "zone_hour": pl.DataFrame(
            {
                "start_zone_agg": ["Z2"],
                "hour_of_week": [10],
                "end_station_id": ["D1"],
                "n": [15],
            }
        ),
        "zone_daypart": pl.DataFrame(
            {
                "start_zone_agg": ["Z3"],
                "daypart": [daypart_10],
                "end_station_id": ["D1"],
                "n": [12],
            }
        ),
        "global": pl.DataFrame({"end_station_id": ["D1"], "n": [1000]}),
    }
    zone_lookup = pl.DataFrame(
        {"station_id": ["S1", "S2", "S3", "S4"], "zone_agg": ["Z1", "Z2", "Z3", "Z4"]}
    )

    cell_tier = od_shares.build_cell_tier(counts, params, zone_lookup=zone_lookup)
    tier_by_station = dict(zip(cell_tier["start_station_id"].to_list(), cell_tier["tier"].to_list()))

    assert tier_by_station["S1"] == "station_hour"
    assert tier_by_station["S2"] == "zone_hour"
    assert tier_by_station["S3"] == "zone_daypart"
    assert tier_by_station["S4"] == "global"


def test_sample_destinations_for_step_uses_assigned_tier():
    """Three origins, each resolved to a different tier, each tier's
    distribution deterministic (single destination, prob=1.0) -- with a
    deterministic distribution the sampled destination is known exactly,
    not just statistically."""
    cell_tier = pl.DataFrame(
        {
            "start_station_id": ["S1", "S2", "S3"],
            "hour_of_week": [10, 10, 10],
            "start_zone_agg": ["Z1", "Z2", "Z3"],
            "daypart": [2, 2, 2],
            "tier": ["station_hour", "zone_hour", "zone_daypart"],
        }
    )
    model = od_shares.ODShareModel(
        cell_tier=cell_tier,
        station_hour_probs=pl.DataFrame(
            {"start_station_id": ["S1"], "hour_of_week": [10], "end_station_id": ["D_station"], "prob": [1.0]}
        ),
        zone_hour_probs=pl.DataFrame(
            {"start_zone_agg": ["Z2"], "hour_of_week": [10], "end_station_id": ["D_zone_hour"], "prob": [1.0]}
        ),
        zone_daypart_probs=pl.DataFrame(
            {"start_zone_agg": ["Z3"], "daypart": [2], "end_station_id": ["D_zone_daypart"], "prob": [1.0]}
        ),
        global_probs=pl.DataFrame({"end_station_id": ["D_global"], "prob": [1.0]}),
    )
    departures = pl.DataFrame({"station_id": ["S1", "S2", "S3", "S_unseen"], "n": [3, 2, 4, 5]})

    trips = od_shares.sample_destinations_for_step(model, departures, hour_of_week=10, daypart=2, rng=np.random.default_rng(0))

    assert trips.height == 3 + 2 + 4 + 5
    dests_by_origin = {}
    for origin, group in trips.group_by("station_id"):
        origin = origin[0] if isinstance(origin, tuple) else origin
        dests_by_origin[origin] = set(group["dest_station_id"].to_list())
    assert dests_by_origin["S1"] == {"D_station"}
    assert dests_by_origin["S2"] == {"D_zone_hour"}
    assert dests_by_origin["S3"] == {"D_zone_daypart"}
    # S_unseen has no cell_tier row at all (never an origin in trips.parquet
    # history) -- falls straight to global, the only tier that doesn't need
    # an origin-specific match.
    assert dests_by_origin["S_unseen"] == {"D_global"}


# ---------------------------------------------------------------------------
# Reroute-on-full-dock (src/sim/simulator.py)
# ---------------------------------------------------------------------------


def _make_line_network(station_ids: list[str], capacities: list[float], spacing_m: float = 200.0):
    """Stations placed along one meridian, spacing_m apart -- exact
    distances are then just (index difference) * spacing_m, so reroute
    hop/distance expectations are known analytically, not approximate."""
    n = len(station_ids)
    deg_per_m = 1.0 / 111_320.0  # standard approximation, degrees latitude per meter
    lat = np.array([i * spacing_m * deg_per_m for i in range(n)])
    lng = np.zeros(n)
    network = simulator.NetworkArrays(
        station_id=np.array(station_ids),
        capacity=np.array(capacities, dtype=float),
        zone_agg=np.array([None] * n, dtype=object),
        lat=lat,
        lng=lng,
        index_of={sid: i for i, sid in enumerate(station_ids)},
    )
    coords_rad = np.column_stack([np.radians(network.lat), np.radians(network.lng)])
    tree = BallTree(coords_rad, metric="haversine")
    return network, tree, coords_rad


def test_find_reroute_target_finds_nearest_with_room():
    """A, B, C full; D (600m away, 3rd nearest) has room -- must be found
    at exactly hop 3, distance ~600m, with a generous cap."""
    network, tree, coords_rad = _make_line_network(["A", "B", "C", "D"], [5, 5, 5, 5])
    inventory = np.array([5.0, 5.0, 5.0, 2.0])

    target_idx, hops, dist_m = simulator.find_reroute_target(
        0, inventory, network.capacity, tree, coords_rad, max_hops=5, max_radius_m=10_000.0
    )
    assert target_idx == network.index_of["D"]
    assert hops == 3
    assert dist_m == pytest.approx(600.0, rel=0.02)


def test_find_reroute_target_respects_hop_cap():
    """Same scenario, but the only station with room is the 3rd-nearest --
    a hop cap of 2 must fail to find it, even though it exists."""
    network, tree, coords_rad = _make_line_network(["A", "B", "C", "D"], [5, 5, 5, 5])
    inventory = np.array([5.0, 5.0, 5.0, 2.0])

    target_idx, _hops, dist_m = simulator.find_reroute_target(
        0, inventory, network.capacity, tree, coords_rad, max_hops=2, max_radius_m=10_000.0
    )
    assert target_idx is None
    assert np.isnan(dist_m)


def test_find_reroute_target_respects_radius_cap():
    """Same scenario, generous hop budget but a radius cap (500m) tighter
    than the only station with room (600m away) -- must also fail."""
    network, tree, coords_rad = _make_line_network(["A", "B", "C", "D"], [5, 5, 5, 5])
    inventory = np.array([5.0, 5.0, 5.0, 2.0])

    target_idx, _hops, dist_m = simulator.find_reroute_target(
        0, inventory, network.capacity, tree, coords_rad, max_hops=10, max_radius_m=500.0
    )
    assert target_idx is None
    assert np.isnan(dist_m)


# ---------------------------------------------------------------------------
# route_departures: conservation + three-way outcome split
# ---------------------------------------------------------------------------


def test_route_departures_conserves_trip_count_and_matches_expected_split():
    """A is full (0 room) and the sole destination of 6 trips; B is full;
    C has exactly 3 free docks. Expected, worked by hand: 0 direct (A had
    no room), 3 rerouted (fill exactly C's 3 free docks, in the order
    checked -- B first (full, skipped), then C), and the remaining 3 lost
    past cap once both B and C are full and no further candidates exist in
    this 3-station network."""
    network, tree, coords_rad = _make_line_network(["A", "B", "C"], [2, 5, 5])
    inventory = np.array([2.0, 5.0, 2.0])  # A: 2/2 full, B: 5/5 full, C: 2/5 (3 free)
    trips = pl.DataFrame({"station_id": ["X"] * 6, "dest_station_id": ["A"] * 6, "tier": ["global"] * 6})

    direct, rerouted, lost_past_cap, trip_log = simulator.route_departures(
        trips, network, inventory, tree, coords_rad, max_hops=5, max_radius_m=10_000.0
    )

    assert trip_log.height == 6
    assert direct.sum() + rerouted.sum() + lost_past_cap.sum() == 6
    assert direct.sum() == 0
    assert rerouted[network.index_of["C"]] == 3
    assert lost_past_cap[network.index_of["A"]] == 3
    assert inventory[network.index_of["C"]] == pytest.approx(5.0)  # C's 3 free docks now filled
    assert inventory[network.index_of["A"]] == pytest.approx(2.0)  # A never received a direct arrival

    outcome_counts = trip_log["outcome"].value_counts().sort("outcome")
    outcomes = dict(zip(outcome_counts["outcome"].to_list(), outcome_counts["count"].to_list()))
    assert outcomes.get("direct", 0) == 0
    assert outcomes["rerouted"] == 3
    assert outcomes["lost_past_cap"] == 3


def test_route_departures_direct_when_room_available():
    """No stress case: destination has room for everyone -- all trips
    direct, no reroute search even attempted."""
    network, tree, coords_rad = _make_line_network(["A", "B"], [10, 10])
    inventory = np.array([2.0, 0.0])
    trips = pl.DataFrame({"station_id": ["X"] * 4, "dest_station_id": ["A"] * 4, "tier": ["global"] * 4})

    direct, rerouted, lost_past_cap, trip_log = simulator.route_departures(
        trips, network, inventory, tree, coords_rad, max_hops=5, max_radius_m=10_000.0
    )

    assert direct[network.index_of["A"]] == 4
    assert rerouted.sum() == 0
    assert lost_past_cap.sum() == 0
    assert inventory[network.index_of["A"]] == pytest.approx(6.0)
    assert (trip_log["outcome"] == "direct").all()
