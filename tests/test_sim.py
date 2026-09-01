"""Synthetic-fixture tests for src/models/od_shares.py and
src/sim/simulator.py -- Phase 7 forward simulator (SPEC.md §4, RUNBOOK
Phase 7). Each test constructs a small scenario where the correct answer is
known analytically, per CLAUDE.md's "every model function gets a test with
a synthetic fixture" rule. None of these touch real data on disk.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import demand  # noqa: E402
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


# ---------------------------------------------------------------------------
# Phase 9 (SPEC.md §8): induced-move injection + demand_multiplier
# ---------------------------------------------------------------------------


def test_induced_step_dicts_none_or_empty_returns_empty_dicts():
    network, _tree, _coords = _make_line_network(["A", "B"], [10, 10])
    calendar = pl.DataFrame({"interval_start": ["t0", "t1"], "hour_of_week": [10, 11]})
    empty = pl.DataFrame(schema={"origin_station_id": pl.String, "dest_station_id": pl.String, "hour_of_week": pl.Int64, "induced_trips_per_hour": pl.Float64})

    for induced in (None, empty):
        induced_in, induced_out = simulator._induced_step_dicts(induced, calendar, network)
        assert induced_in == {}
        assert induced_out == {}


def test_induced_step_dicts_explodes_hour_of_week_across_matching_steps_at_quarter_rate():
    """One induced move, A->C, 8 trips/hour at hour_of_week=10. The
    simulated week has two 15-min steps at hour_of_week=10 (t0, t1) and one
    at hour_of_week=11 (t2) -- expect 8/4=2.0 bikes/step added to C and
    removed from A at BOTH hour-10 steps, and no entry at all for t2 (the
    hour-11 step never matches, same "absent key defaults to zero" contract
    run_simulation already relies on for baseline N)."""
    network, _tree, _coords = _make_line_network(["A", "B", "C"], [10, 10, 10])
    calendar = pl.DataFrame({"interval_start": ["t0", "t1", "t2"], "hour_of_week": [10, 10, 11]})
    induced = pl.DataFrame(
        {"origin_station_id": ["A"], "dest_station_id": ["C"], "hour_of_week": [10], "induced_trips_per_hour": [8.0]}
    )

    induced_in, induced_out = simulator._induced_step_dicts(induced, calendar, network)

    assert set(induced_in.keys()) == {"t0", "t1"}
    assert set(induced_out.keys()) == {"t0", "t1"}
    for t in ("t0", "t1"):
        assert induced_in[t][network.index_of["C"]] == pytest.approx(2.0)
        assert induced_in[t][network.index_of["A"]] == pytest.approx(0.0)
        assert induced_out[t][network.index_of["A"]] == pytest.approx(2.0)
        assert induced_out[t][network.index_of["C"]] == pytest.approx(0.0)


def test_induced_step_dicts_sums_multiple_moves_sharing_a_station():
    """B is simultaneously an origin (funding A) and a destination (funded
    by C) at the same hour -- induced_in/induced_out must each reflect ONLY
    their own side's flows, summed independently, not netted against each
    other."""
    network, _tree, _coords = _make_line_network(["A", "B", "C"], [10, 10, 10])
    calendar = pl.DataFrame({"interval_start": ["t0"], "hour_of_week": [10]})
    induced = pl.DataFrame(
        {
            "origin_station_id": ["B", "C"],
            "dest_station_id": ["A", "B"],
            "hour_of_week": [10, 10],
            "induced_trips_per_hour": [4.0, 4.0],
        }
    )

    induced_in, induced_out = simulator._induced_step_dicts(induced, calendar, network)

    assert induced_in["t0"][network.index_of["A"]] == pytest.approx(1.0)
    assert induced_in["t0"][network.index_of["B"]] == pytest.approx(1.0)
    assert induced_out["t0"][network.index_of["B"]] == pytest.approx(1.0)
    assert induced_out["t0"][network.index_of["C"]] == pytest.approx(1.0)


class _FakeGBT:
    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def _fake_fitted_departures(rate: float, station_ids: list[str]) -> demand.FittedDirection:
    """Mirrors tests/test_demand.py's _fake_fitted -- a GBT stub that
    ignores feature content and always predicts `rate`, so run_step's
    Poisson sampling rate is known exactly (rate * demand_multiplier)."""
    return demand.FittedDirection(
        spec=demand.DEPARTURES,
        gbt=_FakeGBT(rate),
        glm=None,
        station_enc=pl.DataFrame({"station_id": station_ids, "station_id_target_enc": [0.0] * len(station_ids)}),
        station_global_mean=0.0,
        zone_enc=pl.DataFrame({"zone_agg": ["z1"], "zone_agg_target_enc": [0.0]}),
        zone_global_mean=0.0,
        bucket_bias={},
    )


def _empty_od_model() -> od_shares.ODShareModel:
    return od_shares.ODShareModel(
        cell_tier=pl.DataFrame(
            schema={"start_station_id": pl.String, "hour_of_week": pl.Int64, "start_zone_agg": pl.String, "daypart": pl.Int64, "tier": pl.String}
        ),
        station_hour_probs=pl.DataFrame(schema={"start_station_id": pl.String, "hour_of_week": pl.Int64, "end_station_id": pl.String, "prob": pl.Float64}),
        zone_hour_probs=pl.DataFrame(schema={"start_zone_agg": pl.String, "hour_of_week": pl.Int64, "end_station_id": pl.String, "prob": pl.Float64}),
        zone_daypart_probs=pl.DataFrame(schema={"start_zone_agg": pl.String, "daypart": pl.Int64, "end_station_id": pl.String, "prob": pl.Float64}),
        global_probs=pl.DataFrame(schema={"end_station_id": pl.String, "prob": pl.Float64}),
    )


def _global_only_od_model(dest_id: str) -> od_shares.ODShareModel:
    model = _empty_od_model()
    model.global_probs = pl.DataFrame({"end_station_id": [dest_id], "prob": [1.0]})
    return model


def _calendar_row(interval_start: str, hour_of_week: int = 10, daypart: int = 2) -> dict:
    return {
        "interval_start": interval_start,
        "is_holiday": False,
        "hour_of_week": hour_of_week,
        "daypart": daypart,
        "month": 6,
        "temp_c": 20.0,
        "precip_mm": 0.0,
        "precip_lag1h": 0.0,
        "precip_lag2h": 0.0,
        "wind_kph": 5.0,
        "humidity_pct": 50,
    }


def _default_sim_params() -> simulator.SimulationParams:
    return simulator.SimulationParams(
        daypart_hour_edges=[7, 10, 16, 19, 22], max_reroute_radius_m=1000.0, max_reroute_hops=3, seed=0,
        validation_week_start="2025-10-06",
    )


def _make_line_network_with_zone(station_ids: list[str], capacities: list[float], zone: str = "z1"):
    """_make_line_network gives every station a null zone_agg (fine for the
    reroute-only tests above), which polars infers as an Object column and
    can't hash-join against a real String zone_agg -- run_step's feature
    build DOES join on zone_agg (against FittedDirection.zone_enc), so the
    run_step-level tests below need a real string zone instead."""
    network, tree, coords_rad = _make_line_network(station_ids, capacities)
    network.zone_agg = np.array([zone] * network.n, dtype=object)
    return network, tree, coords_rad


def test_run_step_demand_multiplier_zero_forces_zero_departures():
    """Poisson(0) is deterministic -- rate * 0 always samples exactly 0
    departures, regardless of the base GBT rate or rng draw. The cleanest
    analytically-known-answer check that demand_multiplier (Phase 9,
    SPEC.md §8's bootstrap axis (a)) actually reaches the sampling step."""
    network, tree, coords_rad = _make_line_network_with_zone(["A", "B"], [10, 10])
    fitted_dep = _fake_fitted_departures(rate=50.0, station_ids=["A", "B"])
    od_model = _empty_od_model()
    sim_params = _default_sim_params()
    n = 2
    calendar_row = _calendar_row("2025-10-06T10:00:00")

    outcome = simulator.run_step(
        network, np.array([10.0, 10.0]), np.zeros((n, demand.OWN_LAG_HOUR_INTERVALS)), np.zeros(n),
        fitted_dep, od_model, calendar_row, np.zeros(n), np.zeros(n), np.zeros(n),
        tree, coords_rad, sim_params, np.random.default_rng(0), forced_departures=None, forced_trips=None,
        demand_multiplier=0.0,
    )

    assert outcome.departures_sampled.sum() == 0.0
    assert outcome.departures_actual.sum() == 0.0
    assert outcome.lost_no_bike.sum() == 0.0
    assert outcome.trip_log.height == 0


def test_run_step_demand_multiplier_scales_poisson_rate_up():
    """Higher demand_multiplier -> higher effective Poisson rate -> higher
    EXPECTED sampled departures. Averaged over many seeds since a single
    Poisson draw is stochastic; the point is to catch a sign-flip or
    no-op bug in the multiplier wiring, not to re-prove Poisson's mean."""
    network, tree, coords_rad = _make_line_network_with_zone(["A", "B"], [10, 10])
    fitted_dep = _fake_fitted_departures(rate=5.0, station_ids=["A", "B"])
    od_model = _global_only_od_model("B")
    sim_params = _default_sim_params()
    n = 2
    calendar_row = _calendar_row("2025-10-06T10:00:00")

    def _sampled_total(multiplier: float, seed: int) -> float:
        outcome = simulator.run_step(
            network, np.array([1000.0, 1000.0]), np.zeros((n, demand.OWN_LAG_HOUR_INTERVALS)), np.zeros(n),
            fitted_dep, od_model, calendar_row, np.zeros(n), np.zeros(n), np.zeros(n),
            tree, coords_rad, sim_params, np.random.default_rng(seed), forced_departures=None, forced_trips=None,
            demand_multiplier=multiplier,
        )
        return float(outcome.departures_sampled.sum())

    totals_low = np.array([_sampled_total(0.2, seed) for seed in range(30)])
    totals_high = np.array([_sampled_total(3.0, seed) for seed in range(30)])
    assert totals_high.mean() > totals_low.mean()


def test_run_simulation_induced_moves_relocates_bikes_beyond_baseline_n():
    """End-to-end (run_simulation, not just run_step): with demand_multiplier
    -> 0 (no organic trips at all) and n_schedule all zero (no baseline N),
    the ONLY thing that can move a bike is an injected induced move. A
    single-step week (one interval) with A->B, 4 trips/hour induced at that
    step's hour-of-week must show up as exactly +1.0 bike at B and -1.0 at
    A (4/4 quarter-hour rate), and the run's own clip accounting must show
    NO violation (1 bike moving out of a 10-capacity, 5-bike station is
    always in-bounds) -- a direct, analytically-known conservation check."""
    network, tree, coords_rad = _make_line_network_with_zone(["A", "B"], [10, 10])
    fitted_dep = _fake_fitted_departures(rate=5.0, station_ids=["A", "B"])
    od_model = _empty_od_model()
    sim_params = _default_sim_params()
    n = 2

    calendar_weather = pl.DataFrame([_calendar_row("2025-10-06T10:00:00")])
    week = simulator.WeekInputs(
        week_start=datetime(2025, 10, 6),
        week_end=datetime(2025, 10, 13),
        calendar_weather=calendar_weather,
        initial_inventory=np.array([5.0, 5.0]),
        own_lag_1h_seed=np.zeros((n, demand.OWN_LAG_HOUR_INTERVALS)),
        prev_departures_seed=np.zeros(n),
        n_schedule=pl.DataFrame(schema={"interval_start": pl.String, "station_id": pl.String, "inferred_nontrip_in": pl.Float64, "inferred_nontrip_out": pl.Float64}),
        own_lag_1week_table=pl.DataFrame(schema={"interval_start": pl.String, "station_id": pl.String, "dep_own_lag_1week": pl.Float64}),
        actual_departures=pl.DataFrame(schema={"interval_start": pl.String, "station_id": pl.String, "departures": pl.Float64}),
        actual_arrivals_inventory=pl.DataFrame(
            schema={"station_id": pl.String, "interval_start": pl.String, "inventory": pl.Float64, "is_bike_empty": pl.Boolean, "is_dock_full": pl.Boolean}
        ),
        half_cap_seeded_station_ids=[],
    )
    induced_moves = pl.DataFrame(
        {"origin_station_id": ["A"], "dest_station_id": ["B"], "hour_of_week": [10], "induced_trips_per_hour": [4.0]}
    )

    run = simulator.run_simulation(
        network, week, fitted_dep, od_model, sim_params, mode="stochastic",
        induced_moves=induced_moves, demand_multiplier=0.0,
    )

    end_inventory = run.station_intervals.sort("station_id")["inventory"].to_list()
    assert end_inventory == pytest.approx([4.0, 6.0])
    assert run.total_n_bound_violations == 0
    assert run.total_clip_created == 0.0
    assert run.total_clip_destroyed == 0.0
