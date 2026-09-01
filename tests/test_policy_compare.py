"""Synthetic-fixture tests for src/sim/policy_compare.py -- Phase 9
(SPEC.md §8, RUNBOOK Phase 9). Per CLAUDE.md: every model function gets a
test with a synthetic fixture where the answer is known analytically.

load_shared_artifacts/run_one_replicate/run_replicates are intentionally
untested here -- they read real cached models/parquet off disk and run the
actual simulator, exercised end-to-end by the Phase 9 pilot run instead
(same split as allocate.py's main() vs. its unit tests, and simulator.py's
own run_simulation vs. its lower-level synthetic tests in test_sim.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import opt.allocate as alloc  # noqa: E402
import sim.policy_compare as pc  # noqa: E402
import sim.simulator as simulator  # noqa: E402

ALLOC_PARAMS = alloc.AllocationParams(
    dollars_per_point=0.20,
    weekly_budget=10000.0,
    elasticity_tier1=alloc.TierElasticity(a=2.0, b=0.15),
    elasticity_tier2=alloc.TierElasticity(a=3.0, b=0.25),
    max_induced_moves_per_station_hour=3,
    max_move_duration_min=25.0,
    origin_max_low_frac=0.10,
    sweep_a_min=1.0, sweep_a_max=5.0, sweep_a_steps=3,
    sweep_b_min=0.05, sweep_b_max=0.5, sweep_b_steps=3,
)


# ---------------------------------------------------------------------------
# compute_fill_rate_table: system + zone level ONLY, never station level
# ---------------------------------------------------------------------------


def _network(station_ids, zones):
    n = len(station_ids)
    return simulator.NetworkArrays(
        station_id=np.array(station_ids),
        capacity=np.full(n, 20.0),
        zone_agg=np.array(zones, dtype=object),
        lat=np.zeros(n),
        lng=np.zeros(n),
        index_of={sid: i for i, sid in enumerate(station_ids)},
    )


def test_compute_fill_rate_table_system_and_zone_values_known_by_hand():
    """A, B in zone Z1; C in zone Z2. Hand-computed:
    A: fulfilled=10, lost=2   B: fulfilled=6, lost=1   C: fulfilled=3, lost=1
    system: fulfilled=19, lost=4 -> fill_rate=19/23
    Z1 (A+B): fulfilled=16, lost=3 -> fill_rate=16/19
    Z2 (C):   fulfilled=3,  lost=1 -> fill_rate=3/4"""
    network = _network(["A", "B", "C"], ["Z1", "Z1", "Z2"])
    station_intervals = pl.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "direct_arrivals": [10.0, 5.0, 3.0],
            "rerouted_arrivals": [0.0, 1.0, 0.0],
            "lost_no_bike": [2.0, 0.0, 1.0],
            "lost_past_cap_arrivals": [0.0, 1.0, 0.0],
        }
    )
    run = simulator.SimulationRun(
        station_intervals=station_intervals, trip_log=pl.DataFrame(),
        total_n_bound_violations=0, total_clip_created=0.0, total_clip_destroyed=0.0,
    )

    out = pc.compute_fill_rate_table(run, network)

    # Module docstring's hard requirement: system + zone rows ONLY.
    assert set(out["level"].to_list()) == {"system", "zone"}
    assert out.height == 3  # 1 system + 2 zones -- never a per-station row

    by_key = {(r["level"], r["zone_agg"]): r for r in out.iter_rows(named=True)}
    assert by_key[("system", "system")]["fill_rate"] == pytest.approx(19 / 23)
    assert by_key[("zone", "Z1")]["fill_rate"] == pytest.approx(16 / 19)
    assert by_key[("zone", "Z2")]["fill_rate"] == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------
# compute_treated_cell_fill: pooled sum over named (station, hour) cells
# ONLY -- still never a per-station row, restricting the denominator is not
# the same thing as reopening the per-station output ban above.
# ---------------------------------------------------------------------------


def test_compute_treated_cell_fill_restricts_to_named_cells_only():
    """4 station-interval rows across 2 stations x 2 hours-of-week.
    treated_cells names only (A, hour=1) and (B, hour=2) -- hand-computed:
    A@h1: fulfilled=10, lost=2 (INCLUDED)
    A@h2: fulfilled=100, lost=100 (excluded -- wrong hour for A)
    B@h1: fulfilled=100, lost=100 (excluded -- wrong hour for B)
    B@h2: fulfilled=4, lost=1 (INCLUDED)
    Expected: fulfilled=14, lost=3 -- the two 100/100 decoy rows must NOT
    move the result, proving the join actually restricts by (station,
    hour), not just by station or just by hour."""
    station_intervals = pl.DataFrame(
        {
            "station_id": ["A", "A", "B", "B"],
            "interval_start": [1, 2, 1, 2],
            "direct_arrivals": [8.0, 60.0, 60.0, 3.0],
            "rerouted_arrivals": [2.0, 40.0, 40.0, 1.0],
            "lost_no_bike": [2.0, 60.0, 60.0, 1.0],
            "lost_past_cap_arrivals": [0.0, 40.0, 40.0, 0.0],
        }
    )
    calendar_weather = pl.DataFrame({"interval_start": [1, 2], "hour_of_week": [1, 2]})
    treated_cells = pl.DataFrame({"station_id": ["A", "B"], "hour_of_week": [1, 2]})

    fulfilled, lost = pc.compute_treated_cell_fill(station_intervals, calendar_weather, treated_cells)

    assert fulfilled == pytest.approx(14.0)
    assert lost == pytest.approx(3.0)


def test_compute_treated_cell_fill_empty_treated_cells_gives_zero():
    station_intervals = pl.DataFrame(
        {
            "station_id": ["A"], "interval_start": [1],
            "direct_arrivals": [10.0], "rerouted_arrivals": [0.0],
            "lost_no_bike": [2.0], "lost_past_cap_arrivals": [0.0],
        }
    )
    calendar_weather = pl.DataFrame({"interval_start": [1], "hour_of_week": [1]})
    treated_cells = pl.DataFrame({"station_id": [], "hour_of_week": []}, schema={"station_id": pl.String, "hour_of_week": pl.Int64})

    fulfilled, lost = pc.compute_treated_cell_fill(station_intervals, calendar_weather, treated_cells)
    assert fulfilled == 0.0
    assert lost == 0.0


# ---------------------------------------------------------------------------
# sample_demand_multiplier
# ---------------------------------------------------------------------------


def test_sample_demand_multiplier_constant_distribution_is_deterministic():
    """Resampling (with replacement) from a CONSTANT array must always
    return exactly that constant -- no randomness can move a mean away
    from a distribution with zero variance, regardless of seed."""
    ratios = np.full(200, 1.35)
    for seed in range(10):
        assert pc.sample_demand_multiplier(np.random.default_rng(seed), ratios) == pytest.approx(1.35)


def test_sample_demand_multiplier_resample_mean_bounded_by_original_range():
    """Bootstrap resampling with replacement can only draw values that
    already exist in the array, so the resample mean can never fall
    outside [min, max] of the original distribution -- true for every
    possible resample, checked over many seeds, not a statistical claim."""
    rng_data = np.random.default_rng(42)
    ratios = rng_data.uniform(0.5, 1.5, size=500)
    lo, hi = ratios.min(), ratios.max()
    for seed in range(20):
        m = pc.sample_demand_multiplier(np.random.default_rng(seed), ratios)
        assert lo <= m <= hi


# ---------------------------------------------------------------------------
# sample_elasticity_draw
# ---------------------------------------------------------------------------


def test_sample_elasticity_draw_tier2_within_sweep_range():
    rng = np.random.default_rng(0)
    for _ in range(50):
        tier1, tier2 = pc.sample_elasticity_draw(rng, ALLOC_PARAMS)
        assert ALLOC_PARAMS.sweep_a_min <= tier2.a <= ALLOC_PARAMS.sweep_a_max
        assert ALLOC_PARAMS.sweep_b_min <= tier2.b <= ALLOC_PARAMS.sweep_b_max


# ---------------------------------------------------------------------------
# Policy / budget-sweep configuration consistency (added 2026-08-14 with the
# 6th policy + redesigned sweep levels)
# ---------------------------------------------------------------------------


def test_policy_names_includes_full_budget_allocator_variant():
    assert "allocator" in pc.POLICY_NAMES
    assert "allocator_full_budget" in pc.POLICY_NAMES
    assert len(pc.POLICY_NAMES) == 6


def test_budget_sweep_policies_are_a_subset_of_policy_names():
    assert set(pc.BUDGET_SWEEP_POLICIES).issubset(set(pc.POLICY_NAMES))
    assert "do_nothing" in pc.BUDGET_SWEEP_POLICIES  # the $0 anchor


def test_budget_levels_sorted_ascending_from_zero_to_at_least_weekly_budget():
    """Capped at 1x weekly_budget ($10k), not SPEC.md §8's literal 3x --
    a deliberate, explicit scope cut given time constraints (see
    BUDGET_LEVELS_DOLLARS's comment), not an oversight."""
    levels = pc.BUDGET_LEVELS_DOLLARS
    assert levels == tuple(sorted(levels))
    assert levels[0] == 0.0
    assert max(levels) >= ALLOC_PARAMS.weekly_budget


def test_budget_levels_dense_near_the_pilot_observed_plateau():
    """The pilot found the allocator's own spend plateaus somewhere in
    $244-567 across different elasticity draws -- the grid must actually
    resolve that region, not jump straight from $0 to $10k. At least 5
    distinct levels below $1,000."""
    below_1000 = [lv for lv in pc.BUDGET_LEVELS_DOLLARS if lv < 1000.0]
    assert len(below_1000) >= 5


def test_budget_task_path_unique_and_sortable_per_level():
    paths = [pc._budget_task_path(level, 0) for level in pc.BUDGET_LEVELS_DOLLARS]
    assert len(set(paths)) == len(paths)  # every level gets a distinct file
    assert [p.name for p in paths] == sorted(p.name for p in paths)  # filename order matches level order


def test_sample_elasticity_draw_preserves_config_conservative_ratio():
    """tier1 must always be derived from tier2 via the SAME ratio as the
    config defaults (2.0/3.0 on a, 0.15/0.25 on b) -- not drawn
    independently, per the module docstring's rationale (preserve the
    documented tier1-more-conservative-than-tier2 relationship)."""
    a_ratio = ALLOC_PARAMS.elasticity_tier1.a / ALLOC_PARAMS.elasticity_tier2.a
    b_ratio = ALLOC_PARAMS.elasticity_tier1.b / ALLOC_PARAMS.elasticity_tier2.b
    rng = np.random.default_rng(1)
    for _ in range(20):
        tier1, tier2 = pc.sample_elasticity_draw(rng, ALLOC_PARAMS)
        assert tier1.a == pytest.approx(tier2.a * a_ratio)
        assert tier1.b == pytest.approx(tier2.b * b_ratio)
        assert tier1.a < tier2.a  # tier1 stays strictly more conservative
        assert tier1.b < tier2.b
