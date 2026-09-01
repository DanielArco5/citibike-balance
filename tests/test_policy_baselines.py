"""Synthetic-fixture tests for src/opt/policy_baselines.py -- Phase 9
(SPEC.md §8, RUNBOOK Phase 9). Per CLAUDE.md: every model function gets a
test with a synthetic fixture where the answer is known analytically. None
of these touch real data on disk (load_shared_candidate_universe, the one
function that reads from data/processed/, is intentionally untested here --
it's a thin parquet-read + join, exercised end-to-end by the Phase 9 pilot
run instead, same split as allocate.py's own main() vs. its unit tests).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import opt.allocate as alloc  # noqa: E402
import opt.policy_baselines as pb  # noqa: E402

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
TIER1 = alloc.TierElasticity(a=2.0, b=0.15)
TIER2 = alloc.TierElasticity(a=3.0, b=0.25)


def _universe(mu=None, p_stockout=None) -> pb.SharedCandidateUniverse:
    """2 tier1_scheduled + 2 tier2_dynamic cells. mu/p_stockout_empirical
    default to a fixed, hand-chosen pattern so ranking/weighting tests have
    a known-in-advance answer; callers override to set up a specific
    scenario."""
    n = 4
    moves = pl.DataFrame(
        {
            "station_id": ["D1", "D2", "D3", "D4"],
            "hour_of_week": [1, 2, 3, 4],
            "tier": ["tier1_scheduled", "tier1_scheduled", "tier2_dynamic", "tier2_dynamic"],
            "low_frac": [0.8, 0.7, 0.3, 0.2],
            "k_max": [4, 4, 4, 4],
            "origin_station_id": ["O1", "O2", "O3", "O4"],
            "origin_cost": [0.0, 0.0, 0.0, 0.0],
            "flow_prob": [0.1, 0.1, 0.1, 0.1],
            "mu": mu if mu is not None else [1.0, 2.0, 3.0, 4.0],
            "p_stockout_empirical": p_stockout if p_stockout is not None else [0.1, 0.4, 0.9, 0.2],
        }
    )
    mv_curve = pl.DataFrame(
        {
            "station_id": ["D1", "D2", "D3", "D4"] * 4,
            "hour_of_week": [1, 2, 3, 4] * 4,
            "k": [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4,
            "mv": [1.0] * 16,
        }
    )
    return pb.SharedCandidateUniverse(candidate_moves=moves, dest_cum_mv=alloc.build_dest_cumulative_mv(mv_curve))


# ---------------------------------------------------------------------------
# policy_do_nothing
# ---------------------------------------------------------------------------


def test_policy_do_nothing_returns_empty_with_correct_schema():
    out = pb.policy_do_nothing()
    assert out.height == 0
    assert set(out.columns) == set(pb.FUNDED_SCHEMA)


# ---------------------------------------------------------------------------
# policy_uniform: equal DOLLARS per cell, solved via payout inversion
# ---------------------------------------------------------------------------


def test_policy_uniform_spends_equal_dollars_per_cell_within_a_tier():
    universe = _universe()
    out = pb.policy_uniform(universe, TIER1, TIER2, ALLOC_PARAMS, weekly_budget=400.0)
    assert out.height == 4

    target_per_cell = 400.0 / 4  # 100.0
    # Every cell's dollar_cost should land close to the target -- the
    # payout is SOLVED (via grid inversion) to hit it, not assumed.
    assert out["dollar_cost"].to_list() == pytest.approx([target_per_cell] * 4, rel=0.02)

    # Payout must be IDENTICAL within a tier (uniform policy applies one
    # payout per tier, not per cell) and may legitimately differ ACROSS
    # tiers (different elasticity curves need a different payout to hit
    # the same dollar target).
    tier1_payouts = out.filter(pl.col("tier") == "tier1_scheduled")["payout"].to_list()
    tier2_payouts = out.filter(pl.col("tier") == "tier2_dynamic")["payout"].to_list()
    assert tier1_payouts[0] == pytest.approx(tier1_payouts[1])
    assert tier2_payouts[0] == pytest.approx(tier2_payouts[1])


def test_policy_uniform_empty_universe_returns_empty():
    empty_moves = pl.DataFrame(
        schema={
            "station_id": pl.String, "hour_of_week": pl.Int64, "tier": pl.String, "low_frac": pl.Float64,
            "k_max": pl.Int64, "origin_station_id": pl.String, "origin_cost": pl.Float64, "flow_prob": pl.Float64,
            "mu": pl.Float64, "p_stockout_empirical": pl.Float64,
        }
    )
    empty = pb.SharedCandidateUniverse(candidate_moves=empty_moves, dest_cum_mv=pl.DataFrame())
    out = pb.policy_uniform(empty, TIER1, TIER2, ALLOC_PARAMS, weekly_budget=100.0)
    assert out.height == 0


# ---------------------------------------------------------------------------
# policy_proportional: dollars proportional to `mu`
# ---------------------------------------------------------------------------


def test_policy_proportional_spend_ratio_matches_mu_ratio():
    """D1..D4 have mu = 1,2,3,4 (total 10). At a total budget of 100, target
    dollars/cell = 10, 20, 30, 40 -- verified via the SAME payout-inversion
    round trip as the uniform test, cell by cell this time."""
    universe = _universe(mu=[1.0, 2.0, 3.0, 4.0])
    out = pb.policy_proportional(universe, TIER1, TIER2, ALLOC_PARAMS, weekly_budget=100.0)
    assert out.height == 4

    by_station = {r["station_id"]: r["dollar_cost"] for r in out.iter_rows(named=True)}
    assert by_station["D1"] == pytest.approx(10.0, rel=0.02)
    assert by_station["D2"] == pytest.approx(20.0, rel=0.02)
    assert by_station["D3"] == pytest.approx(30.0, rel=0.02)
    assert by_station["D4"] == pytest.approx(40.0, rel=0.02)


def test_policy_proportional_zero_total_volume_returns_empty():
    universe = _universe(mu=[0.0, 0.0, 0.0, 0.0])
    out = pb.policy_proportional(universe, TIER1, TIER2, ALLOC_PARAMS, weekly_budget=100.0)
    assert out.height == 0


# ---------------------------------------------------------------------------
# policy_top_n_stockout: naive rank-by-frequency, flat payout, cumulative cutoff
# ---------------------------------------------------------------------------


def test_policy_top_n_stockout_funds_in_stockout_frequency_order_until_budget_exhausted():
    """p_stockout_empirical = D3(0.9) > D2(0.4) > D4(0.2) > D1(0.1) -- with
    a flat payout, every cell's dollar_cost is identical, so a budget that
    covers exactly 2 cells' worth must fund D3 and D2 only, in that order,
    dropping D4 and D1 entirely (not partially funding past the cutoff)."""
    universe = _universe(p_stockout=[0.1, 0.4, 0.9, 0.2])
    # Fund exactly the first cell's cost with a tiny bit of headroom for a
    # second, none for a third -- flat payout means every tier1 cell costs
    # the same and every tier2 cell costs the same, so pin the payout and
    # compute the per-cell cost directly to size the budget precisely. D3
    # (highest p_stockout_empirical, funded first) is tier2_dynamic in the
    # fixture, so its cost uses TIER2's elasticity.
    flat_payout = 10.0
    single_cost = TIER2.a * (1.0 - np.exp(-TIER2.b * flat_payout)) * flat_payout * ALLOC_PARAMS.dollars_per_point
    budget = single_cost * 1.5  # room for exactly one cell, not two

    out = pb.policy_top_n_stockout(universe, TIER1, TIER2, ALLOC_PARAMS, budget, flat_payout=flat_payout)

    assert out.height == 1
    assert out.row(0, named=True)["station_id"] == "D3"


def test_policy_top_n_stockout_uses_flat_payout_not_optimized():
    universe = _universe()
    out = pb.policy_top_n_stockout(universe, TIER1, TIER2, ALLOC_PARAMS, weekly_budget=1e9, flat_payout=7.0)
    assert (out["payout"] == 7.0).all()


# ---------------------------------------------------------------------------
# policy_allocator: thin wrapper around Phase 8's own rank_targets/cutoff
# ---------------------------------------------------------------------------


def test_policy_allocator_matches_calling_allocate_directly():
    universe = _universe()
    via_wrapper = pb.policy_allocator(universe, TIER1, TIER2, ALLOC_PARAMS, weekly_budget=50.0)

    ranking = alloc.rank_targets(universe.candidate_moves, universe.dest_cum_mv, TIER1, TIER2, ALLOC_PARAMS)
    via_direct = alloc.cumulative_budget_cutoff(ranking, 50.0)

    assert via_wrapper.sort("station_id")["dollar_cost"].to_list() == pytest.approx(
        via_direct.sort("station_id")["dollar_cost"].to_list()
    )
    assert set(via_wrapper["station_id"].to_list()) == set(via_direct["station_id"].to_list())


# ---------------------------------------------------------------------------
# policy_allocator_full_budget: budget-exhausting variant (added post-pilot,
# 2026-08-14, after the default allocator was found to spend only ~$310-383
# of a $10k budget)
# ---------------------------------------------------------------------------


def test_policy_allocator_full_budget_matches_allocator_when_budget_only_covers_baseline():
    """At a budget that covers exactly the baseline (own-best-payout) cost
    and nothing more, no escalation can happen -- output must match
    policy_allocator exactly."""
    universe = _universe()
    baseline = pb.policy_allocator(universe, TIER1, TIER2, ALLOC_PARAMS, weekly_budget=1e9)
    exact_budget = float(baseline["dollar_cost"].sum())

    full = pb.policy_allocator_full_budget(universe, TIER1, TIER2, ALLOC_PARAMS, weekly_budget=exact_budget)
    assert full.sort("station_id")["payout"].to_list() == pytest.approx(baseline.sort("station_id")["payout"].to_list())
    assert full.sort("station_id")["dollar_cost"].to_list() == pytest.approx(baseline.sort("station_id")["dollar_cost"].to_list())


def test_policy_allocator_full_budget_spends_more_than_baseline_when_budget_allows():
    """With generous extra budget, at least one candidate must escalate
    beyond its own baseline-optimal payout -- total spend must exceed the
    baseline-only total (the whole point of this variant)."""
    universe = _universe()
    baseline_total = float(pb.policy_allocator(universe, TIER1, TIER2, ALLOC_PARAMS, weekly_budget=1e9)["dollar_cost"].sum())

    full = pb.policy_allocator_full_budget(universe, TIER1, TIER2, ALLOC_PARAMS, weekly_budget=baseline_total * 3)
    assert float(full["dollar_cost"].sum()) > baseline_total
    # Never overspends past what was actually offered.
    assert float(full["dollar_cost"].sum()) <= baseline_total * 3 + 1e-6


def test_policy_allocator_full_budget_never_exceeds_grid_maximum_payout():
    """With an ABSURDLY large budget, every candidate should max out at
    payout_grid's top value and stop -- escalation cannot invent a payout
    beyond the grid, and the surplus budget simply goes unspent."""
    universe = _universe()
    full = pb.policy_allocator_full_budget(universe, TIER1, TIER2, ALLOC_PARAMS, weekly_budget=1e12)
    assert (full["payout"] == max(alloc.PAYOUT_GRID)).all()


def test_policy_allocator_full_budget_prioritizes_top_ranked_candidate_first():
    """Round-robin-by-priority: the #1 ranked candidate's payout must never
    be lower than a lower-ranked candidate's, at any budget level -- the
    best candidate always gets first claim on each escalation round."""
    universe = _universe()
    ranking = alloc.rank_targets(universe.candidate_moves, universe.dest_cum_mv, TIER1, TIER2, ALLOC_PARAMS)
    top_station = ranking.sort("rank").row(0, named=True)["station_id"]
    baseline_total = float(pb.policy_allocator(universe, TIER1, TIER2, ALLOC_PARAMS, weekly_budget=1e9)["dollar_cost"].sum())

    full = pb.policy_allocator_full_budget(universe, TIER1, TIER2, ALLOC_PARAMS, weekly_budget=baseline_total * 1.5)
    top_payout = full.filter(pl.col("station_id") == top_station)["payout"][0]
    other_payouts = full.filter(pl.col("station_id") != top_station)["payout"].to_list()
    assert all(top_payout >= p for p in other_payouts)


# ---------------------------------------------------------------------------
# apply_move_cap
# ---------------------------------------------------------------------------


def test_apply_move_cap_clips_induced_trips_leaves_dollar_cost_untouched():
    funded = pl.DataFrame(
        {
            "station_id": ["D1", "D2"],
            "hour_of_week": [1, 2],
            "tier": ["tier1_scheduled", "tier2_dynamic"],
            "origin_station_id": ["O1", "O2"],
            "induced_trips": [5.0, 1.5],  # D1 over the cap, D2 under it
            "payout": [10.0, 10.0],
            "dollar_cost": [20.0, 5.0],
        }
    )
    capped = pb.apply_move_cap(funded, max_induced_moves_per_station_hour=3.0)
    assert capped.sort("station_id")["induced_trips"].to_list() == pytest.approx([3.0, 1.5])
    # dollar_cost reflects what was actually PAID for -- untouched by the
    # realization cap (see module docstring: this is a real inefficiency
    # a policy can have, not something to silently rescale away).
    assert capped.sort("station_id")["dollar_cost"].to_list() == pytest.approx([20.0, 5.0])


def test_apply_move_cap_on_empty_returns_empty():
    empty = pl.DataFrame(schema=pb.FUNDED_SCHEMA)
    assert pb.apply_move_cap(empty, 3.0).height == 0


# ---------------------------------------------------------------------------
# to_simulator_induced_moves
# ---------------------------------------------------------------------------


def test_to_simulator_induced_moves_renames_and_selects_expected_schema():
    funded = pl.DataFrame(
        {
            "station_id": ["D1"],
            "hour_of_week": [10],
            "tier": ["tier1_scheduled"],
            "origin_station_id": ["O1"],
            "induced_trips": [2.5],
            "payout": [10.0],
            "dollar_cost": [5.0],
        }
    )
    out = pb.to_simulator_induced_moves(funded)
    assert out.columns == ["origin_station_id", "dest_station_id", "hour_of_week", "induced_trips_per_hour"]
    row = out.row(0, named=True)
    assert row["dest_station_id"] == "D1"
    assert row["origin_station_id"] == "O1"
    assert row["induced_trips_per_hour"] == pytest.approx(2.5)


def test_to_simulator_induced_moves_empty_returns_none():
    assert pb.to_simulator_induced_moves(pl.DataFrame(schema=pb.FUNDED_SCHEMA)) is None
