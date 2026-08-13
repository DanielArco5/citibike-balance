"""Synthetic-fixture tests for src/opt/allocate.py -- Phase 8 Part C
(SPEC.md §7, RUNBOOK Phase 8 Part C). Per CLAUDE.md: every model function
gets a test with a synthetic fixture where the answer is known
analytically. None of these touch real data on disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import opt.allocate as alloc  # noqa: E402
import models.od_shares as od_shares  # noqa: E402
import opt.marginal_value as mv  # noqa: E402

MV_PARAMS = mv.MarginalValueParams(
    min_weeks_for_cross_check=5, relaxation_time_flag_minutes=15.0, first_passage_window_minutes=60.0, mv_k_max=4,
    schedulable_modal_share_threshold=0.7,
)

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
# assign_tiers
# ---------------------------------------------------------------------------


def test_assign_tiers_splits_chronic_schedulable_from_everything_else():
    low_freq_per_cell = pl.DataFrame(
        {
            "station_id": ["S1", "S2", "S3", "S4"],
            "hour_of_week": [1, 2, 3, 4],
            "low_frac": [0.8, 0.8, 0.3, 0.0],  # S1/S2 chronic, S3 non-chronic-but-eligible, S4 never low
        }
    )
    chronic_timing = pl.DataFrame(
        {
            "station_id": ["S1", "S2"],
            "hour_of_week": [1, 2],
            "modal_share": [0.9, 0.4],  # S1 schedulable, S2 chronic-but-erratic
        }
    )
    tiers = alloc.assign_tiers(low_freq_per_cell, chronic_timing, MV_PARAMS)
    by_station = {row["station_id"]: row["tier"] for row in tiers.iter_rows(named=True)}
    assert by_station["S1"] == "tier1_scheduled"
    assert by_station["S2"] == "tier2_dynamic"  # chronic but erratic -> tier2
    assert by_station["S3"] == "tier2_dynamic"  # non-chronic but eligible -> tier2
    assert "S4" not in by_station  # never low -> excluded entirely


# ---------------------------------------------------------------------------
# qualifying_origins
# ---------------------------------------------------------------------------


def test_qualifying_origins_costs_zero_when_never_in_measured_region():
    weekly = pl.DataFrame(
        {
            "station_id": ["O1"] * 3,
            "hour_of_week": [5] * 3,
            "mean_inventory": [10.0, 12.0, 9.0],  # min=9, well above mv_k_max=4
        }
    )
    low_freq_per_cell = pl.DataFrame({"station_id": ["O1"], "hour_of_week": [5], "low_frac": [0.0]})
    mv_curve = pl.DataFrame(schema={"station_id": pl.String, "hour_of_week": pl.Int64, "k": pl.Int64, "mv": pl.Float64})
    origins = alloc.qualifying_origins(weekly, low_freq_per_cell, mv_curve, MV_PARAMS, ALLOC_PARAMS)
    assert origins.height == 1
    row = origins.row(0, named=True)
    assert row["min_inventory"] == pytest.approx(9.0)
    assert row["origin_cost"] == pytest.approx(0.0)


def test_qualifying_origins_uses_measured_mv_when_available():
    """O2 occasionally dips to 3 bikes (still qualifies as an origin,
    low_frac=0.05 <= threshold) -- its cost must come from mv_curve at
    k=3, not be assumed 0 just because it's mostly a surplus station."""
    weekly = pl.DataFrame({"station_id": ["O2"] * 2, "hour_of_week": [5] * 2, "mean_inventory": [3.0, 12.0]})
    low_freq_per_cell = pl.DataFrame({"station_id": ["O2"], "hour_of_week": [5], "low_frac": [0.05]})
    mv_curve = pl.DataFrame({"station_id": ["O2"], "hour_of_week": [5], "k": [3], "mv": [0.42]})
    origins = alloc.qualifying_origins(weekly, low_freq_per_cell, mv_curve, MV_PARAMS, ALLOC_PARAMS)
    row = origins.row(0, named=True)
    assert row["min_inventory"] == pytest.approx(3.0)
    assert row["origin_cost"] == pytest.approx(0.42)


def test_qualifying_origins_excludes_frequently_low_stations():
    weekly = pl.DataFrame({"station_id": ["O3"], "hour_of_week": [5], "mean_inventory": [2.0]})
    low_freq_per_cell = pl.DataFrame({"station_id": ["O3"], "hour_of_week": [5], "low_frac": [0.6]})  # chronic itself
    mv_curve = pl.DataFrame(schema={"station_id": pl.String, "hour_of_week": pl.Int64, "k": pl.Int64, "mv": pl.Float64})
    origins = alloc.qualifying_origins(weekly, low_freq_per_cell, mv_curve, MV_PARAMS, ALLOC_PARAMS)
    assert origins.height == 0


# ---------------------------------------------------------------------------
# build_dest_cumulative_mv
# ---------------------------------------------------------------------------


def test_build_dest_cumulative_mv_running_sum_and_k_max():
    mv_curve = pl.DataFrame(
        {
            "station_id": ["D1", "D1", "D1"],
            "hour_of_week": [1, 1, 1],
            "k": [1, 2, 3],
            "mv": [2.0, 1.0, 0.5],
        }
    )
    result = alloc.build_dest_cumulative_mv(mv_curve)
    result = result.sort("k")
    assert result["cum_mv"].to_list() == pytest.approx([2.0, 3.0, 3.5])
    assert (result["k_max"] == 3).all()


# ---------------------------------------------------------------------------
# build_candidate_moves
# ---------------------------------------------------------------------------


def _od_model_with_flow(pairs: list[tuple[str, int, str, float]]) -> od_shares.ODShareModel:
    station_hour_probs = pl.DataFrame(
        {
            "start_station_id": [p[0] for p in pairs],
            "hour_of_week": [p[1] for p in pairs],
            "end_station_id": [p[2] for p in pairs],
            "prob": [p[3] for p in pairs],
        }
    ) if pairs else pl.DataFrame(
        schema={"start_station_id": pl.String, "hour_of_week": pl.Int64, "end_station_id": pl.String, "prob": pl.Float64}
    )
    empty = pl.DataFrame(schema={"end_station_id": pl.String, "prob": pl.Float64})
    return od_shares.ODShareModel(
        cell_tier=pl.DataFrame(),
        station_hour_probs=station_hour_probs,
        zone_hour_probs=pl.DataFrame(),
        zone_daypart_probs=pl.DataFrame(),
        global_probs=empty,
    )


def test_build_candidate_moves_prefers_lowest_cost_origin_same_zone():
    tiers = pl.DataFrame({"station_id": ["D1"], "hour_of_week": [5], "low_frac": [0.8], "tier": ["tier1_scheduled"]})
    dest_cum_mv = alloc.build_dest_cumulative_mv(pl.DataFrame({"station_id": ["D1"], "hour_of_week": [5], "k": [1], "mv": [1.5]}))
    origins = pl.DataFrame(
        {
            "station_id": ["O1", "O2"],
            "hour_of_week": [5, 5],
            "min_inventory": [10.0, 10.0],
            "origin_cost": [0.5, 0.1],  # O2 is cheaper
        }
    )
    stations = pl.DataFrame({"station_id": ["D1", "O1", "O2"], "zone_agg": ["Z1", "Z1", "Z1"]})
    od_model = _od_model_with_flow([])

    moves = alloc.build_candidate_moves(tiers, dest_cum_mv, origins, stations, od_model)
    assert moves.height == 1
    row = moves.row(0, named=True)
    assert row["origin_station_id"] == "O2"
    assert row["origin_cost"] == pytest.approx(0.1)
    assert row["k_max"] == 1


def test_build_candidate_moves_allows_flow_pair_outside_zone_and_excludes_unconnected():
    tiers = pl.DataFrame({"station_id": ["D1"], "hour_of_week": [5], "low_frac": [0.8], "tier": ["tier1_scheduled"]})
    dest_cum_mv = alloc.build_dest_cumulative_mv(pl.DataFrame({"station_id": ["D1"], "hour_of_week": [5], "k": [1], "mv": [1.5]}))
    origins = pl.DataFrame(
        {
            "station_id": ["O_far_connected", "O_far_unconnected"],
            "hour_of_week": [5, 5],
            "min_inventory": [10.0, 10.0],
            "origin_cost": [0.0, 0.0],
        }
    )
    stations = pl.DataFrame(
        {"station_id": ["D1", "O_far_connected", "O_far_unconnected"], "zone_agg": ["Z1", "Z2", "Z3"]}
    )
    od_model = _od_model_with_flow([("O_far_connected", 5, "D1", 0.2)])

    moves = alloc.build_candidate_moves(tiers, dest_cum_mv, origins, stations, od_model)
    assert moves.height == 1
    assert moves.row(0, named=True)["origin_station_id"] == "O_far_connected"


# ---------------------------------------------------------------------------
# rank_targets
# ---------------------------------------------------------------------------


def _candidate_moves_fixture():
    moves = pl.DataFrame(
        {
            "station_id": ["D_high", "D_low"],
            "hour_of_week": [1, 2],
            "tier": ["tier1_scheduled", "tier1_scheduled"],
            "low_frac": [0.8, 0.6],
            "k_max": [1, 1],
            "origin_station_id": ["O1", "O2"],
            "origin_cost": [0.0, 0.0],
            "flow_prob": [0.1, 0.1],
        }
    )
    mv_curve = pl.DataFrame(
        {"station_id": ["D_high", "D_low"], "hour_of_week": [1, 2], "k": [1, 1], "mv": [5.0, 0.5]}
    )
    return moves, alloc.build_dest_cumulative_mv(mv_curve)


def test_rank_targets_orders_by_net_value_per_dollar():
    moves, dest_cum_mv = _candidate_moves_fixture()
    te = alloc.TierElasticity(a=2.0, b=0.2)
    ranked = alloc.rank_targets(moves, dest_cum_mv, te, te, ALLOC_PARAMS, payout_grid=[5.0, 10.0])
    assert ranked.height == 2
    assert ranked.row(0, named=True)["station_id"] == "D_high"
    assert ranked.row(0, named=True)["rank"] == 1
    assert ranked["net_value_per_dollar"].to_list() == sorted(ranked["net_value_per_dollar"].to_list(), reverse=True)


def test_rank_targets_higher_origin_cost_lowers_net_value():
    moves = pl.DataFrame(
        {
            "station_id": ["D1", "D2"],
            "hour_of_week": [1, 1],
            "tier": ["tier1_scheduled", "tier1_scheduled"],
            "low_frac": [0.8, 0.8],
            "k_max": [1, 1],
            "origin_station_id": ["O1", "O2"],
            "origin_cost": [0.0, 1.5],  # D2's origin is expensive
            "flow_prob": [0.1, 0.1],
        }
    )
    dest_cum_mv = alloc.build_dest_cumulative_mv(
        pl.DataFrame({"station_id": ["D1", "D2"], "hour_of_week": [1, 1], "k": [1, 1], "mv": [2.0, 2.0]})
    )
    te = alloc.TierElasticity(a=2.0, b=0.2)
    ranked = alloc.rank_targets(moves, dest_cum_mv, te, te, ALLOC_PARAMS, payout_grid=[10.0])
    d1 = ranked.filter(pl.col("station_id") == "D1").row(0, named=True)
    d2 = ranked.filter(pl.col("station_id") == "D2").row(0, named=True)
    assert d1["net_value_per_dollar"] > d2["net_value_per_dollar"]


def test_rank_targets_ranking_genuinely_depends_on_elasticity():
    """Regression test for the exact bug caught before reporting the
    sweep: with a FLAT per-trip destination value, induced_trips cancels
    out of net_value_per_dollar algebraically and the ranking becomes
    independent of (a, b) entirely (every sweep draw produced an
    IDENTICAL top-100, Spearman 1.000 with zero variation -- a formula
    artifact, not evidence of real robustness). Using the cumulative,
    saturating mv curve fixes this: D1 has a low ceiling reached
    immediately (k_max=1, mv=3.0); D2 needs many bikes to reach a HIGHER
    ceiling (k_max=4, cumulative mv up to 4.0, built up slowly). At low
    elasticity D1 wins (its one bike is worth more than D2's first);
    at high elasticity (enough induced volume to reach D2's full
    ceiling cheaply) D2 overtakes it -- values below are the exact
    numbers verified by hand before being hardcoded here, not guessed."""
    moves = pl.DataFrame(
        {
            "station_id": ["D1", "D2"],
            "hour_of_week": [1, 2],
            "tier": ["tier1_scheduled", "tier1_scheduled"],
            "low_frac": [0.8, 0.8],
            "k_max": [1, 4],
            "origin_station_id": ["O1", "O2"],
            "origin_cost": [0.0, 0.0],
            "flow_prob": [0.1, 0.1],
        }
    )
    mv_curve = pl.DataFrame(
        {
            "station_id": ["D1", "D2", "D2", "D2", "D2"],
            "hour_of_week": [1, 2, 2, 2, 2],
            "k": [1, 1, 2, 3, 4],
            "mv": [3.0, 0.5, 1.0, 1.0, 1.5],
        }
    )
    dest_cum_mv = alloc.build_dest_cumulative_mv(mv_curve)

    low = alloc.TierElasticity(a=4.0, b=2.0)
    ranked_low = alloc.rank_targets(moves, dest_cum_mv, low, low, ALLOC_PARAMS, payout_grid=[1.0])
    assert ranked_low.filter(pl.col("rank") == 1).row(0, named=True)["station_id"] == "D1"

    high = alloc.TierElasticity(a=4.0, b=5.0)
    ranked_high = alloc.rank_targets(moves, dest_cum_mv, high, high, ALLOC_PARAMS, payout_grid=[1.0])
    assert ranked_high.filter(pl.col("rank") == 1).row(0, named=True)["station_id"] == "D2"


# ---------------------------------------------------------------------------
# cumulative_budget_cutoff
# ---------------------------------------------------------------------------


def test_cumulative_budget_cutoff_respects_budget():
    ranking = pl.DataFrame({"station_id": ["A", "B", "C"], "dollar_cost": [3.0, 4.0, 5.0]})
    cutoff = alloc.cumulative_budget_cutoff(ranking, weekly_budget=7.0)
    assert cutoff["station_id"].to_list() == ["A", "B"]


# ---------------------------------------------------------------------------
# elasticity_grid / spearman_stability / stable_core
# ---------------------------------------------------------------------------


def test_elasticity_grid_shape_matches_config():
    grid = alloc.elasticity_grid(ALLOC_PARAMS)
    assert len(grid) == ALLOC_PARAMS.sweep_a_steps * ALLOC_PARAMS.sweep_b_steps
    a_vals = sorted({a for a, _ in grid})
    assert a_vals[0] == pytest.approx(ALLOC_PARAMS.sweep_a_min)
    assert a_vals[-1] == pytest.approx(ALLOC_PARAMS.sweep_a_max)


def test_spearman_stability_perfect_for_identical_rankings():
    ranking = pl.DataFrame({"station_id": ["A", "B", "C"], "hour_of_week": [1, 2, 3], "rank": [1, 2, 3]})
    sweep_results = {(1.0, 0.1): ranking, (2.0, 0.2): ranking.clone()}
    stability = alloc.spearman_stability(sweep_results)
    assert stability.height == 1
    assert stability.row(0, named=True)["spearman"] == pytest.approx(1.0)


def test_spearman_stability_detects_reversed_ranking():
    ranking_a = pl.DataFrame({"station_id": ["A", "B", "C"], "hour_of_week": [1, 2, 3], "rank": [1, 2, 3]})
    ranking_b = pl.DataFrame({"station_id": ["A", "B", "C"], "hour_of_week": [1, 2, 3], "rank": [3, 2, 1]})
    stability = alloc.spearman_stability({(1.0, 0.1): ranking_a, (2.0, 0.2): ranking_b})
    assert stability.row(0, named=True)["spearman"] == pytest.approx(-1.0)


def test_stable_core_identifies_targets_present_in_every_draw():
    always = pl.DataFrame({"station_id": ["ALWAYS"], "hour_of_week": [1], "tier": ["tier1_scheduled"]})
    sometimes = pl.DataFrame({"station_id": ["SOMETIMES"], "hour_of_week": [2], "tier": ["tier2_dynamic"]})
    draw1 = pl.concat([always, sometimes], how="vertical")
    draw2 = always.clone()
    core = alloc.stable_core({(1.0, 0.1): draw1, (2.0, 0.2): draw2})
    always_row = core.filter(pl.col("station_id") == "ALWAYS").row(0, named=True)
    sometimes_row = core.filter(pl.col("station_id") == "SOMETIMES").row(0, named=True)
    assert always_row["appearance_frac"] == pytest.approx(1.0)
    assert sometimes_row["appearance_frac"] == pytest.approx(0.5)
