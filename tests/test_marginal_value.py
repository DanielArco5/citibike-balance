"""Synthetic-fixture tests for src/opt/marginal_value.py -- Phase 8 Part A
(SPEC.md §7, RUNBOOK Phase 8). Per CLAUDE.md: every model function gets a
test with a synthetic fixture where the answer is known analytically. None
of these touch real data on disk -- functions that read from
UNMET_DEMAND_NET_PATH/INVENTORY_PATH directly (build_month_station_hour_
week_partial, finalize_station_hour_week) aren't unit tested here for that
reason; everything downstream of them takes plain DataFrames/arrays and is.

Covers both the retained stationary-model utilities (mm1k_p0,
relaxation_time_minutes -- kept as the documented record of why that
approach was abandoned, see marginal_value.py's module docstring) and the
current transient first-passage model (build_step_transition_matrix,
hitting_probabilities, mv_curve, check_concavity).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.stats import skellam

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import opt.marginal_value as mv  # noqa: E402
from opt.marginal_value import MarginalValueParams  # noqa: E402

PARAMS = MarginalValueParams(
    min_weeks_for_cross_check=5,
    relaxation_time_flag_minutes=15.0,
    first_passage_window_minutes=60.0,
    mv_k_max=4,
    schedulable_modal_share_threshold=0.7,
)


# ---------------------------------------------------------------------------
# Retained stationary-model utilities (mm1k_p0, relaxation_time_minutes)
# ---------------------------------------------------------------------------


def test_mm1k_p0_matches_hand_computed_value():
    # rho=0.5, K=2: (1-0.5)/(1-0.5^3) = 0.5/0.875
    assert mv.mm1k_p0(0.5, 2) == pytest.approx(0.5 / 0.875)


def test_mm1k_p0_uniform_when_rho_is_one():
    for capacity in (1, 3, 10):
        assert mv.mm1k_p0(1.0, capacity) == pytest.approx(1.0 / (capacity + 1))


def test_mm1k_p0_degenerate_zero_capacity_always_empty():
    assert mv.mm1k_p0(0.3, 0) == 1.0
    assert mv.mm1k_p0(5.0, 0) == 1.0


def test_relaxation_time_matches_closed_form():
    lam, mu = 1.0, 2.0
    expected = 15.0 / (np.sqrt(mu) - np.sqrt(lam)) ** 2
    assert mv.relaxation_time_minutes(lam, mu) == pytest.approx(expected)


def test_relaxation_time_nan_when_rho_at_or_above_one():
    assert np.isnan(mv.relaxation_time_minutes(2.0, 2.0))
    assert np.isnan(mv.relaxation_time_minutes(3.0, 2.0))


# ---------------------------------------------------------------------------
# Transient first-passage model: transition matrix
# ---------------------------------------------------------------------------


def test_build_step_transition_matrix_rows_sum_to_one():
    """Basic probability conservation -- must hold regardless of rates or
    capacity, since every unit of truncated Skellam tail mass is
    renormalized back in, not dropped."""
    for lam, mu, capacity in [(0.5, 1.0, 5), (1.2, 0.8, 10), (2.0, 2.0, 3), (0.05, 0.05, 20)]:
        P = mv.build_step_transition_matrix(lam, mu, capacity)
        assert P.shape == (capacity + 1, capacity + 1)
        assert np.allclose(P.sum(axis=1), 1.0)


def test_build_step_transition_matrix_capacity_one_matches_skellam_cdf():
    """capacity=1 (two states, 0 and 1) is small enough that the transition
    probabilities are exactly a Skellam CDF tail, checkable by hand:
    1->0 requires a net change of -1 or worse; 0->1 requires +1 or better
    (both boundaries, so no interior-state ambiguity to worry about)."""
    lam, mu = 0.8, 1.3
    P = mv.build_step_transition_matrix(lam, mu, 1)
    assert P[1, 0] == pytest.approx(skellam.cdf(-1, lam, mu), abs=1e-6)
    assert P[0, 1] == pytest.approx(1.0 - skellam.cdf(0, lam, mu), abs=1e-6)


def test_build_step_transition_matrix_degenerate_zero_rates_is_identity():
    P = mv.build_step_transition_matrix(0.0, 0.0, 4)
    assert np.allclose(P, np.eye(5))


# ---------------------------------------------------------------------------
# Transient first-passage model: hitting probabilities
# ---------------------------------------------------------------------------


def test_hitting_probabilities_one_step_capacity_one_matches_skellam_cdf():
    """n_steps=1 is just the one-step transition matrix's first column --
    same analytic check as the transition-matrix test above, exercised
    through the public hitting_probabilities entrypoint instead."""
    lam, mu = 0.8, 1.3
    hit = mv.hitting_probabilities(lam, mu, capacity=1, n_steps=1)
    assert hit[0] == 1.0  # already empty -- already "hit" by definition
    assert hit[1] == pytest.approx(skellam.cdf(-1, lam, mu), abs=1e-6)


def test_hitting_probabilities_monotonic_in_time():
    """More time to hit empty can only raise (or hold) the probability,
    for any fixed starting level -- true regardless of the specific rates,
    a basic property of first-passage probabilities."""
    lam, mu, capacity = 0.5, 1.5, 8
    hit1 = mv.hitting_probabilities(lam, mu, capacity, n_steps=1)
    hit4 = mv.hitting_probabilities(lam, mu, capacity, n_steps=4)
    assert np.all(hit4 >= hit1 - 1e-9)


def test_hitting_probabilities_monotonic_in_starting_level():
    """More bikes at the start can only lower (or hold) the chance of
    hitting empty within a fixed window."""
    lam, mu, capacity = 0.5, 1.5, 8
    hit = mv.hitting_probabilities(lam, mu, capacity, n_steps=4)
    assert np.all(np.diff(hit) <= 1e-9)


def test_hitting_probabilities_zero_capacity_always_hit():
    assert mv.hitting_probabilities(1.0, 1.0, 0, n_steps=4) == pytest.approx([1.0])


# ---------------------------------------------------------------------------
# Transient first-passage model: mv_curve, check_concavity
# ---------------------------------------------------------------------------


def test_mv_curve_zero_when_no_historical_net_lost():
    curve = mv.mv_curve(1.0, 2.0, 5, e_net_lost_given_stockout=0.0, n_steps=4)
    assert np.all(curve == 0.0)


def test_mv_curve_matches_hitting_probability_deltas():
    lam, mu, capacity, e_net_lost, n_steps = 0.5, 1.5, 6, 2.0, 4
    curve = mv.mv_curve(lam, mu, capacity, e_net_lost, n_steps)
    hit = mv.hitting_probabilities(lam, mu, capacity, n_steps)
    expected = e_net_lost * (hit[:-1] - hit[1:])
    assert np.allclose(curve, expected)
    assert np.all(curve >= -1e-9)  # each additional bike can't hurt


def test_check_concavity_detects_violation_at_correct_index():
    mv_arr = np.array([5.0, 3.0, 4.0, 2.0])  # MV(3)=4 > MV(2)=3 -- a real violation
    is_concave, first_violation_n = mv.check_concavity(mv_arr)
    assert not is_concave
    assert first_violation_n == 2


def test_check_concavity_true_for_strictly_decreasing():
    mv_arr = np.array([10.0, 7.0, 4.0, 1.0])
    is_concave, violation = mv.check_concavity(mv_arr)
    assert is_concave
    assert violation is None


# ---------------------------------------------------------------------------
# aggregate_station_hour
# ---------------------------------------------------------------------------


def test_aggregate_station_hour_weights_by_n_intervals():
    """Two weeks of the same (station, hour) cell with different
    n_intervals -- the yearly rate must be the INTERVAL-weighted mean, not
    a plain average of the two weekly means."""
    weekly = pl.DataFrame(
        {
            "station_id": ["S1", "S1"],
            "hour_of_week": [10, 10],
            "week_start": ["2025-01-06", "2025-01-13"],
            "capacity": [10, 10],
            "zone_agg": ["Z1", "Z1"],
            "p_stockout": [0.5, 0.0],
            "mean_dep_D_hat": [4.0, 2.0],
            "mean_arr_D_hat": [1.0, 1.0],
            "mean_dep_net_lost": [1.0, 0.0],
            "mean_inventory": [2.0, 5.0],
            "mean_nontrip_in": [0.0, 0.0],
            "mean_nontrip_out": [0.0, 0.0],
            "n_intervals": [1, 3],  # week 1 thin (1 obs), week 2 full (3 obs)
        }
    )
    agg = mv.aggregate_station_hour(weekly)
    row = agg.row(0, named=True)
    # dep_rate = (4*1 + 2*3) / 4 = 10/4 = 2.5, NOT the plain mean (4+2)/2=3.
    # nontrip_out is 0 in this fixture, so mu == dep_rate exactly.
    assert row["mu"] == pytest.approx(2.5)
    assert row["n_weeks"] == 2
    assert row["n_intervals"] == 4


# ---------------------------------------------------------------------------
# hourly_net_lost_given_stockout
# ---------------------------------------------------------------------------


def test_hourly_net_lost_given_stockout_averages_hour_totals_over_stockout_hours_only():
    weekly = pl.DataFrame(
        {
            "station_id": ["S1", "S1", "S1"],
            "hour_of_week": [5, 5, 5],
            "p_stockout": [0.5, 0.0, 0.25],  # middle row never stocked out -- excluded
            "mean_dep_net_lost": [2.0, 0.0, 1.0],
            "n_intervals": [4, 4, 4],
        }
    )
    result = mv.hourly_net_lost_given_stockout(weekly)
    assert result.height == 1
    row = result.row(0, named=True)
    # hour totals: 2.0*4=8.0 and 1.0*4=4.0 -> mean=6.0 (the p_stockout=0 row excluded)
    assert row["e_net_lost_given_stockout"] == pytest.approx(6.0)
    assert row["n_stockout_hour_instances"] == 2


# ---------------------------------------------------------------------------
# cross_check_regression
# ---------------------------------------------------------------------------


def _make_weekly_fixture(
    station_id: str, hour_of_week: int, inventories: list[float], slope: float, intercept: float, n_intervals: int = 4
) -> pl.DataFrame:
    """hour_net_lost (= mean_dep_net_lost * n_intervals) = intercept +
    slope*inventory exactly -- mean_dep_net_lost is back-derived from that
    so cross_check_regression's hour-total computation recovers the known
    slope exactly, not approximately."""
    n = len(inventories)
    hour_totals = [max(0.0, intercept + slope * inv) for inv in inventories]
    mean_net_lost = [t / n_intervals for t in hour_totals]
    return pl.DataFrame(
        {
            "station_id": [station_id] * n,
            "hour_of_week": [hour_of_week] * n,
            "week_start": [f"2025-{i+1:02d}-01" for i in range(n)],
            "mean_inventory": inventories,
            "mean_dep_net_lost": mean_net_lost,
            "n_intervals": [n_intervals] * n,
        }
    )


def test_cross_check_regression_recovers_known_slope():
    """hour_net_lost = 10 - 2*inventory (exact, no noise) across 8 weeks --
    empirical_mv (= -slope) must recover 2.0 exactly."""
    inventories = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 4.5]
    weekly = _make_weekly_fixture("S1", 42, inventories, slope=-2.0, intercept=10.0)
    result = mv.cross_check_regression(weekly, PARAMS)
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["empirical_mv"] == pytest.approx(2.0, abs=1e-9)
    assert row["n_weeks"] == 8


def test_cross_check_regression_skips_cells_below_min_weeks():
    inventories = [1.0, 2.0, 3.0]  # 3 weeks, below PARAMS.min_weeks_for_cross_check=5
    weekly = _make_weekly_fixture("S2", 10, inventories, slope=-1.0, intercept=5.0)
    result = mv.cross_check_regression(weekly, PARAMS)
    assert result.height == 0


def test_cross_check_regression_skips_cells_with_no_inventory_variation():
    inventories = [3.0] * 6  # enough weeks, but zero variation to regress on
    weekly = _make_weekly_fixture("S3", 10, inventories, slope=-1.0, intercept=5.0)
    result = mv.cross_check_regression(weekly, PARAMS)
    assert result.height == 0


# ---------------------------------------------------------------------------
# build_cross_check_report
# ---------------------------------------------------------------------------


def test_build_cross_check_report_matches_midpoint_k_and_computes_ratio():
    cross_check = pl.DataFrame(
        {
            "station_id": ["S1"],
            "hour_of_week": [10],
            "empirical_mv": [1.5],
            "n_weeks": [12],
            "inventory_min": [2.0],
            "inventory_max": [4.0],  # midpoint 3.0 -> k=3
        }
    )
    mv_curve_exploded = pl.DataFrame(
        {
            "station_id": ["S1", "S1", "S1"],
            "hour_of_week": [10, 10, 10],
            "k": [2, 3, 4],
            "mv": [2.0, 1.0, 0.5],
        }
    )
    report = mv.build_cross_check_report(cross_check, mv_curve_exploded)
    assert report.height == 1
    row = report.row(0, named=True)
    assert row["k"] == 3
    assert row["model_mv"] == pytest.approx(1.0)
    assert row["diff"] == pytest.approx(0.5)
    assert row["ratio"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# build_mv_curve_table: truncation at mv_k_max
# ---------------------------------------------------------------------------


def test_build_mv_curve_table_truncates_at_mv_k_max():
    """Per DECISIONS.md's persistence entry: only k=1..mv_k_max is emitted
    -- capacity here (20) is well above mv_k_max (4), so the curve must
    stop at exactly 4 rows, not continue to capacity."""
    station_hour = pl.DataFrame(
        {
            "station_id": ["S1"],
            "hour_of_week": [10],
            "capacity": [20],
            "lam": [0.5],
            "mu": [1.5],
            "e_net_lost_given_stockout": [3.0],
        }
    )
    summary, curve = mv.build_mv_curve_table(station_hour, PARAMS, n_steps=4)
    assert curve.height == PARAMS.mv_k_max
    assert sorted(curve["k"].to_list()) == [1, 2, 3, 4]


def test_build_mv_curve_table_stops_at_capacity_if_smaller_than_mv_k_max():
    station_hour = pl.DataFrame(
        {
            "station_id": ["S1"],
            "hour_of_week": [10],
            "capacity": [2],  # smaller than mv_k_max=4
            "lam": [0.5],
            "mu": [1.5],
            "e_net_lost_given_stockout": [3.0],
        }
    )
    summary, curve = mv.build_mv_curve_table(station_hour, PARAMS, n_steps=4)
    assert curve.height == 2
    assert sorted(curve["k"].to_list()) == [1, 2]


# ---------------------------------------------------------------------------
# eligibility_report
# ---------------------------------------------------------------------------


def test_eligibility_report_flags_cells_by_min_inventory_and_shares_net_lost():
    """S1 (min_inventory=3, eligible) and S2 (min_inventory=10, not
    eligible) each contribute known hour-net-lost totals -- the reported
    eligible share must match the hand-computed fraction exactly."""
    weekly = pl.DataFrame(
        {
            "station_id": ["S1", "S1", "S2", "S2"],
            "hour_of_week": [5, 5, 8, 8],
            "mean_inventory": [3.0, 6.0, 10.0, 12.0],  # S1 min=3 (eligible), S2 min=10 (not)
            "mean_dep_net_lost": [1.0, 0.5, 0.25, 0.1],
            "n_intervals": [4, 4, 4, 4],
        }
    )
    summary, eligible = mv.eligibility_report(weekly, PARAMS)

    assert summary["n_total_cells"] == 2  # (S1,5) and (S2,8)
    assert summary["n_eligible_cells"] == 1
    assert eligible["station_id"].to_list() == ["S1"]

    # hour totals: S1 = (1.0+0.5)*4=6.0, S2 = (0.25+0.1)*4=1.4 -> total=7.4
    s1_total = (1.0 + 0.5) * 4
    s2_total = (0.25 + 0.1) * 4
    assert summary["total_net_lost"] == pytest.approx(s1_total + s2_total)
    assert summary["eligible_net_lost"] == pytest.approx(s1_total)
    assert summary["eligible_net_lost_share"] == pytest.approx(s1_total / (s1_total + s2_total))


# ---------------------------------------------------------------------------
# eligibility_frequency_report
# ---------------------------------------------------------------------------


def test_eligibility_frequency_report_computes_low_frac_and_correct_bucket():
    """S1: low (<=4) in 3 of 4 weeks -> low_frac=0.75 -> '>50%' bucket.
    S2: low in 1 of 4 weeks -> low_frac=0.25 -> '10-25%' bucket boundary
    (<=0.25 inclusive, per the module's bucket definition)."""
    weekly = pl.DataFrame(
        {
            "station_id": ["S1"] * 4 + ["S2"] * 4,
            "hour_of_week": [5] * 4 + [9] * 4,
            "mean_inventory": [2.0, 3.0, 4.0, 8.0, 2.0, 9.0, 9.0, 9.0],  # S1: 3/4 low; S2: 1/4 low
            "mean_dep_net_lost": [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "n_intervals": [4] * 8,
        }
    )
    per_cell, buckets = mv.eligibility_frequency_report(weekly, PARAMS)

    per_cell = per_cell.sort("station_id")
    s1 = per_cell.filter(pl.col("station_id") == "S1").row(0, named=True)
    s2 = per_cell.filter(pl.col("station_id") == "S2").row(0, named=True)
    assert s1["n_weeks"] == 4
    assert s1["n_low_weeks"] == 3
    assert s1["low_frac"] == pytest.approx(0.75)
    assert s2["n_low_weeks"] == 1
    assert s2["low_frac"] == pytest.approx(0.25)

    # S1's cell_net_lost sits entirely in the ">50%" (chronic) bucket,
    # S2's entirely in "10-25%".
    chronic = buckets.filter(pl.col("bucket").str.contains(">50%"))
    ten_25 = buckets.filter(pl.col("bucket").str.contains("10-25%"))
    assert chronic.row(0, named=True)["n_cells"] == 1
    assert chronic.row(0, named=True)["net_lost"] == pytest.approx(s1["cell_net_lost"])
    assert ten_25.row(0, named=True)["n_cells"] == 1
    assert ten_25.row(0, named=True)["net_lost"] == pytest.approx(s2["cell_net_lost"])


def test_eligibility_frequency_report_bucket_shares_sum_to_one():
    weekly = pl.DataFrame(
        {
            "station_id": ["S1"] * 2 + ["S2"] * 2 + ["S3"] * 2,
            "hour_of_week": [1] * 2 + [1] * 2 + [1] * 2,
            "mean_inventory": [2.0, 2.0, 8.0, 8.0, 2.0, 8.0],  # S1 always low, S2 never, S3 half
            "mean_dep_net_lost": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "n_intervals": [4] * 6,
        }
    )
    per_cell, buckets = mv.eligibility_frequency_report(weekly, PARAMS)
    assert buckets["n_cells"].sum() == 3
    assert buckets["cell_share"].sum() == pytest.approx(1.0)
    assert buckets["net_lost_share"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# chronic_timing_summary
# ---------------------------------------------------------------------------


def test_chronic_timing_summary_perfectly_predictable():
    """Same sub-interval position every low week -- std=0, modal_share=1.0,
    the "schedulable" case."""
    per_hour_instance = pl.DataFrame(
        {
            "station_id": ["S1"] * 5,
            "hour_of_week": [10] * 5,
            "week_start": [f"2025-{i+1:02d}-01" for i in range(5)],
            "first_low_position": [2, 2, 2, 2, 2],
        }
    )
    result = mv.chronic_timing_summary(per_hour_instance)
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["n_low_weeks_observed"] == 5
    assert row["mean_position"] == pytest.approx(2.0)
    assert row["std_position"] == pytest.approx(0.0)
    assert row["modal_share"] == pytest.approx(1.0)


def test_chronic_timing_summary_scattered_uniformly():
    """Every one of the 4 slots hit exactly once -- modal_share=0.25 (the
    least-predictable case a 4-slot scale can show), std matches the
    hand-computed sample std of {0,1,2,3}."""
    per_hour_instance = pl.DataFrame(
        {
            "station_id": ["S1"] * 4,
            "hour_of_week": [10] * 4,
            "week_start": [f"2025-{i+1:02d}-01" for i in range(4)],
            "first_low_position": [0, 1, 2, 3],
        }
    )
    result = mv.chronic_timing_summary(per_hour_instance)
    row = result.row(0, named=True)
    assert row["modal_share"] == pytest.approx(0.25)
    assert row["std_position"] == pytest.approx(np.std([0, 1, 2, 3], ddof=1))


def test_chronic_timing_summary_single_observation_std_is_null_not_zero():
    """One low week observed -- variance is genuinely UNDEFINED, not 0;
    must not be silently reported as perfectly predictable via a fake
    zero."""
    per_hour_instance = pl.DataFrame(
        {
            "station_id": ["S1"],
            "hour_of_week": [10],
            "week_start": ["2025-01-01"],
            "first_low_position": [1],
        }
    )
    result = mv.chronic_timing_summary(per_hour_instance)
    row = result.row(0, named=True)
    assert row["n_low_weeks_observed"] == 1
    assert row["std_position"] is None
    assert row["modal_share"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# rebalancing_vs_chronicity_report / rebalancing_by_demand_decile
# ---------------------------------------------------------------------------


def _station_hour_row(station_id, hour_of_week, dep_rate, nontrip_in_rate):
    return {
        "station_id": station_id,
        "hour_of_week": hour_of_week,
        "capacity": 20,
        "zone_agg": "Z1",
        "n_weeks": 40,
        "n_intervals": 160,
        "lam": 1.0,
        "mu": dep_rate,
        "p_stockout_empirical": 0.1,
        "mv_empirical_baseline": 0.5,
        "dep_rate": dep_rate,
        "arr_rate": 1.0,
        "nontrip_in_rate": nontrip_in_rate,
        "nontrip_out_rate": 0.0,
    }


def test_rebalancing_vs_chronicity_report_flags_under_rebalanced_chronic_cells():
    """Chronic cell (S1) gets HALF the rebalancing intensity of an
    otherwise-identical non-chronic cell (S2) at the same demand level --
    the ratio must come out well below 1.0, not masked by raw-rate
    differences."""
    station_hour = pl.DataFrame(
        [
            _station_hour_row("S1", 5, dep_rate=2.0, nontrip_in_rate=0.5),  # chronic, intensity=0.25
            _station_hour_row("S2", 6, dep_rate=2.0, nontrip_in_rate=1.0),  # non-chronic, intensity=0.5
        ]
    )
    low_freq_per_cell = pl.DataFrame(
        {"station_id": ["S1", "S2"], "hour_of_week": [5, 6], "low_frac": [0.8, 0.1]}
    )
    summary, joined = mv.rebalancing_vs_chronicity_report(station_hour, low_freq_per_cell)
    assert summary["n_chronic"] == 1
    assert summary["n_non_chronic"] == 1
    assert summary["chronic_median_rebalancing_intensity"] == pytest.approx(0.25, rel=1e-3)
    assert summary["non_chronic_median_rebalancing_intensity"] == pytest.approx(0.5, rel=1e-3)
    assert summary["intensity_ratio_chronic_over_non_chronic"] == pytest.approx(0.5, rel=1e-2)
    assert joined.filter(pl.col("station_id") == "S1")["is_chronic"].item() is True
    assert joined.filter(pl.col("station_id") == "S2")["is_chronic"].item() is False


def test_rebalancing_by_demand_decile_compares_within_matched_demand_bins():
    """Two demand tiers (dep_rate ~1 vs ~10); within EACH tier, chronic
    cells get systematically less nontrip_in than non-chronic ones -- the
    per-decile comparison must preserve that within-tier gap rather than
    letting the low-demand chronic cells and high-demand non-chronic cells
    average out into a misleading aggregate."""
    rows = []
    for i in range(6):
        rows.append(_station_hour_row(f"C{i}", i, dep_rate=1.0, nontrip_in_rate=0.1))  # low demand, chronic
        rows.append(_station_hour_row(f"N{i}", i + 100, dep_rate=1.0, nontrip_in_rate=0.5))  # low demand, non-chronic
        rows.append(_station_hour_row(f"C{i}h", i + 200, dep_rate=10.0, nontrip_in_rate=1.0))  # high demand, chronic
        rows.append(_station_hour_row(f"N{i}h", i + 300, dep_rate=10.0, nontrip_in_rate=5.0))  # high demand, non-chronic
    station_hour = pl.DataFrame(rows)
    low_freq_rows = []
    for i in range(6):
        low_freq_rows.append({"station_id": f"C{i}", "hour_of_week": i, "low_frac": 0.9})
        low_freq_rows.append({"station_id": f"N{i}", "hour_of_week": i + 100, "low_frac": 0.05})
        low_freq_rows.append({"station_id": f"C{i}h", "hour_of_week": i + 200, "low_frac": 0.9})
        low_freq_rows.append({"station_id": f"N{i}h", "hour_of_week": i + 300, "low_frac": 0.05})
    low_freq_per_cell = pl.DataFrame(low_freq_rows)

    _summary, joined = mv.rebalancing_vs_chronicity_report(station_hour, low_freq_per_cell)
    per_decile = mv.rebalancing_by_demand_decile(joined)

    low_demand = per_decile.filter(pl.col("median_dep_rate") < 5.0)
    high_demand = per_decile.filter(pl.col("median_dep_rate") >= 5.0)
    for tier in (low_demand, high_demand):
        chronic_rate = tier.filter(pl.col("is_chronic"))["median_nontrip_in_rate"].item()
        non_chronic_rate = tier.filter(~pl.col("is_chronic"))["median_nontrip_in_rate"].item()
        assert chronic_rate < non_chronic_rate
