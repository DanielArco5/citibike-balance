"""Synthetic-fixture tests for src/models/censoring.py.

Each test constructs a small hand-built flag sequence or panel where the
correct answer is known by inspection, per CLAUDE.md's non-negotiable that
every model function gets a test with an analytically known answer."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from models.censoring import (  # noqa: E402
    add_run_length,
    assign_strata,
    bucket_run_length,
    compute_capacity_staleness,
    compute_stratum_edges,
    evaluate_recovery,
    make_synthetic_censoring_runs_matched,
    stratum_proportions,
    summarize_coverage_by_decile,
    make_synthetic_censoring_runs,
)


def _station_frame(station_id: str, flags: list[bool]) -> pl.DataFrame:
    n = len(flags)
    return pl.DataFrame(
        {
            "station_id": [station_id] * n,
            "interval_start": list(range(n)),
            "is_bike_empty": flags,
        }
    )


def test_run_length_single_isolated_censored_interval():
    # censored, open, open -> run length 1 at position 0, 0 elsewhere.
    df = _station_frame("A", [True, False, False])
    out = add_run_length(df, "is_bike_empty", "run_len")
    assert out.sort("interval_start")["run_len"].to_list() == [1, 0, 0]


def test_run_length_whole_run_gets_full_length_not_position():
    # open, censored, censored, censored, open -> every censored row gets 3.
    df = _station_frame("A", [False, True, True, True, False])
    out = add_run_length(df, "is_bike_empty", "run_len")
    assert out.sort("interval_start")["run_len"].to_list() == [0, 3, 3, 3, 0]


def test_run_length_two_separate_runs_not_merged():
    df = _station_frame("A", [True, True, False, True, True, True])
    out = add_run_length(df, "is_bike_empty", "run_len")
    assert out.sort("interval_start")["run_len"].to_list() == [2, 2, 0, 3, 3, 3]


def test_run_length_does_not_bridge_across_stations():
    # Station A ends censored, station B starts censored -- must NOT merge
    # into one run just because they're adjacent rows before sorting.
    a = _station_frame("A", [False, True])
    b = _station_frame("B", [True, False])
    df = pl.concat([a, b])
    out = add_run_length(df, "is_bike_empty", "run_len").sort("station_id", "interval_start")
    assert out.filter(pl.col("station_id") == "A")["run_len"].to_list() == [0, 1]
    assert out.filter(pl.col("station_id") == "B")["run_len"].to_list() == [1, 0]


def test_bucket_run_length_edges():
    edges = [1, 4]
    df = pl.DataFrame({"run_len": [1, 2, 4, 5, 10]})
    out = df.with_columns(bucket_run_length(pl.col("run_len"), edges).alias("bucket"))
    assert out["bucket"].to_list() == [0, 1, 1, 2, 2]


def test_capacity_staleness_flags_only_stations_below_peak_throughput():
    # Station A: capacity 10, but one interval has departures+arrivals=12 -- stale.
    # Station B: capacity 10, peak throughput 8 -- not stale.
    panel = pl.DataFrame(
        {
            "station_id": ["A", "A", "B", "B"],
            "departures": [7, 2, 5, 3],
            "arrivals": [5, 1, 3, 2],
            "capacity": [10, 10, 10, 10],
        }
    )
    out = compute_capacity_staleness(panel).sort("station_id")
    assert out["is_capacity_stale"].to_list() == [True, False]
    assert out.filter(pl.col("station_id") == "A")["peak_throughput"].item() == 12


def test_synthetic_censoring_run_caps_at_station_quantile_and_preserves_true_D():
    df = pl.DataFrame(
        {
            "station_id": ["A"] * 20,
            "interval_start": list(range(20)),
            "departures": [10] * 10 + [1] * 10,  # low quantile comes from the tail of 1s
        }
    )
    out = make_synthetic_censoring_runs(
        df, value_col="departures", run_lengths=[3], n_runs_per_length=1, cap_quantile=0.1, rng_seed=0
    )
    assert out.height == 3
    assert out["synthetic_run_length"].to_list() == [3, 3, 3]
    # censored_Y must never exceed true_D, and must equal true_D wherever
    # true_D was already <= the station's cap.
    assert (out["censored_Y"] <= out["true_D"]).all()
    below_cap = out.filter(pl.col("true_D") <= 1)
    assert (below_cap["censored_Y"] == below_cap["true_D"]).all()


def test_evaluate_recovery_perfect_prediction_gives_zero_error():
    censored = pl.DataFrame(
        {
            "station_id": ["A", "A", "B"],
            "interval_start": [0, 1, 0],
            "true_D": [10.0, 12.0, 5.0],
            "censored_Y": [3.0, 4.0, 2.0],
            "synthetic_run_length": [1, 1, 5],
        }
    )
    true_unmet = censored["true_D"] - censored["censored_Y"]
    result = evaluate_recovery(censored, true_unmet, edges=[1, 4])
    overall = result.filter(pl.col("stratum") == "overall")
    assert overall["mae"].item() == pytest.approx(0.0)
    assert overall["bias"].item() == pytest.approx(0.0)


def test_evaluate_recovery_detects_systematic_underprediction():
    censored = pl.DataFrame(
        {
            "station_id": ["A", "A"],
            "interval_start": [0, 1],
            "true_D": [10.0, 10.0],
            "censored_Y": [3.0, 3.0],
            "synthetic_run_length": [1, 1],
        }
    )
    true_unmet = censored["true_D"] - censored["censored_Y"]  # = 7, 7
    under_predicted = true_unmet - 2.0  # consistently 2 too low
    result = evaluate_recovery(censored, under_predicted, edges=[1, 4])
    overall = result.filter(pl.col("stratum") == "overall")
    assert overall["bias"].item() == pytest.approx(-2.0)
    assert overall["mae"].item() == pytest.approx(2.0)


def test_evaluate_recovery_stratifies_by_capacity_stale_flag():
    censored = pl.DataFrame(
        {
            "station_id": ["A", "B"],
            "interval_start": [0, 0],
            "true_D": [10.0, 10.0],
            "censored_Y": [3.0, 3.0],
            "synthetic_run_length": [1, 1],
        }
    )
    stale_flags = pl.DataFrame({"station_id": ["A", "B"], "is_capacity_stale": [True, False]})
    imputed = pl.Series([7.0, 7.0])  # perfect for both
    result = evaluate_recovery(censored, imputed, edges=[1, 4], stale_flags=stale_flags)
    assert set(result["stratum"].unique().to_list()) == {"overall", "run_length_bucket", "capacity_stale"}
    stale_rows = result.filter(pl.col("stratum") == "capacity_stale")
    assert stale_rows.height == 2


def _target_and_pool(pool_has_high_decile: bool):
    # Target (genuinely-censored) population: 5 low-demand rows (D_hat=1),
    # 5 high-demand rows (D_hat=9) -> median edge 5.0 -> two deciles, 50/50.
    target = pl.DataFrame(
        {
            "station_id": ["X"] * 10,
            "interval_start": list(range(10)),
            "departures": [5] * 10,
            "D_hat": [1.0] * 5 + [9.0] * 5,
            "capacity": [10] * 10,
            "hour_of_week": [5] * 10,
        }
    )
    n_high = 10 if pool_has_high_decile else 0
    pool = pl.DataFrame(
        {
            "station_id": [f"s{i}" for i in range(10 + n_high)],
            "interval_start": [0] * (10 + n_high),
            "departures": [5] * (10 + n_high),
            "D_hat": [1.0] * 10 + [9.0] * n_high,
            "capacity": [10] * (10 + n_high),
            "hour_of_week": [5] * (10 + n_high),
        }
    )
    return target, pool


def test_compute_stratum_edges_derived_from_target_not_pool():
    target, pool = _target_and_pool(pool_has_high_decile=False)
    edges = compute_stratum_edges(target, "D_hat", "capacity", n_deciles=2, n_capacity_bands=1)
    assert edges["d_hat_edges"] == pytest.approx(np.quantile(target["D_hat"].to_numpy(), [0.5]))
    assert edges["capacity_edges"].shape == (0,)  # single band -> no interior cut points


def test_assign_strata_buckets_low_and_high_demand_separately():
    target, _ = _target_and_pool(pool_has_high_decile=False)
    edges = compute_stratum_edges(target, "D_hat", "capacity", n_deciles=2, n_capacity_bands=1)
    out = assign_strata(target, "D_hat", "capacity", edges)
    assert out.filter(pl.col("D_hat") == 1.0)["demand_decile"].unique().to_list() == [0]
    assert out.filter(pl.col("D_hat") == 9.0)["demand_decile"].unique().to_list() == [1]
    assert out["capacity_band"].unique().to_list() == [0]


def test_stratum_proportions_sums_to_one_and_matches_known_split():
    target, _ = _target_and_pool(pool_has_high_decile=False)
    edges = compute_stratum_edges(target, "D_hat", "capacity", n_deciles=2, n_capacity_bands=1)
    strata = assign_strata(target, "D_hat", "capacity", edges)
    props = stratum_proportions(strata)
    assert props["target_prop"].sum() == pytest.approx(1.0)
    assert sorted(props["target_prop"].to_list()) == pytest.approx([0.5, 0.5])


def test_matched_sampling_hits_target_proportions_when_supply_available():
    target, pool = _target_and_pool(pool_has_high_decile=True)
    synthetic, coverage = make_synthetic_censoring_runs_matched(
        pool,
        target,
        value_col="departures",
        d_hat_col="D_hat",
        run_lengths=[1],
        n_runs_per_length=10,
        n_deciles=2,
        n_capacity_bands=1,
        rng_seed=0,
    )
    # desired = round(0.5*10) per stratum = 5 each, both strata fully supplied.
    assert synthetic.height == 10
    assert coverage["taken"].sum() == 10
    assert (coverage["taken"] == coverage["desired"]).all()


def test_matched_sampling_reports_shortfall_when_target_stratum_has_no_pool_support():
    # Pool has NO high-demand-decile candidates -- exactly the "real censoring
    # concentrates where uncensored exemplars are scarce" scenario.
    target, pool = _target_and_pool(pool_has_high_decile=False)
    synthetic, coverage = make_synthetic_censoring_runs_matched(
        pool,
        target,
        value_col="departures",
        d_hat_col="D_hat",
        run_lengths=[1],
        n_runs_per_length=10,
        n_deciles=2,
        n_capacity_bands=1,
        rng_seed=0,
    )
    high_decile_coverage = coverage.filter(pl.col("stratum_key").str.starts_with("1_"))
    assert (high_decile_coverage["available"] == 0).all()
    assert (high_decile_coverage["taken"] == 0).all()
    # Shortfall must NOT be silently backfilled from the low-decile stratum --
    # only the low stratum's own (fully-supplied) demand should appear.
    assert synthetic.height == 5
    assert synthetic["true_D"].max() == 5  # only decile-0 rows (departures=5) got drawn; none from decile 1


def test_summarize_coverage_by_decile_computes_fill_rate():
    target, pool = _target_and_pool(pool_has_high_decile=False)
    _, coverage = make_synthetic_censoring_runs_matched(
        pool,
        target,
        value_col="departures",
        d_hat_col="D_hat",
        run_lengths=[1],
        n_runs_per_length=10,
        n_deciles=2,
        n_capacity_bands=1,
        rng_seed=0,
    )
    summary = summarize_coverage_by_decile(coverage).sort("demand_decile")
    assert summary["demand_decile"].to_list() == [0, 1]
    fill_rates = summary["fill_rate"].to_list()
    assert fill_rates[0] == pytest.approx(1.0)  # decile 0 fully supplied
    assert fill_rates[1] == pytest.approx(0.0)  # decile 1 had zero supply


def test_summarize_coverage_by_decile_does_not_double_count_target_mass_across_run_lengths():
    # coverage has one row per (run_length, stratum_key) -- target_prop is
    # the SAME value repeated per run_length, not additive. With 3 run
    # lengths, a naive sum would inflate each decile's target_mass 3x.
    target, pool = _target_and_pool(pool_has_high_decile=True)
    synthetic, coverage = make_synthetic_censoring_runs_matched(
        pool,
        target,
        value_col="departures",
        d_hat_col="D_hat",
        run_lengths=[1, 2, 3],
        n_runs_per_length=10,
        n_deciles=2,
        n_capacity_bands=1,
        rng_seed=0,
    )
    summary = summarize_coverage_by_decile(coverage).sort("demand_decile")
    assert summary["target_mass"].sum() == pytest.approx(1.0)
    assert summary["target_mass"].to_list() == pytest.approx([0.5, 0.5])
    # desired/taken ARE additive across run lengths -- 3 run lengths x
    # desired=5/decile (round(0.5*10)) = 15 desired per decile.
    assert summary["desired_n"].to_list() == [15, 15]
