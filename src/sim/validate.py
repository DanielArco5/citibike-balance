"""Phase 7 validation harness (RUNBOOK Phase 7 -- the hard gate before
Phase 8): replay the held-out week defined by config/params.yaml's
simulation.validation_week_start and compare simulated vs. actual.

**Resolution (DECISIONS.md, Phase 7, 2026-08-12) -- read this before
interpreting any number below.** The original gate ("stockout-minutes
correlation > 0.7," per-INTERVAL) does not validate under stochastic
destination sampling, and structurally cannot: destination is the other
half of a realized trip, sampling it from historical marginals gets
aggregate flow right and realized pairing wrong, and inventory is a
path-dependent accumulation of realized pairings. That is not a bug in this
module or in od_shares.py -- see DECISIONS.md for the full argument and why
more OD conditioning is not the fix. The simulator instead validates on the
quantities Phase 8 actually needs: trip totals, per-zone volume, the
continuous inventory trajectory, and stockout RATE by station-hour-of-week
(what MV(s,t) consumes) -- NOT per-interval stockout timing.

Three run modes, all reported here:

1. **sanity_od**: departures forced to real historical counts, destinations
   drawn from OD shares. Isolates OD/routing/inventory-update machinery
   from demand-model error.
2. **sanity_true_dest**: departures AND destinations forced to real
   history. THE mode that validates the simulator's mechanics (routing,
   capacity constraints, N-replay) -- proves the trajectory is fundamentally
   right when given true trip-level ground truth. Not a mode Phase 8/9 runs
   against, since it requires future ground truth that doesn't exist for a
   forward simulation.
3. **stochastic**: departures sampled from the fitted demand model,
   destinations drawn from OD shares -- the real forward simulation. THE
   mode Phase 8 (marginal value) and Phase 9 (policy comparison) actually
   run. Phase 9 compares stochastic runs under DIFFERENT policies but the
   SAME sampling noise (same seed), so destination-assignment error is
   present in both the baseline and the treatment run and largely cancels
   in the DIFFERENCE Phase 9 reports -- this is an assumption carried
   forward from this validation, not proven here, and should be stated as
   such wherever Phase 9's lift numbers are reported.

Gates that still apply (RUNBOOK Phase 7, restated per the resolution
above): totals within ~5%, per-zone within ~15%, evaluated on
sanity_true_dest (mechanics validation) -- NOT on stochastic, which trades
some of that accuracy for destination-assignment noise by design. The
per-interval stockout-minutes correlation gate is retained as a diagnostic
(printed, not gating) precisely because it demonstrates the threshold-
sensitivity effect: continuous inventory can track well while the binary
flag does not.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace

import numpy as np
import polars as pl

import models.demand as demand
import models.od_shares as od_shares
import sim.simulator as simulator
import utils.progress as progress

TOTAL_TRIPS_GATE = 0.05
PER_ZONE_GATE = 0.15
# Retained as a printed reference point only -- NOT a pass/fail gate on any
# mode. See this module's docstring / DECISIONS.md Phase 7 resolution:
# per-interval stockout-timing correlation cannot survive stochastic
# destination sampling by construction. stockout_rate_by_station_hour below
# is the metric that actually gates against Phase 8's needs.
STOCKOUT_CORR_REFERENCE = 0.7


@dataclass
class GateReport:
    label: str
    total_trips_pct_error: float
    per_zone_wmape: float
    stockout_corr: float
    hourly_profile_wmape: float
    tier_usage: pl.DataFrame
    reroute_outcomes: pl.DataFrame
    reroute_distance_summary: dict


def _zone_lookup(network: simulator.NetworkArrays) -> pl.DataFrame:
    return pl.DataFrame({"station_id": network.station_id, "zone_agg": network.zone_agg})


def _completed_trips(trip_log: pl.DataFrame) -> pl.DataFrame:
    return trip_log.filter(pl.col("outcome").is_in(["direct", "rerouted"]))


def total_trips_pct_error(trip_log: pl.DataFrame, actual: pl.DataFrame) -> tuple[float, int, int]:
    sim_total = _completed_trips(trip_log).height
    actual_total = int(actual["departures"].sum())
    err = abs(sim_total - actual_total) / actual_total if actual_total else float("nan")
    return err, sim_total, actual_total


def per_zone_wmape(trip_log: pl.DataFrame, actual: pl.DataFrame, zone_lookup: pl.DataFrame) -> tuple[float, pl.DataFrame]:
    sim_by_zone = (
        _completed_trips(trip_log)
        .rename({"station_id": "origin_station_id"})
        .join(zone_lookup.rename({"station_id": "origin_station_id"}), on="origin_station_id", how="left")
        .group_by("zone_agg")
        .agg(pl.len().alias("sim_trips"))
    )
    actual_by_zone = (
        actual.join(zone_lookup, on="station_id", how="left")
        .group_by("zone_agg")
        .agg(pl.col("departures").sum().alias("actual_trips"))
    )
    comparison = actual_by_zone.join(sim_by_zone, on="zone_agg", how="left").with_columns(
        pl.col("sim_trips").fill_null(0)
    )
    wmape = demand.compute_wmape(
        comparison["actual_trips"].to_numpy().astype(float), comparison["sim_trips"].to_numpy().astype(float)
    )
    return wmape, comparison.sort("zone_agg")


def stockout_minutes_correlation(
    sim_intervals: pl.DataFrame, actual_inventory: pl.DataFrame, exclude_station_ids: set[str] | None = None
) -> tuple[float, pl.DataFrame]:
    if exclude_station_ids:
        excl = list(exclude_station_ids)
        sim_intervals = sim_intervals.filter(~pl.col("station_id").is_in(excl))
        actual_inventory = actual_inventory.filter(~pl.col("station_id").is_in(excl))

    sim_minutes = sim_intervals.group_by("station_id").agg(
        (pl.col("is_bike_empty").cast(pl.Int32).sum() * simulator.INTERVAL_MINUTES).alias("sim_stockout_min")
    )
    actual_minutes = actual_inventory.group_by("station_id").agg(
        (pl.col("is_bike_empty").cast(pl.Int32).sum() * simulator.INTERVAL_MINUTES).alias("actual_stockout_min")
    )
    comparison = actual_minutes.join(sim_minutes, on="station_id", how="left").with_columns(
        pl.col("sim_stockout_min").fill_null(0)
    )
    x = comparison["actual_stockout_min"].to_numpy().astype(float)
    y = comparison["sim_stockout_min"].to_numpy().astype(float)
    if x.std() == 0 or y.std() == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(x, y)[0, 1])
    return corr, comparison


def continuous_inventory_diagnostics(
    sim_intervals: pl.DataFrame, actual_inventory: pl.DataFrame, exclude_station_ids: set[str] | None = None
) -> dict:
    """Correlate simulated vs. actual CONTINUOUS inventory level per
    station-interval -- not the binary is_bike_empty/is_dock_full flag
    stockout_minutes_correlation uses. Requested alongside the seeding-
    exclusion check: if this correlates well (and |diff| is small in
    absolute bikes) while the binary stockout-minutes correlation is poor,
    the underlying trajectory is fine and the gate is measuring threshold-
    crossing sensitivity, not simulation quality -- a station oscillating
    around 1-2 bikes flips is_bike_empty on tiny differences even when its
    actual inventory path is nearly right. If this ALSO correlates poorly,
    the trajectory is genuinely wrong and the threshold isn't the issue."""
    sim = sim_intervals.select("station_id", "interval_start", pl.col("inventory").alias("sim_inventory"))
    actual = actual_inventory.select("station_id", "interval_start", pl.col("inventory").alias("actual_inventory"))
    if exclude_station_ids:
        excl = list(exclude_station_ids)
        sim = sim.filter(~pl.col("station_id").is_in(excl))
        actual = actual.filter(~pl.col("station_id").is_in(excl))

    comparison = actual.join(sim, on=["station_id", "interval_start"], how="inner")
    x = comparison["actual_inventory"].to_numpy().astype(float)
    y = comparison["sim_inventory"].to_numpy().astype(float)
    abs_diff = np.abs(x - y)
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {
        "n": len(x),
        "corr": corr,
        "mean_abs_diff": float(abs_diff.mean()),
        "median_abs_diff": float(np.median(abs_diff)),
        "p90_abs_diff": float(np.quantile(abs_diff, 0.9)),
        "p99_abs_diff": float(np.quantile(abs_diff, 0.99)),
        "max_abs_diff": float(abs_diff.max()),
    }


def stockout_rate_by_station_hour(
    sim_intervals: pl.DataFrame, actual_inventory: pl.DataFrame, calendar_weather: pl.DataFrame
) -> pl.DataFrame:
    """P(stockout) per (station, hour-of-week), aggregated over the
    held-out week -- the quantity Phase 8's MV(s,t) = P(stockout) x
    E[unmet trips | stockout] (SPEC.md §7) actually consumes, as opposed to
    "was station s empty at this exact 15-minute instant." A single held-out
    week gives each (station, hour-of-week) cell exactly 4 15-min
    sub-observations (one hour = 4 intervals), so the rate here is coarse
    but it's the right kind of coarse: it asks whether the simulator gets
    stockout FREQUENCY right at a given time-of-week, which is what a
    marginal-value estimate needs, not whether it reproduces the exact
    15-minute window a real stockout happened to start in.

    Returns (station_id, hour_of_week, sim_rate, actual_rate) -- one row
    per (station, hour-of-week) cell observed in either side."""
    hour_map = calendar_weather.select("interval_start", "hour_of_week")

    sim_rate = (
        sim_intervals.join(hour_map, on="interval_start", how="left")
        .group_by("station_id", "hour_of_week")
        .agg(pl.col("is_bike_empty").cast(pl.Float64).mean().alias("sim_rate"))
    )
    actual_rate = (
        actual_inventory.join(hour_map, on="interval_start", how="left")
        .group_by("station_id", "hour_of_week")
        .agg(pl.col("is_bike_empty").cast(pl.Float64).mean().alias("actual_rate"))
    )
    return actual_rate.join(sim_rate, on=["station_id", "hour_of_week"], how="left").with_columns(
        pl.col("sim_rate").fill_null(0.0)
    )


def stockout_rate_summary(rate_table: pl.DataFrame) -> dict:
    x = rate_table["actual_rate"].to_numpy().astype(float)
    y = rate_table["sim_rate"].to_numpy().astype(float)
    abs_diff = np.abs(x - y)
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {
        "n_cells": len(x),
        "corr": corr,
        "mean_abs_diff": float(abs_diff.mean()),
        "median_abs_diff": float(np.median(abs_diff)),
        "wmape": demand.compute_wmape(x, y),
    }


def hourly_profile_wmape(trip_log: pl.DataFrame, actual: pl.DataFrame) -> tuple[float, pl.DataFrame]:
    sim_hourly = (
        _completed_trips(trip_log)
        .with_columns(pl.col("interval_start").dt.hour().alias("hour"))
        .group_by("hour")
        .agg(pl.len().alias("sim_trips"))
    )
    actual_hourly = (
        actual.with_columns(pl.col("interval_start").dt.hour().alias("hour"))
        .group_by("hour")
        .agg(pl.col("departures").sum().alias("actual_trips"))
    )
    comparison = actual_hourly.join(sim_hourly, on="hour", how="left").with_columns(pl.col("sim_trips").fill_null(0)).sort("hour")
    wmape = demand.compute_wmape(
        comparison["actual_trips"].to_numpy().astype(float), comparison["sim_trips"].to_numpy().astype(float)
    )
    return wmape, comparison


def tier_usage_fractions(trip_log: pl.DataFrame) -> pl.DataFrame:
    """Authoritative "fraction of destination draws served by each tier"
    number (Phase 7 plan-mode discussion): computed from what the
    simulation actually drew, weighted by real simulated trip volume --
    not the unweighted per-cell count od_shares.py prints at build time."""
    total = trip_log.height
    counts = trip_log.group_by("tier").agg(pl.len().alias("n"))
    return counts.with_columns((pl.col("n") / total).alias("frac")).sort("frac", descending=True)


def reroute_outcome_breakdown(trip_log: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Three separate outcome counts, never collapsed (Phase 7 plan-mode
    discussion): direct / rerouted / lost_past_cap are different-cost
    outcomes. Distance/hop summary covers rerouted trips only -- direct and
    lost_past_cap have no meaningful distance."""
    outcomes = trip_log.group_by("outcome").agg(pl.len().alias("n")).with_columns(
        (pl.col("n") / trip_log.height).alias("frac")
    ).sort("n", descending=True)

    rerouted = trip_log.filter(pl.col("outcome") == "rerouted")
    if rerouted.height == 0:
        dist_summary = {"n": 0}
    else:
        dist_summary = {
            "n": rerouted.height,
            "mean_hops": float(rerouted["hops"].mean()),
            "mean_distance_m": float(rerouted["distance_m"].mean()),
            "p90_distance_m": float(rerouted["distance_m"].quantile(0.9)),
        }
    return outcomes, dist_summary


def build_gate_report(run: simulator.SimulationRun, week: simulator.WeekInputs, network: simulator.NetworkArrays, label: str) -> GateReport:
    zone_lookup = _zone_lookup(network)
    trips_err, sim_total, actual_total = total_trips_pct_error(run.trip_log, week.actual_departures)
    zone_wmape, zone_table = per_zone_wmape(run.trip_log, week.actual_departures, zone_lookup)
    stockout_corr, stockout_table = stockout_minutes_correlation(run.station_intervals, week.actual_arrivals_inventory)
    hourly_wmape, hourly_table = hourly_profile_wmape(run.trip_log, week.actual_departures)
    tier_usage = tier_usage_fractions(run.trip_log)
    reroute_outcomes, reroute_dist = reroute_outcome_breakdown(run.trip_log)

    print(f"\n[validate] === {label} ===")
    print(f"[validate] total trips: sim={sim_total:,} actual={actual_total:,} pct_error={trips_err:.2%} (gate: <= {TOTAL_TRIPS_GATE:.0%}) {'PASS' if trips_err <= TOTAL_TRIPS_GATE else 'FAIL'}")
    print(f"[validate] per-zone WMAPE={zone_wmape:.2%} (gate: <= {PER_ZONE_GATE:.0%}) {'PASS' if zone_wmape <= PER_ZONE_GATE else 'FAIL'}")
    print(
        f"[validate] per-interval stockout-minutes correlation={stockout_corr:.3f} "
        f"(reference only, not gated -- see stockout-rate-by-station-hour below for the Phase-8-relevant metric; "
        f"would read {'>=' if stockout_corr > STOCKOUT_CORR_REFERENCE else '<'} the old {STOCKOUT_CORR_REFERENCE} threshold)"
    )
    print(f"[validate] hourly system profile WMAPE={hourly_wmape:.2%} (diagnostic, not gated)")
    print(f"[validate] OD tier usage (volume-weighted, from actual simulated draws):\n{tier_usage}")
    print(f"[validate] reroute outcome breakdown:\n{reroute_outcomes}")
    print(f"[validate] rerouted-trip distance/hop summary: {reroute_dist}")

    return GateReport(
        label=label,
        total_trips_pct_error=trips_err,
        per_zone_wmape=zone_wmape,
        stockout_corr=stockout_corr,
        hourly_profile_wmape=hourly_wmape,
        tier_usage=tier_usage,
        reroute_outcomes=reroute_outcomes,
        reroute_distance_summary=reroute_dist,
    )


def print_clip_report(run: simulator.SimulationRun, week: simulator.WeekInputs, label: str) -> None:
    """The conservation check requested alongside the sanity_true_dest run:
    clipping to [0, capacity] after baseline-N replay isn't neutral, so
    report the MAGNITUDE (bikes created/destroyed), not just how often it
    happened, against a fleet-size proxy (sum of initial inventory at week
    start across the simulated network) for scale."""
    fleet_proxy = float(week.initial_inventory.sum())
    net = run.total_clip_created - run.total_clip_destroyed
    print(f"\n[validate] === {label}: clipping / conservation report ===")
    print(f"[validate] bound violations (station-interval times clipped): {run.total_n_bound_violations:,}")
    print(f"[validate] bikes created by clipping (negative -> 0):   {run.total_clip_created:>12,.0f}")
    print(f"[validate] bikes destroyed by clipping (over-cap -> cap): {run.total_clip_destroyed:>12,.0f}")
    print(f"[validate] net bikes added to the system by clipping:    {net:>+12,.0f}")
    print(f"[validate] fleet-size proxy (sum of initial inventory):  {fleet_proxy:>12,.0f}")
    print(f"[validate] net clipping as a fraction of fleet size:      {net / fleet_proxy:+.2%}")


def print_trajectory_diagnostics(run: simulator.SimulationRun, week: simulator.WeekInputs, label: str) -> None:
    seeded = set(week.half_cap_seeded_station_ids)
    print(f"\n[validate] === {label}: continuous-inventory trajectory diagnostics ===")
    for excl_label, excl_set in (("all stations", None), (f"excl. {len(seeded)} half-cap-seeded stations", seeded)):
        cont = continuous_inventory_diagnostics(run.station_intervals, week.actual_arrivals_inventory, excl_set)
        stockout_corr, _ = stockout_minutes_correlation(run.station_intervals, week.actual_arrivals_inventory, excl_set)
        print(
            f"[validate] {excl_label} (n={cont['n']:,}): continuous-inventory corr={cont['corr']:.3f}, "
            f"|diff| median={cont['median_abs_diff']:.2f} mean={cont['mean_abs_diff']:.2f} "
            f"p90={cont['p90_abs_diff']:.2f} p99={cont['p99_abs_diff']:.2f} max={cont['max_abs_diff']:.2f} bikes "
            f"|| binary stockout-minutes corr={stockout_corr:.3f}"
        )


def print_stockout_rate_report(run: simulator.SimulationRun, week: simulator.WeekInputs, label: str) -> None:
    rate_table = stockout_rate_by_station_hour(run.station_intervals, week.actual_arrivals_inventory, week.calendar_weather)
    summary = stockout_rate_summary(rate_table)
    print(f"\n[validate] === {label}: stockout RATE by station-hour-of-week (Phase 8's actual input) ===")
    print(
        f"[validate] {summary['n_cells']:,} (station, hour-of-week) cells: corr={summary['corr']:.3f}, "
        f"WMAPE={summary['wmape']:.2%}, mean|diff|={summary['mean_abs_diff']:.3f}, "
        f"median|diff|={summary['median_abs_diff']:.3f} (rate units, 0-1)"
    )


# Six consecutive Mondays inside demand_model's holdout window
# (2025-10-01 onward, config/params.yaml) with no major holiday in any of
# them (Thanksgiving is 2025-11-27, out of range) -- the revisit
# DECISIONS.md's Phase 7 entry flagged: does destination-assignment error
# average out at the (station, hour-of-week) resolution Phase 8 consumes,
# given enough weeks, or does correlation stay flat regardless of how much
# data is pooled? A single week can't test this (each cell gets exactly 4
# sub-observations, one per week); this can, since hour_of_week is a
# well-defined week-relative index that means the same thing (e.g. "Weds
# ~2am") across different calendar weeks, so pooling across weeks is a
# real increase in sample size per cell, not just more weeks side by side.
MULTIWEEK_STARTS = ["2025-10-06", "2025-10-13", "2025-10-20", "2025-10-27", "2025-11-03", "2025-11-10"]
MULTIWEEK_CHECKPOINTS = [1, 2, 4, 6]


def run_multiweek_stockout_rate_trend(
    week_starts: list[str],
    checkpoints: list[int],
    network: simulator.NetworkArrays,
    fitted_dep: demand.FittedDirection,
    od_model: od_shares.ODShareModel,
    sim_params: simulator.SimulationParams,
) -> pl.DataFrame:
    """Runs the STOCHASTIC simulator (the mode Phase 8/9 actually use) over
    each week in week_starts, same seed family throughout (sim_params.seed
    held fixed -- only the real week's underlying data changes, not the RNG
    seed), and correlates simulated vs. actual stockout RATE per (station,
    hour_of_week), POOLED across an increasing number of weeks at each
    checkpoint. Returns one row per checkpoint: n_weeks, n_cells, corr,
    wmape, mean/median abs diff -- if corr rises meaningfully with n_weeks,
    destination error averages out at this resolution (Phase 8's inputs are
    trustworthy); if it stays flat, it doesn't (Phase 8's MV(s,t) inputs are
    noisier than the single-week test suggested, and that needs to be known
    before src/opt/marginal_value.py gets built on top of them)."""
    per_week_cells = []
    for week_key in week_starts:
        t0 = time.monotonic()
        week_params = replace(sim_params, validation_week_start=week_key)
        week = simulator.prepare_week_inputs(network, week_params)
        run = simulator.run_simulation(network, week, fitted_dep, od_model, week_params, mode="stochastic")

        hour_map = week.calendar_weather.select("interval_start", "hour_of_week")
        sim_cell = (
            run.station_intervals.join(hour_map, on="interval_start", how="left")
            .group_by("station_id", "hour_of_week")
            .agg(pl.col("is_bike_empty").cast(pl.Int64).sum().alias("sim_empty"), pl.len().alias("sim_n"))
        )
        actual_cell = (
            week.actual_arrivals_inventory.join(hour_map, on="interval_start", how="left")
            .group_by("station_id", "hour_of_week")
            .agg(pl.col("is_bike_empty").cast(pl.Int64).sum().alias("actual_empty"), pl.len().alias("actual_n"))
        )
        cell = (
            actual_cell.join(sim_cell, on=["station_id", "hour_of_week"], how="left")
            .with_columns(pl.col("sim_empty").fill_null(0), pl.col("sim_n").fill_null(0))
            .with_columns(pl.lit(week_key).alias("week_key"))
        )
        per_week_cells.append(cell)
        print(
            f"[validate] multiweek: {week_key} done in {time.monotonic() - t0:.0f}s, "
            f"peak RSS {progress.peak_rss_mb():.0f} MB"
        )

    all_cells = pl.concat(per_week_cells, how="vertical")

    rows = []
    for n in checkpoints:
        if n > len(week_starts):
            continue
        subset = all_cells.filter(pl.col("week_key").is_in(week_starts[:n]))
        pooled = (
            subset.group_by("station_id", "hour_of_week")
            .agg(
                pl.col("sim_empty").sum().alias("sim_empty"),
                pl.col("sim_n").sum().alias("sim_n"),
                pl.col("actual_empty").sum().alias("actual_empty"),
                pl.col("actual_n").sum().alias("actual_n"),
            )
            .with_columns(
                (pl.col("sim_empty") / pl.col("sim_n")).alias("sim_rate"),
                (pl.col("actual_empty") / pl.col("actual_n")).alias("actual_rate"),
            )
        )
        x = pooled["actual_rate"].to_numpy().astype(float)
        y = pooled["sim_rate"].to_numpy().astype(float)
        abs_diff = np.abs(x - y)
        corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
        rows.append(
            {
                "n_weeks": n,
                "n_cells": pooled.height,
                "corr": corr,
                "wmape": demand.compute_wmape(x, y),
                "mean_abs_diff": float(abs_diff.mean()),
                "median_abs_diff": float(np.median(abs_diff)),
            }
        )
    return pl.DataFrame(rows)


def print_multiweek_stockout_rate_trend(trend: pl.DataFrame) -> None:
    print("\n[validate] === multiweek stockout-rate-by-station-hour trend (stochastic mode) ===")
    print(trend)
    corrs = trend["corr"].to_list()
    if len(corrs) >= 2 and corrs[-1] - corrs[0] > 0.15:
        print(
            f"[validate] correlation rose {corrs[0]:.3f} -> {corrs[-1]:.3f} across {trend['n_weeks'].to_list()} weeks -- "
            "meaningful improvement, consistent with destination error averaging out at this resolution"
        )
    elif len(corrs) >= 2:
        print(
            f"[validate] correlation stayed roughly flat ({corrs[0]:.3f} -> {corrs[-1]:.3f}) across "
            f"{trend['n_weeks'].to_list()} weeks -- NOT consistent with destination error averaging out; "
            "treat Phase 8's MV(s,t) inputs as noisier than the single-week test suggested"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 7 validation harness.")
    p.add_argument(
        "--multiweek", action="store_true",
        help="run the multiweek stockout-rate-by-station-hour trend (stochastic mode, MULTIWEEK_STARTS) "
        "instead of the standard single-week 3-mode validation",
    )
    args = p.parse_args()

    sim_params = simulator.load_simulation_params()
    network = simulator.load_station_network()
    fitted_dep = demand.load_fitted(demand.DEMAND_MODEL_DIR / "departures.pkl")
    od_model = od_shares.load_od_share_model()

    if args.multiweek:
        t0 = time.monotonic()
        trend = run_multiweek_stockout_rate_trend(
            MULTIWEEK_STARTS, MULTIWEEK_CHECKPOINTS, network, fitted_dep, od_model, sim_params
        )
        print_multiweek_stockout_rate_trend(trend)
        print(f"\n[validate] multiweek total wall time {time.monotonic() - t0:.0f}s, peak RSS {progress.peak_rss_mb():.0f} MB")
        return

    week = simulator.prepare_week_inputs(network, sim_params)
    t0 = time.monotonic()
    sanity_od_run = simulator.run_simulation(network, week, fitted_dep, od_model, sim_params, mode="sanity_od")
    sanity_od_report = build_gate_report(sanity_od_run, week, network, "sanity_od (forced actual departures, OD-share destinations)")
    print_clip_report(sanity_od_run, week, "sanity_od")
    print_trajectory_diagnostics(sanity_od_run, week, "sanity_od")
    print_stockout_rate_report(sanity_od_run, week, "sanity_od")

    true_dest_run = simulator.run_simulation(network, week, fitted_dep, od_model, sim_params, mode="sanity_true_dest")
    true_dest_report = build_gate_report(true_dest_run, week, network, "sanity_true_dest (forced actual departures AND true historical destinations)")
    print_clip_report(true_dest_run, week, "sanity_true_dest")
    print_trajectory_diagnostics(true_dest_run, week, "sanity_true_dest")
    print_stockout_rate_report(true_dest_run, week, "sanity_true_dest")

    stochastic_run = simulator.run_simulation(network, week, fitted_dep, od_model, sim_params, mode="stochastic")
    stochastic_report = build_gate_report(stochastic_run, week, network, "stochastic (sampled from fitted demand model)")
    print_clip_report(stochastic_run, week, "stochastic")
    print_trajectory_diagnostics(stochastic_run, week, "stochastic")
    print_stockout_rate_report(stochastic_run, week, "stochastic")

    print(f"\n[validate] === side by side ===")
    print(
        f"{'metric':<28} {'sanity_od':>15} {'sanity_true_dest':>18} {'stochastic':>15}\n"
        f"{'total trips % error':<28} {sanity_od_report.total_trips_pct_error:>14.2%} {true_dest_report.total_trips_pct_error:>17.2%} {stochastic_report.total_trips_pct_error:>14.2%}\n"
        f"{'per-zone WMAPE':<28} {sanity_od_report.per_zone_wmape:>14.2%} {true_dest_report.per_zone_wmape:>17.2%} {stochastic_report.per_zone_wmape:>14.2%}\n"
        f"{'stockout-minutes corr':<28} {sanity_od_report.stockout_corr:>15.3f} {true_dest_report.stockout_corr:>18.3f} {stochastic_report.stockout_corr:>15.3f}\n"
        f"{'hourly profile WMAPE':<28} {sanity_od_report.hourly_profile_wmape:>14.2%} {true_dest_report.hourly_profile_wmape:>17.2%} {stochastic_report.hourly_profile_wmape:>14.2%}\n"
        f"{'net clip / fleet size':<28} "
        f"{(sanity_od_run.total_clip_created - sanity_od_run.total_clip_destroyed) / week.initial_inventory.sum():>14.2%} "
        f"{(true_dest_run.total_clip_created - true_dest_run.total_clip_destroyed) / week.initial_inventory.sum():>17.2%} "
        f"{(stochastic_run.total_clip_created - stochastic_run.total_clip_destroyed) / week.initial_inventory.sum():>14.2%}"
    )
    print(f"\n[validate] total wall time {time.monotonic() - t0:.0f}s, peak RSS {progress.peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
