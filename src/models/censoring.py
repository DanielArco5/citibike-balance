"""Censoring-severity utilities for Phase 5 (SPEC.md §3, RUNBOOK Phase 5).

Three things live here, kept separate from demand.py per the same pattern
inventory.py uses (build vs. validate as separate, independently-runnable
concerns):

1. Run-length + bucketing: the severity proxy that stands in for SPEC §3's
   "stockout minutes / interval minutes" ratio. That ratio isn't computable
   from data/processed/inventory.parquet -- minutes_empty/minutes_full are a
   binary 0-or-15 re-encoding of is_bike_empty/is_dock_full (the LP only
   resolves state at 15-min grid points; see inventory.py's "Coarse
   per-bucket proxy" comment), not a continuous within-interval count. Run
   length -- how many consecutive 15-min intervals the current censored
   spell has lasted -- is the real, data-grounded substitute: a station
   empty for one isolated interval is a weaker censoring signal than one in
   the middle of a multi-hour drought, and that distinction IS resolvable at
   15-min grid granularity.

2. Capacity-staleness flag: DECISIONS.md's "329 stations" pool (capacity
   below the station's own observed single-15-minute-interval peak
   throughput) was left as narrative only in Phase 4, not a stored column.
   Reconstructed here from panel.parquet so Phase 5 can use it -- not to
   drop or reweight stations (DECISIONS.md explicitly rejected that: risks
   discarding legitimate high-turnover stations along with genuinely stale
   ones), but to mark arrival-side (dock-full) outputs as lower-confidence
   downstream. Departure-side (bike-empty) censoring isn't implicated by
   this failure mode -- is_bike_empty only requires capacity to be
   non-null/non-zero (already filtered in prepare_panel), not correctly
   *sized*, unlike is_dock_full which depends on capacity being an accurate
   upper bound.

3. Artificial-censoring recovery test (SPEC.md §3's validation, extended):
   synthetically censor known-uncensored intervals in contiguous runs (not
   scattered single points -- real censoring comes in runs, and the
   run-length bucket correction needs runs to be testable at all), then
   compare imputed unmet demand against the known true gap. Stratified by
   run-length bucket (does the bucket correction earn its complexity?) and
   by the capacity-staleness flag (is arrival-side correction worse there,
   empirically, not just per DECISIONS.md's narrative concern?).

   Two sampling schemes for step 3, both provided here:
   - make_synthetic_censoring_runs: draws run starts uniformly at random
     from uncensored intervals. Cheap, but the sample it produces looks
     like "typical" uncensored moments -- mid-demand, unremarkable hours --
     because that's what most uncensored intervals ARE.
   - make_synthetic_censoring_runs_matched: draws run starts so the
     resulting sample's (predicted-demand decile, hour-of-week, capacity
     band) distribution matches the GENUINELY-censored population's, not
     the uncensored pool's. This matters because real censoring is not
     randomly distributed across those conditions -- stations run empty
     because demand outran supply, so real stockouts concentrate in the
     upper predicted-demand tail. A model can recover held-out demand well
     in typical conditions (interpolation) while still being unreliable
     exactly where real censoring happens (extrapolation into a demand
     regime the uncensored training data barely samples). The unmatched
     test can't distinguish those two cases; the matched one is designed
     to. It also reports a coverage diagnostic: for target strata (e.g. the
     top demand deciles) where few or no uncensored exemplars exist to draw
     from, that's not a sampling bug to route around -- it's the honest
     finding that the extrapolation region has little to no local
     uncensored analog at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "config" / "params.yaml"


@dataclass
class CensoringParams:
    run_length_bucket_edges: list[int]


def load_params(path: Path = PARAMS_PATH) -> CensoringParams:
    cfg = yaml.safe_load(path.read_text())["demand_model"]
    return CensoringParams(run_length_bucket_edges=list(cfg["run_length_bucket_edges"]))


# ---------------------------------------------------------------------------
# Run length + bucketing
# ---------------------------------------------------------------------------


def add_run_length(
    df: pl.DataFrame,
    flag_col: str,
    run_col: str,
    station_col: str = "station_id",
    order_col: str = "interval_start",
) -> pl.DataFrame:
    """For each row, the length (in consecutive intervals) of the unbroken
    run of flag_col==True that the row falls within -- 0 for flag_col==False
    rows. Every row in a censored run gets the FULL run length, not its
    position within it: severity is about total spell duration, not how far
    into it a given interval sits.

    Sorts by (station_col, order_col) internally -- correctness depends on
    chronological order per station, and this shouldn't be a silent
    precondition on the caller."""
    df = df.sort(station_col, order_col)
    df = df.with_columns(
        (
            (pl.col(flag_col) != pl.col(flag_col).shift(1).over(station_col)).fill_null(True)
        )
        .cast(pl.Int32)
        .cum_sum()
        .alias("_run_id")
    )
    df = df.with_columns(pl.len().over("_run_id").alias("_run_len"))
    df = df.with_columns(
        pl.when(pl.col(flag_col)).then(pl.col("_run_len")).otherwise(0).alias(run_col)
    )
    return df.drop(["_run_id", "_run_len"])


def bucket_run_length(run_length: pl.Expr, edges: Sequence[int]) -> pl.Expr:
    """Maps a run-length expression to an integer bucket index via
    upper-inclusive edges, e.g. edges=[1,4] -> 0 (<=1), 1 (2-4), 2 (5+).
    Callers should mask run_length==0 (uncensored) to null themselves --
    this function doesn't special-case it, since "bucket of an uncensored
    row" isn't a meaningful question and different callers may want
    different null-handling."""
    edges = sorted(edges)
    expr = pl.when(run_length <= edges[0]).then(pl.lit(0, dtype=pl.Int32))
    for i, e in enumerate(edges[1:], start=1):
        expr = expr.when(run_length <= e).then(pl.lit(i, dtype=pl.Int32))
    return expr.otherwise(pl.lit(len(edges), dtype=pl.Int32))


# ---------------------------------------------------------------------------
# Capacity staleness flag (DECISIONS.md's 329-station pool)
# ---------------------------------------------------------------------------


def compute_capacity_staleness(usable_panel: pl.DataFrame) -> pl.DataFrame:
    """usable_panel must already be filtered to the same "usable" population
    DECISIONS.md's 329-station figure refers to -- pass the output of
    models.inventory.prepare_panel(), not the raw panel, so null-capacity
    and capacity==0-despite-activity stations (already handled there) aren't
    double-counted here under a different name.

    Definition, verbatim from DECISIONS.md: capacity below the station's own
    observed single-15-minute-interval peak throughput (departures +
    arrivals in one interval). Returns one row per station:
    (station_id, peak_throughput, capacity, is_capacity_stale)."""
    return (
        usable_panel.with_columns((pl.col("departures") + pl.col("arrivals")).alias("_throughput"))
        .group_by("station_id")
        .agg(
            pl.col("_throughput").max().alias("peak_throughput"),
            pl.col("capacity").first().alias("capacity"),
        )
        .with_columns((pl.col("capacity") < pl.col("peak_throughput")).alias("is_capacity_stale"))
    )


# ---------------------------------------------------------------------------
# Artificial-censoring recovery test
# ---------------------------------------------------------------------------


def make_synthetic_censoring_runs(
    df: pl.DataFrame,
    value_col: str,
    run_lengths: Sequence[int],
    n_runs_per_length: int,
    station_col: str = "station_id",
    order_col: str = "interval_start",
    cap_quantile: float = 0.1,
    rng_seed: int = 0,
) -> pl.DataFrame:
    """Synthetically censors contiguous runs of otherwise-uncensored rows.
    df must contain ONLY already-uncensored intervals (true D == value_col,
    the whole point of the test), sorted or sortable by
    (station_col, order_col).

    For each requested run length, picks n_runs_per_length random
    (station, start-position) pairs with enough contiguous room, and caps
    value_col within that run at that station's cap_quantile of value_col
    (simulating "as if the station had a low bike/dock ceiling C for this
    window", so observed Y' = min(true_D, C)) -- low enough that most runs
    produce genuine censoring (true_D > C) without hand-tuning per row.

    Returns the selected rows only, with columns: station_col, order_col,
    true_D (original value_col), censored_Y (post-cap), synthetic_run_length,
    is_capacity_stale is NOT joined here -- callers join it separately so
    this function stays agnostic to which direction (departures/arrivals)
    is being tested."""
    df = df.sort(station_col, order_col)
    rng = np.random.default_rng(rng_seed)

    station_caps = df.group_by(station_col, maintain_order=True).agg(
        pl.col(value_col).quantile(cap_quantile).alias("_cap"), pl.len().alias("_n")
    )
    df = df.join(station_caps, on=station_col)

    # Single partition_by pass, not a filter per (station, run_length) pair --
    # with ~2,270 stations and several run lengths, repeated whole-dataframe
    # filters (checked directly: ~110s on a 60-day/~12M-row subset) don't
    # scale to a full year. station_n reuses the groupby's row counts instead
    # of filtering again just to check eligibility.
    station_frames = {f[station_col][0]: f for f in df.partition_by(station_col, maintain_order=True)}
    station_n = dict(zip(station_caps[station_col].to_list(), station_caps["_n"].to_list()))

    selected_frames = []
    for run_length in run_lengths:
        eligible = [s for s, n in station_n.items() if n > run_length]
        if not eligible:
            continue
        chosen_stations = rng.choice(eligible, size=min(n_runs_per_length, len(eligible)), replace=False)
        for s in chosen_stations:
            sdf = station_frames[s]
            max_start = sdf.height - run_length
            start = int(rng.integers(0, max_start + 1))
            run = sdf.slice(start, run_length)
            selected_frames.append(
                run.with_columns(
                    pl.col(value_col).alias("true_D"),
                    pl.min_horizontal(pl.col(value_col), pl.col("_cap")).alias("censored_Y"),
                    pl.lit(run_length, dtype=pl.Int32).alias("synthetic_run_length"),
                )
            )

    out = pl.concat(selected_frames).select(
        station_col, order_col, "true_D", "censored_Y", "synthetic_run_length"
    )
    return out


def bucket_edges_for(params: CensoringParams) -> list[int]:
    return params.run_length_bucket_edges


def evaluate_recovery(
    censored: pl.DataFrame,
    imputed_unmet: pl.Series,
    edges: Sequence[int],
    stale_flags: pl.DataFrame | None = None,
    station_col: str = "station_id",
) -> pl.DataFrame:
    """Compares imputed unmet demand against the known true gap
    (true_D - censored_Y) on the output of make_synthetic_censoring_runs.
    Reports MAE and bias overall, by run-length bucket, and (if stale_flags
    is given -- station_col, is_capacity_stale) by capacity-staleness flag.
    stale_flags should only be passed for the arrival-side/dock-full test --
    departure-side censoring isn't implicated by capacity staleness (see
    module docstring)."""
    df = censored.with_columns(
        imputed_unmet.alias("imputed_unmet"),
        (pl.col("true_D") - pl.col("censored_Y")).alias("true_unmet"),
    ).with_columns(
        (pl.col("imputed_unmet") - pl.col("true_unmet")).alias("error"),
        bucket_run_length(pl.col("synthetic_run_length"), edges).alias("run_length_bucket"),
    )

    if stale_flags is not None:
        df = df.join(stale_flags.select(station_col, "is_capacity_stale"), on=station_col, how="left")

    aggs = (
        pl.col("error").abs().mean().alias("mae"),
        pl.col("error").mean().alias("bias"),
        pl.len().alias("n"),
    )

    def _summarize(frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
        if not by:
            return frame.select(*aggs)
        return frame.group_by(by).agg(*aggs).sort(by)

    overall = _summarize(df, []).with_columns(pl.lit("overall").alias("stratum"))
    by_bucket = _summarize(df, ["run_length_bucket"]).with_columns(pl.lit("run_length_bucket").alias("stratum"))
    parts = [overall, by_bucket]
    if stale_flags is not None:
        by_stale = _summarize(df, ["is_capacity_stale"]).with_columns(pl.lit("capacity_stale").alias("stratum"))
        parts.append(by_stale)
    return pl.concat(parts, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# Matched (demand-decile / hour-of-week / capacity-band) synthetic censoring
# ---------------------------------------------------------------------------


def compute_stratum_edges(
    target_df: pl.DataFrame,
    d_hat_col: str,
    capacity_col: str,
    n_deciles: int = 10,
    n_capacity_bands: int = 4,
) -> dict:
    """Cut points derived from the TARGET population (genuinely-censored
    rows), not the candidate pool. The point of matching is to describe the
    target's own distribution and ask how much of it the uncensored pool can
    cover -- deriving edges from the pool instead would just describe the
    pool's (very different, by construction) distribution and defeat the
    purpose."""
    d_hat = target_df[d_hat_col].to_numpy()
    capacity = target_df[capacity_col].to_numpy()
    return {
        "d_hat_edges": np.quantile(d_hat, np.linspace(0, 1, n_deciles + 1)[1:-1]),
        "capacity_edges": np.quantile(capacity, np.linspace(0, 1, n_capacity_bands + 1)[1:-1]),
    }


def assign_strata(
    df: pl.DataFrame,
    d_hat_col: str,
    capacity_col: str,
    edges: dict,
    hour_col: str = "hour_of_week",
) -> pl.DataFrame:
    """Adds demand_decile, capacity_band (both via the TARGET-derived edges,
    so a given row means the same bin in both the target and pool
    populations), and a combined stratum_key (exact match on hour_of_week --
    it's already a 168-level discrete column, no binning needed)."""
    demand_decile = np.searchsorted(edges["d_hat_edges"], df[d_hat_col].to_numpy(), side="right")
    capacity_band = np.searchsorted(edges["capacity_edges"], df[capacity_col].to_numpy(), side="right")
    df = df.with_columns(
        pl.Series("demand_decile", demand_decile.astype(np.int32)),
        pl.Series("capacity_band", capacity_band.astype(np.int32)),
    )
    return df.with_columns(
        pl.concat_str(
            [pl.col("demand_decile").cast(pl.Utf8), pl.col("capacity_band").cast(pl.Utf8), pl.col(hour_col).cast(pl.Utf8)],
            separator="_",
        ).alias("stratum_key")
    )


def stratum_proportions(df: pl.DataFrame, stratum_col: str = "stratum_key") -> pl.DataFrame:
    total = df.height
    return df.group_by(stratum_col).agg(pl.len().alias("target_n")).with_columns(
        (pl.col("target_n") / total).alias("target_prop")
    )


def make_synthetic_censoring_runs_matched(
    candidate_pool: pl.DataFrame,
    target_population: pl.DataFrame,
    value_col: str,
    d_hat_col: str,
    run_lengths: Sequence[int],
    n_runs_per_length: int,
    station_col: str = "station_id",
    order_col: str = "interval_start",
    hour_col: str = "hour_of_week",
    capacity_col: str = "capacity",
    n_deciles: int = 10,
    n_capacity_bands: int = 4,
    cap_quantile: float = 0.1,
    max_pool_rows: int = 8_000_000,
    rng_seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Like make_synthetic_censoring_runs, but run starts are drawn so the
    sample's (demand_decile, hour_of_week, capacity_band) composition
    matches target_population's -- not drawn uniformly from candidate_pool.
    Both frames must already carry d_hat_col (this function doesn't call the
    model itself, so callers control exactly which model/features produced
    it) and capacity_col/hour_col.

    Per run length, for each target stratum with target_prop p: desired =
    round(p * n_runs_per_length); if fewer than `desired` eligible
    candidates exist in that stratum (eligible = uncensored rows with room
    for a run of this length ahead of them), all available candidates are
    taken and the shortfall is recorded in the coverage report rather than
    silently backfilled from a different stratum -- backfilling would
    quietly re-introduce the interpolation-vs-extrapolation confound this
    function exists to remove.

    Two things done for memory, both checked directly against a real
    73M-row full-year uncensored pool that OOM-killed (exit 137) a first
    version without them, on a 16GB machine:
    - Only the columns this function actually needs are kept (candidate_pool
      typically arrives as a much wider feature frame). Holding station_id/
      order_col/value_col/d_hat_col/hour_col/capacity_col only, instead of
      ~25 unused feature columns, is roughly a 4-5x reduction on its own.
    - candidate_pool is capped at max_pool_rows via uniform random sample.
      We only ever need up to a few thousand exemplars per stratum; a
      multi-million-row pool is already far more supply than that, so
      capping it costs essentially nothing statistically while bounding
      memory regardless of how large the full uncensored population is.

    Returns (synthetic rows -- same schema as make_synthetic_censoring_runs,
    plus matched_stratum_key -- coverage report: one row per
    (run_length, stratum_key) with target_prop/desired/available/taken)."""
    needed_cols = [station_col, order_col, value_col, d_hat_col, hour_col, capacity_col]
    candidate_pool = candidate_pool.select(needed_cols)
    target_population = target_population.select(needed_cols)
    if candidate_pool.height > max_pool_rows:
        candidate_pool = candidate_pool.sample(n=max_pool_rows, seed=rng_seed)

    edges = compute_stratum_edges(target_population, d_hat_col, capacity_col, n_deciles, n_capacity_bands)
    target = assign_strata(target_population, d_hat_col, capacity_col, edges, hour_col)
    pool = assign_strata(candidate_pool, d_hat_col, capacity_col, edges, hour_col)
    target_props = stratum_proportions(target)

    pool = pool.sort(station_col, order_col)
    station_caps = pool.group_by(station_col, maintain_order=True).agg(
        pl.col(value_col).quantile(cap_quantile).alias("_cap"), pl.len().alias("_station_n")
    )
    pool = pool.join(station_caps, on=station_col)
    pool = pool.with_columns(pl.int_range(pl.len()).over(station_col).alias("_pos"))
    station_frames = {f[station_col][0]: f for f in pool.partition_by(station_col, maintain_order=True)}

    rng = np.random.default_rng(rng_seed)
    target_rows = target_props.iter_rows(named=True)
    target_rows = [r for r in target_rows if r["target_prop"] > 0]

    # Single eligibility pass at the LARGEST run length, not one pass per run
    # length: a position valid there (p <= height - max(run_lengths)) is
    # automatically valid for every SHORTER run length too (the constraint
    # only gets looser), so this one filter + group_by safely covers every
    # run length in the list. Loses at most max(run_lengths) trailing
    # positions per station for the shorter run lengths (a handful of rows
    # out of a station's full-year history) -- negligible, and the
    # 7x-repeated group_by this replaces was the dominant cost in the OOM'd
    # run.
    max_run_length = max(run_lengths)
    eligible = pool.filter(pl.col("_pos") <= (pl.col("_station_n") - max_run_length))
    by_stratum = {k[0] if isinstance(k, tuple) else k: g for k, g in eligible.group_by("stratum_key")}

    selected_frames = []
    coverage_rows = []

    for run_length in run_lengths:
        for row in target_rows:
            stratum = row["stratum_key"]
            desired = int(round(row["target_prop"] * n_runs_per_length))
            candidates = by_stratum.get(stratum)
            available = 0 if candidates is None else candidates.height
            take = min(desired, available)
            coverage_rows.append(
                {
                    "run_length": run_length,
                    "stratum_key": stratum,
                    "target_prop": row["target_prop"],
                    "desired": desired,
                    "available": available,
                    "taken": take,
                }
            )
            if take == 0:
                continue
            chosen = candidates.sample(n=take, with_replacement=False, seed=int(rng.integers(0, 2**31 - 1)))
            for r in chosen.iter_rows(named=True):
                s = r[station_col]
                start = r["_pos"]
                run = station_frames[s].slice(start, run_length)
                selected_frames.append(
                    run.with_columns(
                        pl.col(value_col).alias("true_D"),
                        pl.min_horizontal(pl.col(value_col), pl.col("_cap")).alias("censored_Y"),
                        pl.lit(run_length, dtype=pl.Int32).alias("synthetic_run_length"),
                        pl.lit(stratum, dtype=pl.Utf8).alias("matched_stratum_key"),
                    )
                )

    coverage = pl.DataFrame(coverage_rows) if coverage_rows else pl.DataFrame(
        schema={"run_length": pl.Int64, "stratum_key": pl.Utf8, "target_prop": pl.Float64, "desired": pl.Int64, "available": pl.Int64, "taken": pl.Int64}
    )
    if not selected_frames:
        return (
            pl.DataFrame(
                schema={
                    station_col: pool.schema[station_col],
                    order_col: pool.schema[order_col],
                    "true_D": pl.Float64,
                    "censored_Y": pl.Float64,
                    "synthetic_run_length": pl.Int32,
                    "matched_stratum_key": pl.Utf8,
                }
            ),
            coverage,
        )
    out = pl.concat(selected_frames, how="diagonal_relaxed").select(
        station_col, order_col, "true_D", "censored_Y", "synthetic_run_length", "matched_stratum_key"
    )
    return out, coverage


def summarize_coverage_by_decile(coverage: pl.DataFrame) -> pl.DataFrame:
    """Collapses the matched-sampling coverage diagnostic onto the
    demand-decile axis (dropping capacity_band/hour_of_week detail) -- the
    dimension most likely to expose the extrapolation gap: high predicted
    demand is exactly where genuine censoring happens AND where uncensored
    exemplars are scarcest, by construction. fill_rate well below 1.0 in the
    top deciles means the matched sample couldn't actually be filled there
    -- report that gap, don't paper over it.

    target_mass and desired_n/taken_n are aggregated differently on purpose:
    coverage has one row per (run_length, stratum_key) -- target_prop is the
    SAME value repeated for every run_length (it describes the target
    population, not a specific run length), so it's de-duplicated per
    stratum before summing across deciles. desired_n/taken_n are per-draw
    COUNTS that are genuinely additive across run lengths (total draws
    attempted/achieved for this decile, across every run length tried), so
    those sum directly over all rows."""
    if coverage.height == 0:
        return coverage
    with_decile = coverage.with_columns(
        pl.col("stratum_key").str.split("_").list.get(0).cast(pl.Int32).alias("demand_decile")
    )
    target_mass = (
        with_decile.select("stratum_key", "demand_decile", "target_prop")
        .unique()
        .group_by("demand_decile")
        .agg(pl.col("target_prop").sum().alias("target_mass"))
    )
    draws = with_decile.group_by("demand_decile").agg(
        pl.col("desired").sum().alias("desired_n"), pl.col("taken").sum().alias("taken_n")
    )
    return (
        target_mass.join(draws, on="demand_decile")
        .with_columns(
            pl.when(pl.col("desired_n") > 0)
            .then(pl.col("taken_n") / pl.col("desired_n"))
            .otherwise(None)
            .alias("fill_rate")
        )
        .sort("demand_decile")
    )
