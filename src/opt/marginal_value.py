"""Phase 8 Part A: MV(s,t,k) = expected trips saved by having the k-th bike
at station s, hour-of-week t (SPEC.md §7, RUNBOOK Phase 8).

Per DECISIONS.md's Phase 7 follow-up entry ("forward simulator -- stockout-
timing gate restated"): the forward simulator (src/sim/) cannot supply
P(stockout | s, t) at station-hour resolution, in any run mode, at any
amount of week-pooling tried (pooled correlation plateaus around 0.10
across 1/2/4/6 held-out weeks). MV(s,t) does NOT need destination
information at all -- it only needs "how often is this station empty" and
"how much demand exists when it is," both station-local marginal
quantities -- so this module derives it EMPIRICALLY from Phase 4-6 outputs
instead: `data/processed/unmet_demand_net.parquet` (demand.py's censored-
demand D_hat + substitution.py's neighbor-netted lost-demand estimate) and
`data/processed/inventory.parquet` (Phase 4's LP-reconstructed inventory
and inferred non-trip movement N). No simulator involved.

**Model history -- second attempt, kept honest in the docstring rather
than silently replaced (DECISIONS.md has the full account).** The first
formulation modeled each station-hour as a STATIONARY M/M/1/K queue and
treated "adding the n-th bike" as a capacity-scaling proxy. Checked, not
assumed: median relaxation time across the real network came back ~9,249
minutes (~6.4 days) -- because active rebalancing (Phase 4's inferred N)
pushes ~63% of station-hours to rho=lambda/mu~1 by design (successful
rebalancing IS operating near critical balance), and most station-hours
are low-traffic enough at 15-min granularity that mixing is slow even away
from rho=1. A station-hour that never reaches its stationary distribution
within any policy-relevant window can't be usefully described by that
distribution. `mm1k_p0` and `relaxation_time_minutes` are kept (still
correct, still tested) purely as the retained record of that finding.

**Current model: transient first-passage, not stationary equilibrium.**
The operational question was always "P(this station runs out within the
next hour | it currently has k bikes)," not "what fraction of all time
does it spend at zero." Same birth-death chain (bikes in = birth, bikes
out = death, LATENT demand-model rates -- see below), but instead of its
long-run equilibrium, this computes P(hit 0 within `n_steps` 15-minute
steps | start at k) via the standard first-passage trick: make state 0
absorbing, raise the one-step transition matrix to the n_steps power, and
read off the probability mass that ended up at 0. `hitting_probabilities`
does this exactly (a small matrix power -- station capacities are modest,
so this is cheap per cell even across ~380K station-hour cells).
MV(s,t,k) = [P(hit empty | k) - P(hit empty | k+1)] x E[net_lost |
stockout], where E[net_lost | stockout] is the empirical (Phase 6) average
TOTAL net-lost demand across an hour, conditional on that hour having had
at least one stockout sub-interval -- an hour-granularity quantity to
match hit(k), not the per-15-min-interval rate `mv_empirical_baseline`
uses.

Two methods, cross-checked against each other rather than either one
trusted blindly (same pattern as the LP-vs-greedy and DOT cross-checks
elsewhere in this project):

1. **Transient first-passage (the model half), described above.** Per
   this phase's plan-mode discussion: feed it LATENT demand-model rates,
   not observed arrival/departure counts -- observed arrivals are
   dock-full-censored the same way observed departures are empty-censored,
   and using them would understate stockout risk at exactly the busiest,
   most policy-relevant stations (the ones already running hot enough to
   censor their own arrival counts).

2. **Local-sensitivity cross-check (the empirical half).** Phase 4's LP
   reconstructs REAL inventory per station per Monday-aligned week, so the
   SAME station-hour genuinely had different starting-inventory levels in
   different real calendar weeks (rebalancing schedule variation, prior-
   week demand shocks, etc.) -- not a designed experiment, but real
   variation nonetheless. Regressing observed hour-total net-lost demand
   against observed starting inventory across those ~52 weekly
   observations gives an EMPIRICAL marginal-value estimate with no
   queueing model involved. This regression IS already a first-passage-
   shaped quantity (conditional on a real starting level, over a real
   hour) -- under the earlier stationary model these were mismatched
   quantities (conditional-on-start vs. unconditional-long-run), which is
   why that cross-check read near zero; under this model they measure the
   same thing, so agreement here is informative. Reported per
   (station_id, hour_of_week), not collapsed into one aggregate number
   that would hide exactly the cells where the two disagree.

**Concavity is checked, not assumed, and unlike the stationary blocking
probability it is NOT guaranteed by construction here** -- transient
hitting probabilities for a finite-time window don't have the same
provable convexity-in-capacity property the stationary M/M/1/K blocking
probability does. `check_concavity` reports where it fails, if it does.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.stats import skellam

import models.od_shares as od_shares
import utils.checkpoint as checkpoint
import utils.progress as progress

REPO_ROOT = Path(__file__).resolve().parents[2]
UNMET_DEMAND_NET_PATH = REPO_ROOT / "data" / "processed" / "unmet_demand_net.parquet"
INVENTORY_PATH = REPO_ROOT / "data" / "processed" / "inventory.parquet"
PARAMS_PATH = REPO_ROOT / "config" / "params.yaml"

OUT_DIR = REPO_ROOT / "data" / "processed" / "marginal_value"
STATION_HOUR_WEEK_PARTS_DIR = OUT_DIR / "station_hour_week_parts"
STATION_HOUR_WEEK_PATH = OUT_DIR / "station_hour_week.parquet"
STATION_HOUR_PATH = OUT_DIR / "station_hour.parquet"
MV_CURVE_PATH = OUT_DIR / "mv_curve.parquet"
CROSS_CHECK_PATH = OUT_DIR / "cross_check.parquet"
ELIGIBLE_CELLS_PATH = OUT_DIR / "eligible_cells.parquet"
LOW_FREQUENCY_PATH = OUT_DIR / "low_frequency.parquet"
CHRONIC_TIMING_PATH = OUT_DIR / "chronic_timing.parquet"

INTERVAL_MINUTES = 15.0


@dataclass
class MarginalValueParams:
    min_weeks_for_cross_check: int
    relaxation_time_flag_minutes: float
    first_passage_window_minutes: float
    mv_k_max: int
    schedulable_modal_share_threshold: float


def load_params(path: Path = PARAMS_PATH) -> MarginalValueParams:
    cfg = yaml.safe_load(path.read_text())["marginal_value"]
    return MarginalValueParams(
        min_weeks_for_cross_check=int(cfg["min_weeks_for_cross_check"]),
        relaxation_time_flag_minutes=float(cfg["relaxation_time_flag_minutes"]),
        first_passage_window_minutes=float(cfg["first_passage_window_minutes"]),
        mv_k_max=int(cfg["mv_k_max"]),
        schedulable_modal_share_threshold=float(cfg["schedulable_modal_share_threshold"]),
    )


def first_passage_steps(params: MarginalValueParams) -> int:
    return int(round(params.first_passage_window_minutes / INTERVAL_MINUTES))


# ---------------------------------------------------------------------------
# Stage 1: (station, hour-of-week, week) partial sums, month-chunked
# ---------------------------------------------------------------------------


def all_month_keys() -> list[str]:
    keys = (
        pl.scan_parquet(UNMET_DEMAND_NET_PATH)
        .select(checkpoint.month_key_expr().alias("month_key"))
        .unique()
        .sort("month_key")
        .collect()
    )
    return keys["month_key"].to_list()


def build_month_station_hour_week_partial(month_key: str) -> pl.DataFrame:
    """Partial SUMS (not means) per (station_id, hour_of_week, week_start)
    for this month's rows only. Sums, not means, because calendar weeks
    (`interval_start.dt.truncate("1w")`) don't align to calendar-month
    boundaries -- a week straddling two months would get silently split
    and under-counted if this returned per-month means directly.
    finalize_station_hour_week() re-sums these partials across ALL months
    before dividing, so a split week is combined correctly regardless of
    which month-chunk each of its rows landed in (same pattern as
    demand.py's compute_encodings_and_sample / _finalize_encoding)."""
    start, end = od_shares.month_bounds(month_key)
    unmet_m = (
        pl.scan_parquet(UNMET_DEMAND_NET_PATH)
        .filter(pl.col("interval_start").is_between(start, end, closed="left"))
        .select(
            "station_id", "interval_start", "hour_of_week", "capacity", "zone_agg",
            "is_bike_empty", "dep_D_hat", "arr_D_hat", "dep_net_lost",
        )
        .collect()
    )
    inv_m = (
        pl.scan_parquet(INVENTORY_PATH)
        .filter(pl.col("interval_start").is_between(start, end, closed="left"))
        .select("station_id", "interval_start", "inventory", "inferred_nontrip_in", "inferred_nontrip_out")
        .collect()
    )
    df = unmet_m.join(inv_m, on=["station_id", "interval_start"], how="inner")
    df = df.with_columns(pl.col("interval_start").dt.truncate("1w").alias("week_start"))

    return df.group_by("station_id", "hour_of_week", "week_start").agg(
        pl.col("capacity").first().alias("capacity"),
        pl.col("zone_agg").first().alias("zone_agg"),
        pl.col("is_bike_empty").cast(pl.Int64).sum().alias("_stockout_sum"),
        pl.col("dep_D_hat").sum().alias("_dep_dhat_sum"),
        pl.col("arr_D_hat").sum().alias("_arr_dhat_sum"),
        pl.col("dep_net_lost").sum().alias("_dep_net_lost_sum"),
        pl.col("inventory").sum().alias("_inventory_sum"),
        pl.col("inferred_nontrip_in").sum().alias("_nontrip_in_sum"),
        pl.col("inferred_nontrip_out").sum().alias("_nontrip_out_sum"),
        pl.len().alias("n_intervals"),
    )


def run_stage1(force: bool = False) -> None:
    months = all_month_keys()
    for month_key in months:
        if checkpoint.is_checkpointed(STATION_HOUR_WEEK_PARTS_DIR, month_key) and not force:
            print(f"[marginal_value] stage1 {month_key}: checkpoint exists, skipping")
            continue
        t0 = time.monotonic()
        part = build_month_station_hour_week_partial(month_key)
        checkpoint.write_checkpoint(part, checkpoint.checkpoint_path(STATION_HOUR_WEEK_PARTS_DIR, month_key))
        progress.log_month(month_key, part.height, time.monotonic() - t0, extra="marginal_value stage1")


def finalize_station_hour_week(force: bool = False) -> pl.DataFrame:
    """Combines Stage 1's per-month partial sums into one row per real
    (station_id, hour_of_week, week_start) -- re-summing across months
    first (correct even for weeks split across a month boundary), then
    dividing once to get the week's true means."""
    if STATION_HOUR_WEEK_PATH.exists() and not force:
        print(f"[marginal_value] stage1: reusing cached {STATION_HOUR_WEEK_PATH}")
        return pl.read_parquet(STATION_HOUR_WEEK_PATH)

    combined = (
        pl.scan_parquet(STATION_HOUR_WEEK_PARTS_DIR / "*.parquet")
        .group_by("station_id", "hour_of_week", "week_start")
        .agg(
            pl.col("capacity").first(),
            pl.col("zone_agg").first(),
            pl.col("_stockout_sum").sum(),
            pl.col("_dep_dhat_sum").sum(),
            pl.col("_arr_dhat_sum").sum(),
            pl.col("_dep_net_lost_sum").sum(),
            pl.col("_inventory_sum").sum(),
            pl.col("_nontrip_in_sum").sum(),
            pl.col("_nontrip_out_sum").sum(),
            pl.col("n_intervals").sum(),
        )
        .collect()
    )
    n = pl.col("n_intervals")
    weekly = combined.with_columns(
        (pl.col("_stockout_sum") / n).alias("p_stockout"),
        (pl.col("_dep_dhat_sum") / n).alias("mean_dep_D_hat"),
        (pl.col("_arr_dhat_sum") / n).alias("mean_arr_D_hat"),
        (pl.col("_dep_net_lost_sum") / n).alias("mean_dep_net_lost"),
        (pl.col("_inventory_sum") / n).alias("mean_inventory"),
        (pl.col("_nontrip_in_sum") / n).alias("mean_nontrip_in"),
        (pl.col("_nontrip_out_sum") / n).alias("mean_nontrip_out"),
    ).select(
        "station_id", "hour_of_week", "week_start", "capacity", "zone_agg",
        "p_stockout", "mean_dep_D_hat", "mean_arr_D_hat", "mean_dep_net_lost",
        "mean_inventory", "mean_nontrip_in", "mean_nontrip_out", "n_intervals",
    )
    checkpoint.write_checkpoint(weekly, STATION_HOUR_WEEK_PATH)
    print(f"[marginal_value] stage1: {weekly.height:,} (station, hour-of-week, week) rows -> {STATION_HOUR_WEEK_PATH}")
    return weekly


# ---------------------------------------------------------------------------
# Stage 2: yearly-averaged (station, hour-of-week) rates
# ---------------------------------------------------------------------------


def aggregate_station_hour(weekly: pl.DataFrame) -> pl.DataFrame:
    """Yearly rates per (station, hour_of_week), weighted by n_intervals so
    a station's partial first/last calendar week doesn't get equal weight
    to a full week. `lam`/`mu` are the first-passage model's inputs --
    LATENT demand-model rates plus inferred non-trip movement, per this
    phase's constraint that observed (censored) counts must not feed the
    model directly."""
    w = weekly.with_columns(
        (pl.col("mean_dep_D_hat") * pl.col("n_intervals")).alias("_wdep"),
        (pl.col("mean_arr_D_hat") * pl.col("n_intervals")).alias("_warr"),
        (pl.col("mean_nontrip_in") * pl.col("n_intervals")).alias("_win"),
        (pl.col("mean_nontrip_out") * pl.col("n_intervals")).alias("_wout"),
        (pl.col("p_stockout") * pl.col("n_intervals")).alias("_wstockout"),
        (pl.col("mean_dep_net_lost") * pl.col("n_intervals")).alias("_wnetlost"),
    )
    agg = w.group_by("station_id", "hour_of_week").agg(
        pl.col("capacity").first(),
        pl.col("zone_agg").first(),
        pl.col("n_intervals").sum().alias("n_intervals"),
        pl.len().alias("n_weeks"),
        pl.col("_wdep").sum(),
        pl.col("_warr").sum(),
        pl.col("_win").sum(),
        pl.col("_wout").sum(),
        pl.col("_wstockout").sum(),
        pl.col("_wnetlost").sum(),
    )
    n = pl.col("n_intervals")
    agg = agg.with_columns(
        (pl.col("_wdep") / n).alias("dep_rate"),
        (pl.col("_warr") / n).alias("arr_rate"),
        (pl.col("_win") / n).alias("nontrip_in_rate"),
        (pl.col("_wout") / n).alias("nontrip_out_rate"),
        (pl.col("_wstockout") / n).alias("p_stockout_empirical"),
        (pl.col("_wnetlost") / n).alias("mv_empirical_baseline"),
    )
    return agg.with_columns(
        (pl.col("dep_rate") + pl.col("nontrip_out_rate")).alias("mu"),
        (pl.col("arr_rate") + pl.col("nontrip_in_rate")).alias("lam"),
    ).select(
        "station_id", "hour_of_week", "capacity", "zone_agg", "n_weeks", "n_intervals",
        "lam", "mu", "p_stockout_empirical", "mv_empirical_baseline",
        # Component rates, exposed (not just folded into lam/mu) for
        # rebalancing_vs_chronicity_report -- comparing Phase 4's inferred
        # non-trip inflow against organic trip demand needs them separately,
        # not combined.
        "dep_rate", "arr_rate", "nontrip_in_rate", "nontrip_out_rate",
    )


def hourly_net_lost_given_stockout(weekly: pl.DataFrame) -> pl.DataFrame:
    """E[total net-lost demand across the hour | at least one 15-min
    sub-interval within it was a stockout], per (station_id,
    hour_of_week) -- averaged across the real hour-instances (one row per
    real week in `weekly`) where a stockout occurred. This is at HOUR
    granularity to match hit(k) (a per-window probability), unlike
    aggregate_station_hour's mv_empirical_baseline, which is a per-15-min-
    interval rate. mean_dep_net_lost * n_intervals recovers that
    hour-instance's TOTAL net-lost demand from its already-computed mean
    (n_intervals is normally 4; fewer only for a station's partial first/
    last calendar week)."""
    stockout_hours = weekly.filter(pl.col("p_stockout") > 0).with_columns(
        (pl.col("mean_dep_net_lost") * pl.col("n_intervals")).alias("hour_net_lost")
    )
    return stockout_hours.group_by("station_id", "hour_of_week").agg(
        pl.col("hour_net_lost").mean().alias("e_net_lost_given_stockout"),
        pl.len().alias("n_stockout_hour_instances"),
    )


# ---------------------------------------------------------------------------
# Retained from the abandoned stationary model -- explains why it doesn't
# gate the current one. See module docstring.
# ---------------------------------------------------------------------------


def mm1k_p0(rho: float, capacity: int) -> float:
    """P(0 bikes) for a STATIONARY M/M/1/K birth-death chain truncated at
    K=capacity, rho = lambda/mu. Closed form: (1-rho)/(1-rho^(K+1)) for
    rho != 1, else 1/(K+1). Not used by the current (transient) model --
    kept for relaxation_time_minutes and the tests that document why the
    stationary approach was abandoned."""
    if capacity <= 0:
        return 1.0
    if abs(rho - 1.0) < 1e-9:
        return 1.0 / (capacity + 1)
    return (1.0 - rho) / (1.0 - rho ** (capacity + 1))


def relaxation_time_minutes(lam: float, mu: float, interval_minutes: float = INTERVAL_MINUTES) -> float:
    """Closed-form M/M/1 relaxation-time approximation (spectral gap of
    the birth-death generator in the K->infinity limit):
    1/(sqrt(mu)-sqrt(lam))^2, in units of 1/rate -- rates are per INTERVAL
    here, multiplied by interval_minutes to report minutes. Only defined
    for lam < mu; nan otherwise. Reported alongside the current model
    purely as retained context for why a stationary formulation was
    rejected -- it does not gate or otherwise affect the transient
    first-passage model below."""
    if not (lam < mu) or mu <= 0:
        return float("nan")
    denom = (np.sqrt(mu) - np.sqrt(lam)) ** 2
    if denom <= 0:
        return float("nan")
    return interval_minutes / denom


# ---------------------------------------------------------------------------
# Transient first-passage model
# ---------------------------------------------------------------------------


def build_step_transition_matrix(lam: float, mu: float, capacity: int) -> np.ndarray:
    """One 15-min-step transition matrix for the REFLECTING (not yet
    absorbing) bike-count dynamics: net change over one step is Skellam-
    distributed (difference of independent Poisson(lam) arrivals and
    Poisson(mu) departures -- the same Poisson draw the forward simulator
    itself uses per step, src/sim/simulator.py), clipped at the physical
    bounds [0, capacity]. `w` truncates the Skellam support at a width
    generous for realistic per-15-min rates (almost always < a few units);
    the truncated tail is renormalized back into the kept mass rather than
    silently dropped."""
    n = capacity + 1
    lam = max(0.0, lam)
    mu = max(0.0, mu)
    if lam == 0.0 and mu == 0.0:
        # scipy's skellam.pmf returns nan (not 0) at mu1=mu2=0 -- an actual
        # degenerate point mass at delta=0, not the "no data" case
        # total<=0 below is meant to catch; handled explicitly so it isn't
        # silently masked by NaN propagating through pmf.sum().
        return np.eye(n)
    w = int(np.ceil(6.0 * np.sqrt(lam + mu + 1.0) + 10))
    deltas = np.arange(-w, w + 1)
    pmf = skellam.pmf(deltas, lam, mu)
    total = pmf.sum()
    if not np.isfinite(total) or total <= 0:
        return np.eye(n)  # defensive fallback -- should not occur outside lam=mu=0
    pmf = pmf / total

    P = np.zeros((n, n))
    for k in range(n):
        k2 = np.clip(k + deltas, 0, capacity)
        np.add.at(P[k], k2, pmf)
    return P


def hitting_probabilities(lam: float, mu: float, capacity: int, n_steps: int) -> np.ndarray:
    """P(hit 0 -- stockout -- at some point within n_steps 15-min steps |
    start with k bikes), for k=0..capacity. Standard first-passage trick:
    make state 0 absorbing in the one-step transition matrix, raise it to
    the n_steps power -- the probability mass that ends up AT state 0
    after n_steps is exactly the probability of having been absorbed
    (hit 0) at any point up to n_steps, not just at the final step.

    np.errstate suppresses divide/overflow/invalid RuntimeWarnings from
    the matmul itself -- verified benign, not a correctness issue: for a
    low-traffic real cell (lam=0.133, mu=0.120, capacity=19) that triggers
    them, an independent sequential-multiply computation of the same power
    matched matrix_power's result to 1.7e-16 (floating-point noise). The
    warning fires on intermediate products in the ~1e-20 range once
    absorption probability has mostly saturated by k -- expected given
    most real station-hours have low per-15-min rates (median mu ~0.19,
    see DECISIONS.md), not something to chase per-cell across ~380K
    cells."""
    if capacity <= 0:
        return np.ones(1)
    P = build_step_transition_matrix(lam, mu, capacity)
    P[0, :] = 0.0
    P[0, 0] = 1.0
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Pn = np.linalg.matrix_power(P, n_steps)
    return Pn[:, 0]


def mv_curve(lam: float, mu: float, capacity: int, e_net_lost_given_stockout: float, n_steps: int) -> np.ndarray:
    """MV(n) for n=1..capacity: expected trips saved by having n bikes at
    the start of the window instead of n-1 -- the transient first-passage
    reframing (module docstring), not the earlier stationary
    approximation. MV(n) = [hit(n-1) - hit(n)] * E[net_lost | stockout],
    both at hour-window granularity."""
    if capacity <= 0 or e_net_lost_given_stockout <= 0:
        return np.zeros(max(capacity, 0))
    hit = hitting_probabilities(lam, mu, capacity, n_steps)
    return e_net_lost_given_stockout * (hit[:-1] - hit[1:])


def check_concavity(mv: np.ndarray) -> tuple[bool, int | None]:
    """mv: MV(1..capacity). Diminishing marginal returns is NOT guaranteed
    by construction for the transient model (unlike the stationary
    blocking probability's provable convexity) -- checked via second
    differences, not assumed. Returns (is_concave_everywhere,
    first_violation_n) where first_violation_n (1-based, into mv) is the
    smallest n with MV(n+1) > MV(n) -- marginal value going UP, a genuine
    violation -- or None if none found."""
    if len(mv) < 2:
        return True, None
    diffs = np.diff(mv)
    violations = np.where(diffs > 1e-9)[0]
    if len(violations) == 0:
        return True, None
    return False, int(violations[0]) + 1


# ---------------------------------------------------------------------------
# Stage 3: apply the transient model to every (station, hour-of-week) cell
# ---------------------------------------------------------------------------


def build_mv_curve_table(
    station_hour: pl.DataFrame, params: MarginalValueParams, n_steps: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """station_hour must already have `e_net_lost_given_stockout` joined on
    (see main()). Returns (station_hour_summary, mv_curve_exploded).

    Per DECISIONS.md's "persistence" entry: only k = 1..min(capacity,
    params.mv_k_max) is emitted. Above mv_k_max the model and the
    cross-check regression agree MV is effectively zero within the
    first-passage window -- computing and reporting those near-zero values
    anyway would invite exactly the ratio-of-near-zeros artifact that
    looked like catastrophic disagreement before the cells were segmented
    by k. Absence of a k > mv_k_max row in the output IS the answer for
    that k, not a gap.

    station_hour_summary: one row per (station_id, hour_of_week) with
    relaxation_time_minutes (retained context only, see module docstring),
    and whether the REPORTED (truncated) MV curve is concave everywhere
    (with the first violation n if not -- a real check under this model).

    mv_curve_exploded: one row per (station_id, hour_of_week, k) with
    MV(k), k = 1..min(capacity, mv_k_max)."""
    summary_rows = []
    curve_rows = []
    for row in station_hour.iter_rows(named=True):
        lam, mu, capacity = row["lam"], row["mu"], int(row["capacity"])
        e_net_lost = row["e_net_lost_given_stockout"] or 0.0
        full_curve = mv_curve(lam, mu, capacity, e_net_lost, n_steps)
        k_max = min(capacity, params.mv_k_max)
        reported_curve = full_curve[:k_max]
        is_concave, first_violation = check_concavity(reported_curve)

        summary_rows.append(
            {
                "station_id": row["station_id"],
                "hour_of_week": row["hour_of_week"],
                "relaxation_time_minutes": relaxation_time_minutes(lam, mu),
                "is_concave": is_concave,
                "first_concavity_violation_k": first_violation,
            }
        )
        for k, mv_k in enumerate(reported_curve, start=1):
            curve_rows.append(
                {
                    "station_id": row["station_id"],
                    "hour_of_week": row["hour_of_week"],
                    "k": k,
                    "mv": float(mv_k),
                }
            )

    summary = pl.DataFrame(summary_rows)
    curve = pl.DataFrame(curve_rows) if curve_rows else pl.DataFrame(
        schema={"station_id": pl.String, "hour_of_week": pl.Int16, "k": pl.Int64, "mv": pl.Float64}
    )
    return station_hour.join(summary, on=["station_id", "hour_of_week"], how="left"), curve


def eligibility_report(weekly: pl.DataFrame, params: MarginalValueParams) -> tuple[dict, pl.DataFrame]:
    """Station-hours whose REAL observed inventory ever reached
    <= mv_k_max bikes in at least one of the panel's ~52 real weeks -- per
    the persistence finding (DECISIONS.md), those are the only cells where
    this model estimates a nonzero MV, so they're the only ones eligible
    for incentive-based reallocation. Reports their count/share of all
    (station, hour-of-week) cells and their share of TOTAL net-lost demand
    (Phase 6, hour-window units to match mv_curve) -- if a small eligible
    pool carries most of the recoverable loss, that concentration IS the
    allocation result, not a caveat on it; Phase 8's optimizer only needs
    to search this pool, not all 379K cells.

    Returns (summary dict, eligible_cells DataFrame: station_id,
    hour_of_week, min_inventory)."""
    per_cell_min_inventory = weekly.group_by("station_id", "hour_of_week").agg(
        pl.col("mean_inventory").min().alias("min_inventory")
    )
    eligible_cells = per_cell_min_inventory.filter(pl.col("min_inventory") <= params.mv_k_max)

    weekly_hn = weekly.with_columns((pl.col("mean_dep_net_lost") * pl.col("n_intervals")).alias("hour_net_lost"))
    total_net_lost = float(weekly_hn["hour_net_lost"].sum())
    eligible_net_lost = float(
        weekly_hn.join(eligible_cells.select("station_id", "hour_of_week"), on=["station_id", "hour_of_week"], how="inner")[
            "hour_net_lost"
        ].sum()
    )

    n_total = per_cell_min_inventory.height
    n_eligible = eligible_cells.height
    summary = {
        "n_total_cells": n_total,
        "n_eligible_cells": n_eligible,
        "eligible_cell_fraction": n_eligible / n_total if n_total else float("nan"),
        "total_net_lost": total_net_lost,
        "eligible_net_lost": eligible_net_lost,
        "eligible_net_lost_share": eligible_net_lost / total_net_lost if total_net_lost else float("nan"),
    }
    return summary, eligible_cells


def eligibility_frequency_report(weekly: pl.DataFrame, params: MarginalValueParams) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Refines eligibility_report's lifetime-max criterion into a
    FREQUENCY one: incentive spend is a recurring policy, so a
    (station_id, hour_of_week) cell that ran low (mean_inventory <=
    mv_k_max) once in 52 real weeks is a fundamentally different
    allocation target than one that runs low every week, even though "ever
    reached k <= mv_k_max" treats them identically.

    Returns (per_cell: station_id, hour_of_week, n_weeks, n_low_weeks,
    low_frac, cell_net_lost -- one row per cell; bucket_summary: bucket,
    n_cells, cell_share, net_lost, net_lost_share -- bucketed by low_frac,
    in a fixed low-to-high order, not alphabetical)."""
    weekly_hn = weekly.with_columns(
        (pl.col("mean_dep_net_lost") * pl.col("n_intervals")).alias("hour_net_lost"),
        (pl.col("mean_inventory") <= params.mv_k_max).alias("is_low"),
    )
    per_cell = (
        weekly_hn.group_by("station_id", "hour_of_week")
        .agg(
            pl.len().alias("n_weeks"),
            pl.col("is_low").cast(pl.Int64).sum().alias("n_low_weeks"),
            pl.col("hour_net_lost").sum().alias("cell_net_lost"),
        )
        .with_columns((pl.col("n_low_weeks") / pl.col("n_weeks")).alias("low_frac"))
    )

    per_cell = per_cell.with_columns(
        pl.when(pl.col("low_frac") <= 0.0)
        .then(pl.lit(0))
        .when(pl.col("low_frac") <= 0.10)
        .then(pl.lit(1))
        .when(pl.col("low_frac") <= 0.25)
        .then(pl.lit(2))
        .when(pl.col("low_frac") <= 0.50)
        .then(pl.lit(3))
        .otherwise(pl.lit(4))
        .alias("_bucket_order")
    )

    n_total_cells = per_cell.height
    total_net_lost = float(per_cell["cell_net_lost"].sum())

    bucket_summary = (
        per_cell.group_by("_bucket_order")
        .agg(pl.len().alias("n_cells"), pl.col("cell_net_lost").sum().alias("net_lost"))
        .sort("_bucket_order")
        .with_columns(
            pl.when(pl.col("_bucket_order") == 0)
            .then(pl.lit("never"))
            .when(pl.col("_bucket_order") == 1)
            .then(pl.lit("low 0-10% of weeks"))
            .when(pl.col("_bucket_order") == 2)
            .then(pl.lit("low 10-25% of weeks"))
            .when(pl.col("_bucket_order") == 3)
            .then(pl.lit("low 25-50% of weeks"))
            .otherwise(pl.lit("low >50% of weeks (chronic)"))
            .alias("bucket"),
            (pl.col("n_cells") / n_total_cells).alias("cell_share"),
            (pl.col("net_lost") / total_net_lost if total_net_lost else pl.lit(float("nan"))).alias("net_lost_share"),
        )
        .select("bucket", "n_cells", "cell_share", "net_lost", "net_lost_share")
    )
    return per_cell.drop("_bucket_order"), bucket_summary


# ---------------------------------------------------------------------------
# Stage 4: local-sensitivity cross-check
# ---------------------------------------------------------------------------


def cross_check_regression(weekly: pl.DataFrame, params: MarginalValueParams) -> pl.DataFrame:
    """Per (station_id, hour_of_week) with >= min_weeks_for_cross_check
    real weekly observations AND real inventory variation across them: OLS
    slope of hour-TOTAL net-lost demand (mean_dep_net_lost * n_intervals,
    matching mv_curve's hour-window units -- NOT the per-15-min-interval
    rate) on mean_inventory, computed via the sum-based closed form (fully
    vectorized in polars, no per-group Python loop over ~381K cells).
    empirical_mv = -slope: trips saved per additional bike, over the
    inventory range that station-hour actually experienced across the
    panel's ~52 weeks."""
    df = weekly.with_columns((pl.col("mean_dep_net_lost") * pl.col("n_intervals")).alias("hour_net_lost"))
    df = df.with_columns(
        (pl.col("mean_inventory") * pl.col("hour_net_lost")).alias("_xy"),
        (pl.col("mean_inventory") ** 2).alias("_xx"),
    )
    grouped = df.group_by("station_id", "hour_of_week").agg(
        pl.len().alias("n_weeks"),
        pl.col("mean_inventory").sum().alias("_sx"),
        pl.col("hour_net_lost").sum().alias("_sy"),
        pl.col("_xy").sum().alias("_sxy"),
        pl.col("_xx").sum().alias("_sxx"),
        pl.col("mean_inventory").min().alias("inventory_min"),
        pl.col("mean_inventory").max().alias("inventory_max"),
    )
    n = pl.col("n_weeks")
    denom = n * pl.col("_sxx") - pl.col("_sx") ** 2
    grouped = grouped.filter(
        (n >= params.min_weeks_for_cross_check)
        & (denom.abs() > 1e-6)
        & (pl.col("inventory_max") > pl.col("inventory_min"))
    )
    grouped = grouped.with_columns(
        (-(n * pl.col("_sxy") - pl.col("_sx") * pl.col("_sy")) / denom).alias("empirical_mv")
    )
    return grouped.select("station_id", "hour_of_week", "empirical_mv", "n_weeks", "inventory_min", "inventory_max")


def build_cross_check_report(cross_check: pl.DataFrame, mv_curve_exploded: pl.DataFrame) -> pl.DataFrame:
    """Joins the empirical (real cross-week variation) MV against the
    transient model's MV at the matching k (this station-hour's observed
    mid-range inventory, rounded to the nearest integer bike count) --
    agreement/disagreement reported PER (station_id, hour_of_week), not
    collapsed into one aggregate number that would hide exactly the cells
    where the two disagree."""
    cc = cross_check.with_columns(
        ((pl.col("inventory_min") + pl.col("inventory_max")) / 2.0).round(0).cast(pl.Int64).alias("k")
    )
    joined = cc.join(
        mv_curve_exploded.rename({"mv": "model_mv"}), on=["station_id", "hour_of_week", "k"], how="left"
    )
    return joined.with_columns(
        (pl.col("empirical_mv") - pl.col("model_mv")).alias("diff"),
        pl.when(pl.col("model_mv").abs() > 1e-9)
        .then(pl.col("empirical_mv") / pl.col("model_mv"))
        .otherwise(None)
        .alias("ratio"),
    ).select(
        "station_id", "hour_of_week", "k", "empirical_mv", "model_mv", "diff", "ratio",
        "n_weeks", "inventory_min", "inventory_max",
    )


# ---------------------------------------------------------------------------
# Stage 5b: chronic-cell timing predictability -- does the policy need a
# live inventory feed, or does a standing schedule catch it?
# ---------------------------------------------------------------------------


def build_month_chronic_timing_partial(month_key: str, chronic_cells: pl.DataFrame, mv_k_max: int) -> pl.DataFrame:
    """Raw 15-min inventory, restricted to CHRONIC cells only (inner join
    on station_id + hour_of_week -- chronic_cells: station_id,
    hour_of_week), for this month. Returns one row per (station_id,
    hour_of_week, week_start) among those cells' LOW weeks this month,
    with first_low_position: the sub-interval index (0-3, i.e.
    :00/:15/:30/:45 into the hour) where inventory first crossed
    <= mv_k_max that week. Every week flagged low at the mean_inventory
    level (Stage 6's is_low) has at least one sub-interval <= mv_k_max --
    if all 4 were above it, their mean would be too -- so first_low_
    position is always defined for a low week; this just locates WHICH of
    the 4 slots.

    No month-boundary split-week concern here (unlike Stage 1's weekly
    partial sums): a single hour never spans two calendar months, so each
    (station, hour_of_week, week_start) instance is entirely produced by
    one month's chunk -- callers just concatenate across months, no
    re-summing needed."""
    start, end = od_shares.month_bounds(month_key)
    inv_m = (
        pl.scan_parquet(INVENTORY_PATH)
        .filter(pl.col("interval_start").is_between(start, end, closed="left"))
        .select("station_id", "interval_start", "inventory")
        .with_columns(
            (pl.col("interval_start").dt.weekday() - 1).cast(pl.Int16).alias("dow"),
            pl.col("interval_start").dt.hour().cast(pl.Int16).alias("hour"),
        )
        .with_columns((pl.col("dow") * 24 + pl.col("hour")).alias("hour_of_week"))
        .join(chronic_cells.lazy(), on=["station_id", "hour_of_week"], how="inner")
        .with_columns(
            pl.col("interval_start").dt.truncate("1w").alias("week_start"),
            (pl.col("interval_start").dt.minute() // 15).cast(pl.Int16).alias("position"),
        )
        .collect()
    )
    return (
        inv_m.filter(pl.col("inventory") <= mv_k_max)
        .group_by("station_id", "hour_of_week", "week_start")
        .agg(pl.col("position").min().alias("first_low_position"))
    )


def run_chronic_timing(chronic_cells: pl.DataFrame, params: MarginalValueParams) -> pl.DataFrame:
    parts = [build_month_chronic_timing_partial(mk, chronic_cells, params.mv_k_max) for mk in all_month_keys()]
    return pl.concat(parts, how="vertical")


def chronic_timing_summary(per_hour_instance: pl.DataFrame) -> pl.DataFrame:
    """Per chronic cell: n_low_weeks_observed, mean/std of
    first_low_position (0-3 scale, std=null for a cell with only one
    observed low week -- variance undefined, not zero), and modal_share --
    the fraction of that cell's low weeks whose first-low sub-interval was
    its single most common one. modal_share near 1.0 means the deficit
    recurs at a predictable 15-min clock slot (schedulable with a standing
    incentive, no live inventory feed); near 0.25 means it's effectively
    uniform across the hour's 4 slots (needs state-triggering)."""
    per_cell_std = per_hour_instance.group_by("station_id", "hour_of_week").agg(
        pl.len().alias("n_low_weeks_observed"),
        pl.col("first_low_position").mean().alias("mean_position"),
        pl.col("first_low_position").std().alias("std_position"),
    )
    position_counts = per_hour_instance.group_by("station_id", "hour_of_week", "first_low_position").agg(
        pl.len().alias("n")
    )
    modal = position_counts.group_by("station_id", "hour_of_week").agg(pl.col("n").max().alias("modal_count"))
    per_cell = per_cell_std.join(modal, on=["station_id", "hour_of_week"]).with_columns(
        (pl.col("modal_count") / pl.col("n_low_weeks_observed")).alias("modal_share")
    )
    return per_cell.select(
        "station_id", "hour_of_week", "n_low_weeks_observed", "mean_position", "std_position", "modal_share"
    )


# ---------------------------------------------------------------------------
# Stage 8: chronic cells vs Phase 4's inferred non-trip inflow -- are
# chronic deficits already heavily rebalanced (diminishing returns for
# incentive spend there) or barely rebalanced (genuinely unserved -- the
# strongest possible argument for the incentive mechanism)?
# ---------------------------------------------------------------------------


def rebalancing_vs_chronicity_report(station_hour: pl.DataFrame, low_freq_per_cell: pl.DataFrame) -> tuple[dict, pl.DataFrame]:
    """Crosses chronic-cell classification (eligibility_frequency_report's
    low_frac) against Phase 4's inferred non-trip INFLOW rate
    (nontrip_in_rate -- rebalancing/maintenance bikes brought IN, the
    direction that would relieve a bike deficit; nontrip_out_rate is the
    wrong direction to look at for a low-inventory cell). "At comparable
    demand" means controlling for organic latent departure-trip demand
    (dep_rate) -- NOT mu, which already folds nontrip_out_rate in and
    would be circular to condition on when the thing being measured is
    itself a non-trip quantity.

    Two per-cell views, cross-checked against each other rather than
    either trusted alone: a normalized per-cell ratio (nontrip_in_rate /
    dep_rate, "rebalancing intensity" -- controls for demand at the
    individual-cell level, no binning needed) and, from the returned
    per-cell frame, a demand-decile-stratified comparison the caller can
    build (rebalancing_by_demand_decile) as a second, coarser-but-more-
    interpretable check.

    Returns (summary dict, per-cell joined frame with is_chronic and
    rebalancing_intensity columns for further slicing)."""
    joined = (
        station_hour.join(
            low_freq_per_cell.select("station_id", "hour_of_week", "low_frac"),
            on=["station_id", "hour_of_week"],
            how="inner",
        )
        .with_columns((pl.col("low_frac") > 0.5).alias("is_chronic"))
        .with_columns((pl.col("nontrip_in_rate") / (pl.col("dep_rate") + 1e-6)).alias("rebalancing_intensity"))
    )

    chronic = joined.filter(pl.col("is_chronic"))
    non_chronic = joined.filter(~pl.col("is_chronic"))

    chronic_intensity = float(chronic["rebalancing_intensity"].median())
    non_chronic_intensity = float(non_chronic["rebalancing_intensity"].median())
    summary = {
        "n_chronic": chronic.height,
        "n_non_chronic": non_chronic.height,
        "chronic_median_nontrip_in_rate": float(chronic["nontrip_in_rate"].median()),
        "non_chronic_median_nontrip_in_rate": float(non_chronic["nontrip_in_rate"].median()),
        "chronic_median_dep_rate": float(chronic["dep_rate"].median()),
        "non_chronic_median_dep_rate": float(non_chronic["dep_rate"].median()),
        "chronic_median_rebalancing_intensity": chronic_intensity,
        "non_chronic_median_rebalancing_intensity": non_chronic_intensity,
        "intensity_ratio_chronic_over_non_chronic": (
            chronic_intensity / non_chronic_intensity if non_chronic_intensity else float("nan")
        ),
    }
    return summary, joined


def rebalancing_by_demand_decile(joined: pl.DataFrame) -> pl.DataFrame:
    """Bins cells by dep_rate decile (organic demand, NOT the rebalancing
    quantity itself), then compares median nontrip_in_rate between chronic
    and non-chronic cells WITHIN each decile -- the demand-stratified
    cross-check for rebalancing_vs_chronicity_report's normalized-ratio
    view, guarding against chronic and non-chronic cells simply sitting at
    systematically different demand levels overall."""
    with_decile = joined.filter(pl.col("dep_rate") > 0).with_columns(
        pl.col("dep_rate").qcut(10, allow_duplicates=True).alias("demand_decile")
    )
    per_decile = (
        with_decile.group_by("demand_decile", "is_chronic")
        .agg(
            pl.len().alias("n_cells"),
            pl.col("nontrip_in_rate").median().alias("median_nontrip_in_rate"),
            pl.col("dep_rate").median().alias("median_dep_rate"),
        )
        .sort("median_dep_rate", "is_chronic")
    )
    return per_decile


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    params = load_params()
    n_steps = first_passage_steps(params)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[marginal_value] Stage 1: (station, hour-of-week, week) partial sums, month-chunked...")
    run_stage1()
    weekly = finalize_station_hour_week()

    print("[marginal_value] Stage 2: yearly-averaged rates + E[net_lost|stockout] at hour granularity...")
    station_hour = aggregate_station_hour(weekly)
    hourly_net_lost = hourly_net_lost_given_stockout(weekly)
    station_hour = station_hour.join(hourly_net_lost, on=["station_id", "hour_of_week"], how="left").with_columns(
        pl.col("e_net_lost_given_stockout").fill_null(0.0),
        pl.col("n_stockout_hour_instances").fill_null(0),
    )
    print(f"[marginal_value] {station_hour.height:,} (station, hour-of-week) cells, peak RSS {progress.peak_rss_mb():.0f} MB")

    print(
        f"[marginal_value] Stage 3: transient first-passage MV curve "
        f"(window={params.first_passage_window_minutes:.0f} min, {n_steps} steps) + concavity check..."
    )
    station_hour_summary, mv_curve_exploded = build_mv_curve_table(station_hour, params, n_steps)
    checkpoint.write_checkpoint(station_hour_summary, STATION_HOUR_PATH)
    checkpoint.write_checkpoint(mv_curve_exploded, MV_CURVE_PATH)

    n_cells = station_hour_summary.height
    n_concave = station_hour_summary.filter(pl.col("is_concave")).height
    rt = station_hour_summary["relaxation_time_minutes"].drop_nulls().drop_nans()
    print(
        f"[marginal_value] concavity (real check under the transient model, not guaranteed by construction): "
        f"{n_concave:,}/{n_cells:,} cells ({n_concave/n_cells:.1%}) concave everywhere; "
        f"{n_cells - n_concave:,} have at least one violation -- see station_hour.parquet's "
        "first_concavity_violation_k for where"
    )
    print(
        f"[marginal_value] relaxation time (retained context, does not gate this model): "
        f"median={rt.median():.1f} min, p90={rt.quantile(0.9):.1f} min"
    )

    print("[marginal_value] Stage 4: local-sensitivity cross-check (unit-matched to the transient model)...")
    cross_check = cross_check_regression(weekly, params)
    report = build_cross_check_report(cross_check, mv_curve_exploded)
    checkpoint.write_checkpoint(report, CROSS_CHECK_PATH)

    matched = report.filter(pl.col("model_mv").is_not_null())
    print(
        f"[marginal_value] cross-check: {cross_check.height:,} cells had enough weekly variation to regress; "
        f"{matched.height:,} matched a modeled k. Per-cell agreement -> {CROSS_CHECK_PATH}"
    )
    if matched.height:
        ratio = matched["ratio"].drop_nulls().drop_nans()
        print(
            f"[marginal_value] empirical_mv / model_mv ratio (k <= {params.mv_k_max} only, "
            f"where both methods estimate a nonzero MV): median={ratio.median():.2f}, "
            f"p10={ratio.quantile(0.1):.2f}, p90={ratio.quantile(0.9):.2f} (1.0 = perfect agreement)"
        )

    print(f"[marginal_value] Stage 5: eligibility (station-hours whose real inventory ever reached <= {params.mv_k_max})...")
    eligibility, eligible_cells = eligibility_report(weekly, params)
    checkpoint.write_checkpoint(eligible_cells, ELIGIBLE_CELLS_PATH)
    print(
        f"[marginal_value] eligible cells: {eligibility['n_eligible_cells']:,} / {eligibility['n_total_cells']:,} "
        f"({eligibility['eligible_cell_fraction']:.1%}) of all (station, hour-of-week) cells"
    )
    print(
        f"[marginal_value] eligible cells carry {eligibility['eligible_net_lost_share']:.1%} of total net-lost demand "
        f"({eligibility['eligible_net_lost']:,.0f} / {eligibility['total_net_lost']:,.0f} trips, hour-window units) "
        f"-> {ELIGIBLE_CELLS_PATH}"
    )

    print(
        "[marginal_value] Stage 6: eligibility FREQUENCY (recurring-policy refinement -- how often, not just ever)..."
    )
    low_freq_per_cell, low_freq_buckets = eligibility_frequency_report(weekly, params)
    checkpoint.write_checkpoint(low_freq_per_cell, LOW_FREQUENCY_PATH)
    print(f"[marginal_value] low_frac distribution by bucket -> {LOW_FREQUENCY_PATH}")
    print(low_freq_buckets)
    chronic_share = low_freq_buckets.filter(pl.col("bucket").str.contains("chronic"))["net_lost_share"]
    if chronic_share.height:
        print(
            f"[marginal_value] cells low in >50% of their weeks carry "
            f"{chronic_share.item():.1%} of total net-lost demand"
        )

    chronic_cells_all = low_freq_per_cell.filter(pl.col("low_frac") > 0.5).select("station_id", "hour_of_week", "low_frac")
    n_chronic = chronic_cells_all.height
    n_very = chronic_cells_all.filter(pl.col("low_frac") > 0.8).height
    n_moderate = n_chronic - n_very
    print(
        f"[marginal_value] chronic split: low_frac>0.8: {n_very:,} ({n_very/n_chronic:.1%} of chronic); "
        f"0.5-0.8: {n_moderate:,} ({n_moderate/n_chronic:.1%} of chronic)"
    )

    print(
        f"[marginal_value] Stage 7: chronic-cell timing predictability "
        f"({n_chronic:,} chronic cells -- is the deficit schedulable or does it need state-triggering?)..."
    )
    per_hour_instance = run_chronic_timing(chronic_cells_all.select("station_id", "hour_of_week"), params)
    timing = chronic_timing_summary(per_hour_instance)
    checkpoint.write_checkpoint(timing, CHRONIC_TIMING_PATH)

    modal_share = timing["modal_share"].drop_nulls().drop_nans()
    std_pos = timing["std_position"].drop_nulls().drop_nans()
    n_schedulable = timing.filter(pl.col("modal_share") >= params.schedulable_modal_share_threshold).height
    print(
        f"[marginal_value] within-hour timing across {timing.height:,} chronic cells: "
        f"modal_share median={modal_share.median():.2f}, p25={modal_share.quantile(0.25):.2f}, "
        f"p75={modal_share.quantile(0.75):.2f} | std_position (0-3 scale) median={std_pos.median():.2f} "
        f"-> {CHRONIC_TIMING_PATH}"
    )
    print(
        f"[marginal_value] schedulable (modal_share >= {params.schedulable_modal_share_threshold}): "
        f"{n_schedulable:,} / {timing.height:,} ({n_schedulable/timing.height:.1%}) of chronic cells"
    )


if __name__ == "__main__":
    main()
