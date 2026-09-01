"""Phase 9 (SPEC.md §8, RUNBOOK Phase 9): run the Phase 7 simulator under
incentive policies (src/opt/policy_baselines.py) at the same budget,
bootstrap fill-rate lift over 3 uncertainty axes, and (secondary) sweep
budget for "our allocator."

**Two constraints from Phase 7's DECISIONS.md resolution, both load-bearing
here, neither optional:**

1. **Zone/system level ONLY.** The forward simulator cannot supply
   P(stockout | station, hour-of-week) -- confirmed structurally, not just
   untested (6-week pooled correlation plateaus at ~0.10). No function in
   this module accepts, returns, or checkpoints a per-station fill-rate
   breakdown; `compute_fill_rate_table` only ever emits "system" and
   per-zone rows.
2. **Paired bootstrap, same seed across policies within a replicate.**
   Phase 7's resolution assumes destination-assignment noise is present in
   BOTH the baseline and treatment run and cancels in the reported
   DIFFERENCE -- unproven, stated as such here and in every output this
   module produces, not treated as established. `run_one_replicate` draws
   ONE (demand_multiplier, tier1, tier2, seed) tuple and runs all policies
   against it, never independent draws per policy.

**Six policies, not five -- added post-pilot (2026-08-14).** The 1-replicate
pilot showed `policy_allocator` spends only ~3-4% of the $10k weekly_budget
(all 2,998 candidates funded at their own individually-optimal payout costs
$310-383 total, confirmed directly from ranking_default.parquet: net-value-
per-dollar declines smoothly, never collapses, and the ranked list simply
runs out -- see policy_baselines.policy_allocator_full_budget's docstring).
Per request: report BOTH `allocator` (as-is, the honest "how much does the
optimizer's own logic actually need" number) AND `allocator_full_budget`
(a budget-exhausting variant, same priority order, escalating payouts until
the budget is spent) side by side, rather than silently picking one.

Checkpointing (per replicate, atomic via utils/checkpoint.write_checkpoint)
makes the whole run resumable -- a kill mid-batch loses at most the
in-flight replicate, never previously completed ones.
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

import models.demand as demand
import models.od_shares as od_shares
import opt.allocate as allocate
import opt.policy_baselines as policy_baselines
import sim.simulator as simulator
import utils.checkpoint as checkpoint
import utils.progress as progress

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "policy_compare"
REPLICATES_DIR = OUT_DIR / "replicates"
BUDGET_SWEEP_DIR = OUT_DIR / "budget_sweep_replicates"
TREATED_DIR = OUT_DIR / "treated_replicates"
BOOTSTRAP_RESULTS_PATH = OUT_DIR / "bootstrap_results.parquet"
BUDGET_SWEEP_RESULTS_PATH = OUT_DIR / "budget_sweep_results.parquet"
TREATED_RESULTS_PATH = OUT_DIR / "treated_results.parquet"

# Matches config/params.yaml's simulation.seed default -- one integer per
# replicate seeds BOTH the outer (demand_multiplier, elasticity) draw and
# the simulator's own internal rng for that replicate (shared across all 5
# policies, per the paired-bootstrap requirement above).
BASE_SEED = 0

POLICY_NAMES = ("do_nothing", "uniform", "proportional", "top_n_stockout", "allocator", "allocator_full_budget")

# SPEC.md §8: "Plot lift vs. budget from $0 to 3x" -- scoped to "our
# allocator" only (see the plan: SPEC frames this curve specifically
# around the optimizer, not all 6 policies). do_nothing is the $0 anchor,
# reused from the main bootstrap's own do_nothing result where possible.
#
# Levels are ABSOLUTE DOLLARS, not multiples of weekly_budget -- redesigned
# post-pilot (2026-08-14) around the $244-567 total-spend finding above.
# **Deliberate deviation from SPEC.md §8's literal "$0 to 3x" range, per
# explicit instruction given time constraints (user traveling the
# following morning): capped at 1x weekly_budget ($10,000), not 3x
# ($30,000).** Placement is deliberately concentrated where the curve
# actually bends (dense $0-$1,000, spanning the observed $244-567 natural-
# spend range) with two coarser points (2500, 10000) to confirm the
# plateau stays flat well beyond it -- a shape-finding grid, not a
# uniform one. The main bootstrap (run_replicates) carries the real CIs;
# this sweep is intentionally light (see BUDGET_SWEEP_POLICIES and
# main()'s --budget-sweep-replicates) and running it is subordinate to
# the main run finishing intact -- main() always completes and
# checkpoints the main bootstrap FIRST, sweep second, so an interruption
# during the sweep never touches the main results.
BUDGET_LEVELS_DOLLARS = (0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 750.0, 1000.0, 2500.0, 10000.0)
BUDGET_SWEEP_POLICIES = ("do_nothing", "allocator", "allocator_full_budget")


# ---------------------------------------------------------------------------
# (a) demand-model-residual bootstrap axis
# ---------------------------------------------------------------------------


def compute_holdout_residual_ratios(fitted_dep: demand.FittedDirection) -> np.ndarray:
    """No stored residual artifact exists on disk (checked --
    calibrate_direction_matched recomputes on the fly, never persisted).
    Computed once here, at Phase 9 setup, from the demand model's OWN
    holdout period (config/params.yaml's demand_model.holdout_start_date
    onward) via demand.predict_vs_actual_holdout -- the same uncensored-
    only evaluation held_out_wmape (Phase 5's gate) already uses, factored
    out so this module doesn't re-derive it against a private function.

    Aggregated to DAILY totals before taking a ratio, not a per-15-min-
    interval ratio -- checked directly (not assumed): at 15-min
    granularity, uncensored departure counts are dominated by 0s and small
    integers, so mean(y_i / d_hat_i) is pulled hard toward 0 by that
    zero-inflation (Jensen's-inequality-flavored bias, not a real "demand
    ran 66% low" signal) -- an early version of this function returned a
    mean ratio of 0.34, which is implausible on its face for a model that
    otherwise clears Phase 5's WMAPE gate. sum(y)/sum(d_hat) per day is the
    standard fix: each day aggregates enough intervals that a handful of
    zero-departure 15-min slots can't dominate the ratio, and the result IS
    a meaningful "demand ran X% high/low that day" scale factor.
    sample_demand_multiplier bootstrap-resamples from these ~90 daily
    values (one per holdout day, 3 months)."""
    pipeline_params = demand.load_pipeline_params()
    lag_features_lf = demand.ensure_lag_features_artifact()
    weather = pl.read_parquet(demand.WEATHER_PATH)
    weather_lag_table = demand.compute_weather_lag_table(weather)

    months = demand.all_month_keys()
    holdout_prefix = pipeline_params.holdout_start_date[:7]
    holdout_months = [m for m in months if m >= holdout_prefix]
    holdout_df = demand.build_range_features(holdout_months, weather_lag_table, lag_features_lf)

    y, d_hat, interval_start = demand.predict_vs_actual_holdout(fitted_dep, holdout_df)
    daily = (
        pl.DataFrame({"day": pl.Series(interval_start).dt.truncate("1d"), "y": y, "d_hat": d_hat})
        .group_by("day")
        .agg(pl.col("y").sum().alias("actual"), pl.col("d_hat").sum().alias("predicted"))
        .filter(pl.col("predicted") > 0)
    )
    return (daily["actual"] / daily["predicted"]).to_numpy()


def sample_demand_multiplier(rng: np.random.Generator, residual_ratios: np.ndarray) -> float:
    """Bootstrap-resamples the holdout actual/predicted ratio distribution
    and returns the resample's mean -- a single per-replicate "demand ran
    X% high/low this week" model-uncertainty draw (SPEC.md §8's axis (a)),
    applied via simulator.run_step's demand_multiplier parameter."""
    resample = rng.choice(residual_ratios, size=len(residual_ratios), replace=True)
    return float(resample.mean())


# ---------------------------------------------------------------------------
# (b) elasticity-parameter bootstrap axis
# ---------------------------------------------------------------------------


def sample_elasticity_draw(
    rng: np.random.Generator, alloc_params: allocate.AllocationParams
) -> tuple[allocate.TierElasticity, allocate.TierElasticity]:
    """Draws tier2's (a, b) uniformly from the existing elasticity_sweep
    range (config/params.yaml -- same grid bounds Phase 8's own sweep
    uses), then derives tier1 via the SAME conservative ratio as the
    config defaults (tier1/tier2 = 2.0/3.0 on a, 0.15/0.25 on b) rather
    than drawing tier1 independently -- preserves the documented
    tier1-more-conservative-than-tier2 structural relationship
    (DECISIONS.md's "chronicity vs. non-trip movement" finding) instead of
    risking an inconsistent draw where tier1 is accidentally MORE
    aggressive than tier2."""
    a2 = float(rng.uniform(alloc_params.sweep_a_min, alloc_params.sweep_a_max))
    b2 = float(rng.uniform(alloc_params.sweep_b_min, alloc_params.sweep_b_max))
    a_ratio = alloc_params.elasticity_tier1.a / alloc_params.elasticity_tier2.a
    b_ratio = alloc_params.elasticity_tier1.b / alloc_params.elasticity_tier2.b
    tier1 = allocate.TierElasticity(a=a2 * a_ratio, b=b2 * b_ratio)
    tier2 = allocate.TierElasticity(a=a2, b=b2)
    return tier1, tier2


# ---------------------------------------------------------------------------
# Fill rate (SPEC.md §8) -- system + zone level ONLY, never station level
# ---------------------------------------------------------------------------


def compute_fill_rate_table(run: simulator.SimulationRun, network: simulator.NetworkArrays) -> pl.DataFrame:
    """fill_rate = fulfilled / (fulfilled + lost), SPEC.md §8's definition,
    computed directly from the simulator's own trip accounting (direct +
    rerouted arrivals = fulfilled; lost_no_bike + lost_past_cap_arrivals =
    lost) -- the simulator's reroute mechanic already IS its own
    substitution model, so this doesn't need substitution.py's separate
    historical-panel net-lost definition. Returns "system" (one row) +
    one row per zone -- see module docstring point 1. NEVER per-station."""
    si = run.station_intervals.with_columns(
        (pl.col("direct_arrivals") + pl.col("rerouted_arrivals")).alias("fulfilled"),
        (pl.col("lost_no_bike") + pl.col("lost_past_cap_arrivals")).alias("lost"),
    )
    system = si.select(pl.col("fulfilled").sum(), pl.col("lost").sum()).with_columns(
        pl.lit("system").alias("level"), pl.lit("system").alias("zone_agg")
    )

    zone_table = pl.DataFrame({"station_id": network.station_id, "zone_agg": network.zone_agg})
    by_zone = (
        si.join(zone_table, on="station_id", how="left")
        .group_by("zone_agg")
        .agg(pl.col("fulfilled").sum(), pl.col("lost").sum())
        .with_columns(pl.lit("zone").alias("level"))
    )
    out = pl.concat(
        [system.select("level", "zone_agg", "fulfilled", "lost"), by_zone.select("level", "zone_agg", "fulfilled", "lost")],
        how="vertical",
    )
    return out.with_columns((pl.col("fulfilled") / (pl.col("fulfilled") + pl.col("lost"))).alias("fill_rate"))


# ---------------------------------------------------------------------------
# Shared artifacts (loaded once per worker process, reused across replicates)
# ---------------------------------------------------------------------------


@dataclass
class SharedArtifacts:
    network: simulator.NetworkArrays
    week: simulator.WeekInputs
    fitted_dep: demand.FittedDirection
    od_model: od_shares.ODShareModel
    sim_params: simulator.SimulationParams
    alloc_params: allocate.AllocationParams
    universe: policy_baselines.SharedCandidateUniverse
    residual_ratios: np.ndarray


def load_shared_artifacts() -> SharedArtifacts:
    sim_params = simulator.load_simulation_params()
    network = simulator.load_station_network()
    week = simulator.prepare_week_inputs(network, sim_params)
    fitted_dep = demand.load_fitted(demand.DEMAND_MODEL_DIR / "departures.pkl")
    od_model = od_shares.load_od_share_model()
    alloc_params = allocate.load_params()
    universe = policy_baselines.load_shared_candidate_universe()
    residual_ratios = compute_holdout_residual_ratios(fitted_dep)
    return SharedArtifacts(network, week, fitted_dep, od_model, sim_params, alloc_params, universe, residual_ratios)


def _build_policy(
    name: str,
    universe: policy_baselines.SharedCandidateUniverse,
    tier1: allocate.TierElasticity,
    tier2: allocate.TierElasticity,
    alloc_params: allocate.AllocationParams,
    weekly_budget: float,
) -> pl.DataFrame:
    if name == "do_nothing":
        return policy_baselines.policy_do_nothing()
    if name == "uniform":
        return policy_baselines.policy_uniform(universe, tier1, tier2, alloc_params, weekly_budget)
    if name == "proportional":
        return policy_baselines.policy_proportional(universe, tier1, tier2, alloc_params, weekly_budget)
    if name == "top_n_stockout":
        return policy_baselines.policy_top_n_stockout(universe, tier1, tier2, alloc_params, weekly_budget)
    if name == "allocator":
        return policy_baselines.policy_allocator(universe, tier1, tier2, alloc_params, weekly_budget)
    if name == "allocator_full_budget":
        return policy_baselines.policy_allocator_full_budget(universe, tier1, tier2, alloc_params, weekly_budget)
    raise ValueError(f"unknown policy {name!r}")


def _run_policy_simulation(
    name: str,
    artifacts: SharedArtifacts,
    tier1: allocate.TierElasticity,
    tier2: allocate.TierElasticity,
    weekly_budget: float,
    demand_multiplier: float,
    seed: int,
) -> tuple[simulator.SimulationRun, pl.DataFrame, float]:
    """Builds one policy's funded moves, applies the physical-plausibility
    cap, injects into the simulator, and returns (raw SimulationRun, the
    capped funded-move table, dollar_cost) -- the shared step underneath
    both _simulate_policy's system/zone fill table and
    run_one_replicate_treated's treated-cell fill table (added when the
    system-level comparison turned out to be measuring a ~0.1%-of-system
    effect against system-wide variance -- see that function's docstring),
    so build->cap->inject->simulate isn't duplicated between them."""
    funded = _build_policy(name, artifacts.universe, tier1, tier2, artifacts.alloc_params, weekly_budget)
    capped = policy_baselines.apply_move_cap(funded, artifacts.alloc_params.max_induced_moves_per_station_hour)
    induced_moves = policy_baselines.to_simulator_induced_moves(capped)
    dollar_cost = float(capped["dollar_cost"].sum()) if capped.height else 0.0

    policy_sim_params = dataclasses.replace(artifacts.sim_params, seed=seed)
    run = simulator.run_simulation(
        artifacts.network, artifacts.week, artifacts.fitted_dep, artifacts.od_model, policy_sim_params,
        mode="stochastic", induced_moves=induced_moves, demand_multiplier=demand_multiplier,
    )
    return run, capped, dollar_cost


def _simulate_policy(
    name: str,
    artifacts: SharedArtifacts,
    tier1: allocate.TierElasticity,
    tier2: allocate.TierElasticity,
    weekly_budget: float,
    demand_multiplier: float,
    seed: int,
) -> tuple[pl.DataFrame, float]:
    """Builds one policy's funded moves, applies the physical-plausibility
    cap, injects into the simulator, and returns (fill_rate_table,
    dollar_cost). Shared by both the main bootstrap and the budget sweep so
    the build->cap->inject->simulate->fill_rate sequence isn't duplicated
    between them."""
    run, _, dollar_cost = _run_policy_simulation(
        name, artifacts, tier1, tier2, weekly_budget, demand_multiplier, seed
    )
    return compute_fill_rate_table(run, artifacts.network), dollar_cost


# ---------------------------------------------------------------------------
# Main bootstrap: one shared draw, all 5 policies, per replicate
# ---------------------------------------------------------------------------


@dataclass
class ReplicateDraw:
    demand_multiplier: float
    tier1: allocate.TierElasticity
    tier2: allocate.TierElasticity
    seed: int


def draw_replicate(replicate_idx: int, artifacts: SharedArtifacts, base_seed: int = BASE_SEED) -> ReplicateDraw:
    rng = np.random.default_rng(base_seed + replicate_idx)
    demand_multiplier = sample_demand_multiplier(rng, artifacts.residual_ratios)
    tier1, tier2 = sample_elasticity_draw(rng, artifacts.alloc_params)
    return ReplicateDraw(demand_multiplier=demand_multiplier, tier1=tier1, tier2=tier2, seed=base_seed + replicate_idx)


def run_one_replicate(replicate_idx: int, artifacts: SharedArtifacts, base_seed: int = BASE_SEED) -> pl.DataFrame:
    """The paired-bootstrap unit: ONE (demand_multiplier, tier1, tier2,
    seed) draw, run through every policy in POLICY_NAMES with that SAME
    draw (module docstring point 2). Returns one row per (policy, level) --
    "system" + one per zone, per compute_fill_rate_table."""
    draw = draw_replicate(replicate_idx, artifacts, base_seed)

    rows = []
    for name in POLICY_NAMES:
        fill, dollar_cost = _simulate_policy(
            name, artifacts, draw.tier1, draw.tier2, artifacts.alloc_params.weekly_budget,
            draw.demand_multiplier, draw.seed,
        )
        rows.append(
            fill.with_columns(
                pl.lit(replicate_idx).alias("replicate"),
                pl.lit(name).alias("policy"),
                pl.lit(draw.demand_multiplier).alias("demand_multiplier"),
                pl.lit(draw.tier1.a).alias("tier1_a"), pl.lit(draw.tier1.b).alias("tier1_b"),
                pl.lit(draw.tier2.a).alias("tier2_a"), pl.lit(draw.tier2.b).alias("tier2_b"),
                pl.lit(draw.seed).alias("seed"),
                pl.lit(dollar_cost).alias("dollar_cost"),
            )
        )
    return pl.concat(rows, how="vertical")


# ---------------------------------------------------------------------------
# Treated-cell paired comparison: same replicate/seed, but the fill-rate
# denominator is restricted to the (station, hour) cells a policy actually
# funded, not the whole system.
#
# **Why this exists.** The system-level comparison above pools fill rate
# over the ENTIRE network (~875K trips/week) while every policy here funds
# at most a few thousand induced trips against it -- SPEC.md's own
# candidate pool is 2,998 cells, and `allocator`'s natural spend covers far
# fewer. That's a >99.5%-of-the-denominator dilution of any real effect
# before bootstrap noise even enters the picture, which is a measurement-
# design problem, not (only) a replicate-count problem -- no amount of
# additional bootstrap replicates shrinks a signal-to-noise ratio that's
# capped by the denominator itself. Restricting both the policy run AND a
# same-seed do-nothing run to the SAME treated cells removes that dilution
# directly, and pairing on the SAME seed within a replicate (not just the
# same nominal draw) also cancels the demand-residual/elasticity variance
# the system-level pairing already relied on -- so if THIS comparison is
# still indistinguishable from zero, that's a real power/underpowered
# finding; if it isn't, the system-level "underpowered" conclusion was
# actually a measurement-design artifact, and DECISIONS.md should say so.
#
# **Why this does NOT reopen the Phase 7 station-level restriction.**
# compute_fill_rate_table never emits a per-station row because Phase 7
# found simulated per-(station, hour-of-week) stockout timing doesn't
# correlate with real ground truth at that grain (CLAUDE.md, "confirmed
# simulator noise"). That finding is about matching REALITY at station
# resolution. This comparison never claims to match reality at station
# resolution -- it only compares two runs of the SAME simulator, same
# seed, against each other, and reports ONE pooled number per policy
# summed over every cell that policy touched (never a single station's
# fill rate). Still genuinely noisier than the system-level number per
# cell (same destination-assignment stochasticity Phase 7 flagged applies
# here too, just not diluted away) -- that's exactly why this needs its
# own CI, not an assumption it's automatically cleaner.
def compute_treated_cell_fill(
    station_intervals: pl.DataFrame,
    calendar_weather: pl.DataFrame,
    treated_cells: pl.DataFrame,
) -> tuple[float, float]:
    """Sums fulfilled/lost over exactly the interval_start rows whose
    (station_id, hour_of_week) appears in treated_cells -- the same
    hour-of-week-only join simulator._induced_step_dicts already uses to
    decide which interval_start rows a funded move actually touches, so
    "treated" here means precisely the rows the induced move could have
    affected. treated_cells must carry station_id (the DESTINATION side of
    a funded move -- policy_baselines.FUNDED_SCHEMA's "station_id", before
    to_simulator_induced_moves renames it) and hour_of_week."""
    how_map = calendar_weather.select("interval_start", "hour_of_week").unique()
    tagged = station_intervals.join(how_map, on="interval_start", how="left")
    matched = tagged.join(
        treated_cells.select("station_id", "hour_of_week").unique(),
        on=["station_id", "hour_of_week"],
        how="inner",
    )
    fulfilled = float((matched["direct_arrivals"] + matched["rerouted_arrivals"]).sum())
    lost = float((matched["lost_no_bike"] + matched["lost_past_cap_arrivals"]).sum())
    return fulfilled, lost


TREATED_CELL_POLICIES = ("uniform", "proportional", "top_n_stockout", "allocator", "allocator_full_budget")


def run_one_replicate_treated(replicate_idx: int, artifacts: SharedArtifacts, base_seed: int = BASE_SEED) -> pl.DataFrame:
    """SAME draw as run_one_replicate (identical demand_multiplier, tier1,
    tier2, seed for this replicate_idx), but reports fill rate restricted
    to each policy's own treated cells, paired against a do-nothing run at
    the SAME cells and the SAME seed -- see the module comment above this
    function for why. do_nothing itself has no treated cells and isn't a
    row here (its role is the paired baseline, computed once and reused
    for every policy in this replicate)."""
    draw = draw_replicate(replicate_idx, artifacts, base_seed)

    dn_run, _, _ = _run_policy_simulation(
        "do_nothing", artifacts, draw.tier1, draw.tier2, artifacts.alloc_params.weekly_budget,
        draw.demand_multiplier, draw.seed,
    )

    rows = []
    for name in TREATED_CELL_POLICIES:
        run, capped, dollar_cost = _run_policy_simulation(
            name, artifacts, draw.tier1, draw.tier2, artifacts.alloc_params.weekly_budget,
            draw.demand_multiplier, draw.seed,
        )
        treated_cells = capped.select("station_id", "hour_of_week").unique()
        n_cells = treated_cells.height
        if n_cells == 0:
            fulfilled_p = lost_p = fulfilled_dn = lost_dn = 0.0
        else:
            fulfilled_p, lost_p = compute_treated_cell_fill(run.station_intervals, artifacts.week.calendar_weather, treated_cells)
            fulfilled_dn, lost_dn = compute_treated_cell_fill(dn_run.station_intervals, artifacts.week.calendar_weather, treated_cells)
        rows.append(
            {
                "replicate": replicate_idx,
                "policy": name,
                "n_treated_cells": n_cells,
                "fulfilled_treated": fulfilled_p,
                "lost_treated": lost_p,
                "fill_rate_treated": fulfilled_p / (fulfilled_p + lost_p) if (fulfilled_p + lost_p) > 0 else float("nan"),
                "fulfilled_treated_do_nothing": fulfilled_dn,
                "lost_treated_do_nothing": lost_dn,
                "fill_rate_treated_do_nothing": fulfilled_dn / (fulfilled_dn + lost_dn) if (fulfilled_dn + lost_dn) > 0 else float("nan"),
                "dollar_cost": dollar_cost,
                "demand_multiplier": draw.demand_multiplier,
                "seed": draw.seed,
            }
        )
    return pl.DataFrame(rows)


def _run_treated_task(replicate_idx: int) -> str:
    path = TREATED_DIR / f"{replicate_idx:04d}.parquet"
    if path.exists():
        return f"[policy_compare] treated replicate {replicate_idx}: already checkpointed, skipping"
    t0 = time.monotonic()
    df = run_one_replicate_treated(replicate_idx, _ARTIFACTS)
    checkpoint.write_checkpoint(df, path)
    elapsed = time.monotonic() - t0
    return f"[policy_compare] treated replicate {replicate_idx}: done in {elapsed:.0f}s, peak RSS {progress.peak_rss_mb():.0f} MB -> {path}"


def run_treated_replicates(n_replicates: int, n_workers: int) -> None:
    TREATED_DIR.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(n_replicates) if not (TREATED_DIR / f"{i:04d}.parquet").exists()]
    print(f"[policy_compare] treated-cell: {n_replicates} replicates requested, {len(todo)} not yet checkpointed, {n_workers} worker(s)")
    if not todo:
        return
    if n_workers <= 1:
        _worker_init()
        for i in todo:
            print(_run_treated_task(i))
        return
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as ex:
        for msg in ex.map(_run_treated_task, todo):
            print(msg)


def aggregate_treated_results() -> pl.DataFrame:
    files = sorted(TREATED_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no checkpointed treated-cell replicates in {TREATED_DIR}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical")


_ARTIFACTS: SharedArtifacts | None = None  # set once per worker process by _worker_init


def _worker_init() -> None:
    global _ARTIFACTS
    print(f"[policy_compare] worker loading shared artifacts, peak RSS {progress.peak_rss_mb():.0f} MB before load")
    _ARTIFACTS = load_shared_artifacts()
    print(f"[policy_compare] worker ready, peak RSS {progress.peak_rss_mb():.0f} MB after load")


def _run_replicate_task(replicate_idx: int) -> str:
    path = REPLICATES_DIR / f"{replicate_idx:04d}.parquet"
    if path.exists():
        return f"[policy_compare] replicate {replicate_idx}: already checkpointed, skipping"
    t0 = time.monotonic()
    df = run_one_replicate(replicate_idx, _ARTIFACTS)
    checkpoint.write_checkpoint(df, path)
    elapsed = time.monotonic() - t0
    return f"[policy_compare] replicate {replicate_idx}: done in {elapsed:.0f}s, peak RSS {progress.peak_rss_mb():.0f} MB -> {path}"


def run_replicates(n_replicates: int, n_workers: int) -> None:
    REPLICATES_DIR.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(n_replicates) if not (REPLICATES_DIR / f"{i:04d}.parquet").exists()]
    print(f"[policy_compare] {n_replicates} replicates requested, {len(todo)} not yet checkpointed, {n_workers} worker(s)")
    if not todo:
        return
    if n_workers <= 1:
        _worker_init()
        for i in todo:
            print(_run_replicate_task(i))
        return
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as ex:
        for msg in ex.map(_run_replicate_task, todo):
            print(msg)


def aggregate_bootstrap_results() -> pl.DataFrame:
    files = sorted(REPLICATES_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no checkpointed replicates in {REPLICATES_DIR}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical")


# ---------------------------------------------------------------------------
# Budget sweep: do_nothing + both allocator variants, $0 -> 3x weekly_budget
# (SPEC.md §8, levels redesigned post-pilot -- see BUDGET_LEVELS_DOLLARS)
# ---------------------------------------------------------------------------


def run_one_budget_replicate(replicate_idx: int, artifacts: SharedArtifacts, budget_dollars: float, base_seed: int = BASE_SEED) -> pl.DataFrame:
    """Same paired-draw design as run_one_replicate, restricted to
    BUDGET_SWEEP_POLICIES, at ONE budget level (absolute dollars, not a
    multiplier -- see BUDGET_LEVELS_DOLLARS)."""
    rng = np.random.default_rng(base_seed + replicate_idx)
    demand_multiplier = sample_demand_multiplier(rng, artifacts.residual_ratios)
    tier1, tier2 = sample_elasticity_draw(rng, artifacts.alloc_params)
    seed = base_seed + replicate_idx

    rows = []
    for name in BUDGET_SWEEP_POLICIES:
        fill, dollar_cost = _simulate_policy(name, artifacts, tier1, tier2, budget_dollars, demand_multiplier, seed)
        rows.append(
            fill.with_columns(
                pl.lit(replicate_idx).alias("replicate"),
                pl.lit(name).alias("policy"),
                pl.lit(budget_dollars).alias("weekly_budget"),
                pl.lit(dollar_cost).alias("dollar_cost"),
                pl.lit(seed).alias("seed"),
            )
        )
    return pl.concat(rows, how="vertical")


def _budget_task_path(budget_dollars: float, replicate_idx: int) -> Path:
    return BUDGET_SWEEP_DIR / f"budget{budget_dollars:08.2f}_{replicate_idx:04d}.parquet"


def _run_budget_task(task: tuple[float, int]) -> str:
    budget_dollars, replicate_idx = task
    path = _budget_task_path(budget_dollars, replicate_idx)
    if path.exists():
        return f"[policy_compare] budget ${budget_dollars:.0f} replicate {replicate_idx}: already checkpointed, skipping"
    t0 = time.monotonic()
    df = run_one_budget_replicate(replicate_idx, _ARTIFACTS, budget_dollars)
    checkpoint.write_checkpoint(df, path)
    elapsed = time.monotonic() - t0
    return f"[policy_compare] budget ${budget_dollars:.0f} replicate {replicate_idx}: done in {elapsed:.0f}s -> {path}"


def run_budget_sweep(n_replicates_per_level: int, n_workers: int) -> None:
    BUDGET_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        (level, i)
        for level in BUDGET_LEVELS_DOLLARS
        for i in range(n_replicates_per_level)
        if not _budget_task_path(level, i).exists()
    ]
    print(f"[policy_compare] budget sweep: {len(BUDGET_LEVELS_DOLLARS)} levels x {n_replicates_per_level} replicates x {len(BUDGET_SWEEP_POLICIES)} policies, {len(tasks)} tasks not yet checkpointed, {n_workers} worker(s)")
    if not tasks:
        return
    if n_workers <= 1:
        _worker_init()
        for t in tasks:
            print(_run_budget_task(t))
        return
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as ex:
        for msg in ex.map(_run_budget_task, tasks):
            print(msg)


def aggregate_budget_sweep_results() -> pl.DataFrame:
    files = sorted(BUDGET_SWEEP_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no checkpointed budget-sweep replicates in {BUDGET_SWEEP_DIR}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 9 (SPEC.md §8): policy comparison bootstrap driver.")
    p.add_argument("--n-replicates", type=int, default=10, help="bootstrap replicates for the main 6-policy comparison")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--budget-sweep", action="store_true", help="also run the do_nothing+allocator+allocator_full_budget budget sweep")
    p.add_argument("--budget-sweep-replicates", type=int, default=10, help="replicates PER budget level")
    p.add_argument("--treated-cells", action="store_true", help="also run the treated-cell paired comparison (fill rate restricted to each policy's own funded cells)")
    p.add_argument("--treated-cells-replicates", type=int, default=40, help="replicates for the treated-cell comparison")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_replicates(args.n_replicates, args.n_workers)
    aggregated = aggregate_bootstrap_results()
    checkpoint.write_checkpoint(aggregated, BOOTSTRAP_RESULTS_PATH)
    print(f"[policy_compare] {aggregated['replicate'].n_unique()} replicates aggregated -> {BOOTSTRAP_RESULTS_PATH}")

    if args.budget_sweep:
        run_budget_sweep(args.budget_sweep_replicates, args.n_workers)
        sweep = aggregate_budget_sweep_results()
        checkpoint.write_checkpoint(sweep, BUDGET_SWEEP_RESULTS_PATH)
        print(f"[policy_compare] budget sweep aggregated -> {BUDGET_SWEEP_RESULTS_PATH}")

    if args.treated_cells:
        run_treated_replicates(args.treated_cells_replicates, args.n_workers)
        treated = aggregate_treated_results()
        checkpoint.write_checkpoint(treated, TREATED_RESULTS_PATH)
        print(f"[policy_compare] treated-cell comparison aggregated -> {TREATED_RESULTS_PATH}")


if __name__ == "__main__":
    main()
