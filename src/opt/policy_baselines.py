"""Phase 9 (SPEC.md §8, RUNBOOK Phase 9): the 4 non-"do-nothing" incentive
policies plus a thin wrapper around Phase 8's optimizer, all built on the
SAME candidate-move universe Phase 8's pipeline already produced and
cached to disk (`src/opt/allocate.py`'s `candidate_moves.parquet`) --
loaded ONCE by `load_shared_candidate_universe`, reused by every policy and
every bootstrap replicate in `src/sim/policy_compare.py`.

**Scoping decision, stated here since it isn't inherited from Phase 8:**
baselines 2-4 (uniform, proportional, top-N) are restricted to this SAME
2,998-cell, tier-assigned, origin-matched candidate pool -- NOT the full
~265K `eligible_cells` population -- because origin-matching the full
population at bootstrap scale (potentially hundreds of replicates) is not
tractable, and using the same candidate pool as the optimizer keeps the
comparison apples-to-apples: same set of possible targets, only the
selection/weighting method differs across policies. A "spend across
everything" baseline computed against a different, unmatched-OD-pair
universe would introduce an uncontrolled confound into the fill-rate
comparison, not just a different policy.

**Physical-plausibility cap applied HERE, not in Phase 8:**
config/params.yaml's `allocation.max_induced_moves_per_station_hour` (=3,
the real p90 arrival-throughput-grounded ceiling) is loaded into
`allocate.AllocationParams` but never actually enforced anywhere in
`allocate.rank_targets`/`cumulative_budget_cutoff` -- confirmed by reading
that module, not assumed. This module is the first place induced trips
become actual bike movements in a capacity-constrained network simulation,
so `apply_move_cap` clips `induced_trips` to that cap for every funded
policy, AFTER budget selection (dollar_cost reflects what was actually
paid for; the cap only bounds what the network can plausibly realize --
paying for more than the cap can absorb is a real policy inefficiency,
not something to silently rescale away). This is a Phase-9-specific
correction, not a retroactive fix to Phase 8's ranking formula.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

import opt.allocate as allocate
import opt.marginal_value as mv

REPO_ROOT = Path(__file__).resolve().parents[2]

FUNDED_SCHEMA = {
    "station_id": pl.String,
    "hour_of_week": pl.Int16,
    "tier": pl.String,
    "origin_station_id": pl.String,
    "induced_trips": pl.Float64,
    "payout": pl.Float64,
    "dollar_cost": pl.Float64,
}

# Naive top-N policy: SOURCE: NONE, same honesty convention as the rest of
# this repo's money assumptions (config/params.yaml). A flat, non-optimized
# payout is precisely what makes this baseline naive -- an ops person
# reading a stockout-frequency leaderboard wouldn't run a payout-value
# search per station, they'd pick one middling incentive and fund down the
# list until the money runs out. The middle of the existing PAYOUT_GRID
# (allocate.PAYOUT_GRID = [1,2,5,10,20,30,50]) is as good a "middling"
# choice as any single number here -- not fit to data, by design.
TOP_N_STOCKOUT_FLAT_PAYOUT = 10.0  # points

# Fine grid for inverting dollar_cost(payout) per tier -- dollar_cost(payout)
# = a*(1-exp(-b*payout))*payout*dollars_per_point is monotonic increasing in
# payout (product of two nonnegative, increasing-from-zero functions), so
# np.interp against this precomputed grid solves "the payout that spends $X
# at this cell" for an entire array of cells/targets at once, no per-cell
# root-finding loop. Upper bound generous vs. the existing discrete
# PAYOUT_GRID's max (50) since baselines 2-3 spread budget across far fewer
# cells than the optimizer's ranking would concentrate on, so a single
# cell's target spend can be larger -- especially at the 3x budget-sweep
# level (SPEC.md §8).
_PAYOUT_INVERSION_GRID = np.linspace(0.0, 500.0, 20_000)


def _dollar_cost_curve(a: float, b: float, dollars_per_point: float) -> np.ndarray:
    payout = _PAYOUT_INVERSION_GRID
    induced_trips = a * (1.0 - np.exp(-b * payout))
    return induced_trips * payout * dollars_per_point


def _invert_payout_for_target_dollars(
    target_dollars: np.ndarray | float, a: float, b: float, dollars_per_point: float
) -> np.ndarray:
    """Vectorized inverse of dollar_cost(payout) for a FIXED (a, b, d) --
    used per-tier for policy_uniform (one scalar target) and per-cell for
    policy_proportional (an array of targets, one per cell, same tier)."""
    curve = _dollar_cost_curve(a, b, dollars_per_point)
    return np.interp(target_dollars, curve, _PAYOUT_INVERSION_GRID)


def _empty_funded() -> pl.DataFrame:
    return pl.DataFrame(schema=FUNDED_SCHEMA)


# ---------------------------------------------------------------------------
# Shared candidate universe (loaded once, reused across policies + replicates)
# ---------------------------------------------------------------------------


@dataclass
class SharedCandidateUniverse:
    candidate_moves: pl.DataFrame  # allocate.py's cached candidate_moves.parquet, enriched with mu/p_stockout_empirical
    dest_cum_mv: pl.DataFrame


def load_shared_candidate_universe() -> SharedCandidateUniverse:
    """Reads Phase 8's already-built, already-cached outputs directly
    (candidate_moves.parquet, mv_curve.parquet) rather than re-running
    assign_tiers/qualifying_origins/build_candidate_moves -- that pipeline
    is exactly the "expensive, elasticity-independent, built once" half
    allocate.py's own docstring describes, and it was already built once
    for the Phase 8 default run. dest_cum_mv itself isn't cached (cheap to
    rebuild from mv_curve.parquet, O(cells x k_max) rows)."""
    candidate_moves = pl.read_parquet(allocate.CANDIDATE_MOVES_PATH)
    station_hour = pl.read_parquet(mv.STATION_HOUR_PATH).select(
        "station_id", "hour_of_week", "mu", "p_stockout_empirical"
    )
    enriched = candidate_moves.join(station_hour, on=["station_id", "hour_of_week"], how="left")
    mv_curve = pl.read_parquet(mv.MV_CURVE_PATH)
    dest_cum_mv = allocate.build_dest_cumulative_mv(mv_curve)
    return SharedCandidateUniverse(candidate_moves=enriched, dest_cum_mv=dest_cum_mv)


# ---------------------------------------------------------------------------
# The 5 policies (SPEC.md §8)
# ---------------------------------------------------------------------------


def policy_do_nothing() -> pl.DataFrame:
    """Baseline #1: no incentive spend at all. Baseline non-trip movement
    (Phase 4's inferred N) is still applied by the simulator -- it always
    is, regardless of policy."""
    return _empty_funded()


def policy_uniform(
    universe: SharedCandidateUniverse,
    tier1: allocate.TierElasticity,
    tier2: allocate.TierElasticity,
    alloc_params: allocate.AllocationParams,
    weekly_budget: float,
) -> pl.DataFrame:
    """Baseline #2: budget split EQUALLY (in dollars) across every
    candidate cell. Solved as one payout-inversion per tier (not per
    payout-grid search -- baselines don't do value optimization), then
    applied uniformly to every cell of that tier."""
    cm = universe.candidate_moves
    if cm.height == 0:
        return _empty_funded()
    target_per_cell = weekly_budget / cm.height

    payout_tier1 = float(_invert_payout_for_target_dollars(target_per_cell, tier1.a, tier1.b, alloc_params.dollars_per_point))
    payout_tier2 = float(_invert_payout_for_target_dollars(target_per_cell, tier2.a, tier2.b, alloc_params.dollars_per_point))

    out = cm.with_columns(
        pl.when(pl.col("tier") == "tier1_scheduled").then(pl.lit(payout_tier1)).otherwise(pl.lit(payout_tier2)).alias("payout"),
        pl.when(pl.col("tier") == "tier1_scheduled").then(pl.lit(tier1.a)).otherwise(pl.lit(tier2.a)).alias("_a"),
        pl.when(pl.col("tier") == "tier1_scheduled").then(pl.lit(tier1.b)).otherwise(pl.lit(tier2.b)).alias("_b"),
    ).with_columns(
        (pl.col("_a") * (1.0 - (-pl.col("_b") * pl.col("payout")).exp())).alias("induced_trips")
    ).with_columns(
        (pl.col("induced_trips") * pl.col("payout") * alloc_params.dollars_per_point).alias("dollar_cost")
    )
    return out.select(list(FUNDED_SCHEMA))


def policy_proportional(
    universe: SharedCandidateUniverse,
    tier1: allocate.TierElasticity,
    tier2: allocate.TierElasticity,
    alloc_params: allocate.AllocationParams,
    weekly_budget: float,
) -> pl.DataFrame:
    """Baseline #3: budget split proportional to each cell's OWN historical
    trip volume -- `mu` (arrival rate at the destination cell,
    station_hour.parquet, the destination's own demand rate) is the
    natural read of SPEC.md §8's "trip volume." Per-cell payout inversion
    is vectorized within each tier (one dollar_cost(payout) curve per
    tier, np.interp against the whole tier's target array at once)."""
    cm = universe.candidate_moves.with_columns(pl.col("mu").fill_null(0.0))
    total_w = float(cm["mu"].sum())
    if total_w <= 0.0 or cm.height == 0:
        return _empty_funded()
    cm = cm.with_columns((pl.col("mu") / total_w * weekly_budget).alias("_target_dollars"))

    out_frames = []
    for tier_name, elas in (("tier1_scheduled", tier1), ("tier2_dynamic", tier2)):
        tier_rows = cm.filter(pl.col("tier") == tier_name)
        if tier_rows.height == 0:
            continue
        targets = tier_rows["_target_dollars"].to_numpy()
        payouts = _invert_payout_for_target_dollars(targets, elas.a, elas.b, alloc_params.dollars_per_point)
        induced = elas.a * (1.0 - np.exp(-elas.b * payouts))
        dollar_cost = induced * payouts * alloc_params.dollars_per_point
        out_frames.append(
            tier_rows.with_columns(
                pl.Series("payout", payouts, dtype=pl.Float64),
                pl.Series("induced_trips", induced, dtype=pl.Float64),
                pl.Series("dollar_cost", dollar_cost, dtype=pl.Float64),
            )
        )
    if not out_frames:
        return _empty_funded()
    out = pl.concat(out_frames, how="vertical")
    return out.select(list(FUNDED_SCHEMA))


def policy_top_n_stockout(
    universe: SharedCandidateUniverse,
    tier1: allocate.TierElasticity,
    tier2: allocate.TierElasticity,
    alloc_params: allocate.AllocationParams,
    weekly_budget: float,
    flat_payout: float = TOP_N_STOCKOUT_FLAT_PAYOUT,
) -> pl.DataFrame:
    """Baseline #4, the naive "obvious" policy: rank candidate cells by
    `p_stockout_empirical` (stockout FREQUENCY -- exactly what SPEC.md §8
    names) descending, fund a FIXED flat payout per cell in rank order
    until the budget runs out (`allocate.cumulative_budget_cutoff`'s same
    cumulative-cost-cutoff logic, reused directly). No value-per-dollar
    search -- that omission is precisely what makes this baseline naive,
    unlike policy 5."""
    cm = universe.candidate_moves.with_columns(pl.col("p_stockout_empirical").fill_null(0.0))
    if cm.height == 0:
        return _empty_funded()
    ranked = cm.sort("p_stockout_empirical", descending=True).with_columns(
        pl.when(pl.col("tier") == "tier1_scheduled").then(pl.lit(tier1.a)).otherwise(pl.lit(tier2.a)).alias("_a"),
        pl.when(pl.col("tier") == "tier1_scheduled").then(pl.lit(tier1.b)).otherwise(pl.lit(tier2.b)).alias("_b"),
        pl.lit(flat_payout).alias("payout"),
    ).with_columns(
        (pl.col("_a") * (1.0 - (-pl.col("_b") * pl.col("payout")).exp())).alias("induced_trips")
    ).with_columns(
        (pl.col("induced_trips") * pl.col("payout") * alloc_params.dollars_per_point).alias("dollar_cost")
    )
    funded = allocate.cumulative_budget_cutoff(ranked, weekly_budget)
    return funded.select(list(FUNDED_SCHEMA))


def policy_allocator(
    universe: SharedCandidateUniverse,
    tier1: allocate.TierElasticity,
    tier2: allocate.TierElasticity,
    alloc_params: allocate.AllocationParams,
    weekly_budget: float,
) -> pl.DataFrame:
    """Baseline #5, "our optimizer": a thin wrapper around Phase 8's own
    rank_targets + cumulative_budget_cutoff, re-run per bootstrap
    replicate's elasticity draw -- candidate generation is the expensive,
    elasticity-independent part (already loaded once via
    load_shared_candidate_universe); ranking is cheap and elasticity-
    dependent, exactly the split allocate.py's own docstring describes."""
    if universe.candidate_moves.height == 0:
        return _empty_funded()
    ranking = allocate.rank_targets(universe.candidate_moves, universe.dest_cum_mv, tier1, tier2, alloc_params)
    funded = allocate.cumulative_budget_cutoff(ranking, weekly_budget)
    return funded.select(list(FUNDED_SCHEMA))


def policy_allocator_full_budget(
    universe: SharedCandidateUniverse,
    tier1: allocate.TierElasticity,
    tier2: allocate.TierElasticity,
    alloc_params: allocate.AllocationParams,
    weekly_budget: float,
    payout_grid: list[float] = allocate.PAYOUT_GRID,
) -> pl.DataFrame:
    """Baseline #6, budget-EXHAUSTING variant of policy_allocator -- added
    after the Phase 9 pilot showed the default allocator spends only
    ~3-4% of the $10k weekly_budget (all 2,998 candidates funded at their
    own individually-optimal payout costs ~$310-383 total; the ranking
    NEVER exhausts a budget this large -- confirmed directly from
    ranking_default.parquet, not assumed: net-value-per-dollar declines
    smoothly from ~12.4 (top decile) to ~2.45 (bottom decile), still
    positive (0.81) at rank 2998 -- there's no collapse, the pool just
    runs out). policy_allocator answers "how much does the allocator's own
    per-candidate optimum actually need"; this answers "how much lift is
    available if the full budget is spent, prioritized the same way the
    allocator already prioritizes."

    Keeps rank_targets' own priority order untouched (still highest
    net-value-per-dollar first) and funds every candidate at its own best
    single-grid payout exactly as policy_allocator does, via the same
    cumulative_budget_cutoff. If budget remains afterward (which, per the
    finding above, it currently always does), makes repeated top-down
    passes over that SAME ranked list, escalating each candidate to the
    next payout_grid level it can still afford, one rung per candidate per
    pass -- round-robin by priority, so the best-ranked candidates get
    first claim on each round of additional spend rather than the budget
    being dumped entirely into the single best candidate. Repeats until no
    candidate can afford its next rung (budget exhausted) or every
    candidate is maxed at payout_grid's top."""
    if universe.candidate_moves.height == 0:
        return _empty_funded()
    sorted_grid = sorted(payout_grid)
    ranking = allocate.rank_targets(universe.candidate_moves, universe.dest_cum_mv, tier1, tier2, alloc_params, payout_grid=sorted_grid)
    funded = allocate.cumulative_budget_cutoff(ranking, weekly_budget)
    if funded.height == 0:
        return _empty_funded()

    remaining = float(weekly_budget - funded["dollar_cost"].sum())
    records = funded.select(list(FUNDED_SCHEMA)).to_dicts()

    progressed = remaining > 0
    while progressed:
        progressed = False
        for rec in records:
            if remaining <= 0:
                break
            cur_idx = sorted_grid.index(rec["payout"])
            if cur_idx + 1 >= len(sorted_grid):
                continue
            next_payout = sorted_grid[cur_idx + 1]
            a = tier1.a if rec["tier"] == "tier1_scheduled" else tier2.a
            b = tier1.b if rec["tier"] == "tier1_scheduled" else tier2.b
            next_induced = a * (1.0 - np.exp(-b * next_payout))
            next_cost = next_induced * next_payout * alloc_params.dollars_per_point
            incremental = next_cost - rec["dollar_cost"]
            if 0.0 < incremental <= remaining:
                remaining -= incremental
                rec["payout"] = next_payout
                rec["induced_trips"] = next_induced
                rec["dollar_cost"] = next_cost
                progressed = True

    out = pl.DataFrame(records, schema=FUNDED_SCHEMA) if records else _empty_funded()
    return out


# ---------------------------------------------------------------------------
# Physical-plausibility cap + simulator schema adapter
# ---------------------------------------------------------------------------


def apply_move_cap(funded: pl.DataFrame, max_induced_moves_per_station_hour: float) -> pl.DataFrame:
    """See module docstring's "physical-plausibility cap" note. dollar_cost
    is left untouched (that's what was actually paid for); only
    induced_trips -- what the simulator is asked to physically realize --
    is clipped."""
    if funded.height == 0:
        return funded
    return funded.with_columns(
        pl.min_horizontal(pl.col("induced_trips"), pl.lit(float(max_induced_moves_per_station_hour))).alias("induced_trips")
    )


def to_simulator_induced_moves(funded: pl.DataFrame) -> pl.DataFrame | None:
    """Adapts a funded-policy table (station_id, hour_of_week, tier,
    origin_station_id, induced_trips, payout, dollar_cost) to
    src/sim/simulator.py's run_simulation(induced_moves=...) schema
    (origin_station_id, dest_station_id, hour_of_week,
    induced_trips_per_hour). Returns None (not an empty DataFrame) for an
    empty/do-nothing policy -- run_simulation's induced_moves parameter
    treats None as "no injection at all," its own no-op default."""
    if funded.height == 0:
        return None
    return funded.rename({"station_id": "dest_station_id", "induced_trips": "induced_trips_per_hour"}).select(
        "origin_station_id", "dest_station_id", "hour_of_week", "induced_trips_per_hour"
    )
