"""Phase 8 Part C: greedy marginal-value-per-dollar allocator (SPEC.md §7,
RUNBOOK Phase 8 Part C).

**Structured around the elasticity problem, not despite it.** Part B's
config/params.yaml documents, in the loudest terms this project uses
anywhere, that the elasticity curve (a, b) is a GUESS with no data source
-- trips.parquet has no payout field, so there is no way to fit it from
what this project has. Rather than produce a single dollar-denominated
allocation that would silently inherit that guess's precision, this
module's PRIMARY output is a RANKING of (station, hour, tier) targets by
net-value-per-dollar, and a stress test of that ranking across the entire
plausible elasticity range (config's `allocation.elasticity_sweep`). The
STABLE CORE -- targets that stay in the top 100 under every (a, b) draw in
the sweep -- is the actual recommendation: station-hours worth funding
regardless of what the true elasticity turns out to be. A fixed budget-
cutoff allocation is also produced (`cumulative_budget_cutoff`), but it is
explicitly secondary, since weekly_budget is an equally uncalibrated
guess (see config) and a specific cutoff inherits both guesses at once.

**Pipeline, in two cost tiers by design (candidate generation is
expensive and elasticity-independent; ranking is cheap and elasticity-
dependent, so they're separated to make the sweep tractable):**

1. `assign_tiers`: every destination-eligible cell (low_frac > 0, i.e. it
   was ever measurably low -- see DECISIONS.md's mv_k_max resolution)
   gets Tier 1 (chronic AND schedulable, DECISIONS.md's "schedulability"
   finding) or Tier 2 (everything else).
2. `qualifying_origins`: a candidate origin must be SURPLUS -- per
   CLAUDE.md's origin caveat, that means its OWN low_frac is low
   (`allocation.origin_max_low_frac`), not just that it isn't currently a
   destination target. Its cost (SPEC §7: a move costs the origin its own
   MV) is looked up from the SAME mv_curve.parquet used for destinations,
   at the origin's own worst historical level -- capped at mv_k_max, same
   as everywhere else in this project. Most qualifying origins never
   dropped into the measured (k <= mv_k_max) region at all, so their cost
   is correctly 0, not assumed 0.
3. `build_candidate_moves`: one best (lowest-cost) origin per destination,
   restricted to plausible OD pairs (same zone_agg OR real historical
   station-to-station flow at that hour, from od_shares) -- NOT
   max_move_duration_min, which this module has no calibrated travel-time
   model to apply; that gap is stated, not silently ignored. Destinations
   are PREFILTERED to the top PREFILTER_N by ceiling MV before this join
   -- the full eligible-destination population is ~265K cells, and given
   how concentrated net-lost demand is in the chronic tail (DECISIONS.md),
   prefiltering is extremely unlikely to exclude a real top-100 contender.
4. `rank_targets`: cheap, elasticity-dependent. Candidate moves are
   (origin, dest, payout_level) tuples per RUNBOOK -- payout is searched
   over a small discrete grid, not continuously optimized. Returns
   destinations ranked by net-value-per-dollar. THIS is the module's
   primary output for a single (a, b) draw.
5. `run_elasticity_sweep` / `spearman_stability` / `stable_core`: re-run
   step 4 across the full (a, b) grid (step 1-3's expensive candidate
   table is built once and reused), then measure how much the ranking
   actually moves.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.stats import spearmanr

import models.od_shares as od_shares
import opt.marginal_value as mv
import utils.checkpoint as checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "config" / "params.yaml"
STATIONS_PATH = od_shares.STATIONS_PATH

OUT_DIR = REPO_ROOT / "data" / "processed" / "allocate"
CANDIDATE_MOVES_PATH = OUT_DIR / "candidate_moves.parquet"
RANKING_PATH = OUT_DIR / "ranking_default.parquet"
BUDGET_CUTOFF_PATH = OUT_DIR / "budget_cutoff_default.parquet"
SWEEP_STABILITY_PATH = OUT_DIR / "sweep_stability.parquet"
SWEEP_APPEARANCE_PATH = OUT_DIR / "sweep_appearance.parquet"

TOP_N = 100
PREFILTER_N = 3000  # candidate destinations considered before origin-matching -- see module docstring
PAYOUT_GRID = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]  # points; a discrete search, not continuous optimization


@dataclass
class TierElasticity:
    a: float
    b: float


@dataclass
class AllocationParams:
    dollars_per_point: float
    weekly_budget: float
    elasticity_tier1: TierElasticity
    elasticity_tier2: TierElasticity
    max_induced_moves_per_station_hour: int
    max_move_duration_min: float
    origin_max_low_frac: float
    sweep_a_min: float
    sweep_a_max: float
    sweep_a_steps: int
    sweep_b_min: float
    sweep_b_max: float
    sweep_b_steps: int


def load_params(path: Path = PARAMS_PATH) -> AllocationParams:
    cfg = yaml.safe_load(path.read_text())["allocation"]
    sweep = cfg["elasticity_sweep"]
    return AllocationParams(
        dollars_per_point=float(cfg["dollars_per_point"]),
        weekly_budget=float(cfg["weekly_budget"]),
        elasticity_tier1=TierElasticity(float(cfg["elasticity_tier1_scheduled"]["a"]), float(cfg["elasticity_tier1_scheduled"]["b"])),
        elasticity_tier2=TierElasticity(float(cfg["elasticity_tier2_dynamic"]["a"]), float(cfg["elasticity_tier2_dynamic"]["b"])),
        max_induced_moves_per_station_hour=int(cfg["max_induced_moves_per_station_hour"]),
        max_move_duration_min=float(cfg["max_move_duration_min"]),
        origin_max_low_frac=float(cfg["origin_max_low_frac"]),
        sweep_a_min=float(sweep["a_min"]),
        sweep_a_max=float(sweep["a_max"]),
        sweep_a_steps=int(sweep["a_steps"]),
        sweep_b_min=float(sweep["b_min"]),
        sweep_b_max=float(sweep["b_max"]),
        sweep_b_steps=int(sweep["b_steps"]),
    )


# ---------------------------------------------------------------------------
# Tier assignment (destinations) and origin qualification
# ---------------------------------------------------------------------------


def assign_tiers(low_freq_per_cell: pl.DataFrame, chronic_timing: pl.DataFrame, mv_params: mv.MarginalValueParams) -> pl.DataFrame:
    """station_id, hour_of_week, low_frac, tier -- 'tier1_scheduled' for
    chronic (low_frac > 0.5) AND schedulable (modal_share >=
    schedulable_modal_share_threshold) cells, 'tier2_dynamic' for every
    other cell that was ever measurably low (low_frac > 0). Cells with
    low_frac == 0 are excluded -- they've never been in the measured
    (k <= mv_k_max) region, so this model has no nonzero MV to offer them
    as destinations (DECISIONS.md's mv_k_max resolution)."""
    joined = low_freq_per_cell.filter(pl.col("low_frac") > 0.0).join(
        chronic_timing.select("station_id", "hour_of_week", "modal_share"),
        on=["station_id", "hour_of_week"],
        how="left",
    )
    return joined.with_columns(
        pl.when((pl.col("low_frac") > 0.5) & (pl.col("modal_share") >= mv_params.schedulable_modal_share_threshold))
        .then(pl.lit("tier1_scheduled"))
        .otherwise(pl.lit("tier2_dynamic"))
        .alias("tier")
    ).select("station_id", "hour_of_week", "low_frac", "tier")


def qualifying_origins(
    weekly: pl.DataFrame, low_freq_per_cell: pl.DataFrame, mv_curve: pl.DataFrame, mv_params: mv.MarginalValueParams, params: AllocationParams
) -> pl.DataFrame:
    """station_id, hour_of_week, min_inventory, origin_cost. Per CLAUDE.md's
    origin caveat: only counts as a surplus origin if its OWN low_frac is
    <= origin_max_low_frac (rarely low itself, not just "not currently a
    destination"). origin_cost applies the SAME k <= mv_k_max caveat used
    for destinations, at the origin's own worst historical level
    (min_inventory, rounded, capped at mv_k_max) -- looked up from
    mv_curve.parquet if that level was ever in the measured region, else
    0. Most qualifying origins (low_frac already small) will have
    min_inventory > mv_k_max and correctly cost 0; the few borderline
    ones (occasionally dip low despite mostly being surplus) get a real,
    measured, nonzero cost instead of an assumed-zero one."""
    min_inv = weekly.group_by("station_id", "hour_of_week").agg(pl.col("mean_inventory").min().alias("min_inventory"))
    origins = low_freq_per_cell.filter(pl.col("low_frac") <= params.origin_max_low_frac).join(
        min_inv, on=["station_id", "hour_of_week"], how="left"
    )
    origins = origins.with_columns(
        pl.min_horizontal(pl.col("min_inventory").round(0), pl.lit(float(mv_params.mv_k_max)))
        .clip(lower_bound=1.0)
        .cast(pl.Int64)
        .alias("k_lookup")
    )
    origins = origins.join(
        mv_curve.rename({"k": "k_lookup", "mv": "origin_cost"}),
        on=["station_id", "hour_of_week", "k_lookup"],
        how="left",
    ).with_columns(pl.col("origin_cost").fill_null(0.0))
    return origins.select("station_id", "hour_of_week", "min_inventory", "origin_cost")


# ---------------------------------------------------------------------------
# Candidate move generation (expensive, elasticity-independent -- built once)
# ---------------------------------------------------------------------------


def build_dest_cumulative_mv(mv_curve: pl.DataFrame) -> pl.DataFrame:
    """station_id, hour_of_week, k, cum_mv, k_max. cum_mv(k) = MV(1) +
    MV(2) + ... + MV(k): the cumulative value of delivering k bikes to
    this destination, using the ACTUAL concave curve from mv_curve.parquet
    (not a single flat per-trip value -- see rank_targets' docstring for
    why that matters). k_max is the largest k actually measured for that
    cell (<= mv_k_max); callers must saturate cum_mv at k_max's value for
    any k beyond it, since MV isn't measured (and per DECISIONS.md, is
    negligible) past that point."""
    sorted_curve = mv_curve.sort(["station_id", "hour_of_week", "k"])
    with_cum = sorted_curve.with_columns(
        pl.col("mv").cum_sum().over(["station_id", "hour_of_week"]).alias("cum_mv")
    )
    k_max = with_cum.group_by("station_id", "hour_of_week").agg(pl.col("k").max().alias("k_max"))
    return with_cum.join(k_max, on=["station_id", "hour_of_week"], how="left").select(
        "station_id", "hour_of_week", "k", "cum_mv", "k_max"
    )


def build_candidate_moves(
    tiers: pl.DataFrame,
    dest_cum_mv: pl.DataFrame,
    origins: pl.DataFrame,
    stations: pl.DataFrame,
    od_model: od_shares.ODShareModel,
    prefilter_n: int = PREFILTER_N,
) -> pl.DataFrame:
    """One row per destination with its single best (lowest-cost, then
    highest-real-flow) qualifying origin. See module docstring for the
    prefiltering and plausible-OD-pair rationale. Prefilters by k_max's
    cum_mv (the destination's full-saturation ceiling value) as a ranking
    proxy for "how valuable could this destination possibly be" -- the
    actual per-draw value used at ranking time is elasticity-dependent
    (see rank_targets) and computed fresh from dest_cum_mv, not from this
    prefilter score. Returns station_id (dest), hour_of_week, tier,
    low_frac, k_max, origin_station_id, origin_cost, flow_prob (0 if the
    pair has no station-level historical flow -- same_zone_agg pairs are
    still included via the OR filter)."""
    ceiling = dest_cum_mv.sort(["station_id", "hour_of_week", "k"]).group_by(
        ["station_id", "hour_of_week"], maintain_order=True
    ).last().select("station_id", "hour_of_week", "cum_mv", "k_max")
    dest = tiers.join(ceiling, on=["station_id", "hour_of_week"], how="inner")
    dest = dest.sort("cum_mv", descending=True).head(prefilter_n)

    dest = dest.join(stations, on="station_id", how="left").rename({"zone_agg": "dest_zone_agg"})
    origins_z = origins.join(stations, on="station_id", how="left").rename(
        {"station_id": "origin_station_id", "zone_agg": "origin_zone_agg"}
    )

    pairs = dest.join(origins_z, on="hour_of_week", how="inner")
    pairs = pairs.filter(pl.col("origin_station_id") != pl.col("station_id"))

    flow = od_model.station_hour_probs.rename({"start_station_id": "origin_station_id", "end_station_id": "station_id"})
    pairs = pairs.join(
        flow.select("origin_station_id", "hour_of_week", "station_id", "prob"),
        on=["origin_station_id", "hour_of_week", "station_id"],
        how="left",
    ).with_columns(pl.col("prob").fill_null(0.0).alias("flow_prob"))

    plausible = pairs.filter((pl.col("origin_zone_agg") == pl.col("dest_zone_agg")) | (pl.col("flow_prob") > 0))

    best_origin = (
        plausible.sort(
            ["station_id", "hour_of_week", "origin_cost", "flow_prob"],
            descending=[False, False, False, True],
        )
        .group_by(["station_id", "hour_of_week"], maintain_order=True)
        .first()
    )
    return best_origin.select(
        "station_id", "hour_of_week", "tier", "low_frac", "k_max",
        "origin_station_id", "origin_cost", "flow_prob",
    )


# ---------------------------------------------------------------------------
# Ranking (cheap, elasticity-dependent -- re-run per sweep draw)
# ---------------------------------------------------------------------------


def rank_targets(
    candidate_moves: pl.DataFrame,
    dest_cum_mv: pl.DataFrame,
    tier1: TierElasticity,
    tier2: TierElasticity,
    params: AllocationParams,
    payout_grid: list[float] = PAYOUT_GRID,
) -> pl.DataFrame:
    """Evaluates net-value-per-dollar for each candidate move across
    payout_grid (candidate moves are (origin, dest, payout_level) tuples,
    RUNBOOK Phase 8 Part C), keeps each move's best payout level, and
    returns ALL candidate destinations ranked by net-value-per-dollar
    descending -- the PRIMARY output of this module for one (tier1,
    tier2) elasticity pair. Not dependent on weekly_budget at all; that
    only decides a cutoff, computed separately.

    Trips-value uses the destination's CUMULATIVE, SATURATING mv curve
    (build_dest_cumulative_mv), not a single flat per-trip value -- this
    is deliberate, not cosmetic. With a flat per-trip value, both
    trips_value and dollar_cost scale linearly in induced_trips, so
    induced_trips cancels out of the trips_value/dollar_cost ratio
    algebraically and net_value_per_dollar stops depending on the
    elasticity curve (a, b) AT ALL -- confirmed directly (an earlier
    version of this function did exactly that, and every elasticity draw
    in the sweep produced an IDENTICAL ranking, Spearman 1.000 with zero
    variation -- a formula artifact caught by checking the sweep's
    suspiciously perfect result before trusting it, not a real robustness
    finding). Using cumulative, k-capped value restores genuine
    diminishing returns: value saturates hard at k_max (consistent with
    the k<=mv_k_max finding), cost keeps rising linearly with payout, so
    there is a real interior trade-off and the elasticity curve genuinely
    determines which payout level -- and therefore which targets -- rank
    highest."""
    grid = pl.concat(
        [candidate_moves.with_columns(pl.lit(p).alias("payout")) for p in payout_grid], how="vertical"
    )
    grid = grid.with_columns(
        pl.when(pl.col("tier") == "tier1_scheduled").then(pl.lit(tier1.a)).otherwise(pl.lit(tier2.a)).alias("_a"),
        pl.when(pl.col("tier") == "tier1_scheduled").then(pl.lit(tier1.b)).otherwise(pl.lit(tier2.b)).alias("_b"),
    )
    grid = grid.with_columns(
        (pl.col("_a") * (1.0 - (-pl.col("_b") * pl.col("payout")).exp())).alias("induced_trips")
    ).drop("_a", "_b")
    grid = grid.with_columns(
        pl.min_horizontal(pl.col("induced_trips").round(0), pl.col("k_max"))
        .clip(lower_bound=1.0)
        .cast(pl.Int64)
        .alias("k_lookup")
    )
    grid = grid.join(
        dest_cum_mv.select("station_id", "hour_of_week", "k", "cum_mv").rename({"k": "k_lookup"}),
        on=["station_id", "hour_of_week", "k_lookup"],
        how="left",
    )
    grid = grid.with_columns(
        (pl.col("cum_mv") - pl.col("induced_trips") * pl.col("origin_cost")).alias("trips_value"),
        (pl.col("induced_trips") * pl.col("payout") * params.dollars_per_point).alias("dollar_cost"),
    )
    grid = grid.filter(pl.col("dollar_cost") > 0)
    grid = grid.with_columns(
        (pl.col("trips_value") - pl.col("dollar_cost")).alias("net_value"),
        ((pl.col("trips_value") - pl.col("dollar_cost")) / pl.col("dollar_cost")).alias("net_value_per_dollar"),
    )
    best_per_move = (
        grid.sort(["station_id", "hour_of_week", "net_value_per_dollar"], descending=[False, False, True])
        .group_by(["station_id", "hour_of_week"], maintain_order=True)
        .first()
    )
    ranked = best_per_move.sort("net_value_per_dollar", descending=True).with_row_index("rank", offset=1)
    return ranked


def cumulative_budget_cutoff(ranking: pl.DataFrame, weekly_budget: float) -> pl.DataFrame:
    """SECONDARY output, explicitly provisional: how far down the ranking
    the CONFIGURED (guessed) weekly_budget reaches, via cumulative
    dollar_cost. Inherits both the elasticity guess AND the budget guess
    at once -- the ranking above, not this cutoff, is what should be
    trusted across elasticity uncertainty."""
    return ranking.with_columns(pl.col("dollar_cost").cum_sum().alias("cumulative_cost")).filter(
        pl.col("cumulative_cost") <= weekly_budget
    )


# ---------------------------------------------------------------------------
# Elasticity sweep and stability analysis
# ---------------------------------------------------------------------------


def elasticity_grid(params: AllocationParams) -> list[tuple[float, float]]:
    a_vals = np.linspace(params.sweep_a_min, params.sweep_a_max, params.sweep_a_steps)
    b_vals = np.linspace(params.sweep_b_min, params.sweep_b_max, params.sweep_b_steps)
    return [(float(a), float(b)) for a, b in itertools.product(a_vals, b_vals)]


def run_elasticity_sweep(
    candidate_moves: pl.DataFrame,
    dest_cum_mv: pl.DataFrame,
    params: AllocationParams,
    payout_grid: list[float] = PAYOUT_GRID,
    top_n: int = TOP_N,
) -> dict[tuple[float, float], pl.DataFrame]:
    """For each (a, b) draw in the sweep grid, applied UNIFORMLY to both
    tiers (testing "what if we're wrong about the specific numbers,"
    not re-deriving the tier1-more-conservative-than-tier2 relationship
    for every draw -- see module docstring), produces a fresh ranking and
    keeps its top_n. The expensive candidate_moves/dest_cum_mv tables are
    built ONCE by the caller and reused across every draw here."""
    results = {}
    for a, b in elasticity_grid(params):
        te = TierElasticity(a=a, b=b)
        ranked = rank_targets(candidate_moves, dest_cum_mv, te, te, params, payout_grid)
        results[(a, b)] = ranked.head(top_n)
    return results


def spearman_stability(sweep_results: dict[tuple[float, float], pl.DataFrame]) -> pl.DataFrame:
    """Pairwise Spearman rank correlation between every pair of draws'
    top-N rankings, restricted to targets common to both (a target
    outside one draw's top-N has no rank there to correlate). One row per
    draw pair -- summarize with .select(pl.col("spearman").min/median/max())
    for the direct answer to "how stable is the ranking across the
    plausible elasticity range.\""""
    keys = list(sweep_results.keys())
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            df_i = sweep_results[keys[i]].select("station_id", "hour_of_week", "rank").rename({"rank": "rank_i"})
            df_j = sweep_results[keys[j]].select("station_id", "hour_of_week", "rank").rename({"rank": "rank_j"})
            common = df_i.join(df_j, on=["station_id", "hour_of_week"], how="inner")
            if common.height < 2:
                corr = float("nan")
            else:
                corr, _p = spearmanr(common["rank_i"].to_numpy(), common["rank_j"].to_numpy())
            rows.append(
                {
                    "a_i": keys[i][0], "b_i": keys[i][1], "a_j": keys[j][0], "b_j": keys[j][1],
                    "n_common": common.height, "spearman": float(corr),
                }
            )
    return pl.DataFrame(rows)


def stable_core(sweep_results: dict[tuple[float, float], pl.DataFrame]) -> pl.DataFrame:
    """Every target that appears in ANY draw's top-N, with
    appearance_frac = fraction of draws it appeared in. appearance_frac
    == 1.0 (every single draw) is the STABLE CORE -- per this phase's
    framing, the actual recommendation: station-hours worth funding
    regardless of what the true elasticity is. Lower fractions show
    exactly which targets "move around" instead of being buried."""
    n_draws = len(sweep_results)
    all_appearances = pl.concat(
        [df.select("station_id", "hour_of_week", "tier") for df in sweep_results.values()], how="vertical"
    )
    counts = all_appearances.group_by("station_id", "hour_of_week", "tier").agg(pl.len().alias("n_appearances"))
    return counts.with_columns((pl.col("n_appearances") / n_draws).alias("appearance_frac")).sort(
        "appearance_frac", descending=True
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    mv_params = mv.load_params()
    alloc_params = load_params()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weekly = mv.finalize_station_hour_week()
    low_freq_per_cell, _buckets = mv.eligibility_frequency_report(weekly, mv_params)
    chronic_timing = pl.read_parquet(mv.CHRONIC_TIMING_PATH)
    mv_curve = pl.read_parquet(mv.MV_CURVE_PATH)
    stations = pl.read_parquet(STATIONS_PATH).select("station_id", pl.col("zone_agglomerative").alias("zone_agg"))
    od_model = od_shares.load_od_share_model()

    print("[allocate] assigning tiers + qualifying origins...")
    tiers = assign_tiers(low_freq_per_cell, chronic_timing, mv_params)
    origins = qualifying_origins(weekly, low_freq_per_cell, mv_curve, mv_params, alloc_params)
    print(f"[allocate] {tiers.height:,} destination-eligible cells, {origins.height:,} qualifying origins")

    dest_cum_mv = build_dest_cumulative_mv(mv_curve)

    print(f"[allocate] building candidate moves (prefiltered to top {PREFILTER_N} destinations by ceiling MV)...")
    candidate_moves = build_candidate_moves(tiers, dest_cum_mv, origins, stations, od_model)
    checkpoint.write_checkpoint(candidate_moves, CANDIDATE_MOVES_PATH)
    print(f"[allocate] {candidate_moves.height:,} candidate moves -> {CANDIDATE_MOVES_PATH}")

    print("[allocate] ranking at the default (config) elasticity...")
    default_ranking = rank_targets(
        candidate_moves, dest_cum_mv, alloc_params.elasticity_tier1, alloc_params.elasticity_tier2, alloc_params
    )
    checkpoint.write_checkpoint(default_ranking, RANKING_PATH)
    print(default_ranking.head(20))

    cutoff = cumulative_budget_cutoff(default_ranking, alloc_params.weekly_budget)
    checkpoint.write_checkpoint(cutoff, BUDGET_CUTOFF_PATH)
    print(
        f"[allocate] SECONDARY, provisional: at the guessed weekly_budget (${alloc_params.weekly_budget:,.0f}), "
        f"{cutoff.height:,} of {default_ranking.height:,} ranked targets would be funded -> {BUDGET_CUTOFF_PATH}"
    )

    grid = elasticity_grid(alloc_params)
    print(f"[allocate] sweeping elasticity: {len(grid)} draws (a in [{alloc_params.sweep_a_min},{alloc_params.sweep_a_max}], "
          f"b in [{alloc_params.sweep_b_min},{alloc_params.sweep_b_max}])...")
    sweep_results = run_elasticity_sweep(candidate_moves, dest_cum_mv, alloc_params)

    stability = spearman_stability(sweep_results)
    checkpoint.write_checkpoint(stability, SWEEP_STABILITY_PATH)
    sp = stability["spearman"].drop_nulls().drop_nans()
    print(
        f"[allocate] Spearman rank correlation across {len(sweep_results)} draws "
        f"({stability.height:,} pairs): min={sp.min():.3f} median={sp.median():.3f} max={sp.max():.3f} -> {SWEEP_STABILITY_PATH}"
    )

    core = stable_core(sweep_results)
    checkpoint.write_checkpoint(core, SWEEP_APPEARANCE_PATH)
    always = core.filter(pl.col("appearance_frac") == 1.0)
    print(
        f"[allocate] STABLE CORE (appear in every draw's top {TOP_N}): {always.height:,} targets -- "
        f"the actual recommendation, independent of the true elasticity -> {SWEEP_APPEARANCE_PATH}"
    )
    print(always.select("station_id", "hour_of_week", "tier").head(20))


if __name__ == "__main__":
    main()
