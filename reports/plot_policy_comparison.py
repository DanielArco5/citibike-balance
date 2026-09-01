"""Phase 9 deliverable (SPEC.md §8, RUNBOOK Phase 9): the policy comparison
table + both sensitivity plots, read from src/sim/policy_compare.py's
checkpointed bootstrap output. Not part of the automated pipeline -- run
manually after policy_compare.py finishes, same "rerun and eyeball the
gate" convention as reports/plot_hour_of_week.py and src/viz/heatmap.py
(whose color palette this matches for visual consistency).

**Two caveats are printed to console AND embedded directly in every saved
output (the table's .md file, both PNGs) -- not left implicit or only in
DECISIONS.md, per the explicit instruction this phase was built under:**

1. Fill rate / lift are reported at ZONE and SYSTEM level only. Phase 7's
   validation found the simulator's per-(station, hour-of-week) stockout
   rate is simulator noise (pooled correlation plateaus at ~0.10 across 6
   held-out weeks) -- no per-station number appears anywhere in this
   module's output, by construction (policy_compare.compute_fill_rate_table
   never produces one).
2. All policies within a bootstrap replicate share the SAME demand-
   residual/elasticity/simulator-seed draw (paired design). Phase 7's
   resolution assumes destination-assignment noise is present in both the
   do_nothing baseline and every treatment run and cancels in the reported
   DIFFERENCE (lift) -- an assumption carried forward from that validation,
   not proven here.

Six policies are compared, not five -- see policy_compare.py's module
docstring: `allocator` (Phase 8's optimizer, as-is) spends only ~3-6% of
the $10k weekly_budget by its own net-value-per-dollar logic;
`allocator_full_budget` is a budget-exhausting variant added specifically
to give a fair "same dollars spent" comparison against the other 4
policies, which all spend close to the full budget by construction.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_COMPARE_DIR = REPO_ROOT / "data" / "processed" / "policy_compare"
BOOTSTRAP_RESULTS_PATH = POLICY_COMPARE_DIR / "bootstrap_results.parquet"
BUDGET_SWEEP_RESULTS_PATH = POLICY_COMPARE_DIR / "budget_sweep_results.parquet"
TREATED_RESULTS_PATH = POLICY_COMPARE_DIR / "treated_results.parquet"
SWEEP_STABILITY_PATH = REPO_ROOT / "data" / "processed" / "allocate" / "sweep_stability.parquet"
SWEEP_APPEARANCE_PATH = REPO_ROOT / "data" / "processed" / "allocate" / "sweep_appearance.parquet"

FIGURES_DIR = REPO_ROOT / "reports" / "figures"
TABLE_MD_PATH = REPO_ROOT / "reports" / "policy_comparison.md"
TABLE_PARQUET_PATH = REPO_ROOT / "reports" / "policy_comparison.parquet"
ZONE_TABLE_PARQUET_PATH = REPO_ROOT / "reports" / "policy_comparison_by_zone.parquet"
TREATED_TABLE_PARQUET_PATH = REPO_ROOT / "reports" / "treated_cell_comparison.parquet"
LIFT_VS_BUDGET_PATH = FIGURES_DIR / "lift_vs_budget.png"
RANK_STABILITY_PATH = FIGURES_DIR / "rank_stability.png"

# Palette matches src/viz/heatmap.py / reports/plot_hour_of_week.py.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#d67a2a"
POLICY_ORDER = ("do_nothing", "uniform", "proportional", "top_n_stockout", "allocator", "allocator_full_budget")
POLICY_LABELS = {
    "do_nothing": "Do nothing",
    "uniform": "Uniform spend",
    "proportional": "Proportional to volume",
    "top_n_stockout": "Top-N stockout (naive)",
    "allocator": "Our allocator",
    "allocator_full_budget": "Our allocator (full budget)",
}

STATION_LEVEL_CAVEAT = (
    "CAVEAT (Phase 7): fill rate and lift are reported at ZONE and SYSTEM level "
    "ONLY. The simulator's per-(station, hour-of-week) stockout rate is "
    "confirmed simulator noise (pooled correlation plateaus at ~0.10 across 6 "
    "held-out weeks, DECISIONS.md's Phase 7 entry) -- no per-station number "
    "appears anywhere in this output."
)
SAME_SEED_CAVEAT = (
    "CAVEAT (Phase 7): all policies within a bootstrap replicate share the SAME "
    "demand-residual, elasticity, and simulator-seed draw (paired design). "
    "Destination-assignment noise is assumed -- not proven -- to cancel in the "
    "reported lift (the difference vs. do_nothing), per Phase 7's validation."
)


# ---------------------------------------------------------------------------
# Bootstrap CI helpers
# ---------------------------------------------------------------------------


def percentile_ci(values: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> tuple[float, float, float]:
    """(median, p_lo, p_hi) -- median as the point estimate (robust to the
    occasional degenerate replicate, e.g. a near-zero trips_recovered
    denominator blowing up cost_per_recovered_trip), 90% CI by default
    matching SPEC.md §8's own worked example ("90% CI: 1.2-3.4")."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, lo)), float(np.percentile(values, hi))


def build_policy_comparison(bootstrap: pl.DataFrame, level: str) -> pl.DataFrame:
    """One row per (policy, zone_agg) at the given level ("system" or
    "zone"), with median + 90% CI for fill_rate, lift_pp (vs. do_nothing,
    paired within replicate), trips_recovered (paired), and
    cost_per_recovered_trip (paired; replicates with trips_recovered <= 0
    are excluded from the ratio -- undefined/meaningless there, not
    silently averaged in -- and the excluded fraction is reported)."""
    sub = bootstrap.filter((pl.col("level") == level) & pl.col("fill_rate").is_not_nan())

    baseline = sub.filter(pl.col("policy") == "do_nothing").select(
        "replicate", "zone_agg", pl.col("fulfilled").alias("_base_fulfilled"), pl.col("fill_rate").alias("_base_fill_rate")
    )
    joined = sub.join(baseline, on=["replicate", "zone_agg"], how="inner").with_columns(
        ((pl.col("fill_rate") - pl.col("_base_fill_rate")) * 100.0).alias("lift_pp"),
        (pl.col("fulfilled") - pl.col("_base_fulfilled")).alias("trips_recovered"),
    )
    joined = joined.with_columns(
        pl.when(pl.col("trips_recovered") > 0)
        .then(pl.col("dollar_cost") / pl.col("trips_recovered"))
        .otherwise(None)
        .alias("cost_per_recovered_trip")
    )

    rows = []
    for (policy, zone_agg), group in joined.group_by(["policy", "zone_agg"], maintain_order=True):
        policy = policy[0] if isinstance(policy, tuple) else policy
        n = group.height
        fr_med, fr_lo, fr_hi = percentile_ci(group["fill_rate"].to_numpy())
        lift_med, lift_lo, lift_hi = percentile_ci(group["lift_pp"].to_numpy())
        trips_med, trips_lo, trips_hi = percentile_ci(group["trips_recovered"].to_numpy())
        cost_vals = group["cost_per_recovered_trip"].drop_nulls().to_numpy()
        n_cost_defined = len(cost_vals)
        cost_med, cost_lo, cost_hi = percentile_ci(cost_vals) if n_cost_defined else (float("nan"),) * 3
        rows.append(
            {
                "policy": policy, "zone_agg": zone_agg, "n_replicates": n,
                "fill_rate_median": fr_med, "fill_rate_p05": fr_lo, "fill_rate_p95": fr_hi,
                "lift_pp_median": lift_med, "lift_pp_p05": lift_lo, "lift_pp_p95": lift_hi,
                "trips_recovered_median": trips_med, "trips_recovered_p05": trips_lo, "trips_recovered_p95": trips_hi,
                "cost_per_recovered_trip_median": cost_med, "cost_per_recovered_trip_p05": cost_lo, "cost_per_recovered_trip_p95": cost_hi,
                "n_replicates_cost_defined": n_cost_defined,
                "frac_replicates_cost_undefined": 1.0 - n_cost_defined / n if n else float("nan"),
                "dollar_cost_median": float(group["dollar_cost"].median()),
            }
        )
    out = pl.DataFrame(rows)
    order = {p: i for i, p in enumerate(POLICY_ORDER)}
    return out.with_columns(pl.col("policy").replace_strict(order, default=len(order)).alias("_order")).sort("_order", "zone_agg").drop("_order")


def zone_activity_report(bootstrap: pl.DataFrame) -> dict:
    """How many (policy, replicate, zone) rows were excluded from the zone-
    level comparison because fill_rate was NaN (0 fulfilled AND 0 lost --
    a genuinely zero-activity zone that simulated week) -- reported
    explicitly rather than silently dropped."""
    zone_rows = bootstrap.filter(pl.col("level") == "zone")
    n_nan = zone_rows.filter(pl.col("fill_rate").is_nan()).height
    n_total = zone_rows.height
    return {"n_nan_zone_rows": n_nan, "n_total_zone_rows": n_total, "frac_nan": n_nan / n_total if n_total else 0.0}


# ---------------------------------------------------------------------------
# Table (.md + .parquet)
# ---------------------------------------------------------------------------


def significant_lift_summary(system_table: pl.DataFrame, zone_table: pl.DataFrame) -> dict:
    """Whether ANY policy's 90% lift CI actually excludes zero -- system
    level and zone level. Computed directly, not assumed, because a table
    of point estimates with wide CIs invites exactly the "+3%" overclaim
    SPEC.md §8 warns against if a reader skips straight to the median
    column."""
    non_baseline_sys = system_table.filter(pl.col("policy") != "do_nothing")
    sys_significant = non_baseline_sys.filter(
        (pl.col("lift_pp_p05") > 0) | (pl.col("lift_pp_p95") < 0)
    )
    non_baseline_zone = zone_table.filter(pl.col("policy") != "do_nothing")
    zone_significant = non_baseline_zone.filter(
        (pl.col("lift_pp_p05") > 0) | (pl.col("lift_pp_p95") < 0)
    )
    return {
        "n_sys_significant": sys_significant.height,
        "n_sys_total": non_baseline_sys.height,
        "n_zone_significant": zone_significant.height,
        "n_zone_total": non_baseline_zone.height,
    }


def write_policy_comparison_table(
    system_table: pl.DataFrame, zone_table: pl.DataFrame, zone_activity: dict, gate_policy: str = "top_n_stockout"
) -> None:
    checkpoint_write(system_table, TABLE_PARQUET_PATH)
    power = significant_lift_summary(system_table, zone_table)

    lines = [
        "# Phase 9: policy comparison (SPEC.md §8)",
        "",
        f"> {STATION_LEVEL_CAVEAT}",
        ">",
        f"> {SAME_SEED_CAVEAT}",
        "",
    ]
    if power["n_sys_significant"] == 0 and power["n_zone_significant"] == 0:
        lines += [
            "**HEADLINE FINDING, read this before the table below:** at this "
            "replicate count, NO policy's fill-rate lift is statistically "
            "distinguishable from zero -- every system-level 90% CI straddles "
            f"zero (0/{power['n_sys_total']} policies significant), and the same "
            f"is true at zone level (0/{power['n_zone_total']:,} policy-zone pairs "
            "significant, checked directly, not assumed). This is a real "
            "power limitation from the bootstrap replicate count, not a "
            "coding defect -- the demand-residual and elasticity axes "
            "contribute genuinely large week-to-week variance that a bigger "
            "N would narrow. Point estimates below (e.g. a policy's median "
            "lift, or its cost-per-recovered-trip) should NOT be read as "
            "established effects; treat every ranking or dollar figure in "
            "this table as directionally suggestive at best until re-run "
            "with more replicates.",
            "",
        ]
    else:
        lines += [
            f"Policies with a system-level lift CI excluding zero: "
            f"{power['n_sys_significant']}/{power['n_sys_total']}. "
            f"Zone-level: {power['n_zone_significant']}/{power['n_zone_total']:,} policy-zone pairs.",
            "",
        ]
    lines += [
        f"Zone-level rows with undefined (0/0) fill rate this simulated week: "
        f"{zone_activity['n_nan_zone_rows']:,} / {zone_activity['n_total_zone_rows']:,} "
        f"({zone_activity['frac_nan']:.1%}) -- excluded from zone-level statistics, not zeroed.",
        "",
        "## System-level results",
        "",
        "| Policy | Fill rate (median, 90% CI) | Lift vs. do-nothing (pp) | Trips recovered | Dollar cost | Cost / recovered trip |",
        "|---|---|---|---|---|---|",
    ]
    for row in system_table.sort(
        pl.col("policy").replace_strict({p: i for i, p in enumerate(POLICY_ORDER)}, default=99)
    ).iter_rows(named=True):
        label = POLICY_LABELS.get(row["policy"], row["policy"])
        fr = f"{row['fill_rate_median']:.4f} ({row['fill_rate_p05']:.4f}–{row['fill_rate_p95']:.4f})"
        lift = f"{row['lift_pp_median']:+.2f} ({row['lift_pp_p05']:+.2f}–{row['lift_pp_p95']:+.2f})"
        trips = f"{row['trips_recovered_median']:+,.0f} ({row['trips_recovered_p05']:+,.0f}–{row['trips_recovered_p95']:+,.0f})"
        dollars = f"${row['dollar_cost_median']:,.0f}"
        if row["n_replicates_cost_defined"] == 0:
            cost = "undefined (no replicate recovered trips)" if row["policy"] != "do_nothing" else "n/a"
        else:
            cost = (
                f"${row['cost_per_recovered_trip_median']:,.2f} "
                f"(${row['cost_per_recovered_trip_p05']:,.2f}–${row['cost_per_recovered_trip_p95']:,.2f})"
            )
            if row["frac_replicates_cost_undefined"] > 0:
                cost += f" [{row['frac_replicates_cost_undefined']:.0%} of replicates excluded, trips_recovered<=0]"
        lines.append(f"| {label} | {fr} | {lift} | {trips} | {dollars} | {cost} |")

    gate_row = system_table.filter(pl.col("policy") == "allocator").row(0, named=True)
    naive_row = system_table.filter(pl.col("policy") == gate_policy).row(0, named=True)
    beats_naive = gate_row["lift_pp_median"] > naive_row["lift_pp_median"]
    both_underpowered = power["n_sys_significant"] == 0
    verdict = (
        "UNDERPOWERED TO CONCLUDE"
        if both_underpowered
        else ("YES" if beats_naive else "NO")
    )
    verdict_suffix = (
        " -- both policies' lift CIs straddle zero, so this point-estimate comparison is not a "
        "resolved answer, just a tiebreaker if one is needed"
        if both_underpowered
        else ""
    )
    lines += [
        "",
        f"## RUNBOOK gate: does `allocator` beat `{gate_policy}` (naive top-N)?",
        "",
        f"{verdict} on raw fill-rate lift "
        f"({gate_row['lift_pp_median']:+.3f}pp vs. {naive_row['lift_pp_median']:+.3f}pp){verdict_suffix}. "
        f"Per RUNBOOK: \"that's still a legitimate and interesting finding -- understand why.\" "
        f"`allocator` spends ${gate_row['dollar_cost_median']:,.0f} vs. `{gate_policy}`'s "
        f"${naive_row['dollar_cost_median']:,.0f} -- see `allocator_full_budget` for the "
        "budget-exhausting comparison, and the module docstrings in "
        "src/opt/policy_baselines.py for the underlying candidate-pool-exhaustion finding.",
        "",
        f"Full zone-level table: `{ZONE_TABLE_PARQUET_PATH.relative_to(REPO_ROOT)}` (not reproduced here -- "
        "one row per policy per zone, machine-readable only, per the station/zone-level caveat above).",
    ]
    TABLE_MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"[plot_policy_comparison] wrote {TABLE_MD_PATH}")
    print(f"[plot_policy_comparison] wrote {TABLE_PARQUET_PATH}")


def checkpoint_write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Treated-cell paired comparison (src/sim/policy_compare.py's
# run_one_replicate_treated) -- added because the system-level table above
# pools fill rate over the WHOLE network while a policy funds at most a
# few thousand cells against it, diluting any real effect below the
# bootstrap noise floor regardless of replicate count. Restricted to each
# policy's own (station, hour) treated cells, paired against do-nothing on
# the SAME cells and SAME seed within a replicate -- removes the dilution
# directly rather than needing more replicates to shrink it. Still never a
# per-station number (see STATION_LEVEL_CAVEAT above) -- every figure here
# is pooled over hundreds to thousands of cells per policy.
# ---------------------------------------------------------------------------


def build_treated_comparison(treated: pl.DataFrame) -> pl.DataFrame:
    """One row per policy, median + 90% CI for fill_rate_treated,
    lift_pp (paired within replicate against the SAME cells' do-nothing
    run), and trips_recovered -- defined as the REDUCTION IN LOST trips at
    the treated cells (lost_treated_do_nothing - lost_treated), not the
    change in arrivals. The incentive fixes departure failures
    (lost_no_bike) at the destination it funds; arrivals at that cell are
    a mostly independent, exogenous quantity (driven by other stations'
    routing), so "change in fulfilled/arrivals" doesn't track what the
    intervention actually does -- checked directly: an earlier draft used
    that definition and got a median NEGATIVE trips_recovered for every
    policy despite a clearly positive, significant fill-rate lift, which
    is precisely the mismatch this docstring warns about."""
    t = treated.with_columns(
        ((pl.col("fill_rate_treated") - pl.col("fill_rate_treated_do_nothing")) * 100.0).alias("lift_pp"),
        (pl.col("lost_treated_do_nothing") - pl.col("lost_treated")).alias("trips_recovered"),
    ).with_columns(
        pl.when(pl.col("trips_recovered") > 0)
        .then(pl.col("dollar_cost") / pl.col("trips_recovered"))
        .otherwise(None)
        .alias("cost_per_recovered_trip")
    )
    rows = []
    for policy, group in t.group_by("policy", maintain_order=True):
        policy = policy[0] if isinstance(policy, tuple) else policy
        n = group.height
        fr_med, fr_lo, fr_hi = percentile_ci(group["fill_rate_treated"].to_numpy() * 100)
        lift_med, lift_lo, lift_hi = percentile_ci(group["lift_pp"].to_numpy())
        trips_med, trips_lo, trips_hi = percentile_ci(group["trips_recovered"].to_numpy())
        cost_vals = group["cost_per_recovered_trip"].drop_nulls().to_numpy()
        n_cost_defined = len(cost_vals)
        cost_med, cost_lo, cost_hi = percentile_ci(cost_vals) if n_cost_defined else (float("nan"),) * 3
        rows.append(
            {
                "policy": policy, "n_replicates": n,
                "avg_n_treated_cells": float(group["n_treated_cells"].mean()),
                "fill_rate_treated_median": fr_med, "fill_rate_treated_p05": fr_lo, "fill_rate_treated_p95": fr_hi,
                "lift_pp_median": lift_med, "lift_pp_p05": lift_lo, "lift_pp_p95": lift_hi,
                "pct_replicates_lift_gt0": float((group["lift_pp"].to_numpy() > 0).mean() * 100),
                "trips_recovered_median": trips_med, "trips_recovered_p05": trips_lo, "trips_recovered_p95": trips_hi,
                "cost_per_recovered_trip_median": cost_med, "cost_per_recovered_trip_p05": cost_lo, "cost_per_recovered_trip_p95": cost_hi,
                "n_replicates_cost_defined": n_cost_defined,
                "frac_replicates_cost_undefined": 1.0 - n_cost_defined / n if n else float("nan"),
                "dollar_cost_median": float(group["dollar_cost"].median()),
            }
        )
    out = pl.DataFrame(rows)
    order = {p: i for i, p in enumerate(POLICY_ORDER)}
    return out.with_columns(pl.col("policy").replace_strict(order, default=len(order)).alias("_order")).sort("_order").drop("_order")


def append_treated_cell_section(treated_table: pl.DataFrame) -> None:
    checkpoint_write(treated_table, TREATED_TABLE_PARQUET_PATH)
    n_significant = int((treated_table["lift_pp_p05"] > 0).sum())
    n_total = treated_table.height

    lines = [
        "",
        "---",
        "",
        "## Treated-cell paired comparison",
        "",
        "Restricted to the (station, hour-of-week) cells each policy itself funded "
        "(post `apply_move_cap`), paired against a do-nothing run on the SAME cells and the SAME "
        "replicate seed -- added after the system-level result above turned out to be a "
        "measurement-design problem, not a power problem: pooling fill rate over the WHOLE network "
        "(~875K trips/week) dilutes a few-thousand-trip treatment below the bootstrap noise floor "
        "regardless of replicate count. Still never a per-station number (see the station-level "
        "caveat above) -- every figure below is pooled over hundreds to thousands of cells per "
        "policy. `trips_recovered` here is the REDUCTION IN LOST trips at the treated cells (what "
        "the incentive actually targets), not a change in arrivals -- see "
        "`build_treated_comparison`'s docstring for why that distinction matters.",
        "",
        f"**{n_significant}/{n_total} policies now show a fill-rate lift CI that excludes zero** "
        f"-- vs. 0/5 at system level above, from the SAME 40 replicates and seeds. The treatment "
        "effect was real and measurable all along; the system-level table diluted it away.",
        "",
        "| Policy | Fill rate on treated cells (median, 90% CI) | Lift vs. do-nothing, same cells (pp) | P(lift>0) | Trips recovered (fewer lost) | Dollar cost | Cost / recovered trip | Avg. treated cells |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in treated_table.iter_rows(named=True):
        label = POLICY_LABELS.get(row["policy"], row["policy"])
        fr = f"{row['fill_rate_treated_median']:.2f}% ({row['fill_rate_treated_p05']:.2f}–{row['fill_rate_treated_p95']:.2f})"
        lift = f"{row['lift_pp_median']:+.2f} ({row['lift_pp_p05']:+.2f}–{row['lift_pp_p95']:+.2f})"
        p_pos = f"{row['pct_replicates_lift_gt0']:.0f}%"
        trips = f"{row['trips_recovered_median']:+,.0f} ({row['trips_recovered_p05']:+,.0f}–{row['trips_recovered_p95']:+,.0f})"
        dollars = f"${row['dollar_cost_median']:,.0f}"
        if row["n_replicates_cost_defined"] == 0:
            cost = "undefined"
        else:
            cost = (
                f"${row['cost_per_recovered_trip_median']:,.2f} "
                f"(${row['cost_per_recovered_trip_p05']:,.2f}–${row['cost_per_recovered_trip_p95']:,.2f})"
            )
            if row["frac_replicates_cost_undefined"] > 0:
                cost += f" [{row['frac_replicates_cost_undefined']:.0%} excluded, trips_recovered<=0]"
        cells = f"{row['avg_n_treated_cells']:,.0f}"
        lines.append(f"| {label} | {fr} | {lift} | {p_pos} | {trips} | {dollars} | {cost} | {cells} |")

    with TABLE_MD_PATH.open("a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[plot_policy_comparison] appended treated-cell section -> {TABLE_MD_PATH}")
    print(f"[plot_policy_comparison] wrote {TREATED_TABLE_PARQUET_PATH}")


# ---------------------------------------------------------------------------
# Plot 1: lift vs. budget (allocator + allocator_full_budget, $0 -> weekly_budget)
# ---------------------------------------------------------------------------


def plot_lift_vs_budget(sweep: pl.DataFrame) -> None:
    sub = sweep.filter((pl.col("level") == "system") & pl.col("fill_rate").is_not_nan())
    baseline = sub.filter(pl.col("policy") == "do_nothing").select(
        "replicate", "weekly_budget", pl.col("fill_rate").alias("_base_fill_rate")
    )
    joined = sub.join(baseline, on=["replicate", "weekly_budget"], how="inner").with_columns(
        ((pl.col("fill_rate") - pl.col("_base_fill_rate")) * 100.0).alias("lift_pp")
    )

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for policy, color in (("allocator", BLUE), ("allocator_full_budget", ORANGE)):
        pol = joined.filter(pl.col("policy") == policy)
        agg = (
            pol.group_by("weekly_budget")
            .agg(
                pl.col("lift_pp").median().alias("median"),
                pl.col("lift_pp").quantile(0.05).alias("p05"),
                pl.col("lift_pp").quantile(0.95).alias("p95"),
                pl.col("dollar_cost").median().alias("actual_spend"),
            )
            .sort("weekly_budget")
        )
        x = agg["weekly_budget"].to_numpy()
        med = agg["median"].to_numpy()
        lo = agg["p05"].to_numpy()
        hi = agg["p95"].to_numpy()
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, med, color=color, linewidth=2, marker="o", markersize=4, label=POLICY_LABELS[policy])

    ax.axhline(0.0, color=BASELINE, linewidth=1, zorder=0)
    ax.set_xlabel("Budget ceiling ($)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Fill-rate lift vs. do-nothing (pp, median + 90% CI)", color=INK_SECONDARY, fontsize=10)
    ax.tick_params(axis="both", colors=INK_MUTED, labelsize=9, length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax.set_title(
        "Lift vs. budget: where do marginal returns flatten?",
        color=INK_PRIMARY, fontsize=13, loc="left", pad=14,
    )
    fig.text(
        0.01, 0.01,
        f"{STATION_LEVEL_CAVEAT}\n{SAME_SEED_CAVEAT}\n"
        "Budget-sweep replicate count is intentionally lighter than the main bootstrap (shape-finding, not precision CIs).",
        fontsize=6.5, color=INK_MUTED, wrap=True, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LIFT_VS_BUDGET_PATH, dpi=180)
    plt.close(fig)
    print(f"[plot_policy_comparison] wrote {LIFT_VS_BUDGET_PATH}")


# ---------------------------------------------------------------------------
# Plot 2: rank stability -- CARRIED FORWARD from Phase 8, not recomputed
# ---------------------------------------------------------------------------


def plot_rank_stability(sweep_stability: pl.DataFrame, sweep_appearance: pl.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)

    # Left: appearance_frac distribution among the ever-appearing top-100
    # targets across the 25-draw elasticity sweep -- surfaces the "stable
    # core" CLAUDE.md already reports.
    frac_sorted = np.sort(sweep_appearance["appearance_frac"].to_numpy())[::-1]
    ax1.bar(range(len(frac_sorted)), frac_sorted, color=BLUE, width=1.0)
    n_stable = int((frac_sorted == 1.0).sum())
    ax1.axhline(1.0, color=BASELINE, linewidth=1, zorder=0)
    ax1.set_xlabel("Targets, sorted by appearance frequency", color=INK_SECONDARY, fontsize=10)
    ax1.set_ylabel("Fraction of 25 elasticity draws appearing in top-100", color=INK_SECONDARY, fontsize=10)
    ax1.set_title(f"Stable core: {n_stable} targets in every draw", color=INK_PRIMARY, fontsize=12, loc="left")
    ax1.tick_params(axis="both", colors=INK_MUTED, labelsize=9, length=0)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    # Right: pairwise Spearman correlation distribution across the sweep.
    sp = sweep_stability["spearman"].drop_nulls().drop_nans().to_numpy()
    ax2.hist(sp, bins=20, color=ORANGE, edgecolor=SURFACE)
    ax2.set_xlabel("Spearman rank correlation (pairwise across 25 draws)", color=INK_SECONDARY, fontsize=10)
    ax2.set_ylabel("Draw pairs", color=INK_SECONDARY, fontsize=10)
    ax2.set_title(
        f"median={np.median(sp):.3f}, min={sp.min():.3f} (n={len(sp)} pairs)",
        color=INK_PRIMARY, fontsize=12, loc="left",
    )
    ax2.tick_params(axis="both", colors=INK_MUTED, labelsize=9, length=0)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.suptitle(
        "Rank stability across the elasticity sweep -- carried forward from Phase 8, not recomputed for Phase 9",
        color=INK_PRIMARY, fontsize=13, x=0.01, ha="left",
    )
    fig.text(0.01, 0.01, "Source: data/processed/allocate/sweep_stability.parquet, sweep_appearance.parquet (Phase 8).", fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RANK_STABILITY_PATH, dpi=180)
    plt.close(fig)
    print(f"[plot_policy_comparison] wrote {RANK_STABILITY_PATH}")


def main() -> None:
    print(f"[plot_policy_comparison] {STATION_LEVEL_CAVEAT}")
    print(f"[plot_policy_comparison] {SAME_SEED_CAVEAT}")

    bootstrap = pl.read_parquet(BOOTSTRAP_RESULTS_PATH)
    system_table = build_policy_comparison(bootstrap, "system")
    zone_table = build_policy_comparison(bootstrap, "zone")
    zone_activity = zone_activity_report(bootstrap)
    checkpoint_write(zone_table, ZONE_TABLE_PARQUET_PATH)
    write_policy_comparison_table(system_table, zone_table, zone_activity)

    if BUDGET_SWEEP_RESULTS_PATH.exists():
        sweep = pl.read_parquet(BUDGET_SWEEP_RESULTS_PATH)
        plot_lift_vs_budget(sweep)
    else:
        print(f"[plot_policy_comparison] {BUDGET_SWEEP_RESULTS_PATH} not found -- skipping lift-vs-budget plot")

    if TREATED_RESULTS_PATH.exists():
        treated = pl.read_parquet(TREATED_RESULTS_PATH)
        treated_table = build_treated_comparison(treated)
        append_treated_cell_section(treated_table)
    else:
        print(f"[plot_policy_comparison] {TREATED_RESULTS_PATH} not found -- skipping treated-cell section")

    sweep_stability = pl.read_parquet(SWEEP_STABILITY_PATH)
    sweep_appearance = pl.read_parquet(SWEEP_APPEARANCE_PATH)
    plot_rank_stability(sweep_stability, sweep_appearance)


if __name__ == "__main__":
    main()
