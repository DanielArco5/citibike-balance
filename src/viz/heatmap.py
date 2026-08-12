"""Two-panel zone x hour-of-week heatmaps of net-lost demand (SPEC.md §5),
plus a per-dock-normalized version and a per-station scatter map of net-lost
demand.

Reads data/processed/unmet_demand_net.parquet (src/models/substitution.py's
output, chunked per month -- see the Phase 6 memory-safety restructure at
/Users/danielcrown1/.claude/plans/lucky-leaping-zebra.md) and
data/processed/stations_zoned.parquet (zone capacity + zone/station
centroids). Not part of the automated pipeline -- run manually after
substitution.py, same "rerun and eyeball the gate" pattern as
reports/plot_hour_of_week.py.

The worst-zone-hours map used to be a folium (Leaflet) HTML map with tile
imagery; replaced with a matplotlib scatter of station coordinates colored
by net-lost demand, since folium's tile servers aren't reachable in this
environment -- no network dependency, same OrRd/PuBu palette as the
zone x hour-of-week panels for visual consistency.

Sanity check per SPEC.md §5, worth checking on every run, not just once:
residential outer-zone stations should bleed bikes (dep_net_lost) 7-9am and
go dock-starved (arr_net_lost) 6-8pm; Midtown/FiDi-type zones should be the
mirror image. main() prints each top zone's peak hour per direction so this
is checkable from the console output, not just by eyeballing the PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
UNMET_NET_DIR = REPO_ROOT / "data" / "processed" / "unmet_demand_net"
# Legacy single-file fallback (e.g. after a manual/consolidated run).
UNMET_NET_PATH = REPO_ROOT / "data" / "processed" / "unmet_demand_net.parquet"
STATIONS_PATH = REPO_ROOT / "data" / "processed" / "stations_zoned.parquet"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

TOP_N_ZONES = 40

# Palette matches reports/plot_hour_of_week.py.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

DEP_CMAP = "OrRd"  # bike-starved
ARR_CMAP = "PuBu"  # dock-starved


def load_unmet_net() -> pl.LazyFrame:
    """Scans the per-month checkpoint directory (falls back to the single
    consolidated file if the directory doesn't exist)."""
    if UNMET_NET_DIR.exists():
        return pl.scan_parquet(UNMET_NET_DIR / "*.parquet")
    return pl.scan_parquet(UNMET_NET_PATH)


def _as_lazy(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """aggregate_zone_hour/n_weeks_covered/aggregate_station_net_lost accept
    either an eager DataFrame (small synthetic fixtures, tests/test_heatmap.py)
    or a LazyFrame (production, a scan over the monthly checkpoint directory)
    -- normalizing here means the FULL unmet table is never eagerly
    materialized on the production path; only the small aggregated result
    (a few thousand rows) is collected, at the end of each function."""
    return df.lazy() if isinstance(df, pl.DataFrame) else df


# ---------------------------------------------------------------------------
# Data prep (pure functions -- unit-tested in tests/test_heatmap.py)
# ---------------------------------------------------------------------------


def n_weeks_covered(unmet: pl.DataFrame | pl.LazyFrame) -> float:
    bounds = _as_lazy(unmet).select(
        pl.col("interval_start").min().alias("_min"), pl.col("interval_start").max().alias("_max")
    ).collect()
    span = bounds["_max"][0] - bounds["_min"][0]
    return span.total_seconds() / (7 * 24 * 3600)


def aggregate_zone_hour(unmet: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """(zone_agg, hour_of_week) -> expected dep_net_lost / arr_net_lost PER
    WEEK (SPEC.md §5: "expected lost trips per week") -- summed across every
    station in the zone and every occurrence of that hour-of-week across the
    panel's date range, then divided by the number of weeks covered."""
    weeks = n_weeks_covered(unmet)
    return (
        _as_lazy(unmet)
        .group_by(["zone_agg", "hour_of_week"])
        .agg(
            pl.col("dep_net_lost").sum().alias("dep_net_lost_total"),
            pl.col("arr_net_lost").sum().alias("arr_net_lost_total"),
        )
        .with_columns(
            (pl.col("dep_net_lost_total") / weeks).alias("dep_net_lost_per_week"),
            (pl.col("arr_net_lost_total") / weeks).alias("arr_net_lost_per_week"),
        )
        .collect()
    )


def aggregate_station_net_lost(unmet: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Per-station (not per-zone) counterpart to aggregate_zone_hour, for the
    matplotlib station scatter -- the folium map it replaces plotted zone
    CENTROIDS, but the replacement is explicitly per-station, so it needs its
    own (station_id) aggregation rather than reusing aggregate_zone_hour's
    (zone_agg, hour_of_week) table."""
    weeks = n_weeks_covered(unmet)
    return (
        _as_lazy(unmet)
        .group_by("station_id")
        .agg(
            pl.col("dep_net_lost").sum().alias("dep_net_lost_total"),
            pl.col("arr_net_lost").sum().alias("arr_net_lost_total"),
        )
        .with_columns(
            (pl.col("dep_net_lost_total") / weeks).alias("dep_net_lost_per_week"),
            (pl.col("arr_net_lost_total") / weeks).alias("arr_net_lost_per_week"),
        )
        .collect()
    )


def zone_capacity(stations: pl.DataFrame, zone_col: str = "zone_agglomerative") -> pl.DataFrame:
    """Total docks per zone -- summed over DISTINCT stations, not over the
    (zone, hour) panel rows, which would double-count every station once
    per interval it appears in."""
    return stations.group_by(zone_col).agg(pl.col("capacity").sum().alias("zone_capacity")).rename(
        {zone_col: "zone_agg"}
    )


# Jersey City/Hoboken is a physically separate Citi Bike sub-fleet (its own
# PATH/light-rail-anchored network across the Hudson), out of scope for the
# NYC commuter-dipole analysis -- see DECISIONS.md, "Jersey City/Hoboken
# exclusion from zone rankings". Threshold calibrated against real station
# coordinates, not guessed: Manhattan's westernmost stations (Battery Park
# City) reach lng -74.017; Hoboken's easternmost (12 St & Sinatra Dr N) sit
# at lng -74.024 -- the Hudson River itself creates that gap, and -74.02
# sits safely in the middle of it. Brooklyn's Bay Ridge reaches a similar
# longitude (-74.024) but is at lat < 40.66, well south of the lat>=40.68
# floor, so it isn't caught; checked directly against the full station list,
# not assumed.
JC_HOBOKEN_LNG_MAX = -74.02
JC_HOBOKEN_LAT_MIN = 40.68


def is_jc_hoboken_zone(stations: pl.DataFrame, zone_col: str = "zone_agglomerative") -> pl.DataFrame:
    """One row per zone: (zone_agg, is_jc_hoboken), by zone CENTROID (mean of
    its member stations' lat/lng) against the JC_HOBOKEN_* bounding box."""
    centroids = stations.group_by(zone_col).agg(
        pl.col("lat").mean().alias("_lat"), pl.col("lng").mean().alias("_lng")
    )
    return centroids.select(
        pl.col(zone_col).alias("zone_agg"),
        ((pl.col("_lng") < JC_HOBOKEN_LNG_MAX) & (pl.col("_lat") >= JC_HOBOKEN_LAT_MIN)).alias("is_jc_hoboken"),
    )


def exclude_jc_hoboken(agg: pl.DataFrame, jc_flags: pl.DataFrame) -> pl.DataFrame:
    return (
        agg.join(jc_flags, on="zone_agg", how="left")
        .filter(~pl.col("is_jc_hoboken").fill_null(False))
        .drop("is_jc_hoboken")
    )


def add_per_dock_columns(agg: pl.DataFrame, zone_cap: pl.DataFrame) -> pl.DataFrame:
    return agg.join(zone_cap, on="zone_agg", how="left").with_columns(
        (pl.col("dep_net_lost_per_week") / pl.col("zone_capacity")).alias("dep_net_lost_per_dock"),
        (pl.col("arr_net_lost_per_week") / pl.col("zone_capacity")).alias("arr_net_lost_per_dock"),
    )


def aggregate_zone_hour_per_dock_ci(
    unmet: pl.DataFrame | pl.LazyFrame, zone_cap: pl.DataFrame, z_score: float = 1.96
) -> pl.DataFrame:
    """Per (zone_agg, hour_of_week): mean, standard error, and (mean -
    z_score*SE) LOWER BOUND of the per-dock weekly net-lost rate -- computed
    from each zone-hour's own sample of individual calendar-week
    observations (~52 for a full year), not a single grand-total/weeks-
    covered point estimate the way add_per_dock_columns's plain mean is.
    dep_net_lost_per_dock/arr_net_lost_per_dock here are numerically the
    same quantity add_per_dock_columns produces (mean of equally-weighted
    per-week values == grand total / weeks covered) -- this function adds
    the SE/lower-bound columns on top, it doesn't change the point estimate.

    Why this exists (see DECISIONS.md, "lower-bound-of-CI ranking replaces
    the per-dock small-zone-noise caveat"): ranking zone-hours by the raw
    per-dock MEAN is exactly as vulnerable to small-sample noise as raw
    volume is to zone-size bias -- a 3-4-station zone with one or two
    genuine stockout weeks and ~50 weeks of zero produces a mean that looks
    "high" per dock purely from a tiny, noisy sample. Ranking by the lower
    confidence bound instead penalizes exactly that: a wide interval (few,
    inconsistent weekly observations) pulls the lower bound toward zero (or
    below) regardless of how high the point estimate is, while a zone with
    a real, WEEK-TO-WEEK-CONSISTENT peak keeps a tight interval and a lower
    bound close to its mean. No dock-count or station-count threshold
    anywhere in the computation -- zone size never enters the formula.

    n_weeks < 2 zone-hours (can't estimate a standard error from one
    observation) get a null lower bound, sorting last -- filling their SE
    with 0 would treat a single observation as perfectly reliable, the
    opposite of the intent here."""
    per_week = (
        _as_lazy(unmet)
        .with_columns(pl.col("interval_start").dt.truncate("1w").alias("_week"))
        .group_by(["zone_agg", "hour_of_week", "_week"])
        .agg(
            pl.col("dep_net_lost").sum().alias("_dep_week"),
            pl.col("arr_net_lost").sum().alias("_arr_week"),
        )
        .collect()
        .join(zone_cap, on="zone_agg", how="left")
        .with_columns(
            (pl.col("_dep_week") / pl.col("zone_capacity")).alias("_dep_per_dock_week"),
            (pl.col("_arr_week") / pl.col("zone_capacity")).alias("_arr_per_dock_week"),
        )
    )
    stats = per_week.group_by(["zone_agg", "hour_of_week"]).agg(
        pl.col("_dep_per_dock_week").mean().alias("dep_net_lost_per_dock"),
        pl.col("_dep_per_dock_week").std(ddof=1).alias("_dep_std"),
        pl.col("_arr_per_dock_week").mean().alias("arr_net_lost_per_dock"),
        pl.col("_arr_per_dock_week").std(ddof=1).alias("_arr_std"),
        pl.len().alias("n_weeks"),
    )
    n_sqrt = pl.col("n_weeks").cast(pl.Float64).sqrt()
    stats = stats.with_columns(
        (pl.col("_dep_std") / n_sqrt).alias("dep_se"), (pl.col("_arr_std") / n_sqrt).alias("arr_se")
    ).with_columns(
        pl.when(pl.col("n_weeks") >= 2)
        .then(pl.col("dep_net_lost_per_dock") - z_score * pl.col("dep_se"))
        .otherwise(None)
        .alias("dep_net_lost_per_dock_lb"),
        pl.when(pl.col("n_weeks") >= 2)
        .then(pl.col("arr_net_lost_per_dock") - z_score * pl.col("arr_se"))
        .otherwise(None)
        .alias("arr_net_lost_per_dock_lb"),
    )
    return stats.select(
        "zone_agg",
        "hour_of_week",
        "dep_net_lost_per_dock",
        "dep_se",
        "dep_net_lost_per_dock_lb",
        "arr_net_lost_per_dock",
        "arr_se",
        "arr_net_lost_per_dock_lb",
        "n_weeks",
    )


def top_zones_by_lower_bound(agg_ci: pl.DataFrame, dep_lb_col: str, arr_lb_col: str, n: int = TOP_N_ZONES) -> list[str]:
    """Zones ranked by their single BEST (highest) lower-bound zone-hour
    across either direction, not a sum across all 168 hours the way
    top_zones() works for point estimates -- summing would let many
    mediocre/negative-lower-bound hours dilute a zone's one genuine
    reliable peak. A zone qualifies for the top of the list by having at
    least one hour where, after the CI penalty, we're still confident the
    rate is genuinely elevated -- the same criterion the top-10 zone-hour
    table ranks on directly."""
    scored = agg_ci.with_columns(
        pl.max_horizontal(pl.col(dep_lb_col).fill_null(float("-inf")), pl.col(arr_lb_col).fill_null(float("-inf"))).alias(
            "_score"
        )
    )
    best = scored.group_by("zone_agg").agg(pl.col("_score").max().alias("_best")).sort("_best", descending=True)
    return best.head(n)["zone_agg"].to_list()


def top_zones(agg: pl.DataFrame, dep_col: str, arr_col: str, n: int = TOP_N_ZONES) -> list[str]:
    """Zones ranked by combined (dep + arr) weekly total, descending -- one
    consistent row order for both panels of a given figure so a reader can
    compare the same zone's two directions side by side. The raw and
    per-dock figures each get their OWN top_zones() call with their own
    columns, so they can (and often will) surface different zones -- that
    difference is the point of the per-dock view (SPEC.md §5 calls it "the
    more actionable" one: a small zone with few docks but high loss-per-dock
    won't rank highly on a raw total but should on a per-dock one)."""
    totals = (
        agg.group_by("zone_agg")
        .agg((pl.col(dep_col).fill_null(0) + pl.col(arr_col).fill_null(0)).sum().alias("total"))
        .sort("total", descending=True)
    )
    return totals.head(n)["zone_agg"].to_list()


def build_matrix(agg: pl.DataFrame, value_col: str, zone_order: list[str]) -> np.ndarray:
    """len(zone_order) x 168 dense matrix, 0-filled for any (zone, hour)
    cell absent from agg (a zone with zero loss at that hour-of-week)."""
    zone_row = {z: i for i, z in enumerate(zone_order)}
    mat = np.zeros((len(zone_order), 168))
    sub = agg.filter(pl.col("zone_agg").is_in(zone_order))
    for row in sub.iter_rows(named=True):
        zi = zone_row.get(row["zone_agg"])
        if zi is None:
            continue
        h = int(row["hour_of_week"])
        val = row[value_col]
        mat[zi, h] = val if val is not None else 0.0
    return mat


def hour_of_week_to_label(how: int) -> str:
    return f"{DAY_LABELS[how // 24]} {how % 24:02d}:00"


def peak_hour(agg: pl.DataFrame, zone: str, value_col: str) -> tuple[str, float]:
    """The single worst hour-of-week for one zone/direction -- used for both
    the folium popups and the console commuter-dipole sanity check."""
    rows = agg.filter(pl.col("zone_agg") == zone).sort(value_col, descending=True).head(1)
    if rows.height == 0:
        return "n/a", 0.0
    r = rows.row(0, named=True)
    return hour_of_week_to_label(int(r["hour_of_week"])), float(r[value_col] or 0.0)


# ---------------------------------------------------------------------------
# Static two-panel heatmap
# ---------------------------------------------------------------------------


def plot_two_panel_heatmap(
    mat_dep: np.ndarray, mat_arr: np.ndarray, zone_order: list[str], title_suffix: str, cbar_label: str, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, max(6, 0.22 * len(zone_order))), facecolor=SURFACE)
    for ax, mat, cmap, subtitle in (
        (axes[0], mat_dep, DEP_CMAP, "Bike-starved (departures)"),
        (axes[1], mat_arr, ARR_CMAP, "Dock-starved (arrivals)"),
    ):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_facecolor(SURFACE)
        for d in range(1, 7):
            ax.axvline(d * 24 - 0.5, color=GRID, linewidth=0.8, zorder=1)
        ax.set_xlim(-0.5, 167.5)
        ax.set_xticks([d * 24 + 12 for d in range(7)])
        ax.set_xticklabels(DAY_LABELS, color=INK_SECONDARY, fontsize=9)
        ax.set_yticks(range(len(zone_order)))
        ax.set_yticklabels(zone_order, color=INK_SECONDARY, fontsize=6.5)
        ax.tick_params(length=0)
        ax.set_title(subtitle, color=INK_PRIMARY, fontsize=12, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=8, colors=INK_MUTED)
        cbar.set_label(cbar_label, color=INK_SECONDARY, fontsize=9)

    fig.suptitle(
        f"Net-lost trips by zone x hour-of-week{title_suffix}", color=INK_PRIMARY, fontsize=14, x=0.02, ha="left"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[heatmap] wrote {out_path}")


# ---------------------------------------------------------------------------
# Per-station scatter map of net-lost demand (folium replacement)
# ---------------------------------------------------------------------------


def plot_station_scatter(station_agg: pl.DataFrame, stations: pl.DataFrame, out_path: Path) -> None:
    """Two-panel matplotlib scatter of station coordinates colored by
    net-lost trips/week -- replaces the folium/Leaflet map (tile servers
    aren't reachable in this environment). No network dependency. Same
    OrRd/PuBu cmaps as plot_two_panel_heatmap's panels, for visual
    consistency across the two figures."""
    merged = station_agg.join(stations.select("station_id", "lat", "lng"), on="station_id", how="left").filter(
        pl.col("lat").is_not_null()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor=SURFACE)
    for ax, value_col, cmap, subtitle in (
        (axes[0], "dep_net_lost_per_week", DEP_CMAP, "Bike-starved (departures)"),
        (axes[1], "arr_net_lost_per_week", ARR_CMAP, "Dock-starved (arrivals)"),
    ):
        vals = merged[value_col].fill_null(0.0).to_numpy()
        sc = ax.scatter(
            merged["lng"].to_numpy(), merged["lat"].to_numpy(), c=vals, cmap=cmap, s=14, alpha=0.85, edgecolors="none"
        )
        ax.set_facecolor(SURFACE)
        ax.set_title(subtitle, color=INK_PRIMARY, fontsize=12, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=8, colors=INK_MUTED)
        cbar.set_label("Net-lost trips / week", color=INK_SECONDARY, fontsize=9)

    fig.suptitle("Net-lost trips by station", color=INK_PRIMARY, fontsize=14, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[heatmap] wrote {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    unmet = load_unmet_net()
    stations = pl.read_parquet(STATIONS_PATH)
    zone_cap = zone_capacity(stations)

    # Jersey City/Hoboken excluded from every ranking/panel below -- see
    # DECISIONS.md, "Jersey City/Hoboken exclusion from zone rankings". Not
    # excluded from the per-station scatter map, which isn't zone-ranked.
    jc_flags = is_jc_hoboken_zone(stations)

    # Per-dock, LOWER-CONFIDENCE-BOUND ranking is the PRIMARY metric (see
    # DECISIONS.md, "lower-bound-of-CI ranking replaces the per-dock
    # small-zone-noise caveat"): raw volume buries small residential zones
    # behind huge Midtown ones; the plain per-dock MEAN overcorrects and lets
    # small-sample noise (a couple of stockout weeks out of ~52) dominate.
    # Ranking by mean - 1.96*SE penalizes exactly that noise with no
    # dock-count or station-count threshold. Primary output keeps the
    # unsuffixed filename; the raw-volume view is kept as a secondary
    # reference, not the default read.
    agg_ci = aggregate_zone_hour_per_dock_ci(unmet, zone_cap)
    agg_ci = exclude_jc_hoboken(agg_ci, jc_flags)

    zone_order_dock = top_zones_by_lower_bound(agg_ci, "dep_net_lost_per_dock_lb", "arr_net_lost_per_dock_lb")
    mat_dep_dock = build_matrix(agg_ci, "dep_net_lost_per_dock", zone_order_dock)
    mat_arr_dock = build_matrix(agg_ci, "arr_net_lost_per_dock", zone_order_dock)
    plot_two_panel_heatmap(
        mat_dep_dock,
        mat_arr_dock,
        zone_order_dock,
        "",
        "Net-lost trips / week / dock",
        FIGURES_DIR / "net_lost_heatmap.png",
    )

    agg = aggregate_zone_hour(unmet)
    agg = add_per_dock_columns(agg, zone_cap)
    agg = exclude_jc_hoboken(agg, jc_flags)

    zone_order_raw = top_zones(agg, "dep_net_lost_per_week", "arr_net_lost_per_week")
    mat_dep = build_matrix(agg, "dep_net_lost_per_week", zone_order_raw)
    mat_arr = build_matrix(agg, "arr_net_lost_per_week", zone_order_raw)
    plot_two_panel_heatmap(
        mat_dep,
        mat_arr,
        zone_order_raw,
        " (by raw volume, secondary)",
        "Net-lost trips / week",
        FIGURES_DIR / "net_lost_heatmap_by_raw_volume.png",
    )

    station_agg = aggregate_station_net_lost(unmet)
    plot_station_scatter(station_agg, stations, FIGURES_DIR / "net_lost_station_map.png")

    # Commuter-dipole sanity check (SPEC.md §5): print, don't just plot, so
    # it's checkable from console output. Expected shape: a zone's peak
    # bike-starved hour lands in the AM commute window and its peak
    # dock-starved hour lands in the PM window (residential/outer profile),
    # OR the mirror image (core/business-district profile). Zones that show
    # neither aren't necessarily wrong, but the TOP zones overall should show
    # this dipole somewhere, or something upstream is broken. Ranked by
    # per-dock lower confidence bound (primary metric) now, not raw volume
    # or the plain per-dock mean.
    print("\n[heatmap] commuter-dipole sanity check -- top 10 zones by per-dock lower-bound:")
    for zone in zone_order_dock[:10]:
        dep_hour, dep_val = peak_hour(agg, zone, "dep_net_lost_per_week")
        arr_hour, arr_val = peak_hour(agg, zone, "arr_net_lost_per_week")
        print(f"  {zone}: peak bike-starved {dep_hour} ({dep_val:.1f}/wk), peak dock-starved {arr_hour} ({arr_val:.1f}/wk)")


if __name__ == "__main__":
    main()
