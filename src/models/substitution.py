"""Substitution netting: gross unmet demand -> displaced vs. net-lost, per
SPEC.md §3 step 4.

SPEC.md's framing: "the rider walked 200m and took a bike from the next
station. Estimate this: for each stockout event at s, compare neighbor
station departures against their own counterfactual. The uplift at
neighbors is displaced, not lost." This module implements exactly that,
plus three extensions the naive version needs to be honest:

1. **Two directions, not one, and NOT the same interpretation.** SPEC.md
   §3 step 4 only describes the bike-stockout (departures) side. Per this
   phase's brief: a blocked departure may be a genuinely lost trip; a
   blocked arrival is a rider re-routing to the next station with an open
   dock (SPEC.md §4's forward-simulation framing: "no dock -> arrival must
   re-route... count it as a degraded trip"), which is closer to "found a
   substitute" by default. To get an actual net_lost number for the
   dock-starved side of the §5 heatmap (which requires it), this module
   runs the SAME counterfactual-uplift methodology symmetrically on
   arrivals -- checking whether neighbor stations' ARRIVALS rise above
   their own counterfactual during a dock-full event. The result means
   something different for each direction: arr_net_lost is not "abandoned
   trip," it's "demand that didn't even complete via a nearby reroute" --
   expected to be small, and should be reported/labeled as such, not
   treated as the departure-side quantity's twin.

2. **Counterfactual = the neighbor's own D_hat.** Already fit and
   calibrated in demand.py -- D_hat for a station's own uncensored
   conditions IS the counterfactual "what would this neighbor's flow look
   like without a nearby stockout pulling in extra riders." No second
   model needed.

3. **Proportional allocation, not full double-counting.** If two stockouts
   both sit within 400m of the same neighbor, naively crediting each with
   the neighbor's FULL uplift would double-count it and overstate total
   displacement whenever stockouts cluster spatially/temporally (which
   they do -- commute-hour system stress hits many nearby stations at
   once). Each neighbor's uplift at a given interval is split across all
   competing nearby stockouts in proportion to their own gross_unmet, so
   the neighbor's uplift is never attributed more than once in total.

Caveat that belongs in the writeup, not just here: "neighbor demand exceeds
its own model-predicted baseline" is CORRELATIONAL evidence of
substitution, not proof of it -- a neighbor can run hot for unrelated
reasons (its own local event, weather microclimate, etc.) at the same time
a nearby station stocks out. This is an estimate of an upper bound on
displacement, same spirit as SPEC.md's "honest error bars over false
precision."
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.neighbors import BallTree

REPO_ROOT = Path(__file__).resolve().parents[2]
# Production input: demand.py's per-month checkpoints (see the Phase 6
# memory-safety restructure, ~/.claude/plans/lucky-
# leaping-zebra.md). Read via pl.scan_parquet(dir/"*.parquet") -- this
# module's algorithm was already bounded (neighbor pairs precomputed once,
# joined only against stockout rows, never a full station x neighbor x
# interval materialization), so only the I/O layer changes here, not
# estimate_substitution/add_substitution.
UNMET_DEMAND_DIR = REPO_ROOT / "data" / "processed" / "unmet_demand"
# Legacy single-file fallback, e.g. after demand.py's optional --consolidate.
UNMET_DEMAND_PATH = REPO_ROOT / "data" / "processed" / "unmet_demand.parquet"
STATIONS_PATH = REPO_ROOT / "data" / "processed" / "stations_zoned.parquet"
OUT_PATH = REPO_ROOT / "data" / "processed" / "unmet_demand_net.parquet"

def load_unmet_demand() -> pl.DataFrame:
    """Scans the per-month checkpoint directory (falls back to the single
    consolidated file if the directory doesn't exist -- e.g. a manual/
    small-fixture run). No column pruning: this module's OUTPUT
    (unmet_demand_net.parquet) is add_substitution's input frame plus 4 new
    columns, so every column downstream consumers need (heatmap.py's
    zone_agg/hour_of_week, in particular) has to survive the read -- the
    checkpoint schema is already narrow (18 cols), so pruning bought little
    memory headroom for real correctness risk."""
    if UNMET_DEMAND_DIR.exists():
        return pl.scan_parquet(UNMET_DEMAND_DIR / "*.parquet").collect()
    return pl.scan_parquet(UNMET_DEMAND_PATH).collect()

EARTH_RADIUS_M = 6_371_000.0
SUBSTITUTION_RADIUS_M = 400.0

DIRECTIONS = (
    # (prefix, observed_col, censor_col)
    ("dep", "departures", "is_bike_empty"),
    ("arr", "arrivals", "is_dock_full"),
)


# ---------------------------------------------------------------------------
# Spatial neighbors
# ---------------------------------------------------------------------------


def build_neighbor_pairs(
    stations: pl.DataFrame,
    radius_m: float = SUBSTITUTION_RADIUS_M,
    lat_col: str = "lat",
    lng_col: str = "lng",
    station_col: str = "station_id",
) -> pl.DataFrame:
    """All (station, neighbor) pairs with great-circle distance <= radius_m,
    excluding self-pairs. BallTree with a haversine metric -- accounts for
    Earth's curvature (irrelevant at 400m in practice, but it's the correct
    tool and already available via scikit-learn, no new dependency)."""
    lat = np.radians(stations[lat_col].to_numpy())
    lng = np.radians(stations[lng_col].to_numpy())
    coords = np.column_stack([lat, lng])
    tree = BallTree(coords, metric="haversine")
    radius_rad = radius_m / EARTH_RADIUS_M
    indices, distances = tree.query_radius(coords, r=radius_rad, return_distance=True)

    station_ids = stations[station_col].to_list()
    pair_station, pair_neighbor, pair_dist = [], [], []
    for i, (idxs, dists) in enumerate(zip(indices, distances)):
        for j, d in zip(idxs, dists):
            if j == i:
                continue
            pair_station.append(station_ids[i])
            pair_neighbor.append(station_ids[j])
            pair_dist.append(float(d) * EARTH_RADIUS_M)

    return pl.DataFrame(
        {"station_id": pair_station, "neighbor_id": pair_neighbor, "distance_m": pair_dist}
    )


# ---------------------------------------------------------------------------
# Substitution / displacement estimation
# ---------------------------------------------------------------------------


def estimate_substitution(
    unmet: pl.DataFrame,
    neighbor_pairs: pl.DataFrame,
    gross_col: str,
    observed_col: str,
    dhat_col: str,
    censor_col: str,
    out_prefix: str,
    station_col: str = "station_id",
    order_col: str = "interval_start",
) -> pl.DataFrame:
    """For every (station, interval) with gross_col > 0 ("a stockout"),
    sums each same-interval neighbor's uplift = max(0, observed - D_hat)
    within radius_m (neighbor_pairs already encodes the radius), zeroing a
    neighbor's contribution if IT was itself censored at that interval (its
    observed count is then a censored lower bound, not clean evidence of
    absorbing displaced riders). Splits each neighbor's uplift
    proportionally across every stockout competing for it at that instant
    (see module docstring, point 3) before capping {prefix}_displaced at
    the station's own gross_col (can't displace more than 100% of your own
    unmet demand). Returns (station_id, interval_start,
    {prefix}_displaced, {prefix}_net_lost) -- one row per stockout
    (station, interval); callers left-join this back onto the full unmet
    table and fill 0 for non-stockout rows."""
    uplift_tbl = unmet.select(
        station_col,
        order_col,
        pl.when(pl.col(censor_col))
        .then(0.0)
        .otherwise((pl.col(observed_col) - pl.col(dhat_col)).clip(lower_bound=0.0))
        .alias("uplift"),
    )

    stockout = unmet.filter(pl.col(gross_col) > 0).select(
        station_col, order_col, pl.col(gross_col).alias("gross_unmet")
    )
    if stockout.height == 0:
        return pl.DataFrame(
            schema={
                station_col: unmet.schema[station_col],
                order_col: unmet.schema[order_col],
                f"{out_prefix}_displaced": pl.Float64,
                f"{out_prefix}_net_lost": pl.Float64,
            }
        )

    edges = stockout.join(neighbor_pairs, on=station_col)
    edges = edges.join(
        uplift_tbl.rename({station_col: "neighbor_id", "uplift": "neighbor_uplift"}),
        left_on=["neighbor_id", order_col],
        right_on=["neighbor_id", order_col],
        how="left",
    ).with_columns(pl.col("neighbor_uplift").fill_null(0.0))

    competing_totals = edges.group_by(["neighbor_id", order_col]).agg(
        pl.col("gross_unmet").sum().alias("total_competing_gross_unmet")
    )
    edges = edges.join(competing_totals, on=["neighbor_id", order_col])
    edges = edges.with_columns(
        pl.when(pl.col("total_competing_gross_unmet") > 0)
        .then(pl.col("neighbor_uplift") * pl.col("gross_unmet") / pl.col("total_competing_gross_unmet"))
        .otherwise(0.0)
        .alias("attributed_uplift")
    )

    displaced = edges.group_by([station_col, order_col]).agg(
        pl.col("attributed_uplift").sum().alias("displaced_raw")
    )

    result = (
        stockout.join(displaced, on=[station_col, order_col], how="left")
        .with_columns(pl.col("displaced_raw").fill_null(0.0))
        .with_columns(
            pl.min_horizontal(pl.col("displaced_raw"), pl.col("gross_unmet")).alias(f"{out_prefix}_displaced")
        )
        .with_columns(
            (pl.col("gross_unmet") - pl.col(f"{out_prefix}_displaced")).alias(f"{out_prefix}_net_lost")
        )
    )
    return result.select(station_col, order_col, f"{out_prefix}_displaced", f"{out_prefix}_net_lost")


def add_substitution(unmet: pl.DataFrame, neighbor_pairs: pl.DataFrame) -> pl.DataFrame:
    """Runs estimate_substitution for both directions and joins the results
    back onto the full unmet table, filling 0 for non-stockout rows (0
    gross_unmet -> 0 displaced -> 0 net_lost, definitionally)."""
    out = unmet
    for prefix, observed_col, censor_col in DIRECTIONS:
        sub = estimate_substitution(
            unmet,
            neighbor_pairs,
            gross_col=f"{prefix}_gross_unmet",
            observed_col=observed_col,
            dhat_col=f"{prefix}_D_hat",
            censor_col=censor_col,
            out_prefix=prefix,
        )
        out = out.join(sub, on=["station_id", "interval_start"], how="left").with_columns(
            pl.col(f"{prefix}_displaced").fill_null(0.0),
            pl.col(f"{prefix}_net_lost").fill_null(0.0),
        )
    return out


def summarize_displaced_vs_lost(unmet_net: pl.DataFrame) -> pl.DataFrame:
    """System-wide: what fraction of gross unmet demand is displaced
    (absorbed by a nearby station) vs. net-lost, per direction. Reported
    separately per module docstring point 1 -- arr's net_lost is not the
    departures-side quantity's twin."""
    rows = []
    for prefix, _observed_col, _censor_col in DIRECTIONS:
        gross = unmet_net[f"{prefix}_gross_unmet"].sum()
        displaced = unmet_net[f"{prefix}_displaced"].sum()
        net_lost = unmet_net[f"{prefix}_net_lost"].sum()
        rows.append(
            {
                "direction": prefix,
                "gross_unmet": gross,
                "displaced": displaced,
                "net_lost": net_lost,
                "frac_displaced": displaced / gross if gross else None,
                "frac_net_lost": net_lost / gross if gross else None,
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    unmet = load_unmet_demand()
    stations = pl.read_parquet(STATIONS_PATH)

    neighbor_pairs = build_neighbor_pairs(stations)
    print(
        f"[substitution] {neighbor_pairs.height:,} station-neighbor pairs within "
        f"{SUBSTITUTION_RADIUS_M:.0f}m ({stations.height:,} stations, "
        f"{neighbor_pairs.height / stations.height:.1f} neighbors/station avg)"
    )

    unmet_net = add_substitution(unmet, neighbor_pairs)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    unmet_net.write_parquet(OUT_PATH)
    print(f"[substitution] wrote {OUT_PATH} ({unmet_net.height:,} rows)")

    summary = summarize_displaced_vs_lost(unmet_net)
    print("[substitution] system-wide gross unmet -> displaced vs. net-lost, by direction:")
    print(summary)


if __name__ == "__main__":
    main()
