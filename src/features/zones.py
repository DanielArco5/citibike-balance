"""Assign each station to a substitution-neighborhood zone, two ways, per
SPEC.md §2:
  (a) H3 resolution 8 (~460m edge) -- fast, reproducible, no tuning.
  (b) Agglomerative clustering (complete linkage) on projected (planar)
      coordinates, distance_threshold=700m.

Both take the station table built by src/ingest/gbfs.py (station_id, name,
lat, lng, capacity) and add a zone column. Neither zone id is comparable
across methods -- they're two independent partitions of the same stations.

Why not DBSCAN: an earlier version used DBSCAN(eps=350m). eps controls
neighbor distance, not cluster diameter -- on a contiguous grid like Citi
Bike's dense core, DBSCAN chains transitively through it (a is within eps
of b, b within eps of c, ...) producing one ~1,150-station "zone" spanning
Brooklyn to Queens, nothing like a walkable neighborhood. Complete-linkage
agglomerative clustering merges by the *maximum* pairwise distance between
clusters, so distance_threshold directly bounds cluster diameter -- no
cluster can end up wider than the threshold, by construction.
"""
from __future__ import annotations

from pathlib import Path

import h3
import numpy as np
import polars as pl
from pyproj import Transformer
from sklearn.cluster import AgglomerativeClustering

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIONS_PATH = REPO_ROOT / "data" / "interim" / "stations.parquet"
OUT_PATH = REPO_ROOT / "data" / "processed" / "stations_zoned.parquet"

H3_RESOLUTION = 8

# NYC sits in UTM zone 18N. Projecting here (rather than treating lat/lng
# as if it were planar) is what makes a Euclidean distance_threshold of
# 700 actually mean 700 meters on the ground.
_WGS84_TO_UTM18N = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)

# The walkable substitution radius per SPEC.md §2 ("~300-400m walking
# distance"). Complete linkage bounds *diameter*, not radius, so 700m here
# caps a cluster at roughly the same walkable span as a ~350m-radius circle.
AGGLOMERATIVE_DISTANCE_THRESHOLD_M = 700.0


def _project(stations: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lngs = stations["lng"].to_numpy()
    lats = stations["lat"].to_numpy()
    return _WGS84_TO_UTM18N.transform(lngs, lats)


def assign_h3_zones(stations: pl.DataFrame, resolution: int = H3_RESOLUTION) -> pl.DataFrame:
    cells = [
        h3.latlng_to_cell(lat, lng, resolution)
        for lat, lng in zip(stations["lat"], stations["lng"])
    ]
    return stations.with_columns(pl.Series("zone_h3", cells))


def assign_agglomerative_zones(
    stations: pl.DataFrame,
    distance_threshold_m: float = AGGLOMERATIVE_DISTANCE_THRESHOLD_M,
) -> pl.DataFrame:
    xs, ys = _project(stations)
    coords = np.column_stack([xs, ys])

    model = AgglomerativeClustering(
        n_clusters=None,
        linkage="complete",
        distance_threshold=distance_threshold_m,
    ).fit(coords)

    zone_ids = [f"agglom_{label}" for label in model.labels_]
    return stations.with_columns(pl.Series("zone_agglomerative", zone_ids))


def zone_distribution(stations: pl.DataFrame, zone_col: str) -> pl.DataFrame:
    return (
        stations.group_by(zone_col)
        .agg(pl.len().alias("n_stations"))
        .sort("n_stations", descending=True)
    )


def _pairwise_max_distance(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    diffs = coords[:, None, :] - coords[None, :, :]
    return float(np.sqrt((diffs**2).sum(-1)).max())


def zone_diameters_m(stations: pl.DataFrame, zone_col: str) -> pl.DataFrame:
    """Max pairwise distance (meters) between stations within each zone --
    the empirical diameter, used to sanity-check a distance-threshold
    clustering actually respects the threshold it was given."""
    xs, ys = _project(stations)
    by_zone: dict[str, list[tuple[float, float]]] = {}
    for zid, x, y in zip(stations[zone_col].to_list(), xs, ys):
        by_zone.setdefault(zid, []).append((x, y))

    rows = [
        (zid, len(pts), _pairwise_max_distance(np.array(pts)))
        for zid, pts in by_zone.items()
    ]
    return pl.DataFrame(rows, schema=["zone", "n_stations", "diameter_m"], orient="row").sort(
        "diameter_m", descending=True
    )


def report_zone_summary(stations: pl.DataFrame, zone_col: str, label: str) -> None:
    dist = zone_distribution(stations, zone_col)
    counts = dist["n_stations"]
    n_zones = dist.height
    n_singleton = counts.filter(counts == 1).len()
    print(f"[zones] {label}: {n_zones:,} zones over {stations.height:,} stations")
    print(
        f"[zones] {label}: stations/zone -- min {counts.min()}, "
        f"p25 {int(counts.quantile(0.25))}, median {int(counts.median())}, "
        f"p75 {int(counts.quantile(0.75))}, max {counts.max()}, "
        f"mean {counts.mean():.2f}"
    )
    print(f"[zones] {label}: {n_singleton:,} singleton zones ({n_singleton / n_zones:.1%} of zones)")


def report_diameter_summary(
    stations: pl.DataFrame, zone_col: str, label: str, threshold_m: float
) -> None:
    diam = zone_diameters_m(stations, zone_col)
    max_diameter = diam["diameter_m"][0]
    max_zone = diam["zone"][0]
    print(
        f"[zones] {label}: largest zone diameter {max_diameter:.0f}m "
        f"(zone {max_zone}, {diam['n_stations'][0]} stations), threshold {threshold_m:.0f}m"
    )
    if max_diameter > threshold_m + 1e-6:
        print(
            f"[zones] ⚠ {label}: a zone exceeds the {threshold_m:.0f}m distance "
            "threshold -- clustering did not bound diameter as expected, investigate."
        )


def main() -> None:
    stations = pl.read_parquet(STATIONS_PATH)

    stations = assign_h3_zones(stations)
    stations = assign_agglomerative_zones(stations)

    report_zone_summary(stations, "zone_h3", "H3 res 8")
    report_zone_summary(stations, "zone_agglomerative", "Agglomerative (complete, 700m)")
    report_diameter_summary(
        stations,
        "zone_agglomerative",
        "Agglomerative (complete, 700m)",
        AGGLOMERATIVE_DISTANCE_THRESHOLD_M,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stations.write_parquet(OUT_PATH)
    print(f"[zones] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
