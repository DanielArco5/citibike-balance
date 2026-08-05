"""Station table from GBFS station_information.json, reconciled against
trip data.

Per SCHEMA.md's Phase 0 crosswalk finding: GBFS's `station_id` field is a
UUID and is NOT the join key against trip data. Trip data's
`start_station_id`/`end_station_id` (e.g. "5406.02") matches GBFS's
`short_name` field. This module uses `short_name` as the canonical
`station_id` in its output -- joining on the UUID would silently produce
zero matches.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
TRIPS_PATH = REPO_ROOT / "data" / "interim" / "trips.parquet"
RAW_CACHE_PATH = REPO_ROOT / "data" / "raw" / "gbfs" / "station_information.json"
OUT_PATH = REPO_ROOT / "data" / "interim" / "stations.parquet"

STATION_INFORMATION_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"


def fetch_station_information(cache_path: Path = RAW_CACHE_PATH) -> dict:
    """Return the raw GBFS station_information payload, fetching once and
    caching to disk. GBFS station_information changes rarely (new stations
    added occasionally) so a stale cache is fine for this pipeline -- delete
    the file to force a refresh."""
    if cache_path.exists():
        print(f"[gbfs] using cached {cache_path}")
        return json.loads(cache_path.read_text())

    print(f"[gbfs] fetching {STATION_INFORMATION_URL}")
    resp = requests.get(STATION_INFORMATION_URL, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))
    print(f"[gbfs] cached raw response to {cache_path}")
    return payload


def build_station_table(payload: dict) -> pl.DataFrame:
    """station_id (== GBFS short_name, the trip-data join key), name, lat,
    lng, capacity -- one row per GBFS station."""
    stations = payload["data"]["stations"]
    df = pl.DataFrame(stations).select(
        pl.col("short_name").alias("station_id"),
        pl.col("name"),
        pl.col("lat"),
        pl.col("lon").alias("lng"),
        pl.col("capacity"),
    )
    n = df.height
    n_dupe = n - df.select(pl.col("station_id").n_unique()).item()
    if n_dupe:
        print(f"[gbfs] WARNING: {n_dupe} duplicate short_name values in station_information")
    print(f"[gbfs] station table: {n:,} stations")
    return df


def trip_station_activity(trips_path: Path = TRIPS_PATH) -> pl.DataFrame:
    """One row per station_id seen in trips.parquet, with `activity` =
    departures + arrivals (trip-legs touching that station). This is the
    volume metric used to weight the capacity-match reconciliation --
    a station with no capacity match is worse news if it's high-traffic."""
    lf = pl.scan_parquet(trips_path)
    departures = lf.group_by("start_station_id").agg(pl.len().alias("departures")).rename(
        {"start_station_id": "station_id"}
    )
    arrivals = lf.group_by("end_station_id").agg(pl.len().alias("arrivals")).rename(
        {"end_station_id": "station_id"}
    )
    return (
        departures.join(arrivals, on="station_id", how="full", coalesce=True)
        .with_columns(
            pl.col("departures").fill_null(0),
            pl.col("arrivals").fill_null(0),
        )
        .with_columns((pl.col("departures") + pl.col("arrivals")).alias("activity"))
        .select("station_id", "activity")
        .collect()
    )


def reconcile_capacity(stations: pl.DataFrame, activity: pl.DataFrame) -> dict:
    """Report how many trip stations have no capacity match, and what
    fraction of trip volume (departures + arrivals) they represent."""
    joined = activity.join(stations.select("station_id", "capacity"), on="station_id", how="left")

    total_stations = joined.height
    total_activity = joined["activity"].sum()

    unmatched = joined.filter(pl.col("capacity").is_null())
    n_unmatched = unmatched.height
    unmatched_activity = unmatched["activity"].sum()

    frac_stations = n_unmatched / total_stations if total_stations else 0.0
    frac_activity = unmatched_activity / total_activity if total_activity else 0.0

    print(
        f"[gbfs] trip stations with no capacity match: {n_unmatched:,} / "
        f"{total_stations:,} ({frac_stations:.2%} of stations)"
    )
    print(
        f"[gbfs] trip volume represented by unmatched stations: "
        f"{unmatched_activity:,} / {total_activity:,} ({frac_activity:.2%} of activity)"
    )
    if frac_activity > 0.02:
        print(
            f"[gbfs] ⚠ {frac_activity:.2%} of trip volume touches a station with no "
            "capacity data. This is not negligible -- the demand/censoring model "
            "in §3 cannot compute stockout status for these stations without "
            "capacity, and they will need to be dropped or backfilled explicitly, "
            "not silently ignored."
        )

    return {
        "total_stations": total_stations,
        "total_activity": total_activity,
        "n_unmatched": n_unmatched,
        "unmatched_activity": unmatched_activity,
        "frac_stations_unmatched": frac_stations,
        "frac_activity_unmatched": frac_activity,
        "unmatched_station_ids": unmatched["station_id"].to_list(),
    }


def main() -> None:
    payload = fetch_station_information()
    stations = build_station_table(payload)

    activity = trip_station_activity()
    reconcile_capacity(stations, activity)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stations.write_parquet(OUT_PATH)
    print(f"[gbfs] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
