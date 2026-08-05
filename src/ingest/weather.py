"""Hourly NYC weather from Open-Meteo's historical archive, for the date
range covered by trips.parquet. Cached to parquet -- this is a network call
against a third-party API, not something to re-fetch on every run.

Weather is a single point series (one grid cell), not per-station: Open-
Meteo's archive resolution is far coarser than station spacing, so there's
nothing to gain from querying per-station. The anchor point is Central Park,
a conventional "NYC weather" reference point.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
TRIPS_PATH = REPO_ROOT / "data" / "interim" / "trips.parquet"
OUT_PATH = REPO_ROOT / "data" / "interim" / "weather.parquet"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
NYC_LAT, NYC_LNG = 40.7789, -73.9692  # Central Park

HOURLY_VARS = ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m"]

RENAME = {
    "temperature_2m": "temp_c",
    "precipitation": "precip_mm",
    "wind_speed_10m": "wind_kph",
    "relative_humidity_2m": "humidity_pct",
}


def trip_date_range(trips_path: Path = TRIPS_PATH) -> tuple[dt.date, dt.date]:
    bounds = (
        pl.scan_parquet(trips_path)
        .select(pl.col("started_at").min().alias("min_ts"), pl.col("started_at").max().alias("max_ts"))
        .collect()
        .row(0, named=True)
    )
    return bounds["min_ts"].date(), bounds["max_ts"].date()


def fetch_hourly_weather(start_date: dt.date, end_date: dt.date) -> pl.DataFrame:
    """One row per hour in [start_date, end_date], local America/New_York
    wall-clock time -- matching trips.parquet's naive local timestamps, so
    the two can be joined on hour without a timezone conversion step."""
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LNG,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "America/New_York",
    }
    print(f"[weather] fetching {start_date} .. {end_date} from {ARCHIVE_URL}")
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    hourly = payload["hourly"]
    df = pl.DataFrame({"timestamp": hourly["time"], **{v: hourly[v] for v in HOURLY_VARS}})
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M")
    ).rename(RENAME)

    n = df.height
    n_null = df.select(pl.any_horizontal(pl.all().is_null())).sum().item()
    print(f"[weather] fetched {n:,} hourly rows, {n_null:,} with a null field")
    return df


def _covers_range(cached: pl.DataFrame, start_date: dt.date, end_date: dt.date) -> bool:
    lo = cached["timestamp"].min()
    hi = cached["timestamp"].max()
    return lo.date() <= start_date and hi.date() >= end_date


def get_weather(cache_path: Path = OUT_PATH) -> pl.DataFrame:
    start_date, end_date = trip_date_range()

    if cache_path.exists():
        cached = pl.read_parquet(cache_path)
        if _covers_range(cached, start_date, end_date):
            print(f"[weather] using cached {cache_path} (covers {start_date} .. {end_date})")
            return cached
        print(f"[weather] cache at {cache_path} doesn't cover {start_date} .. {end_date}, refetching")

    df = fetch_hourly_weather(start_date, end_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    print(f"[weather] wrote {cache_path}")
    return df


def main() -> None:
    get_weather()


if __name__ == "__main__":
    main()
