"""Build the (station, 15-min interval) panel -- the spine of everything
downstream, per SPEC.md §2.

Dense, not sparse: every station gets one row per 15-min interval between
its OWN first and last observed trip (not the global date range), including
intervals with zero departures and zero arrivals. Padding a station's whole
active lifetime with rows it never earned would bias every hour-of-week
average toward zero for stations that opened late or closed early; padding
before it opened / after it closed would fabricate "the station existed but
did nothing" rows that never happened.

Capacity/zone join: 158 of 2,428 trip-active stations (6.5% of stations,
2.83% of trip volume -- see src/ingest/gbfs.py's reconcile_capacity) have no
match in the current GBFS station_information feed, so no capacity and no
zone. Decision: KEEP them, with capacity/zone_h3/zone_agg left null, rather
than dropping them. Reasons:
  1. Dropping would break sum(departures) == sum(arrivals) == trip count --
     a trip touching a dropped station loses its departure or arrival leg
     (or both, if both endpoints are unmatched), silently underscoring the
     very invariant this module exists to guarantee.
  2. It would also quietly understate demand in whatever H3/agglomerative
     zone those stations sit in, at the panel-construction phase where
     nothing yet needs capacity.
  3. Capacity is only required to compute stockout status (SPEC.md §3/§4).
     That's a later phase's problem to solve explicitly (skip or backfill
     null-capacity station-intervals), not this one's to paper over.
This means downstream code MUST NOT treat a null capacity as "no cap" or
coerce it to 0/inf -- it must filter or handle it explicitly.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
TRIPS_PATH = REPO_ROOT / "data" / "interim" / "trips.parquet"
STATIONS_PATH = REPO_ROOT / "data" / "processed" / "stations_zoned.parquet"
WEATHER_PATH = REPO_ROOT / "data" / "interim" / "weather.parquet"
OUT_PATH = REPO_ROOT / "data" / "processed" / "panel.parquet"

INTERVAL = "15m"
INTERVAL_MINUTES = 15


def floor_to_interval(col: pl.Expr, interval: str = INTERVAL) -> pl.Expr:
    return col.dt.truncate(interval)


def build_departures(trips: pl.LazyFrame, interval: str = INTERVAL) -> pl.LazyFrame:
    return trips.group_by(
        pl.col("start_station_id").alias("station_id"),
        floor_to_interval(pl.col("started_at"), interval).alias("interval_start"),
    ).agg(pl.len().alias("departures"))


def build_arrivals(trips: pl.LazyFrame, interval: str = INTERVAL) -> pl.LazyFrame:
    return trips.group_by(
        pl.col("end_station_id").alias("station_id"),
        floor_to_interval(pl.col("ended_at"), interval).alias("interval_start"),
    ).agg(pl.len().alias("arrivals"))


def combine_observed(departures: pl.LazyFrame, arrivals: pl.LazyFrame) -> pl.LazyFrame:
    """Only intervals with nonzero departures or arrivals -- sparse by
    construction. This is also what defines each station's observed
    lifetime: its min/max interval_start here IS its first/last trip."""
    return (
        departures.join(arrivals, on=["station_id", "interval_start"], how="full", coalesce=True)
        .with_columns(
            pl.col("departures").fill_null(0),
            pl.col("arrivals").fill_null(0),
        )
    )


def station_bounds(observed: pl.LazyFrame) -> pl.DataFrame:
    return (
        observed.group_by("station_id")
        .agg(
            pl.col("interval_start").min().alias("first_interval"),
            pl.col("interval_start").max().alias("last_interval"),
        )
        .collect()
    )


def build_dense_grid(bounds: pl.DataFrame, interval: str = INTERVAL) -> pl.DataFrame:
    """One row per station per interval in [first_interval, last_interval],
    inclusive -- via a row-wise datetime range + explode, not a full
    station x global-calendar cross join, so a station that only lived for
    six weeks doesn't get padded across the whole year."""
    return (
        bounds.select(
            "station_id",
            pl.datetime_ranges(
                "first_interval", "last_interval", interval=interval, closed="both"
            ).alias("interval_start"),
        )
        .explode("interval_start")
    )


def assemble_counts(dense_grid: pl.DataFrame, observed: pl.LazyFrame) -> pl.DataFrame:
    return (
        dense_grid.lazy()
        .join(observed, on=["station_id", "interval_start"], how="left")
        .with_columns(
            pl.col("departures").fill_null(0).cast(pl.UInt32),
            pl.col("arrivals").fill_null(0).cast(pl.UInt32),
        )
        .collect()
    )


def join_station_metadata(panel: pl.DataFrame, stations: pl.DataFrame) -> pl.DataFrame:
    """Left join capacity + zones. Deliberately left join, not inner --
    see module docstring on the 158 unmatched stations."""
    meta = stations.select(
        "station_id",
        "capacity",
        "zone_h3",
        pl.col("zone_agglomerative").alias("zone_agg"),
    )
    return panel.join(meta, on="station_id", how="left")


def us_federal_holidays(min_date: dt.date, max_date: dt.date) -> set[dt.date]:
    """US federal holidays (New Year's, MLK, Presidents, Memorial, Juneteenth,
    Independence, Labor, Columbus, Veterans, Thanksgiving, Christmas) via
    pandas' calendar. Not NYC-specific (no NYC public-school or transit
    observances beyond the federal set) -- documented approximation, easy to
    swap for a curated list later if the demand model needs finer granularity."""
    from pandas.tseries.holiday import USFederalHolidayCalendar

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=min_date, end=max_date)
    return set(holidays.date)


def add_calendar_features(panel: pl.DataFrame) -> pl.DataFrame:
    min_date = panel["interval_start"].min().date()
    max_date = panel["interval_start"].max().date()
    holidays = us_federal_holidays(min_date, max_date)

    panel = panel.with_columns(
        pl.col("interval_start").dt.hour().cast(pl.Int16).alias("hour"),
        # polars weekday() is ISO: 1=Mon..7=Sun. Rebase to 0=Mon..6=Sun.
        # Both weekday() and hour() come back as Int8, which overflows
        # (max 127) once dow*24 is computed below for dow >= 6 -- cast to
        # Int16 up front rather than at the multiply.
        (pl.col("interval_start").dt.weekday().cast(pl.Int16) - 1).alias("dow"),
        pl.col("interval_start").dt.month().alias("month"),
        pl.col("interval_start").dt.date().is_in(holidays).alias("is_holiday"),
    )
    return panel.with_columns(
        (pl.col("dow") * 24 + pl.col("hour")).alias("hour_of_week"),
    )


def join_weather(panel: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """Weather is hourly; join each 15-min interval to the hour it falls in.
    Both sides are naive local (America/New_York) timestamps -- see
    src/ingest/weather.py -- so no timezone conversion is needed here."""
    w = weather.rename({"timestamp": "weather_hour"})
    return (
        panel.with_columns(pl.col("interval_start").dt.truncate("1h").alias("weather_hour"))
        .join(w, on="weather_hour", how="left")
        .drop("weather_hour")
    )


def report_capacity_coverage(panel: pl.DataFrame) -> None:
    n_rows = panel.height
    n_null_rows = panel.filter(pl.col("capacity").is_null()).height
    n_stations = panel["station_id"].n_unique()
    n_null_stations = panel.filter(pl.col("capacity").is_null())["station_id"].n_unique()
    print(
        f"[panel] capacity coverage: {n_null_stations:,} / {n_stations:,} stations "
        f"({n_null_stations / n_stations:.2%}) have null capacity -- "
        f"{n_null_rows:,} / {n_rows:,} panel rows ({n_null_rows / n_rows:.2%}). "
        "Kept, not dropped -- see module docstring. Downstream stockout logic "
        "must handle null capacity explicitly."
    )


def run_assertions(panel: pl.DataFrame, trips_lf: pl.LazyFrame, bounds: pl.DataFrame) -> None:
    n_trips = trips_lf.select(pl.len()).collect().item()
    dep_sum = panel["departures"].sum()
    arr_sum = panel["arrivals"].sum()
    assert dep_sum == n_trips, f"sum(departures)={dep_sum:,} != trip count {n_trips:,}"
    assert arr_sum == n_trips, f"sum(arrivals)={arr_sum:,} != trip count {n_trips:,}"
    print(f"[panel] assertion passed: sum(departures) == sum(arrivals) == trip count ({n_trips:,})")

    row_counts = panel.group_by("station_id").agg(pl.len().alias("n_rows"))
    check = bounds.join(row_counts, on="station_id").with_columns(
        (
            (pl.col("last_interval") - pl.col("first_interval")).dt.total_minutes()
            // INTERVAL_MINUTES
            + 1
        ).alias("expected_rows")
    )
    gapped = check.filter(pl.col("n_rows") != pl.col("expected_rows"))
    assert gapped.height == 0, (
        f"{gapped.height} stations have missing intervals between their first "
        f"and last observed trip: {gapped['station_id'].to_list()[:10]}"
    )
    print("[panel] assertion passed: no missing intervals per station within its observed lifetime")

    actual_bounds = panel.group_by("station_id").agg(
        pl.col("interval_start").min().alias("actual_first"),
        pl.col("interval_start").max().alias("actual_last"),
    )
    mismatch = bounds.join(actual_bounds, on="station_id").filter(
        (pl.col("first_interval") != pl.col("actual_first"))
        | (pl.col("last_interval") != pl.col("actual_last"))
    )
    assert mismatch.height == 0, (
        f"{mismatch.height} stations have panel rows outside their observed "
        "first/last trip -- pre-opening or post-closing rows were zero-filled"
    )
    print("[panel] assertion passed: no station-intervals before opening / after closing")


def build_panel(
    trips_lf: pl.LazyFrame,
    stations: pl.DataFrame,
    weather: pl.DataFrame,
    interval: str = INTERVAL,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (panel, bounds) -- bounds is returned alongside so callers
    (main() and tests) can run the same assertions against it."""
    departures = build_departures(trips_lf, interval)
    arrivals = build_arrivals(trips_lf, interval)
    observed = combine_observed(departures, arrivals)

    bounds = station_bounds(observed)
    dense_grid = build_dense_grid(bounds, interval)
    panel = assemble_counts(dense_grid, observed)

    panel = join_station_metadata(panel, stations)
    panel = add_calendar_features(panel)
    panel = join_weather(panel, weather)

    return panel, bounds


def main() -> None:
    trips_lf = pl.scan_parquet(TRIPS_PATH)
    stations = pl.read_parquet(STATIONS_PATH)
    weather = pl.read_parquet(WEATHER_PATH)

    panel, bounds = build_panel(trips_lf, stations, weather)

    run_assertions(panel, trips_lf, bounds)
    report_capacity_coverage(panel)

    n_rows = panel.height
    mem_mb = panel.estimated_size("mb")
    print(f"[panel] {n_rows:,} rows, {panel['station_id'].n_unique():,} stations")
    print(f"[panel] in-memory footprint: {mem_mb:,.1f} MB")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(OUT_PATH)
    print(f"[panel] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
