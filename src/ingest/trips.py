"""Normalize Citi Bike monthly trip CSVs into one parquet with a stable schema.

Era layouts are taken verbatim from SCHEMA.md (Phase 0). Two eras are known:
  Era A (pre-transition, e.g. 2019-09): space-separated column names,
    integer station ids, rider demographics, no ride_id.
  Era B (post-transition, e.g. 2021-01 onward): ride_id present, underscored
    column names, decimal-formatted string station ids, no demographics.

A file whose columns match neither era raises immediately -- never guessed,
never silently coerced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
TRIPS_DIR = REPO_ROOT / "data" / "raw" / "trips"
OUT_PATH = REPO_ROOT / "data" / "interim" / "trips.parquet"

TARGET_SCHEMA = [
    "ride_id",
    "started_at",
    "ended_at",
    "start_station_id",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "rideable_type",
    "member_casual",
]

# Required columns per era, per SCHEMA.md. A file must contain all of these
# (extra columns are fine and ignored) to be recognized as that era.
ERA_A_REQUIRED = {
    "tripduration",
    "starttime",
    "stoptime",
    "start station id",
    "start station latitude",
    "start station longitude",
    "end station id",
    "end station latitude",
    "end station longitude",
    "bikeid",
    "usertype",
}

ERA_B_REQUIRED = {
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_id",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
}

# Timestamps differ only in fractional-second digit count between eras
# (Era A: 4 digits, Era B: 3) -- %.f in polars' strptime matches either.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S%.f"


def detect_era(path: Path) -> str:
    cols = set(pl.scan_csv(path, n_rows=0).collect_schema().names())
    if ERA_B_REQUIRED <= cols:
        return "B"
    if ERA_A_REQUIRED <= cols:
        return "A"
    raise ValueError(
        f"{path.name}: columns match neither Era A nor Era B per SCHEMA.md.\n"
        f"Got: {sorted(cols)}\n"
        f"Era A needs: {sorted(ERA_A_REQUIRED)}\n"
        f"Era B needs: {sorted(ERA_B_REQUIRED)}"
    )


def normalize_era_a(path: Path) -> pl.LazyFrame:
    lf = pl.scan_csv(
        path,
        schema_overrides={"start station id": pl.Utf8, "end station id": pl.Utf8},
    )
    return lf.select(
        # No native ride_id in Era A -- synthesize a stable one from bike +
        # start/end time. Documented, not silent: downstream code must not
        # treat this as an authoritative Citi Bike ride_id.
        pl.concat_str(
            [pl.col("bikeid").cast(pl.Utf8), pl.col("starttime"), pl.col("stoptime")],
            separator="-",
        ).hash().cast(pl.Utf8).alias("ride_id"),
        pl.col("starttime").alias("started_at"),
        pl.col("stoptime").alias("ended_at"),
        pl.col("start station id").alias("start_station_id"),
        pl.col("end station id").alias("end_station_id"),
        pl.col("start station latitude").alias("start_lat"),
        pl.col("start station longitude").alias("start_lng"),
        pl.col("end station latitude").alias("end_lat"),
        pl.col("end station longitude").alias("end_lng"),
        # rideable_type was not recorded pre-transition -- null, not a guess.
        pl.lit(None, dtype=pl.Utf8).alias("rideable_type"),
        # usertype -> member_casual per Citi Bike's own published mapping
        # (Subscriber == annual member, Customer == day-pass/casual rider).
        pl.when(pl.col("usertype") == "Subscriber")
        .then(pl.lit("member"))
        .when(pl.col("usertype") == "Customer")
        .then(pl.lit("casual"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("member_casual"),
    )


def normalize_era_b(path: Path) -> pl.LazyFrame:
    lf = pl.scan_csv(
        path,
        schema_overrides={"start_station_id": pl.Utf8, "end_station_id": pl.Utf8},
    )
    return lf.select(
        pl.col("ride_id"),
        pl.col("started_at"),
        pl.col("ended_at"),
        pl.col("start_station_id"),
        pl.col("end_station_id"),
        pl.col("start_lat"),
        pl.col("start_lng"),
        pl.col("end_lat"),
        pl.col("end_lng"),
        pl.col("rideable_type"),
        pl.col("member_casual"),
    )


def build_raw_lazyframe(csv_paths: list[Path]) -> pl.LazyFrame:
    frames = []
    for path in csv_paths:
        era = detect_era(path)
        print(f"[ingest] {path.name}: detected Era {era}")
        frames.append(normalize_era_a(path) if era == "A" else normalize_era_b(path))
    lf = pl.concat(frames, how="vertical")
    return lf.with_columns(
        pl.col("started_at").str.strptime(pl.Datetime, TIMESTAMP_FORMAT, strict=False),
        pl.col("ended_at").str.strptime(pl.Datetime, TIMESTAMP_FORMAT, strict=False),
    )


def _row_count(lf: pl.LazyFrame) -> int:
    return lf.select(pl.len()).collect().item()


def apply_filters(lf: pl.LazyFrame) -> pl.LazyFrame:
    n = _row_count(lf)
    print(f"[ingest] rows after era-normalize + timestamp parse: {n:,}")

    lf = lf.filter(pl.col("started_at").is_not_null() & pl.col("ended_at").is_not_null())
    n_next = _row_count(lf)
    print(
        f"[ingest] dropped {n - n_next:,} rows: null/unparseable started_at or "
        f"ended_at -> {n_next:,} remain"
    )
    n = n_next

    lf = lf.filter(pl.col("start_station_id").is_not_null() & pl.col("end_station_id").is_not_null())
    n_next = _row_count(lf)
    print(
        f"[ingest] dropped {n - n_next:,} rows: null start_station_id or "
        f"end_station_id (e.g. trip ended outside a dock) -> {n_next:,} remain"
    )
    n = n_next

    duration_s = (pl.col("ended_at") - pl.col("started_at")).dt.total_seconds()
    lf = lf.filter(duration_s >= 60)
    n_next = _row_count(lf)
    print(
        f"[ingest] dropped {n - n_next:,} rows: trip duration < 60s (Citi Bike "
        f"treats these as false starts/re-docks) -> {n_next:,} remain"
    )
    n = n_next

    is_short_self_loop = (pl.col("start_station_id") == pl.col("end_station_id")) & (
        duration_s < 120
    )
    lf = lf.filter(~is_short_self_loop)
    n_next = _row_count(lf)
    print(
        f"[ingest] dropped {n - n_next:,} rows: self-loop trip < 2min (likely "
        f"test/rebalancing check, not a real ride) -> {n_next:,} remain"
    )

    return lf


def run_assertions(lf: pl.LazyFrame) -> None:
    duration_s = (pl.col("ended_at") - pl.col("started_at")).dt.total_seconds()
    stats = lf.select(
        null_started=pl.col("started_at").is_null().sum(),
        null_ended=pl.col("ended_at").is_null().sum(),
        bad_order=(duration_s <= 0).sum(),
        null_start_station=pl.col("start_station_id").is_null().sum(),
        null_end_station=pl.col("end_station_id").is_null().sum(),
    ).collect().row(0, named=True)

    assert stats["null_started"] == 0, f"{stats['null_started']} rows have null started_at"
    assert stats["null_ended"] == 0, f"{stats['null_ended']} rows have null ended_at"
    assert stats["bad_order"] == 0, f"{stats['bad_order']} rows have ended_at <= started_at"
    assert (
        stats["null_start_station"] == 0
    ), f"{stats['null_start_station']} rows have null start_station_id"
    assert (
        stats["null_end_station"] == 0
    ), f"{stats['null_end_station']} rows have null end_station_id"
    print(
        "[ingest] assertions passed: no null timestamps, ended_at > started_at, "
        "station_ids resolve"
    )


def print_monthly_totals(parquet_path: Path) -> None:
    monthly = (
        pl.scan_parquet(parquet_path)
        .with_columns(pl.col("started_at").dt.strftime("%Y-%m").alias("year_month"))
        .group_by("year_month")
        .agg(pl.len().alias("trip_count"))
        .sort("year_month")
        .collect()
    )
    print("[ingest] monthly trip totals (eyeball against Citi Bike's published counts):")
    print(monthly)


def main() -> None:
    csv_paths = sorted(TRIPS_DIR.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {TRIPS_DIR}")

    lf = build_raw_lazyframe(csv_paths)
    lf = apply_filters(lf)
    run_assertions(lf)

    final_lf = lf.select(TARGET_SCHEMA)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_lf.sink_parquet(OUT_PATH)
    print(f"[ingest] wrote {OUT_PATH}")

    print_monthly_totals(OUT_PATH)


if __name__ == "__main__":
    main()
