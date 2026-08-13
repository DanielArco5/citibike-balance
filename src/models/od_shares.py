"""Destination-choice model for the Phase 7 forward simulator (SPEC.md §4:
"destination choice model, e.g. multinomial on historical OD shares
conditioned on hour"). No such artifact existed before this phase --
trips.parquet has raw (start, end) pairs but nothing in src/features or
src/models aggregates them into an OD table.

Backoff hierarchy, confirmed in the Phase 7 plan-mode discussion (see
/Users/danielcrown1/.claude/plans/read-spec-md-4-forward-wise-meadow.md):
origin x hour-of-week is ~2,270 x 168 = ~381K cells, most too thin to trust
directly (many stations see only a handful of trips at a given
hour-of-week across a whole year). Rather than smoothing every cell with a
Laplace prior, each cell picks ONE tier -- the most specific one with at
least `min_trips_per_cell` observed historical trips:

    origin x hour-of-week -> zone x hour-of-week -> zone x daypart -> global

Each step coarsens exactly one dimension (origin: station -> zone; time:
hour-of-week -> daypart -> none), so the fallback degrades gracefully
rather than jumping straight to "no information." `global` drops both
dimensions -- the system-wide marginal destination distribution over all
trips -- and is always available, so every cell resolves to SOME tier.

This module only builds and stores the four tier probability tables plus
the per-cell tier assignment. It does NOT decide, at build time, which
fraction of simulated trips end up drawing from each tier -- that depends
on simulated departure VOLUME per cell, which only exists once the
simulator runs. src/sim/simulator.py counts that during the run; the
`cell_tier` diagnostic printed by main() here is an unweighted, build-time
sanity check only (fraction of CELLS by tier, not fraction of TRIPS).

Memory: every table here is built from group_by aggregations over real
observed trips, so it is sparse by construction -- never a dense
origin x hour x destination array (that would be ~2,270 x 168 x 2,270 ~=
861M cells, ~3.4GB at float32, almost all zero). trips.parquet (~45M rows,
2024-12-31 through 2025-12-31) is read month-chunked, same pattern as
src/models/demand.py's Stage 0 pre-pass, so this never holds more than one
month of raw trip rows in memory at once."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml

import utils.checkpoint as checkpoint
import utils.progress as progress

REPO_ROOT = Path(__file__).resolve().parents[2]
TRIPS_PATH = REPO_ROOT / "data" / "interim" / "trips.parquet"
STATIONS_PATH = REPO_ROOT / "data" / "processed" / "stations_zoned.parquet"
INVENTORY_PATH = REPO_ROOT / "data" / "processed" / "inventory.parquet"
PARAMS_PATH = REPO_ROOT / "config" / "params.yaml"

OD_SHARES_DIR = REPO_ROOT / "data" / "processed" / "od_shares"
CELL_TIER_PATH = OD_SHARES_DIR / "cell_tier.parquet"
STATION_HOUR_PATH = OD_SHARES_DIR / "station_hour_probs.parquet"
ZONE_HOUR_PATH = OD_SHARES_DIR / "zone_hour_probs.parquet"
ZONE_DAYPART_PATH = OD_SHARES_DIR / "zone_daypart_probs.parquet"
GLOBAL_PATH = OD_SHARES_DIR / "global_probs.parquet"

TIERS = ("station_hour", "zone_hour", "zone_daypart", "global")


@dataclass
class ODShareParams:
    min_trips_per_cell: int
    daypart_hour_edges: list[int]


def load_od_share_params(path: Path = PARAMS_PATH) -> ODShareParams:
    cfg = yaml.safe_load(path.read_text())["simulation"]
    return ODShareParams(
        min_trips_per_cell=int(cfg["min_trips_per_cell"]),
        daypart_hour_edges=list(cfg["daypart_hour_edges"]),
    )


def n_dayparts(edges: list[int]) -> int:
    return 2 * (len(edges) + 1)  # weekday/weekend x hour buckets


def daypart_expr(edges: list[int], hour_of_week_col: str = "hour_of_week") -> pl.Expr:
    """0..n_dayparts-1: is_weekend * n_hour_buckets + hour_bucket. dow is
    embedded in hour_of_week (dow*24 + hour, 0=Mon..6=Sun -- see
    src/features/panel.py), so both pieces come from the one column."""
    hour = pl.col(hour_of_week_col) % 24
    dow = pl.col(hour_of_week_col) // 24
    is_weekend = (dow >= 5).cast(pl.Int16)
    n_hour_buckets = len(edges) + 1
    bucket = pl.lit(0, dtype=pl.Int16)
    for i, e in enumerate(edges):
        bucket = pl.when(hour >= e).then(pl.lit(i + 1, dtype=pl.Int16)).otherwise(bucket)
    return is_weekend * n_hour_buckets + bucket


def month_bounds(month_key: str) -> tuple[datetime, datetime]:
    start = datetime.strptime(month_key, "%Y-%m")
    end = datetime(start.year + 1, 1, 1) if start.month == 12 else datetime(start.year, start.month + 1, 1)
    return start, end


def all_month_keys() -> list[str]:
    keys = (
        pl.scan_parquet(TRIPS_PATH)
        .select(checkpoint.month_key_expr("started_at").alias("month_key"))
        .unique()
        .sort("month_key")
        .collect()
    )
    return keys["month_key"].to_list()


def load_zone_lookup() -> pl.DataFrame:
    """station_id -> zone_agg, from stations_zoned.parquet. Renamed from
    zone_agglomerative here for the same reason src/features/panel.py
    renames it on the way into panel.parquet: one consistent name
    ("zone_agg") across every module that uses this zoning."""
    return pl.read_parquet(STATIONS_PATH).select(
        "station_id", pl.col("zone_agglomerative").alias("zone_agg")
    )


def load_usable_station_ids() -> pl.DataFrame:
    """The ~2,270-station population inventory.py's prepare_panel() kept
    (drops null-capacity and capacity==0-despite-activity stations -- see
    inventory.py). inventory.parquet's distinct station_id set IS exactly
    that population, since every row in it passed through prepare_panel."""
    return pl.scan_parquet(INVENTORY_PATH).select("station_id").unique().collect()


# ---------------------------------------------------------------------------
# Per-month counting (production path: month-chunked, never the full year
# of raw trips resident at once)
# ---------------------------------------------------------------------------


def build_month_od_counts(
    month_key: str, zone_lookup: pl.DataFrame, daypart_hour_edges: list[int], usable_station_ids: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    """usable_station_ids restricts DESTINATIONS only, not origins: a trip
    landing at a station outside the simulator's ~2,270-station network has
    no analog in simulation (it isn't a station the simulator tracks
    capacity for), so those trips are dropped before counting rather than
    left to produce an unroutable destination later. Origins are left
    unfiltered -- the simulator only ever samples departures from network
    stations in the first place, so a stray non-network origin in the
    historical count simply never gets looked up; filtering it here would
    only shift the zone/daypart/global marginals for no benefit."""
    start, end = month_bounds(month_key)
    trips = (
        pl.scan_parquet(TRIPS_PATH)
        .filter(pl.col("started_at").is_between(start, end, closed="left"))
        .select("start_station_id", "end_station_id", "started_at")
        .collect()
    )
    trips = trips.join(usable_station_ids, left_on="end_station_id", right_on="station_id", how="inner")
    trips = trips.with_columns(
        (pl.col("started_at").dt.weekday() - 1).cast(pl.Int16).alias("dow"),
        pl.col("started_at").dt.hour().cast(pl.Int16).alias("hour"),
    ).with_columns((pl.col("dow") * 24 + pl.col("hour")).alias("hour_of_week"))
    trips = trips.with_columns(daypart_expr(daypart_hour_edges).alias("daypart"))
    trips = trips.join(
        zone_lookup.rename({"station_id": "start_station_id", "zone_agg": "start_zone_agg"}),
        on="start_station_id",
        how="left",
    )

    station_hour = trips.group_by("start_station_id", "hour_of_week", "end_station_id").agg(pl.len().alias("n"))
    zoned = trips.filter(pl.col("start_zone_agg").is_not_null())
    zone_hour = zoned.group_by("start_zone_agg", "hour_of_week", "end_station_id").agg(pl.len().alias("n"))
    zone_daypart = zoned.group_by("start_zone_agg", "daypart", "end_station_id").agg(pl.len().alias("n"))
    global_counts = trips.group_by("end_station_id").agg(pl.len().alias("n"))

    return {
        "station_hour": station_hour,
        "zone_hour": zone_hour,
        "zone_daypart": zone_daypart,
        "global": global_counts,
    }


_KEY_COLS = {
    "station_hour": ["start_station_id", "hour_of_week", "end_station_id"],
    "zone_hour": ["start_zone_agg", "hour_of_week", "end_station_id"],
    "zone_daypart": ["start_zone_agg", "daypart", "end_station_id"],
    "global": ["end_station_id"],
}


def build_od_counts(
    params: ODShareParams | None = None,
    zone_lookup: pl.DataFrame | None = None,
    usable_station_ids: pl.DataFrame | None = None,
) -> dict[str, pl.DataFrame]:
    """One pass over every month in trips.parquet, accumulating raw
    (key..., end_station_id) -> n counts per tier. Never holds more than
    one month of raw trip rows at once; the per-month count tables
    themselves are tiny (bounded by distinct OD pairs observed, not by
    trip volume) so concatenating a year's worth of them is cheap."""
    params = params or load_od_share_params()
    zone_lookup = zone_lookup if zone_lookup is not None else load_zone_lookup()
    usable_station_ids = usable_station_ids if usable_station_ids is not None else load_usable_station_ids()
    months = all_month_keys()
    parts: dict[str, list[pl.DataFrame]] = {k: [] for k in TIERS}

    for month_key in months:
        t0 = time.monotonic()
        month_counts = build_month_od_counts(month_key, zone_lookup, params.daypart_hour_edges, usable_station_ids)
        n_trips = month_counts["global"]["n"].sum()
        for k, v in month_counts.items():
            parts[k].append(v)
        progress.log_month(month_key, int(n_trips), time.monotonic() - t0, extra="od_shares count pass")

    return {k: pl.concat(v).group_by(_KEY_COLS[k]).agg(pl.col("n").sum()) for k, v in parts.items()}


# ---------------------------------------------------------------------------
# Normalization (counts -> probabilities) and cell-tier assignment
# ---------------------------------------------------------------------------


def _normalize(counts: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    totals = counts.group_by(group_cols).agg(pl.col("n").sum().alias("_total"))
    return counts.join(totals, on=group_cols).with_columns((pl.col("n") / pl.col("_total")).alias("prob")).drop("_total")


def _normalize_global(counts: pl.DataFrame) -> pl.DataFrame:
    total = counts["n"].sum()
    return counts.with_columns((pl.col("n") / total).alias("prob"))


def build_cell_tier(counts: dict[str, pl.DataFrame], params: ODShareParams, zone_lookup: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per (station_id, hour_of_week) observed as an origin in
    trips.parquet, naming the tier its destination draws should use.
    Station tier wins if the origin's OWN total at that hour-of-week meets
    the threshold; else its zone's total at that hour-of-week; else its
    zone's total at that daypart; else global (always available -- nothing
    further to back off to, so it's never itself thresholded).

    zone_lookup: station_id, zone_agg (defaults to the real stations_zoned
    table; overridable so tests/test_sim.py can exercise the backoff logic
    against a synthetic fixture instead of real station data)."""
    zone_lookup = zone_lookup if zone_lookup is not None else load_zone_lookup()
    station_totals = counts["station_hour"].group_by("start_station_id", "hour_of_week").agg(
        pl.col("n").sum().alias("station_hour_n")
    )
    zone_lookup = zone_lookup.rename({"station_id": "start_station_id", "zone_agg": "start_zone_agg"})
    cells = station_totals.join(zone_lookup, on="start_station_id", how="left")
    cells = cells.with_columns(daypart_expr(params.daypart_hour_edges).alias("daypart"))

    zone_hour_totals = counts["zone_hour"].group_by("start_zone_agg", "hour_of_week").agg(
        pl.col("n").sum().alias("zone_hour_n")
    )
    cells = cells.join(zone_hour_totals, on=["start_zone_agg", "hour_of_week"], how="left")

    zone_daypart_totals = counts["zone_daypart"].group_by("start_zone_agg", "daypart").agg(
        pl.col("n").sum().alias("zone_daypart_n")
    )
    cells = cells.join(zone_daypart_totals, on=["start_zone_agg", "daypart"], how="left")

    cells = cells.with_columns(
        pl.when(pl.col("station_hour_n") >= params.min_trips_per_cell)
        .then(pl.lit("station_hour"))
        .when(pl.col("zone_hour_n").fill_null(0) >= params.min_trips_per_cell)
        .then(pl.lit("zone_hour"))
        .when(pl.col("zone_daypart_n").fill_null(0) >= params.min_trips_per_cell)
        .then(pl.lit("zone_daypart"))
        .otherwise(pl.lit("global"))
        .alias("tier")
    )
    return cells.select(
        "start_station_id", "hour_of_week", "start_zone_agg", "daypart", "tier", "station_hour_n"
    )


# ---------------------------------------------------------------------------
# Build + persist
# ---------------------------------------------------------------------------


def build_and_save_od_shares(force: bool = False) -> None:
    if not force and CELL_TIER_PATH.exists():
        print(f"[od_shares] cached artifacts exist at {OD_SHARES_DIR}, skipping build (use --force to rebuild)")
        return

    params = load_od_share_params()
    zone_lookup = load_zone_lookup()
    t0 = time.monotonic()
    counts = build_od_counts(params, zone_lookup=zone_lookup)

    station_hour_probs = _normalize(counts["station_hour"], ["start_station_id", "hour_of_week"])
    zone_hour_probs = _normalize(counts["zone_hour"], ["start_zone_agg", "hour_of_week"])
    zone_daypart_probs = _normalize(counts["zone_daypart"], ["start_zone_agg", "daypart"])
    global_probs = _normalize_global(counts["global"])
    cell_tier = build_cell_tier(counts, params, zone_lookup=zone_lookup)

    OD_SHARES_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint.write_checkpoint(cell_tier, CELL_TIER_PATH)
    checkpoint.write_checkpoint(
        station_hour_probs.select("start_station_id", "hour_of_week", "end_station_id", "prob"), STATION_HOUR_PATH
    )
    checkpoint.write_checkpoint(
        zone_hour_probs.select("start_zone_agg", "hour_of_week", "end_station_id", "prob"), ZONE_HOUR_PATH
    )
    checkpoint.write_checkpoint(
        zone_daypart_probs.select("start_zone_agg", "daypart", "end_station_id", "prob"), ZONE_DAYPART_PATH
    )
    checkpoint.write_checkpoint(global_probs.select("end_station_id", "prob"), GLOBAL_PATH)

    elapsed = time.monotonic() - t0
    print(f"[od_shares] built in {elapsed:.1f}s, peak RSS {progress.peak_rss_mb():.0f} MB")
    print(
        f"[od_shares] table sizes: station_hour={station_hour_probs.height:,} "
        f"zone_hour={zone_hour_probs.height:,} zone_daypart={zone_daypart_probs.height:,} "
        f"global={global_probs.height:,}"
    )
    tier_counts = cell_tier.group_by("tier").agg(pl.len().alias("n_cells")).sort("n_cells", descending=True)
    print(
        f"[od_shares] cell-tier assignment across {cell_tier.height:,} observed origin-hour cells "
        "(UNWEIGHTED by trip volume -- see module docstring; the authoritative "
        "volume-weighted tier-usage fraction comes from the simulator run, not this build step):"
    )
    print(tier_counts)


# ---------------------------------------------------------------------------
# Load + runtime lookup model
# ---------------------------------------------------------------------------


@dataclass
class ODShareModel:
    cell_tier: pl.DataFrame
    station_hour_probs: pl.DataFrame
    zone_hour_probs: pl.DataFrame
    zone_daypart_probs: pl.DataFrame
    global_probs: pl.DataFrame


def load_od_share_model() -> ODShareModel:
    return ODShareModel(
        cell_tier=pl.read_parquet(CELL_TIER_PATH),
        station_hour_probs=pl.read_parquet(STATION_HOUR_PATH),
        zone_hour_probs=pl.read_parquet(ZONE_HOUR_PATH),
        zone_daypart_probs=pl.read_parquet(ZONE_DAYPART_PATH),
        global_probs=pl.read_parquet(GLOBAL_PATH),
    )


def _sample_grouped(rows: pl.DataFrame, probs: pl.DataFrame, rows_key: str, probs_key: str, rng: np.random.Generator) -> pl.DataFrame:
    """rows: station_id, n, <rows_key> (grouping key -- start_station_id for
    the station_hour tier, start_zone_agg for zone_hour/zone_daypart).
    probs: <probs_key>, end_station_id, prob, already filtered to this
    step's hour_of_week/daypart. ONE batched rng.choice() draw per distinct
    key value, covering every origin that shares that key's distribution
    at once -- draws are i.i.d. per trip regardless of grouping, so pooling
    origins that share a distribution changes nothing about the result,
    only the number of rng calls (~544 zones max, not thousands of
    individual origins)."""
    out_station, out_dest = [], []
    for key_value, group in rows.group_by(rows_key):
        key_value = key_value[0] if isinstance(key_value, tuple) else key_value
        key_probs = probs.filter(pl.col(probs_key) == key_value)
        if key_probs.height == 0:
            raise RuntimeError(
                f"cell_tier assigned {rows_key}={key_value!r} to this tier but no destination "
                "distribution exists for it at this step -- cell_tier and the probs tables were "
                "built from different counts, investigate before proceeding"
            )
        total_n = int(group["n"].sum())
        dests = rng.choice(key_probs["end_station_id"].to_numpy(), size=total_n, p=key_probs["prob"].to_numpy())
        origins = np.repeat(group["station_id"].to_numpy(), group["n"].to_numpy())
        out_station.append(origins)
        out_dest.append(dests)
    n_total = sum(len(a) for a in out_station)
    return pl.DataFrame(
        {
            "station_id": np.concatenate(out_station),
            "dest_station_id": np.concatenate(out_dest),
            "tier": pl.Series([rows["tier"][0]] * n_total, dtype=pl.String),
        }
    )


def sample_destinations_for_step(
    model: ODShareModel,
    departures: pl.DataFrame,
    hour_of_week: int,
    daypart: int,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """departures: station_id, n (n = simulated departure count at that
    station this step; n>0 rows only). Returns one row per individual
    departure: station_id (origin), dest_station_id, tier -- resolved
    through the backoff hierarchy for THIS step's hour_of_week/daypart.
    Every origin at a given hour-of-week shares the same tier (a property
    of the cell, not of an individual trip), so tier lookup is one join,
    not a per-trip decision."""
    tier_slice = model.cell_tier.filter(pl.col("hour_of_week") == hour_of_week).select(
        "start_station_id", "start_zone_agg", "tier"
    )
    dep = departures.join(tier_slice.rename({"start_station_id": "station_id"}), on="station_id", how="left")
    # Origins with zero trips in trips.parquet's history at ALL (never seen
    # as an origin, at any hour) get no cell_tier row -- fall straight to
    # global, the only tier that doesn't need an origin-specific match.
    dep = dep.with_columns(pl.col("tier").fill_null("global"))

    out_frames = []
    station_rows = dep.filter(pl.col("tier") == "station_hour")
    if station_rows.height > 0:
        probs = model.station_hour_probs.filter(pl.col("hour_of_week") == hour_of_week)
        out_frames.append(_sample_grouped(station_rows, probs, "station_id", "start_station_id", rng))

    zone_hour_rows = dep.filter(pl.col("tier") == "zone_hour")
    if zone_hour_rows.height > 0:
        probs = model.zone_hour_probs.filter(pl.col("hour_of_week") == hour_of_week)
        out_frames.append(_sample_grouped(zone_hour_rows, probs, "start_zone_agg", "start_zone_agg", rng))

    zone_daypart_rows = dep.filter(pl.col("tier") == "zone_daypart")
    if zone_daypart_rows.height > 0:
        probs = model.zone_daypart_probs.filter(pl.col("daypart") == daypart)
        out_frames.append(_sample_grouped(zone_daypart_rows, probs, "start_zone_agg", "start_zone_agg", rng))

    global_rows = dep.filter(pl.col("tier") == "global")
    if global_rows.height > 0:
        total_n = int(global_rows["n"].sum())
        dests = rng.choice(
            model.global_probs["end_station_id"].to_numpy(), size=total_n, p=model.global_probs["prob"].to_numpy()
        )
        origins = np.repeat(global_rows["station_id"].to_numpy(), global_rows["n"].to_numpy())
        out_frames.append(
            pl.DataFrame({"station_id": origins, "dest_station_id": dests, "tier": pl.Series(["global"] * total_n, dtype=pl.String)})
        )

    if not out_frames:
        return pl.DataFrame(schema={"station_id": pl.String, "dest_station_id": pl.String, "tier": pl.String})
    return pl.concat(out_frames, how="vertical")


def main() -> None:
    build_and_save_od_shares(force=False)


if __name__ == "__main__":
    main()
