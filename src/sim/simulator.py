"""Phase 7 forward simulator: discrete-event (fixed 15-min step) simulation
over the station network, per SPEC.md §4 and RUNBOOK Phase 7. This is the
hard gate before Phase 8 -- see validate.py for the held-out-week replay
that decides whether it passes.

Design decisions locked in during the Phase 7 plan-mode discussion (see
~/.claude/plans/read-spec-md-4-forward-wise-meadow.md)
are documented inline at the point they're implemented, not repeated here.
Two structural choices worth stating up front:

1. Departures drive the simulation; arrivals EMERGE. demand.py fits two
   independent directions (departures censored by is_bike_empty, arrivals
   censored by is_dock_full) for the historical reconstruction/imputation
   use case in Phases 4-6. The simulator only samples from the departures
   direction -- arrivals are whatever departures get routed to (directly,
   via reroute, or lost), so simulated departures and arrivals stay
   internally consistent by construction. Independently sampling arrivals
   from the arrivals-direction model as well would double-model the same
   physical events and there'd be no way to reconcile the two.

2. own_lag_1week is real historical data, not simulated state, for the
   ENTIRE simulated week. Because the simulation window is exactly one
   week, "7 days before now" always falls before the window starts, for
   every step -- there is no simulated week-ago data to reference, by
   construction, not just at the start. This is a one-week-simulation-
   specific simplification: a multi-week simulator would need to switch to
   true simulated lag once past week 1.

3. neighbor_demand uses the PREVIOUS step's simulated departures, not the
   current step's. demand.py's historical neighbor_demand is same-instant
   (all stations' current-step values are already known, since it's
   observed history) -- forward simulation can't replicate that without a
   circular dependency (station A's demand draw depends on station B's
   same-step draw, which depends on A's). Lagging by one step (15 min)
   breaks the cycle at the cost of a small, documented mismatch with the
   feature's historical definition.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from sklearn.neighbors import BallTree

import models.demand as demand
import models.od_shares as od_shares
import utils.progress as progress

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "config" / "params.yaml"
STATIONS_PATH = od_shares.STATIONS_PATH
INVENTORY_PATH = demand.INVENTORY_PATH
PANEL_PATH = demand.PANEL_PATH
WEATHER_PATH = demand.WEATHER_PATH

EARTH_RADIUS_M = 6_371_000.0
INTERVAL_MINUTES = 15
STEPS_PER_WEEK = 7 * 24 * 4
BOUNDARY_EPS = 1e-6

OWN_LAG_HOUR_INTERVALS = demand.OWN_LAG_HOUR_INTERVALS  # 4
OWN_LAG_WEEK_TIMEDELTA = timedelta(days=7)


@dataclass
class SimulationParams:
    daypart_hour_edges: list[int]
    max_reroute_radius_m: float
    max_reroute_hops: int
    seed: int
    validation_week_start: str


def load_simulation_params(path: Path = PARAMS_PATH) -> SimulationParams:
    cfg = yaml.safe_load(path.read_text())["simulation"]
    return SimulationParams(
        daypart_hour_edges=list(cfg["daypart_hour_edges"]),
        max_reroute_radius_m=float(cfg["max_reroute_radius_m"]),
        max_reroute_hops=int(cfg["max_reroute_hops"]),
        seed=int(cfg["seed"]),
        validation_week_start=cfg["validation_week_start"],
    )


# ---------------------------------------------------------------------------
# Station network
# ---------------------------------------------------------------------------


@dataclass
class NetworkArrays:
    """Fixed-order arrays over the simulated station population -- the
    ~2,270-station "usable" set inventory.py's prepare_panel() kept (see
    od_shares.load_usable_station_ids). Position in these arrays IS a
    station's identity for the rest of this module; index_of maps back to
    station_id for joins against polars tables built fresh each step."""

    station_id: np.ndarray
    capacity: np.ndarray
    zone_agg: np.ndarray  # object array, entries may be None
    lat: np.ndarray
    lng: np.ndarray
    index_of: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.station_id)


def load_station_network() -> NetworkArrays:
    usable = od_shares.load_usable_station_ids()
    stations = pl.read_parquet(STATIONS_PATH).select(
        "station_id", "lat", "lng", "capacity", pl.col("zone_agglomerative").alias("zone_agg")
    )
    network = stations.join(usable, on="station_id", how="inner").sort("station_id")
    missing = usable.height - network.height
    if missing:
        print(
            f"[simulator] {missing:,} usable stations (per inventory.parquet) have no match in "
            f"{STATIONS_PATH.name} -- excluded from the simulated network (no lat/lng/capacity to simulate with)."
        )

    station_id = network["station_id"].to_numpy()
    return NetworkArrays(
        station_id=station_id,
        capacity=network["capacity"].to_numpy().astype(float),
        zone_agg=network["zone_agg"].to_numpy(),
        lat=network["lat"].to_numpy(),
        lng=network["lng"].to_numpy(),
        index_of={sid: i for i, sid in enumerate(station_id)},
    )


def build_reroute_tree(network: NetworkArrays) -> BallTree:
    """Haversine BallTree over the network's lat/lng -- same pattern as
    src/models/substitution.py's build_neighbor_pairs, but queried
    per-arrival at simulation time (k-nearest) rather than precomputed as
    fixed-radius pairs, since "nearest with a free dock" depends on
    time-varying inventory, not just geography."""
    coords = np.column_stack([np.radians(network.lat), np.radians(network.lng)])
    return BallTree(coords, metric="haversine")


def find_reroute_target(
    dest_idx: int,
    inventory: np.ndarray,
    capacity: np.ndarray,
    tree: BallTree,
    coords_rad: np.ndarray,
    max_hops: int,
    max_radius_m: float,
) -> tuple[int | None, int, float]:
    """Nearest-neighbor search outward from a full destination, checked in
    increasing-distance order, for a station with a free dock (inventory <
    capacity). Stops at whichever binds first: max_hops candidates
    checked, or the next candidate's distance exceeds max_radius_m --
    confirmed in the Phase 7 plan-mode discussion. Returns (station_idx,
    hop_count, distance_m) on success, or (None, hops_checked, nan) if
    nothing within the cap has room -- caller counts that trip fully lost,
    not degraded."""
    k = min(max_hops + 1, len(coords_rad))  # +1 for self, always the nearest to itself at distance 0
    dist_rad, idx = tree.query(coords_rad[dest_idx : dest_idx + 1], k=k)
    hops = 0
    for d_rad, cand_idx in zip(dist_rad[0], idx[0]):
        cand_idx = int(cand_idx)
        if cand_idx == dest_idx:
            continue
        dist_m = float(d_rad) * EARTH_RADIUS_M
        if dist_m > max_radius_m or hops >= max_hops:
            return None, hops, float("nan")
        hops += 1
        if inventory[cand_idx] < capacity[cand_idx] - BOUNDARY_EPS:
            return cand_idx, hops, dist_m
    return None, hops, float("nan")


# ---------------------------------------------------------------------------
# Held-out week input assembly -- everything below is scan+filter against
# the full-year artifacts (panel.parquet, inventory.parquet, ~76M rows
# each), never a full read. See config/params.yaml's simulation section and
# the Phase 7 memory-plan discussion.
# ---------------------------------------------------------------------------


@dataclass
class WeekInputs:
    week_start: datetime
    week_end: datetime
    initial_inventory: np.ndarray  # aligned to network order
    n_schedule: pl.DataFrame  # station_id, interval_start, nontrip_in, nontrip_out
    calendar_weather: pl.DataFrame  # interval_start, hour_of_week, month, is_holiday, daypart, weather cols
    own_lag_1week_table: pl.DataFrame  # interval_start (THIS week's), station_id, dep_own_lag_1week
    own_lag_1h_seed: np.ndarray  # shape (n_stations, OWN_LAG_HOUR_INTERVALS), aligned to network order
    prev_departures_seed: np.ndarray  # aligned to network order
    actual_departures: pl.DataFrame  # station_id, interval_start, departures -- real history, for no-op mode + validation
    actual_arrivals_inventory: pl.DataFrame  # station_id, interval_start, inventory, is_bike_empty, is_dock_full -- for validation
    half_cap_seeded_station_ids: list[str]  # network stations with no inventory.parquet row at week_start


def _aligned_from_table(table: pl.DataFrame, value_col: str, network: NetworkArrays, default: float = 0.0) -> np.ndarray:
    """table: station_id, value_col -- one row per station (already
    deduplicated by caller). Returns an array in network.station_id order,
    filling `default` for any network station absent from table (e.g. a
    station too young to have real data in the seed window)."""
    lookup = dict(zip(table["station_id"].to_list(), table[value_col].to_list()))
    return np.array([lookup.get(sid, default) for sid in network.station_id], dtype=float)


def prepare_week_inputs(network: NetworkArrays, params: SimulationParams) -> WeekInputs:
    week_start = datetime.strptime(params.validation_week_start, "%Y-%m-%d")
    week_end = week_start + timedelta(days=7)
    print(f"[simulator] held-out validation week: {week_start.date()} to {week_end.date()}")

    # --- initial inventory (state at the first simulated interval) -------
    inv0 = (
        pl.scan_parquet(INVENTORY_PATH)
        .filter(pl.col("interval_start") == week_start)
        .select("station_id", "inventory")
        .collect()
    )
    initial_inventory = _aligned_from_table(inv0, "inventory", network, default=np.nan)
    missing_mask = np.isnan(initial_inventory)
    half_cap_seeded_station_ids = [sid for sid, missing in zip(network.station_id, missing_mask) if missing]
    if half_cap_seeded_station_ids:
        print(
            f"[simulator] {len(half_cap_seeded_station_ids)} network stations missing an inventory.parquet row "
            "at week start -- seeded at half capacity"
        )
        half_cap = network.capacity / 2.0
        initial_inventory = np.where(missing_mask, half_cap, initial_inventory)

    # --- baseline N schedule (Phase 4's inferred non-trip movement) ------
    n_schedule = (
        pl.scan_parquet(INVENTORY_PATH)
        .filter(pl.col("interval_start").is_between(week_start, week_end, closed="left"))
        .select("station_id", "interval_start", "inferred_nontrip_in", "inferred_nontrip_out")
        .collect()
    )

    # --- calendar + weather (real, exogenous -- same for every station) --
    panel_week = (
        pl.scan_parquet(PANEL_PATH)
        .filter(pl.col("interval_start").is_between(week_start, week_end, closed="left"))
        .select("interval_start", "hour_of_week", "month", "is_holiday", "temp_c", "precip_mm", "wind_kph", "humidity_pct")
        .group_by("interval_start")
        .agg(pl.all().first())
        .collect()
    )
    weather = pl.read_parquet(WEATHER_PATH)
    weather_lag_table = demand.compute_weather_lag_table(weather)
    calendar_weather = (
        panel_week.with_columns(pl.col("interval_start").dt.truncate("1h").alias("weather_hour"))
        .join(weather_lag_table, on="weather_hour", how="left")
        .drop("weather_hour")
        .with_columns(od_shares.daypart_expr(params.daypart_hour_edges).alias("daypart"))
        .sort("interval_start")
    )
    n_steps = calendar_weather.height
    if n_steps != STEPS_PER_WEEK:
        print(f"[simulator] WARNING: expected {STEPS_PER_WEEK} steps in the validation week, found {n_steps} (gaps in panel.parquet?)")

    # --- own_lag_1week: real data from exactly 7 days before each step ---
    lag_week_start, lag_week_end = week_start - OWN_LAG_WEEK_TIMEDELTA, week_end - OWN_LAG_WEEK_TIMEDELTA
    own_lag_1week_table = (
        pl.scan_parquet(PANEL_PATH)
        .filter(pl.col("interval_start").is_between(lag_week_start, lag_week_end, closed="left"))
        .select("station_id", "interval_start", "departures")
        .with_columns((pl.col("interval_start") + pl.duration(days=7)).alias("interval_start"))
        .rename({"departures": "dep_own_lag_1week"})
        .collect()
    )

    # --- own_lag_1h seed: real departures in the 4 intervals before week_start
    seed_start = week_start - timedelta(minutes=INTERVAL_MINUTES * OWN_LAG_HOUR_INTERVALS)
    seed_rows = (
        pl.scan_parquet(PANEL_PATH)
        .filter(pl.col("interval_start").is_between(seed_start, week_start, closed="left"))
        .select("station_id", "interval_start", "departures")
        .collect()
        .sort("interval_start")
    )
    own_lag_1h_seed = np.zeros((network.n, OWN_LAG_HOUR_INTERVALS))
    for j, (_, group) in enumerate(seed_rows.group_by("interval_start", maintain_order=True)):
        arr = _aligned_from_table(group, "departures", network, default=0.0)
        if j < OWN_LAG_HOUR_INTERVALS:
            own_lag_1h_seed[:, j] = arr

    # --- neighbor_demand seed: real departures at the interval right before week_start
    prev_step = (
        pl.scan_parquet(PANEL_PATH)
        .filter(pl.col("interval_start") == week_start - timedelta(minutes=INTERVAL_MINUTES))
        .select("station_id", "departures")
        .collect()
    )
    prev_departures_seed = _aligned_from_table(prev_step, "departures", network, default=0.0)

    # --- ground truth for the no-op sanity run + validation -------------
    # Scoped to the usable/network station population (od_shares's
    # load_usable_station_ids, exactly inventory.parquet's station set),
    # NOT the raw panel: panel.parquet still contains the null/zero-
    # capacity stations inventory.py's prepare_panel() drops (~2.2% of a
    # week's real trip volume, confirmed against the 2025-10-06 validation
    # week). Comparing the simulator's totals against unscoped panel
    # totals would fail it for volume the simulated network structurally
    # can never produce, at stations it doesn't model at all.
    usable_station_ids = od_shares.load_usable_station_ids()
    actual_departures = (
        pl.scan_parquet(PANEL_PATH)
        .filter(pl.col("interval_start").is_between(week_start, week_end, closed="left"))
        .select("station_id", "interval_start", "departures")
        .collect()
        .join(usable_station_ids, on="station_id", how="inner")
    )
    actual_arrivals_inventory = (
        pl.scan_parquet(INVENTORY_PATH)
        .filter(pl.col("interval_start").is_between(week_start, week_end, closed="left"))
        .select("station_id", "interval_start", "inventory", "is_bike_empty", "is_dock_full")
        .collect()
    )

    return WeekInputs(
        week_start=week_start,
        week_end=week_end,
        initial_inventory=initial_inventory,
        n_schedule=n_schedule,
        calendar_weather=calendar_weather,
        own_lag_1week_table=own_lag_1week_table,
        own_lag_1h_seed=own_lag_1h_seed,
        prev_departures_seed=prev_departures_seed,
        actual_departures=actual_departures,
        actual_arrivals_inventory=actual_arrivals_inventory,
        half_cap_seeded_station_ids=half_cap_seeded_station_ids,
    )


def load_true_destination_trips(week_start: datetime, week_end: datetime, network: NetworkArrays) -> pl.DataFrame:
    """Real (origin, destination) pairs for the held-out week, from
    trips.parquet directly -- NOT drawn from od_shares. Only used by the
    "sanity_true_dest" diagnostic mode (see run_simulation), which forces
    BOTH departure counts and destinations to real history, to isolate
    OD-destination-assignment error from everything else (baseline-N-replay
    clipping, in particular).

    Restricted to trips whose destination is in the simulated network, same
    restriction od_shares.py applies when building the OD tables -- a trip
    ending at a station the simulator doesn't track has no routable analog
    here. This means the forced departure count for THIS mode is the count
    of trips with a routable destination, not panel.parquet's raw
    `departures` column (which the no-op "sanity_od" mode uses) -- a small,
    documented difference between the two sanity modes' baselines."""
    usable_station_ids = od_shares.load_usable_station_ids()
    trips = (
        pl.scan_parquet(od_shares.TRIPS_PATH)
        .filter(pl.col("started_at").is_between(week_start, week_end, closed="left"))
        .select("start_station_id", "end_station_id", "started_at")
        .with_columns(pl.col("started_at").dt.truncate(f"{INTERVAL_MINUTES}m").alias("interval_start"))
        .rename({"start_station_id": "station_id", "end_station_id": "dest_station_id"})
        .select("station_id", "interval_start", "dest_station_id")
        .collect()
    )
    n_before = trips.height
    trips = trips.join(usable_station_ids.rename({"station_id": "dest_station_id"}), on="dest_station_id", how="inner")
    dropped_frac = 1.0 - trips.height / n_before if n_before else 0.0
    print(
        f"[simulator] true-destination trips: {trips.height:,} / {n_before:,} kept "
        f"({dropped_frac:.2%} dropped -- destination outside the simulated network)"
    )
    return trips


# ---------------------------------------------------------------------------
# Per-step simulation
# ---------------------------------------------------------------------------


@dataclass
class StepOutcome:
    interval_start: datetime
    station_id: np.ndarray
    departures_sampled: np.ndarray
    departures_actual: np.ndarray
    lost_no_bike: np.ndarray
    direct_arrivals: np.ndarray
    rerouted_arrivals: np.ndarray
    lost_past_cap_arrivals: np.ndarray
    n_in: np.ndarray
    n_out: np.ndarray
    inventory: np.ndarray  # end-of-step
    is_bike_empty: np.ndarray
    is_dock_full: np.ndarray
    n_bound_violations: int  # stations pushed outside [0, capacity] by baseline N replay this step, pre-clip
    clip_created: float  # bikes added back by clipping negative inventory up to 0, this step (>= 0)
    clip_destroyed: float  # bikes discarded by clipping over-capacity inventory down to capacity, this step (>= 0)
    trip_log: pl.DataFrame  # station_id (origin), dest_station_id, tier, outcome, hops, distance_m


def _build_step_features(
    network: NetworkArrays,
    calendar_row: dict,
    own_lag_1h_buffer: np.ndarray,
    own_lag_1week_arr: np.ndarray,
    prev_departures: np.ndarray,
) -> pl.DataFrame:
    tmp = pl.DataFrame({"station_id": network.station_id, "zone_agg": network.zone_agg, "prev_dep": prev_departures})
    zone_totals = tmp.group_by("zone_agg").agg(pl.col("prev_dep").sum().alias("zone_total"))
    tmp = tmp.join(zone_totals, on="zone_agg", how="left")
    neighbor_demand = (tmp["zone_total"] - tmp["prev_dep"]).to_numpy()

    n = network.n
    return pl.DataFrame(
        {
            "station_id": network.station_id,
            "zone_agg": network.zone_agg,
            "is_holiday": [bool(calendar_row["is_holiday"])] * n,
            "hour_of_week": [int(calendar_row["hour_of_week"])] * n,
            "month": [int(calendar_row["month"])] * n,
            "temp_c": [calendar_row["temp_c"]] * n,
            "precip_mm": [calendar_row["precip_mm"]] * n,
            "precip_lag1h": [calendar_row["precip_lag1h"]] * n,
            "precip_lag2h": [calendar_row["precip_lag2h"]] * n,
            "wind_kph": [calendar_row["wind_kph"]] * n,
            "humidity_pct": [calendar_row["humidity_pct"]] * n,
            "capacity": network.capacity,
            "dep_own_lag_1h": own_lag_1h_buffer.mean(axis=1),
            "dep_own_lag_1week": own_lag_1week_arr,
            "dep_neighbor_demand": neighbor_demand,
        }
    )


def route_departures(
    trips: pl.DataFrame,
    network: NetworkArrays,
    inventory: np.ndarray,
    tree: BallTree,
    coords_rad: np.ndarray,
    max_hops: int,
    max_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]:
    """Routes every trip in `trips` (station_id origin, dest_station_id,
    tier -- one row per departure that got a bike, from
    od_shares.sample_destinations_for_step) to its destination, direct if
    there's room, else a capped reroute search, else fully lost. Mutates
    `inventory` in place -- room at a destination depends on how many
    OTHER trips already landed there this step, so destinations are
    processed one at a time, not all vectorized against a single
    pre-arrival snapshot.

    Conservation invariant (tests/test_sim.py exercises this directly):
    len(trips) == direct.sum() + rerouted.sum() + lost_past_cap.sum() --
    every trip that got a bike this step lands in exactly one of those
    three buckets. rerouted/direct are counted at the station a bike
    actually landed at (the reroute TARGET, not the original destination);
    lost_past_cap is counted at the original (full) destination, since
    that's the station whose fullness caused the loss.

    Returns (direct, rerouted, lost_past_cap) as network-order arrays plus
    a trip_log DataFrame (station_id, dest_station_id, tier, outcome,
    hops, distance_m) -- one row per input trip, same length as `trips`."""
    n = network.n
    direct = np.zeros(n)
    rerouted = np.zeros(n)
    lost_past_cap = np.zeros(n)

    dest_counts = trips.group_by("dest_station_id").agg(pl.len().alias("count")).sort("dest_station_id")
    outcome_rows = []
    for dest_id, count in zip(dest_counts["dest_station_id"].to_list(), dest_counts["count"].to_list()):
        dest_idx = network.index_of.get(dest_id)
        if dest_idx is None:
            # Should not happen -- od_shares filters destinations to the
            # usable network at build time. Fail loudly rather than
            # silently dropping trips if that invariant ever breaks.
            raise RuntimeError(f"destination {dest_id!r} not in the simulated network")

        room = int(max(0.0, network.capacity[dest_idx] - inventory[dest_idx]))
        n_direct = min(count, room)
        inventory[dest_idx] += n_direct
        direct[dest_idx] += n_direct
        for _ in range(n_direct):
            outcome_rows.append((dest_id, "direct", 0, 0.0))

        for _ in range(count - n_direct):
            target_idx, hops, dist_m = find_reroute_target(
                dest_idx, inventory, network.capacity, tree, coords_rad, max_hops, max_radius_m
            )
            if target_idx is None:
                lost_past_cap[dest_idx] += 1
                outcome_rows.append((dest_id, "lost_past_cap", hops, float("nan")))
            else:
                inventory[target_idx] += 1
                rerouted[target_idx] += 1
                outcome_rows.append((dest_id, "rerouted", hops, dist_m))

    outcome_df = pl.DataFrame(outcome_rows, schema=["dest_station_id", "outcome", "hops", "distance_m"], orient="row")
    trip_log = trips.sort("dest_station_id").with_columns(
        outcome_df["outcome"], outcome_df["hops"], outcome_df["distance_m"]
    )
    return direct, rerouted, lost_past_cap, trip_log


def _clip_forced_trips(forced_trips: pl.DataFrame, actual_departures: np.ndarray, network: NetworkArrays) -> pl.DataFrame:
    cap_df = pl.DataFrame({"station_id": network.station_id, "cap_n": actual_departures.astype(np.int64)})
    joined = forced_trips.join(cap_df, on="station_id", how="left").with_columns(pl.col("cap_n").fill_null(0))
    joined = joined.with_columns(pl.int_range(0, pl.len()).over("station_id").alias("_rn"))
    clipped = joined.filter(pl.col("_rn") < pl.col("cap_n")).drop("_rn", "cap_n")
    return clipped.with_columns(pl.lit("true_history").alias("tier"))


def run_step(
    network: NetworkArrays,
    inventory: np.ndarray,
    own_lag_1h_buffer: np.ndarray,
    prev_departures: np.ndarray,
    fitted_dep: demand.FittedDirection,
    od_model: od_shares.ODShareModel,
    calendar_row: dict,
    own_lag_1week_arr: np.ndarray,
    n_in_arr: np.ndarray,
    n_out_arr: np.ndarray,
    tree: BallTree,
    coords_rad: np.ndarray,
    sim_params: SimulationParams,
    rng: np.random.Generator,
    forced_departures: np.ndarray | None,
    forced_trips: pl.DataFrame | None = None,
    demand_multiplier: float = 1.0,
) -> StepOutcome:
    """Mutates inventory, own_lag_1h_buffer, prev_departures in place --
    callers own the arrays and read them back via the returned StepOutcome
    if a snapshot is needed; the caller's loop just keeps going.

    forced_trips (station_id, dest_station_id -- this step's real historical
    trips, "sanity_true_dest" mode only): when given, bypasses
    od_shares.sample_destinations_for_step entirely and routes these exact
    (origin, destination) pairs instead of a drawn destination. If simulated
    inventory can't cover every forced trip (actual_departures <
    forced_departures at some station), only the first actual_departures[i]
    of that station's forced trips are taken -- which specific ones is
    arbitrary (order carries no meaning here), same philosophy as
    route_departures not caring about intra-destination-group order.

    demand_multiplier (Phase 9, SPEC.md §8's bootstrap axis (a) -- demand
    model residual uncertainty): a single scalar applied to the GBT rate
    prediction before Poisson sampling, ignored (1.0) outside a Phase 9
    bootstrap replicate. Ignored entirely when forced_departures is set
    (sanity_od/sanity_true_dest replay real counts, not model output)."""
    interval_start = calendar_row["interval_start"]
    n = network.n

    feat = _build_step_features(network, calendar_row, own_lag_1h_buffer, own_lag_1week_arr, prev_departures)
    d_hat = demand.predict_gbt(fitted_dep, feat) * demand_multiplier

    if forced_departures is not None:
        sampled = forced_departures.astype(float)
    else:
        sampled = rng.poisson(np.maximum(d_hat, 0.0)).astype(float)

    actual_departures = np.minimum(sampled, inventory)
    lost_no_bike = sampled - actual_departures
    inventory -= actual_departures

    departures_df = pl.DataFrame(
        {"station_id": network.station_id, "n": actual_departures.astype(np.int64)}
    ).filter(pl.col("n") > 0)

    if departures_df.height > 0:
        if forced_trips is not None:
            trips = _clip_forced_trips(forced_trips, actual_departures, network)
        else:
            trips = od_shares.sample_destinations_for_step(
                od_model, departures_df, int(calendar_row["hour_of_week"]), int(calendar_row["daypart"]), rng
            )
        direct, rerouted, lost_past_cap, trip_log = route_departures(
            trips, network, inventory, tree, coords_rad, sim_params.max_reroute_hops, sim_params.max_reroute_radius_m
        )
    else:
        direct = np.zeros(n)
        rerouted = np.zeros(n)
        lost_past_cap = np.zeros(n)
        trip_log = pl.DataFrame(
            schema={
                "station_id": pl.String, "dest_station_id": pl.String, "tier": pl.String,
                "outcome": pl.String, "hops": pl.Int64, "distance_m": pl.Float64,
            }
        )

    # --- baseline non-trip movement (Phase 4's inferred N) --------------
    inventory += n_in_arr - n_out_arr
    # Documented edge case (Phase 7 plan): N was fit to keep the ACTUAL
    # historical trajectory in bounds, not this (different) simulated one
    # -- replaying it against simulated flow can push a station out of
    # bounds. Clip and move on, but don't silently absorb it without a
    # trace -- run_simulation sums this across all steps and reports one
    # total at the end (per-step prints would flood 672 steps x 2 runs).
    #
    # Clipping is NOT neutral (flagged explicitly when this diagnostic was
    # requested): clamping a negative pre-clip value up to 0 conjures bikes
    # the flow balance said shouldn't be there (clip_created); clamping an
    # over-capacity pre-clip value down to capacity discards bikes the flow
    # balance said SHOULD be there (clip_destroyed). Both break conservation
    # and both are tracked, not just counted, so the magnitude -- not just
    # the frequency -- of the effect is visible.
    n_bound_violations = int(((inventory < -BOUNDARY_EPS) | (inventory > network.capacity + BOUNDARY_EPS)).sum())
    pre_clip = inventory.copy()
    inventory[:] = np.clip(inventory, 0.0, network.capacity)
    delta = inventory - pre_clip
    clip_created = float(delta[delta > 0].sum())
    clip_destroyed = float(-delta[delta < 0].sum())

    own_lag_1h_buffer[:, :-1] = own_lag_1h_buffer[:, 1:]
    own_lag_1h_buffer[:, -1] = actual_departures
    prev_departures[:] = actual_departures

    is_bike_empty = inventory <= BOUNDARY_EPS
    is_dock_full = inventory >= network.capacity - BOUNDARY_EPS

    return StepOutcome(
        interval_start=interval_start,
        station_id=network.station_id,
        departures_sampled=sampled,
        departures_actual=actual_departures,
        lost_no_bike=lost_no_bike,
        direct_arrivals=direct,
        rerouted_arrivals=rerouted,
        lost_past_cap_arrivals=lost_past_cap,
        n_in=n_in_arr,
        n_out=n_out_arr,
        inventory=inventory.copy(),
        is_bike_empty=is_bike_empty,
        is_dock_full=is_dock_full,
        n_bound_violations=n_bound_violations,
        clip_created=clip_created,
        clip_destroyed=clip_destroyed,
        trip_log=trip_log.with_columns(pl.lit(interval_start).alias("interval_start")),
    )


# ---------------------------------------------------------------------------
# Full-week driver
# ---------------------------------------------------------------------------


@dataclass
class SimulationRun:
    station_intervals: pl.DataFrame
    trip_log: pl.DataFrame
    total_n_bound_violations: int
    total_clip_created: float  # bikes added back by clipping, summed across the whole run
    total_clip_destroyed: float  # bikes discarded by clipping, summed across the whole run


MODES = ("stochastic", "sanity_od", "sanity_true_dest")


def _induced_step_dicts(
    induced_moves: pl.DataFrame | None,
    calendar_weather: pl.DataFrame,
    network: NetworkArrays,
) -> tuple[dict, dict]:
    """Builds (induced_in_by_step, induced_out_by_step), same shape as
    run_simulation's n_in_by_step/n_out_by_step, from a Phase 9 policy's
    induced-move schedule (src/opt/policy_baselines.py:
    to_simulator_induced_moves -- origin_station_id, dest_station_id,
    hour_of_week, induced_trips_per_hour). Each row is exploded across
    every interval_start in the simulated week whose hour_of_week matches,
    at 1/4 the hourly rate per 15-min step -- the same hour-of-week-only
    granularity Phase 8's MV(s,t) already established as sufficient
    (DECISIONS.md's "schedulability" finding: deficits recur at the same
    clock time almost every week, no sub-hour targeting needed). Returns
    ({}, {}) when induced_moves is None/empty (the do-nothing policy),
    so callers can treat that case identically to no injection at all."""
    if induced_moves is None or induced_moves.height == 0:
        return {}, {}

    how_map = calendar_weather.select("interval_start", "hour_of_week").unique()
    exploded = induced_moves.select(
        "origin_station_id", "dest_station_id", "hour_of_week", "induced_trips_per_hour"
    ).join(how_map, on="hour_of_week", how="inner").with_columns(
        (pl.col("induced_trips_per_hour") / 4.0).alias("induced_per_step")
    )

    in_table = exploded.group_by("dest_station_id", "interval_start").agg(
        pl.col("induced_per_step").sum()
    ).rename({"dest_station_id": "station_id"})
    out_table = exploded.group_by("origin_station_id", "interval_start").agg(
        pl.col("induced_per_step").sum()
    ).rename({"origin_station_id": "station_id"})

    def _dict_from(table: pl.DataFrame) -> dict:
        out = {}
        for interval_start, group in table.group_by("interval_start", maintain_order=True):
            interval_start = interval_start[0] if isinstance(interval_start, tuple) else interval_start
            out[interval_start] = _aligned_from_table(group, "induced_per_step", network, default=0.0)
        return out

    return _dict_from(in_table), _dict_from(out_table)


def run_simulation(
    network: NetworkArrays,
    week: WeekInputs,
    fitted_dep: demand.FittedDirection,
    od_model: od_shares.ODShareModel,
    sim_params: SimulationParams,
    mode: str,
    induced_moves: pl.DataFrame | None = None,
    demand_multiplier: float = 1.0,
) -> SimulationRun:
    """Three modes (see the Phase 7 plan-mode discussion and its follow-up
    on isolating clipping from destination fidelity):

    - "stochastic": departures sampled from the fitted demand model,
      destinations drawn from OD shares. The real forward simulation.
    - "sanity_od": departures forced to real historical counts, destinations
      drawn from OD shares. Isolates OD/routing/inventory-update machinery
      from demand-model error.
    - "sanity_true_dest": departures AND destinations forced to real
      history (load_true_destination_trips) -- isolates baseline-N-replay
      clipping (and everything else stateful about inventory update) from
      OD-destination-assignment error specifically. If this ALSO fails
      badly, the bug is not destination fidelity.

    induced_moves/demand_multiplier (Phase 9, SPEC.md §8): both default to
    a no-op (None / 1.0) so every existing call site (Phase 7's gate,
    Phase 8) is unaffected. induced_moves is layered onto n_in_arr/n_out_arr
    exactly like baseline N (SPEC.md §7: "incentives move bikes, they don't
    create them" -- a zero-sum relocation), so it inherits the SAME
    clip-to-[0, capacity] accounting already tracked for N; no changes to
    run_step or route_departures are needed. demand_multiplier threads
    straight through to run_step's Poisson rate."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    rng = np.random.default_rng(sim_params.seed)
    tree = build_reroute_tree(network)
    coords_rad = np.column_stack([np.radians(network.lat), np.radians(network.lng)])

    inventory = week.initial_inventory.copy()
    own_lag_1h_buffer = week.own_lag_1h_seed.copy()
    prev_departures = week.prev_departures_seed.copy()

    # Precompute per-step lookup dicts ONCE rather than re-filtering these
    # ~1.5M-row (station x week) tables on every one of the 672 steps --
    # same O(steps) vs. O(steps x table size) concern as everywhere else in
    # this repo that touches a full week/month/year table in a loop.
    def _per_step_dict(table: pl.DataFrame, value_col: str) -> dict:
        out = {}
        for interval_start, group in table.group_by("interval_start", maintain_order=True):
            interval_start = interval_start[0] if isinstance(interval_start, tuple) else interval_start
            out[interval_start] = _aligned_from_table(group, value_col, network, default=0.0)
        return out

    forced_trips_by_step: dict | None = None
    if mode == "sanity_od":
        forced_by_step = _per_step_dict(week.actual_departures, "departures")
    elif mode == "sanity_true_dest":
        true_dest_trips = load_true_destination_trips(week.week_start, week.week_end, network)
        forced_trips_by_step = {}
        forced_by_step = {}
        for interval_start, group in true_dest_trips.group_by("interval_start", maintain_order=True):
            interval_start = interval_start[0] if isinstance(interval_start, tuple) else interval_start
            forced_trips_by_step[interval_start] = group.select("station_id", "dest_station_id")
            counts = group.group_by("station_id").agg(pl.len().alias("n"))
            forced_by_step[interval_start] = _aligned_from_table(counts, "n", network, default=0.0)
    else:
        forced_by_step = None

    n_in_by_step = _per_step_dict(week.n_schedule, "inferred_nontrip_in")
    n_out_by_step = _per_step_dict(week.n_schedule, "inferred_nontrip_out")
    own_lag_1week_by_step = _per_step_dict(week.own_lag_1week_table, "dep_own_lag_1week")
    induced_in_by_step, induced_out_by_step = _induced_step_dicts(induced_moves, week.calendar_weather, network)

    step_outcomes: list[StepOutcome] = []
    t0 = time.monotonic()
    for i, calendar_row in enumerate(week.calendar_weather.iter_rows(named=True)):
        interval_start = calendar_row["interval_start"]
        n_in_arr = n_in_by_step.get(interval_start, np.zeros(network.n))
        n_out_arr = n_out_by_step.get(interval_start, np.zeros(network.n))
        if induced_in_by_step:
            n_in_arr = n_in_arr + induced_in_by_step.get(interval_start, np.zeros(network.n))
        if induced_out_by_step:
            n_out_arr = n_out_arr + induced_out_by_step.get(interval_start, np.zeros(network.n))
        own_lag_1week_arr = own_lag_1week_by_step.get(interval_start, np.zeros(network.n))
        forced = forced_by_step[interval_start] if forced_by_step is not None else None
        forced_trips = (
            forced_trips_by_step.get(interval_start, pl.DataFrame(schema={"station_id": pl.String, "dest_station_id": pl.String}))
            if forced_trips_by_step is not None
            else None
        )

        outcome = run_step(
            network, inventory, own_lag_1h_buffer, prev_departures,
            fitted_dep, od_model, calendar_row, own_lag_1week_arr, n_in_arr, n_out_arr,
            tree, coords_rad, sim_params, rng, forced, forced_trips,
            demand_multiplier=demand_multiplier,
        )
        step_outcomes.append(outcome)

        if (i + 1) % 96 == 0:  # once per simulated day
            elapsed = time.monotonic() - t0
            print(
                f"[simulator] {mode}: "
                f"{i + 1}/{len(week.calendar_weather)} steps ({elapsed:.0f}s elapsed), "
                f"peak RSS {progress.peak_rss_mb():.0f} MB"
            )

    station_intervals = pl.concat(
        [
            pl.DataFrame(
                {
                    "station_id": o.station_id,
                    "interval_start": [o.interval_start] * network.n,
                    "departures_sampled": o.departures_sampled,
                    "departures_actual": o.departures_actual,
                    "lost_no_bike": o.lost_no_bike,
                    "direct_arrivals": o.direct_arrivals,
                    "rerouted_arrivals": o.rerouted_arrivals,
                    "lost_past_cap_arrivals": o.lost_past_cap_arrivals,
                    "n_in": o.n_in,
                    "n_out": o.n_out,
                    "inventory": o.inventory,
                    "is_bike_empty": o.is_bike_empty,
                    "is_dock_full": o.is_dock_full,
                }
            )
            for o in step_outcomes
        ],
        how="vertical",
    )
    trip_log = pl.concat([o.trip_log for o in step_outcomes], how="diagonal_relaxed")

    total_n_violations = sum(o.n_bound_violations for o in step_outcomes)
    n_steps_with_violations = sum(1 for o in step_outcomes if o.n_bound_violations > 0)
    total_clip_created = sum(o.clip_created for o in step_outcomes)
    total_clip_destroyed = sum(o.clip_destroyed for o in step_outcomes)
    if total_n_violations:
        fleet_proxy = float(week.initial_inventory.sum())
        print(
            f"[simulator] baseline N replay pushed inventory out of [0, capacity] "
            f"{total_n_violations:,} station-interval times across {n_steps_with_violations}/{len(step_outcomes)} "
            "steps (clipped each time) -- expected per the Phase 7 plan, since N was fit to the ACTUAL "
            "historical trajectory, not this simulated one"
        )
        print(
            f"[simulator] clipping is NOT neutral: {total_clip_created:,.0f} bikes created (clipped up from "
            f"negative), {total_clip_destroyed:,.0f} bikes destroyed (clipped down from over-capacity), "
            f"net {total_clip_created - total_clip_destroyed:+,.0f} over the week -- vs. a network fleet-size "
            f"proxy (sum of initial inventory at week start) of {fleet_proxy:,.0f} bikes "
            f"({(total_clip_created - total_clip_destroyed) / fleet_proxy:+.1%} of fleet size, net)"
        )

    elapsed = time.monotonic() - t0
    print(
        f"[simulator] {mode} run complete: "
        f"{elapsed:.0f}s, peak RSS {progress.peak_rss_mb():.0f} MB"
    )
    return SimulationRun(
        station_intervals=station_intervals,
        trip_log=trip_log,
        total_n_bound_violations=total_n_violations,
        total_clip_created=total_clip_created,
        total_clip_destroyed=total_clip_destroyed,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 7 forward simulator -- see validate.py for the validation harness.")
    p.add_argument("--mode", choices=MODES, default="sanity_od", help="which of the three run modes to execute (default: sanity_od)")
    args = p.parse_args()

    sim_params = load_simulation_params()
    network = load_station_network()
    week = prepare_week_inputs(network, sim_params)
    fitted_dep = demand.load_fitted(demand.DEMAND_MODEL_DIR / "departures.pkl")
    od_model = od_shares.load_od_share_model()

    run = run_simulation(network, week, fitted_dep, od_model, sim_params, mode=args.mode)
    print(run.station_intervals.select(pl.col("departures_actual", "direct_arrivals", "rerouted_arrivals", "lost_past_cap_arrivals", "lost_no_bike").sum()))


if __name__ == "__main__":
    main()
