"""Infer per-station-interval inventory I(s,t) and non-trip bike movement
N(s,t), per SPEC.md §4 and the RUNBOOK Phase 4 design-fork discussion.

Originally framed as "operator rebalancing" (R). Renamed after the DOT
cross-check (validation #2) came back ~1.4x HIGH against DOT's reported
rebalancing volume -- the wrong direction for a minimal-correction method,
which should only ever underestimate real rebalancing (see the structural
limitation noted below). Diagnosis (full writeup in DECISIONS.md): this
quantity isn't only rebalancing. Anything that removes or adds a bike
without a matching trip -- operator rebalancing, but also maintenance
pulls, broken-bike removal, e-bike battery swaps -- produces the identical
signature in the data: inventory changes arrivals/departures don't explain.
The flow-balance equation can't tell these apart, so the honest name for
what it identifies is non-trip bike movement, not rebalancing specifically.
Rebalancing is a subset of it. This is still the right quantity for Phase
5's censoring correction -- a bike pulled for maintenance affects station
availability exactly like a bike pulled by a truck -- it just isn't
DOT's number, and shouldn't be compared to DOT's as if it were an apples-
to-apples check (see validate_against_dot's docstring).

N is unobserved. Flow balance is:

    I(s,t+1) = I(s,t) + arrivals(s,t) - departures(s,t) + N(s,t)

The naive trajectory (N=0) drifts outside [0, capacity] -- physically
impossible -- and those violations are the non-trip-movement signal. This
module infers the minimal N that keeps every station in bounds, via a
per-station LP over Monday-aligned calendar weeks, rather than a
per-station-day greedy clip. Recap of why (full reasoning was a plan-mode
discussion, not repeated here):

- L1-minimal N is naturally sparse/bursty (a few large corrections), which
  matches how trucks and depot visits actually operate and is what makes
  the DOT cross-check interpretable at all (as a lower bound on our number,
  not an equality -- see above). Greedy corrects at every violating
  instant, which doesn't look like anything countable as a discrete event.
- The window is one week, not one day or one year: daily resets would force
  a spurious correction at every midnight purely from anchor mismatch; a
  single year-long LP would hide within-year drift.
- The anchor (starting inventory) is a free LP variable penalized toward a
  prior, not fixed -- but the penalty weight must exceed 1.0 (N's per-unit
  cost), or the LP finds it cheaper to silently redefine the week's start
  than to credit real non-trip movement (caught by a unit test, see
  ReconstructionParams and tests/test_inventory.py). At weight > 1, the
  prior only matters as a genuine tie-breaker -- station-weeks with zero
  violations under any anchor, where N=0 and anchor=prior are already the
  cheapest answer with nothing to trade off. Each week's ending inventory
  becomes the next week's prior, so trajectories have continuity across the
  year instead of resetting at every boundary.

A structural limitation, true of ANY minimal-|N| method (this one, greedy,
or otherwise): it's a lower bound on real non-trip movement. Operators and
depots act preemptively and sometimes redundantly -- a truck tops off a
station before it would have hit zero, then organic demand partially undoes
that within the hour. That pair of moves is invisible to a minimal-
correction estimator by construction. Not a bug to chase away by tuning --
and it's why the ~1.4x-high DOT ratio was worth stopping to diagnose rather
than accepting: this floor predicts <=1, so >1 meant the estimator was
capturing something real that the label "rebalancing" didn't cover.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import yaml
from scipy.optimize import linprog

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = REPO_ROOT / "data" / "processed" / "panel.parquet"
DOT_REBALANCE_PATH = REPO_ROOT / "data" / "interim" / "dot_monthly_rebalancing.parquet"
PARAMS_PATH = REPO_ROOT / "config" / "params.yaml"
OUT_PATH = REPO_ROOT / "data" / "processed" / "inventory.parquet"

INTERVAL_MINUTES = 15
BOUNDARY_EPS = 1e-6  # LP solver tolerance for "exactly 0" / "exactly capacity"


@dataclass
class ReconstructionParams:
    anchor_prior_weight: float = 2.0  # must be > 1.0 -- see module docstring
    week1_anchor_fraction_of_capacity: float = 0.5


def load_params(path: Path = PARAMS_PATH) -> ReconstructionParams:
    cfg = yaml.safe_load(path.read_text())["inventory_reconstruction"]
    return ReconstructionParams(
        anchor_prior_weight=cfg["anchor_prior_weight"],
        week1_anchor_fraction_of_capacity=cfg["week1_anchor_fraction_of_capacity"],
    )


# ---------------------------------------------------------------------------
# Per-station-week LP
# ---------------------------------------------------------------------------


def solve_station_week_lp(
    net_flow: np.ndarray,
    capacity: float,
    anchor_prior: float,
    anchor_prior_weight: float,
) -> dict:
    """One LP per (station, week). Variables: I[0..W] (inventory at the
    start of each interval, W+1 points spanning the week), N_in[0..W-1],
    N_out[0..W-1] (both >= 0, decomposed rather than signed so the L1 cost
    is a plain linear sum -- N for non-trip movement, see module docstring),
    and u/v (>=0 slack pair linearizing the anchor's |I[0] - prior| penalty).

    Equality constraints, one per interval plus one for the anchor:
        I[t+1] - I[t] - N_in[t] + N_out[t] = net_flow[t]   for t in 0..W-1
        I[0] - u + v = anchor_prior
    Bounds carry 0 <= I[t] <= capacity directly (not constraint rows) --
    keeps the constraint matrix sparse (<=4 nonzeros/row) so this stays fast
    even solved ~127k times (one per station per week) across the panel.

    Objective: minimize sum(N_in) + sum(N_out) + anchor_prior_weight*(u+v).
    """
    W = len(net_flow)
    n_I = W + 1
    idx_Nin = lambda t: n_I + t  # noqa: E731
    idx_Nout = lambda t: n_I + W + t  # noqa: E731
    idx_u = n_I + 2 * W
    idx_v = idx_u + 1
    n_vars = idx_v + 1

    c = np.zeros(n_vars)
    c[n_I : n_I + 2 * W] = 1.0
    c[idx_u] = anchor_prior_weight
    c[idx_v] = anchor_prior_weight

    rows, cols, data = [], [], []
    for t in range(W):
        rows += [t, t, t, t]
        cols += [t + 1, t, idx_Nin(t), idx_Nout(t)]
        data += [1.0, -1.0, -1.0, 1.0]
    rows += [W, W, W]
    cols += [0, idx_u, idx_v]
    data += [1.0, -1.0, 1.0]
    b_eq = np.concatenate([net_flow, [anchor_prior]])

    A_eq = sp.csr_matrix((data, (rows, cols)), shape=(W + 1, n_vars))
    bounds = [(0, capacity)] * n_I + [(0, None)] * (2 * W) + [(0, None), (0, None)]

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP infeasible/failed (W={W}, capacity={capacity}): {res.message}")

    x = res.x
    return {
        "I": x[0:n_I],
        "N_in": x[n_I : n_I + W],
        "N_out": x[n_I + W : n_I + 2 * W],
    }


# ---------------------------------------------------------------------------
# Per-station driver (sequential over weeks -- anchor chaining requires it;
# independent across stations -- see reconstruct_inventory for that axis)
# ---------------------------------------------------------------------------


def reconstruct_station(
    interval_start: np.ndarray,
    net_flow: np.ndarray,
    week_start: np.ndarray,
    capacity: float,
    params: ReconstructionParams,
) -> dict:
    """interval_start/net_flow/week_start must already be sorted
    chronologically for one station. Returns arrays aligned to
    interval_start: inventory, nontrip_in, nontrip_out."""
    boundaries = np.flatnonzero(np.diff(week_start) != 0) + 1
    week_groups = np.split(np.arange(len(week_start)), boundaries)

    out_inventory = np.empty(len(interval_start))
    out_nontrip_in = np.empty(len(interval_start))
    out_nontrip_out = np.empty(len(interval_start))

    prior = capacity * params.week1_anchor_fraction_of_capacity
    for group_idx in week_groups:
        week_net_flow = net_flow[group_idx]
        solved = solve_station_week_lp(week_net_flow, capacity, prior, params.anchor_prior_weight)
        out_inventory[group_idx] = solved["I"][:-1]
        out_nontrip_in[group_idx] = solved["N_in"]
        out_nontrip_out[group_idx] = solved["N_out"]
        prior = solved["I"][-1]

    return {"inventory": out_inventory, "nontrip_in": out_nontrip_in, "nontrip_out": out_nontrip_out}


# ---------------------------------------------------------------------------
# Greedy per-station-day clip -- cheap sanity diff against the LP, not the
# primary estimator. See module docstring for why.
# ---------------------------------------------------------------------------


def reconstruct_station_greedy(net_flow: np.ndarray, capacity: float, anchor: float) -> dict:
    inventory = np.empty(len(net_flow))
    nontrip_in = np.zeros(len(net_flow))
    nontrip_out = np.zeros(len(net_flow))
    level = anchor
    for t, flow in enumerate(net_flow):
        inventory[t] = level
        naive_next = level + flow
        if naive_next < 0:
            nontrip_in[t] = -naive_next
            level = 0.0
        elif naive_next > capacity:
            nontrip_out[t] = naive_next - capacity
            level = capacity
        else:
            level = naive_next
    return {"inventory": inventory, "nontrip_in": nontrip_in, "nontrip_out": nontrip_out}


# ---------------------------------------------------------------------------
# Full-panel driver
# ---------------------------------------------------------------------------


def _report_dropped_stations(panel: pl.DataFrame, dropped: pl.DataFrame, reason: str) -> None:
    """Same accounting gbfs.py's reconcile_capacity() uses for the 158
    null-capacity stations: count, which stations, and what share of trip
    volume they represent -- so exclusions are one auditable number, not a
    silent row-count drop."""
    n_dropped_stations = dropped["station_id"].n_unique()
    dropped_activity = (dropped["departures"] + dropped["arrivals"]).sum()
    total_activity = (panel["departures"] + panel["arrivals"]).sum()
    frac_activity = dropped_activity / total_activity if total_activity else 0.0
    ids = dropped["station_id"].unique().sort().to_list()
    print(
        f"[inventory] dropping {n_dropped_stations:,} stations ({dropped.height:,} rows) "
        f"{reason}: {ids}"
    )
    print(
        f"[inventory]   trip volume represented: {dropped_activity:,} / {total_activity:,} "
        f"({frac_activity:.2%} of activity)"
    )


def prepare_panel(panel: pl.DataFrame) -> pl.DataFrame:
    null_cap = panel.filter(pl.col("capacity").is_null())
    _report_dropped_stations(panel, null_cap, "with null capacity -- an LP can't bound inventory without one")
    usable = panel.filter(pl.col("capacity").is_not_null())

    # capacity == 0 for a station with real trip activity is not a gray-area
    # capacity error -- it's impossible (every single trip would "violate"
    # a zero-width bound) and forces the LP to invent a non-trip-movement
    # event on nearly every active interval. Confirmed on real data: 3 such
    # stations, one with 146K+ trips/year, contributed 7.4% of total
    # inferred N before this filter existed. Same treatment as null
    # capacity -- dropped, not imputed, and logged the same way. See
    # DECISIONS.md for the broader, still-open question of how many OTHER
    # stations have stale (not just zero) capacity that this filter doesn't
    # catch.
    zero_cap_active = usable.filter((pl.col("capacity") == 0) & ((pl.col("departures") > 0) | (pl.col("arrivals") > 0)))
    zero_cap_station_ids = zero_cap_active["station_id"].unique().to_list()
    zero_cap_all_rows = usable.filter(pl.col("station_id").is_in(zero_cap_station_ids))
    _report_dropped_stations(usable, zero_cap_all_rows, "with capacity == 0 despite observed trip activity")
    usable = usable.filter(~pl.col("station_id").is_in(zero_cap_station_ids))

    cap_check = usable.group_by("station_id").agg(pl.col("capacity").n_unique().alias("n_cap"))
    bad = cap_check.filter(pl.col("n_cap") > 1)
    assert bad.height == 0, (
        f"{bad.height} stations have a time-varying capacity in the panel -- "
        "this code assumes constant capacity per station; investigate before proceeding"
    )

    return usable.with_columns(
        (pl.col("arrivals").cast(pl.Int64) - pl.col("departures").cast(pl.Int64)).alias("net_flow"),
        pl.col("interval_start").dt.truncate("1w").alias("week_start"),
    ).sort("station_id", "interval_start")


def reconstruct_inventory(
    panel: pl.DataFrame, params: ReconstructionParams, method: str = "lp"
) -> pl.DataFrame:
    usable = prepare_panel(panel)
    station_frames = usable.partition_by("station_id", maintain_order=True)

    out_frames = []
    t0 = time.time()
    for i, sdf in enumerate(station_frames):
        station_id = sdf["station_id"][0]
        capacity = float(sdf["capacity"][0])
        net_flow = sdf["net_flow"].to_numpy().astype(np.float64)
        interval_start = sdf["interval_start"].to_numpy()

        if method == "lp":
            week_start = sdf["week_start"].to_numpy()
            result = reconstruct_station(interval_start, net_flow, week_start, capacity, params)
        elif method == "greedy":
            anchor = capacity * params.week1_anchor_fraction_of_capacity
            result = reconstruct_station_greedy(net_flow, capacity, anchor)
        else:
            raise ValueError(f"unknown method {method!r}")

        out_frames.append(
            pl.DataFrame(
                {
                    "station_id": [station_id] * len(interval_start),
                    "interval_start": interval_start,
                    "inventory": result["inventory"],
                    "capacity": [capacity] * len(interval_start),
                    "inferred_nontrip_in": result["nontrip_in"],
                    "inferred_nontrip_out": result["nontrip_out"],
                }
            )
        )
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(station_frames) - i - 1) / rate
            print(
                f"[inventory] {method}: {i+1:,}/{len(station_frames):,} stations "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )

    result = pl.concat(out_frames)
    return result.with_columns(
        (pl.col("inventory") <= BOUNDARY_EPS).alias("is_bike_empty"),
        (pl.col("inventory") >= pl.col("capacity") - BOUNDARY_EPS).alias("is_dock_full"),
    ).with_columns(
        # Coarse per-bucket proxy, not a continuous within-interval estimate:
        # the LP only resolves state at 15-min grid points, so it cannot
        # distinguish a station empty for 2 minutes from one empty for the
        # full interval (the exact resolution limitation SPEC.md §2 flags
        # for hourly simulation, one order of magnitude smaller here).
        # Documented, not hidden -- Phase 5's censoring correction inherits
        # this ceiling and should treat "heavily censored" with that in mind.
        pl.when(pl.col("is_bike_empty")).then(INTERVAL_MINUTES).otherwise(0).alias("minutes_empty"),
        pl.when(pl.col("is_dock_full")).then(INTERVAL_MINUTES).otherwise(0).alias("minutes_full"),
    ).select(
        "station_id",
        "interval_start",
        "inventory",
        "is_bike_empty",
        "is_dock_full",
        "minutes_empty",
        "minutes_full",
        "inferred_nontrip_in",
        "inferred_nontrip_out",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_bounds(inventory: pl.DataFrame, panel: pl.DataFrame) -> None:
    cap = panel.select("station_id", "capacity").unique()
    joined = inventory.join(cap, on="station_id", how="left")
    violations = joined.filter(
        (pl.col("inventory") < -BOUNDARY_EPS) | (pl.col("inventory") > pl.col("capacity") + BOUNDARY_EPS)
    )
    assert violations.height == 0, f"{violations.height} station-intervals violate [0, capacity] after correction"
    print(f"[inventory] validation 1/3 PASSED: bounds never violated ({inventory.height:,} station-intervals checked)")


def validate_against_dot(inventory: pl.DataFrame, dot_totals_path: Path = DOT_REBALANCE_PATH) -> pl.DataFrame:
    """NOT an equality check. DOT's monthly figure is Citi Bike's own count
    of rebalancing specifically (trucks/vans/valets/Bike Angels -- see
    src/ingest/dot_reports.py). Our inferred_nontrip_* is a superset: any
    bike removed or added without a matching trip, which also includes
    maintenance pulls, broken-bike removal, and e-bike battery swaps (see
    module docstring, DECISIONS.md). So the expected relationship is
    inferred >= DOT (ratio >= 1), not inferred == DOT -- a ratio near or
    below 1 would be the surprising result here, not one comfortably above
    it. Flagged accordingly below: low ratios are flagged, high-but-
    plausible ones aren't."""
    dot = pl.read_parquet(dot_totals_path)

    ours = (
        inventory.with_columns(pl.col("interval_start").dt.truncate("1mo").dt.date().alias("month_start"))
        .group_by("month_start")
        .agg((pl.col("inferred_nontrip_in") + pl.col("inferred_nontrip_out")).sum().alias("inferred_nontrip_total"))
    )

    comparison = dot.join(ours, on="month_start", how="left").sort("month_start")
    comparison = comparison.with_columns(
        (pl.col("inferred_nontrip_total") / pl.col("rebalanced_bikes_total")).alias("ratio")
    )
    print("[inventory] validation 2/3 -- system-wide monthly inferred non-trip movement vs DOT's rebalancing-only figure:")
    print(
        "  NOTE: 2024-12 is a single day of panel coverage (panel starts "
        "2024-12-31) against a full-month DOT figure -- not a fair comparison, excluded from the read."
    )
    print(
        "  NOTE: ratio >= 1 is EXPECTED (our figure is a superset of DOT's -- see this "
        "function's docstring), not a mismatch to explain away."
    )
    for row in comparison.iter_rows(named=True):
        if row["inferred_nontrip_total"] is None:
            print(
                f"  {row['month_start']}: DOT={row['rebalanced_bikes_total']:,}  "
                "inferred=MISSING -- no panel coverage for this month, not a modeling result"
            )
            continue
        is_dec_2024 = row["month_start"].month == 12 and row["month_start"].year == 2024
        flag = "" if is_dec_2024 else (
            "  <-- BELOW DOT's own rebalancing-only figure, shouldn't happen for a superset" if row["ratio"] < 0.95
            else ("  <-- implausibly high, investigate" if row["ratio"] > 5 else "")
        )
        print(
            f"  {row['month_start']}: inferred_nontrip={row['inferred_nontrip_total']:,.0f}  "
            f"DOT_rebalancing={row['rebalanced_bikes_total']:,}  ratio={row['ratio']:.2f}{flag}"
        )
    return comparison


def validate_against_gbfs() -> None:
    print(
        "[inventory] validation 3/3 SKIPPED: GBFS snapshots on disk start 2026-08-04; "
        "the panel covers 2024-12-31 through 2025-12-31. Zero overlap -- confirmed by "
        "date range, not assumed. No ground-truth inventory exists for this panel."
    )


def main() -> None:
    params = load_params()
    panel = pl.read_parquet(PANEL_PATH)

    inventory = reconstruct_inventory(panel, params, method="lp")

    # Write BEFORE validating. The LP loop is the expensive part (hours,
    # not seconds); a bug in a downstream validation step must never cost
    # a rerun of the reconstruction itself to fix. Learned this the hard
    # way -- see DECISIONS.md, the anchor-penalty-bug entry's neighbor.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_parquet(OUT_PATH)
    print(f"[inventory] wrote {OUT_PATH} ({inventory.height:,} rows)")

    run_validations(panel)


def run_validations(panel: pl.DataFrame | None = None) -> None:
    """Standalone from main() on purpose: lets validation be rerun against
    the already-written parquet without repeating the LP reconstruction."""
    inventory = pl.read_parquet(OUT_PATH)
    if panel is None:
        panel = pl.read_parquet(PANEL_PATH)

    validate_bounds(inventory, panel)
    validate_against_dot(inventory)
    validate_against_gbfs()


if __name__ == "__main__":
    main()
