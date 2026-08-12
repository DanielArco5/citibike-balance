"""Synthetic-fixture tests for src/models/substitution.py.

Each test constructs a small hand-built (station, interval) situation where
the correct displaced/net_lost split is known by inspection, per CLAUDE.md's
non-negotiable that every model function gets a test with an analytically
known answer."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.substitution import (  # noqa: E402
    add_substitution,
    build_neighbor_pairs,
    estimate_substitution,
    summarize_displaced_vs_lost,
)


def test_build_neighbor_pairs_finds_close_excludes_self_and_far():
    # A and B ~150m apart (within 400m); C is ~5km away.
    stations = pl.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "lat": [40.7300, 40.7313, 40.7750],
            "lng": [-73.9900, -73.9900, -73.9900],
        }
    )
    pairs = build_neighbor_pairs(stations, radius_m=400.0)
    pair_set = set(zip(pairs["station_id"].to_list(), pairs["neighbor_id"].to_list()))
    assert ("A", "B") in pair_set
    assert ("B", "A") in pair_set
    assert ("A", "C") not in pair_set
    assert ("A", "A") not in pair_set
    assert all(d <= 400.0 for d in pairs["distance_m"].to_list())


def _unmet_row(station_id, t, gross, observed, d_hat, censored):
    return {
        "station_id": station_id,
        "interval_start": t,
        "dep_gross_unmet": gross,
        "departures": observed,
        "dep_D_hat": d_hat,
        "is_bike_empty": censored,
    }


def test_estimate_substitution_full_displacement_when_uplift_covers_gross_unmet():
    unmet = pl.DataFrame(
        [
            _unmet_row("A", 0, 5.0, 3.0, 8.0, True),  # gross_unmet = 5 at A
            _unmet_row("B", 0, 0.0, 12.0, 2.0, False),  # neighbor uplift = 10
        ]
    )
    pairs = pl.DataFrame({"station_id": ["A"], "neighbor_id": ["B"], "distance_m": [100.0]})
    result = estimate_substitution(
        unmet, pairs, gross_col="dep_gross_unmet", observed_col="departures",
        dhat_col="dep_D_hat", censor_col="is_bike_empty", out_prefix="dep",
    )
    row = result.filter(pl.col("station_id") == "A")
    assert row["dep_displaced"].item() == pytest.approx(5.0)  # capped at gross_unmet, not full uplift of 10
    assert row["dep_net_lost"].item() == pytest.approx(0.0)


def test_estimate_substitution_partial_displacement_when_uplift_insufficient():
    unmet = pl.DataFrame(
        [
            _unmet_row("A", 0, 5.0, 3.0, 8.0, True),
            _unmet_row("B", 0, 0.0, 4.0, 2.0, False),  # neighbor uplift = 2
        ]
    )
    pairs = pl.DataFrame({"station_id": ["A"], "neighbor_id": ["B"], "distance_m": [100.0]})
    result = estimate_substitution(
        unmet, pairs, gross_col="dep_gross_unmet", observed_col="departures",
        dhat_col="dep_D_hat", censor_col="is_bike_empty", out_prefix="dep",
    )
    row = result.filter(pl.col("station_id") == "A")
    assert row["dep_displaced"].item() == pytest.approx(2.0)
    assert row["dep_net_lost"].item() == pytest.approx(3.0)


def test_estimate_substitution_ignores_censored_neighbor_uplift():
    unmet = pl.DataFrame(
        [
            _unmet_row("A", 0, 5.0, 3.0, 8.0, True),
            # B looks like it has uplift (12 - 2 = 10), but B is ITSELF censored,
            # so its observed count is a lower bound, not clean evidence -- ignore it.
            _unmet_row("B", 0, 0.0, 12.0, 2.0, True),
        ]
    )
    pairs = pl.DataFrame({"station_id": ["A"], "neighbor_id": ["B"], "distance_m": [100.0]})
    result = estimate_substitution(
        unmet, pairs, gross_col="dep_gross_unmet", observed_col="departures",
        dhat_col="dep_D_hat", censor_col="is_bike_empty", out_prefix="dep",
    )
    row = result.filter(pl.col("station_id") == "A")
    assert row["dep_displaced"].item() == pytest.approx(0.0)
    assert row["dep_net_lost"].item() == pytest.approx(5.0)


def test_estimate_substitution_splits_uplift_proportionally_across_competing_stockouts():
    # A (gross=6) and B (gross=2) both have N as a neighbor; N's uplift = 4.
    # Split proportional to gross_unmet: A gets 4*6/8=3, B gets 4*2/8=1.
    unmet = pl.DataFrame(
        [
            _unmet_row("A", 0, 6.0, 1.0, 5.0, True),
            _unmet_row("B", 0, 2.0, 1.0, 3.0, True),
            _unmet_row("N", 0, 0.0, 14.0, 10.0, False),  # uplift = 4
        ]
    )
    pairs = pl.DataFrame(
        {"station_id": ["A", "B"], "neighbor_id": ["N", "N"], "distance_m": [100.0, 150.0]}
    )
    result = estimate_substitution(
        unmet, pairs, gross_col="dep_gross_unmet", observed_col="departures",
        dhat_col="dep_D_hat", censor_col="is_bike_empty", out_prefix="dep",
    ).sort("station_id")
    a = result.filter(pl.col("station_id") == "A")
    b = result.filter(pl.col("station_id") == "B")
    assert a["dep_displaced"].item() == pytest.approx(3.0)
    assert a["dep_net_lost"].item() == pytest.approx(3.0)
    assert b["dep_displaced"].item() == pytest.approx(1.0)
    assert b["dep_net_lost"].item() == pytest.approx(1.0)
    # Neighbor's uplift (4) is never attributed more than once in total.
    assert (a["dep_displaced"].item() + b["dep_displaced"].item()) == pytest.approx(4.0)


def test_estimate_substitution_no_stockouts_returns_empty_with_correct_schema():
    unmet = pl.DataFrame([_unmet_row("A", 0, 0.0, 3.0, 2.0, False)])
    pairs = pl.DataFrame({"station_id": [], "neighbor_id": [], "distance_m": []})
    result = estimate_substitution(
        unmet, pairs, gross_col="dep_gross_unmet", observed_col="departures",
        dhat_col="dep_D_hat", censor_col="is_bike_empty", out_prefix="dep",
    )
    assert result.height == 0
    assert set(result.columns) == {"station_id", "interval_start", "dep_displaced", "dep_net_lost"}


def test_add_substitution_fills_zero_for_non_stockout_rows():
    unmet = pl.DataFrame(
        {
            "station_id": ["A", "B"],
            "interval_start": [0, 0],
            "dep_gross_unmet": [0.0, 0.0],
            "departures": [3.0, 3.0],
            "dep_D_hat": [2.0, 2.0],
            "is_bike_empty": [False, False],
            "arr_gross_unmet": [0.0, 0.0],
            "arrivals": [3.0, 3.0],
            "arr_D_hat": [2.0, 2.0],
            "is_dock_full": [False, False],
        }
    )
    pairs = pl.DataFrame({"station_id": ["A"], "neighbor_id": ["B"], "distance_m": [100.0]})
    out = add_substitution(unmet, pairs)
    assert out["dep_displaced"].to_list() == [0.0, 0.0]
    assert out["dep_net_lost"].to_list() == [0.0, 0.0]
    assert out["arr_displaced"].to_list() == [0.0, 0.0]
    assert out["arr_net_lost"].to_list() == [0.0, 0.0]


def test_summarize_displaced_vs_lost_computes_fractions_per_direction():
    unmet_net = pl.DataFrame(
        {
            "dep_gross_unmet": [10.0, 0.0],
            "dep_displaced": [4.0, 0.0],
            "dep_net_lost": [6.0, 0.0],
            "arr_gross_unmet": [5.0, 5.0],
            "arr_displaced": [5.0, 0.0],
            "arr_net_lost": [0.0, 5.0],
        }
    )
    summary = summarize_displaced_vs_lost(unmet_net).sort("direction")
    dep = summary.filter(pl.col("direction") == "dep")
    arr = summary.filter(pl.col("direction") == "arr")
    assert dep["frac_displaced"].item() == pytest.approx(0.4)
    assert dep["frac_net_lost"].item() == pytest.approx(0.6)
    assert arr["frac_displaced"].item() == pytest.approx(0.5)
    assert arr["frac_net_lost"].item() == pytest.approx(0.5)
