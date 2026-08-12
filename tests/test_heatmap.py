"""Synthetic-fixture tests for src/viz/heatmap.py's pure data-transform
functions. Plotting/folium-rendering functions aren't unit-tested here,
matching the project's existing convention (reports/plot_hour_of_week.py has
no test file) -- only the numeric transforms that feed them, since those are
the parts with an analytically knowable answer."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viz import heatmap  # noqa: E402
from viz.heatmap import (  # noqa: E402
    add_per_dock_columns,
    aggregate_station_net_lost,
    aggregate_zone_hour,
    aggregate_zone_hour_per_dock_ci,
    build_matrix,
    exclude_jc_hoboken,
    hour_of_week_to_label,
    is_jc_hoboken_zone,
    n_weeks_covered,
    peak_hour,
    plot_station_scatter,
    top_zones,
    top_zones_by_lower_bound,
    zone_capacity,
)


def test_heatmap_module_does_not_import_folium():
    # folium's tile servers aren't reachable in this environment -- the
    # worst-zone-hours map was replaced with plot_station_scatter
    # (matplotlib, no network dependency). Regression guard against
    # reintroducing the import.
    assert "folium" not in sys.modules or not hasattr(heatmap, "folium")
    assert not hasattr(heatmap, "build_worst_zone_hours_map")


def test_n_weeks_covered_computes_span_in_weeks():
    unmet = pl.DataFrame(
        {"interval_start": [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 15)]}  # exactly 2 weeks apart
    )
    assert n_weeks_covered(unmet) == pytest.approx(2.0)


def test_aggregate_zone_hour_sums_and_divides_by_weeks():
    # 2 stations in zone Z at hour_of_week=5, spanning exactly 1 week ->
    # weeks=1, so per_week == raw sum.
    unmet = pl.DataFrame(
        {
            "zone_agg": ["Z", "Z"],
            "hour_of_week": [5, 5],
            "dep_net_lost": [3.0, 4.0],
            "arr_net_lost": [1.0, 2.0],
            "interval_start": [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 8)],
        }
    )
    agg = aggregate_zone_hour(unmet)
    row = agg.filter((pl.col("zone_agg") == "Z") & (pl.col("hour_of_week") == 5))
    assert row["dep_net_lost_total"].item() == pytest.approx(7.0)
    assert row["dep_net_lost_per_week"].item() == pytest.approx(7.0)
    assert row["arr_net_lost_per_week"].item() == pytest.approx(3.0)


def test_aggregate_zone_hour_accepts_lazyframe_same_as_eager():
    # Production path passes a LazyFrame (a scan over the monthly checkpoint
    # directory); this must produce the same result as the eager path.
    unmet = pl.DataFrame(
        {
            "zone_agg": ["Z", "Z"],
            "hour_of_week": [5, 5],
            "dep_net_lost": [3.0, 4.0],
            "arr_net_lost": [1.0, 2.0],
            "interval_start": [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 8)],
        }
    )
    eager = aggregate_zone_hour(unmet)
    lazy = aggregate_zone_hour(unmet.lazy())
    assert eager.equals(lazy.select(eager.columns))


def test_aggregate_station_net_lost_sums_and_divides_by_weeks():
    unmet = pl.DataFrame(
        {
            "station_id": ["A", "A", "B"],
            "dep_net_lost": [3.0, 4.0, 10.0],
            "arr_net_lost": [1.0, 2.0, 0.0],
            "interval_start": [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 8), dt.datetime(2025, 1, 1)],
        }
    )
    agg = aggregate_station_net_lost(unmet)
    a = agg.filter(pl.col("station_id") == "A")
    b = agg.filter(pl.col("station_id") == "B")
    assert a["dep_net_lost_per_week"].item() == pytest.approx(7.0)
    assert a["arr_net_lost_per_week"].item() == pytest.approx(3.0)
    assert b["dep_net_lost_per_week"].item() == pytest.approx(10.0)


def test_plot_station_scatter_writes_a_file(tmp_path):
    station_agg = pl.DataFrame(
        {"station_id": ["A", "B"], "dep_net_lost_per_week": [5.0, 10.0], "arr_net_lost_per_week": [1.0, 2.0]}
    )
    stations = pl.DataFrame({"station_id": ["A", "B"], "lat": [40.73, 40.75], "lng": [-73.99, -73.98]})
    out_path = tmp_path / "net_lost_station_map.png"
    plot_station_scatter(station_agg, stations, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_is_jc_hoboken_zone_uses_lng_and_lat_thresholds():
    stations = pl.DataFrame(
        {
            "station_id": ["A", "B", "C", "D"],
            # A: Hoboken (12 St & Sinatra Dr N) -- west of the Hudson, in the lat band.
            # B: Manhattan Battery Park City -- east of the Hudson gap.
            # C: Brooklyn Bay Ridge -- similar lng to Hoboken but well south (lat < 40.68).
            # D: Manhattan Midtown -- nowhere near the cutoff.
            "lat": [40.7530, 40.7100, 40.6328, 40.7584],
            "lng": [-74.0240, -74.0165, -74.0244, -73.9759],
            "zone_agglomerative": ["hoboken_zone", "manhattan_zone", "brooklyn_zone", "midtown_zone"],
        }
    )
    flags = is_jc_hoboken_zone(stations).sort("zone_agg")
    flag_map = dict(zip(flags["zone_agg"].to_list(), flags["is_jc_hoboken"].to_list()))
    assert flag_map["hoboken_zone"] is True
    assert flag_map["manhattan_zone"] is False
    assert flag_map["brooklyn_zone"] is False
    assert flag_map["midtown_zone"] is False


def test_exclude_jc_hoboken_drops_flagged_zones_only():
    agg = pl.DataFrame({"zone_agg": ["Z1", "Z2", "Z3"], "hour_of_week": [0, 0, 0], "dep_net_lost": [1.0, 2.0, 3.0]})
    jc_flags = pl.DataFrame({"zone_agg": ["Z1", "Z2"], "is_jc_hoboken": [True, False]})
    out = exclude_jc_hoboken(agg, jc_flags)
    # Z1 dropped (flagged), Z2 kept (flagged False), Z3 kept (unflagged -> fill_null(False)).
    assert sorted(out["zone_agg"].to_list()) == ["Z2", "Z3"]


def test_aggregate_zone_hour_per_dock_ci_penalizes_noisy_small_sample():
    # Both zones have the SAME per-dock point estimate (0.5) but very
    # different week-to-week consistency -- RELIABLE has zero variance
    # (same value every week), NOISY has one big spike among mostly-zero
    # weeks. The lower bound must separate them even though the means tie.
    mondays = [
        dt.datetime(2025, 1, 6, 8),
        dt.datetime(2025, 1, 13, 8),
        dt.datetime(2025, 1, 20, 8),
        dt.datetime(2025, 1, 27, 8),
    ]
    rows = []
    for t, val in zip(mondays, [5.0, 5.0, 5.0, 5.0]):
        rows.append(
            {"zone_agg": "RELIABLE", "hour_of_week": 8, "interval_start": t, "dep_net_lost": val, "arr_net_lost": 0.0}
        )
    for t, val in zip(mondays, [0.0, 0.0, 0.0, 20.0]):
        rows.append(
            {"zone_agg": "NOISY", "hour_of_week": 8, "interval_start": t, "dep_net_lost": val, "arr_net_lost": 0.0}
        )
    unmet = pl.DataFrame(rows)
    zone_cap = pl.DataFrame({"zone_agg": ["RELIABLE", "NOISY"], "zone_capacity": [10, 10]})

    stats = aggregate_zone_hour_per_dock_ci(unmet, zone_cap)
    reliable = stats.filter(pl.col("zone_agg") == "RELIABLE").row(0, named=True)
    noisy = stats.filter(pl.col("zone_agg") == "NOISY").row(0, named=True)

    # Same point estimate (0.5 net-lost/dock/week either way)...
    assert reliable["dep_net_lost_per_dock"] == pytest.approx(0.5)
    assert noisy["dep_net_lost_per_dock"] == pytest.approx(0.5)
    # ...but the lower bound must separate them: zero-variance RELIABLE
    # keeps its full mean (se=0), one-spike-in-four NOISY gets pulled
    # negative (mean 0.5, se 0.5, 0.5 - 1.96*0.5 = -0.48).
    assert reliable["dep_net_lost_per_dock_lb"] == pytest.approx(0.5)
    assert noisy["dep_net_lost_per_dock_lb"] == pytest.approx(-0.48)
    assert reliable["dep_net_lost_per_dock_lb"] > noisy["dep_net_lost_per_dock_lb"]


def test_aggregate_zone_hour_per_dock_ci_null_lower_bound_for_single_week():
    # Can't estimate a standard error from one observation -- the lower
    # bound must be null, not silently 0-filled (which would treat a single
    # noisy week as perfectly reliable).
    unmet = pl.DataFrame(
        {
            "zone_agg": ["Z"],
            "hour_of_week": [8],
            "interval_start": [dt.datetime(2025, 1, 6, 8)],
            "dep_net_lost": [5.0],
            "arr_net_lost": [0.0],
        }
    )
    zone_cap = pl.DataFrame({"zone_agg": ["Z"], "zone_capacity": [10]})
    row = aggregate_zone_hour_per_dock_ci(unmet, zone_cap).row(0, named=True)
    assert row["n_weeks"] == 1
    assert row["dep_net_lost_per_dock_lb"] is None


def test_top_zones_by_lower_bound_ranks_reliable_over_noisy_despite_tied_mean():
    agg_ci = pl.DataFrame(
        {
            "zone_agg": ["NOISY", "RELIABLE"],
            "hour_of_week": [8, 8],
            "dep_net_lost_per_dock_lb": [-0.48, 0.5],
            "arr_net_lost_per_dock_lb": [None, None],
        }
    )
    order = top_zones_by_lower_bound(agg_ci, "dep_net_lost_per_dock_lb", "arr_net_lost_per_dock_lb", n=2)
    assert order == ["RELIABLE", "NOISY"]


def test_zone_capacity_sums_distinct_stations_not_panel_rows():
    # Same station appears twice in stations table would be a data bug, but
    # the real risk this guards against is summing per-INTERVAL panel rows
    # (which would multiply by row count) -- zone_capacity must operate on
    # the one-row-per-station stations table, not a (station, interval) panel.
    stations = pl.DataFrame(
        {"station_id": ["A", "B", "C"], "zone_agglomerative": ["Z1", "Z1", "Z2"], "capacity": [20, 30, 15]}
    )
    zc = zone_capacity(stations)
    assert zc.filter(pl.col("zone_agg") == "Z1")["zone_capacity"].item() == 50
    assert zc.filter(pl.col("zone_agg") == "Z2")["zone_capacity"].item() == 15


def test_add_per_dock_columns_divides_by_zone_capacity():
    agg = pl.DataFrame(
        {"zone_agg": ["Z1"], "hour_of_week": [0], "dep_net_lost_per_week": [10.0], "arr_net_lost_per_week": [5.0]}
    )
    zone_cap = pl.DataFrame({"zone_agg": ["Z1"], "zone_capacity": [20]})
    out = add_per_dock_columns(agg, zone_cap)
    assert out["dep_net_lost_per_dock"].item() == pytest.approx(0.5)
    assert out["arr_net_lost_per_dock"].item() == pytest.approx(0.25)


def test_top_zones_ranks_by_combined_total_descending():
    agg = pl.DataFrame(
        {
            "zone_agg": ["Z1", "Z1", "Z2", "Z3"],
            "hour_of_week": [0, 1, 0, 0],
            "dep_net_lost_per_week": [10.0, 5.0, 1.0, 100.0],
            "arr_net_lost_per_week": [0.0, 0.0, 0.0, 0.0],
        }
    )
    # Z1 total=15, Z2 total=1, Z3 total=100
    order = top_zones(agg, "dep_net_lost_per_week", "arr_net_lost_per_week", n=3)
    assert order == ["Z3", "Z1", "Z2"]


def test_top_zones_can_diverge_from_raw_ranking_on_per_dock_columns():
    # Z1 has a huge raw total but is huge per-dock too here; the point of this
    # test is just that top_zones() is generic over which columns it's given.
    agg = pl.DataFrame(
        {
            "zone_agg": ["Z1", "Z2"],
            "hour_of_week": [0, 0],
            "dep_net_lost_per_dock": [1.0, 50.0],
            "arr_net_lost_per_dock": [0.0, 0.0],
        }
    )
    order = top_zones(agg, "dep_net_lost_per_dock", "arr_net_lost_per_dock", n=2)
    assert order == ["Z2", "Z1"]


def test_build_matrix_places_values_at_correct_zone_row_and_hour_column():
    agg = pl.DataFrame(
        {"zone_agg": ["Z1", "Z2"], "hour_of_week": [3, 100], "dep_net_lost_per_week": [7.0, 9.0]}
    )
    zone_order = ["Z2", "Z1"]  # deliberately not the same order as the input
    mat = build_matrix(agg, "dep_net_lost_per_week", zone_order)
    assert mat.shape == (2, 168)
    assert mat[0, 100] == pytest.approx(9.0)  # Z2 is row 0
    assert mat[1, 3] == pytest.approx(7.0)  # Z1 is row 1
    # every other cell is 0, not null/nan
    assert mat.sum() == pytest.approx(16.0)


def test_build_matrix_ignores_zones_not_in_zone_order():
    agg = pl.DataFrame({"zone_agg": ["Zignored"], "hour_of_week": [0], "dep_net_lost_per_week": [999.0]})
    mat = build_matrix(agg, "dep_net_lost_per_week", ["Z1"])
    assert mat.shape == (1, 168)
    assert mat.sum() == 0.0


def test_hour_of_week_to_label_matches_day_and_hour():
    assert hour_of_week_to_label(0) == "Mon 00:00"
    assert hour_of_week_to_label(31) == "Tue 07:00"  # 24 + 7
    assert hour_of_week_to_label(167) == "Sun 23:00"


def test_peak_hour_finds_the_max_row_for_that_zone():
    agg = pl.DataFrame(
        {
            "zone_agg": ["Z1", "Z1", "Z2"],
            "hour_of_week": [10, 31, 5],
            "dep_net_lost_per_week": [2.0, 8.0, 100.0],
        }
    )
    label, val = peak_hour(agg, "Z1", "dep_net_lost_per_week")
    assert label == hour_of_week_to_label(31)
    assert val == pytest.approx(8.0)


def test_peak_hour_returns_na_for_unknown_zone():
    agg = pl.DataFrame({"zone_agg": ["Z1"], "hour_of_week": [0], "dep_net_lost_per_week": [1.0]})
    label, val = peak_hour(agg, "Zmissing", "dep_net_lost_per_week")
    assert label == "n/a"
    assert val == 0.0
