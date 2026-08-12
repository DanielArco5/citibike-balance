"""Synthetic-fixture tests for src/models/demand.py, per CLAUDE.md's
non-negotiable that every model function gets a test with an analytically
known answer.

Feature-building helpers (add_weather_lags/add_own_lags/add_neighbor_demand)
are tested directly against small hand-built frames. The fit/calibrate/
impute pipeline is tested with a stubbed-out gradient booster (fixed,
known predictions) rather than an actually-trained model, so the
bias-correction bookkeeping in impute_unmet_demand can be checked
deterministically without depending on sklearn convergence on tiny data."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import demand  # noqa: E402
from models.censoring import CensoringParams  # noqa: E402


def test_add_weather_lags_shifts_precip_by_hour():
    weather = pl.DataFrame(
        {
            "timestamp": [
                dt.datetime(2025, 1, 1, 0),
                dt.datetime(2025, 1, 1, 1),
                dt.datetime(2025, 1, 1, 2),
                dt.datetime(2025, 1, 1, 3),
            ],
            "precip_mm": [0.0, 1.0, 2.0, 3.0],
        }
    )
    df = pl.DataFrame({"interval_start": [dt.datetime(2025, 1, 1, 2, 5)]})
    out = demand.add_weather_lags(df, weather)
    # interval falls in hour=2 (precip=2.0); lag1h should be hour=1's precip (1.0),
    # lag2h should be hour=0's precip (0.0).
    assert out["precip_lag1h"].item() == pytest.approx(1.0)
    assert out["precip_lag2h"].item() == pytest.approx(0.0)


def test_add_own_lags_rolling_mean_excludes_current_interval():
    df = pl.DataFrame(
        {
            "station_id": ["A"] * 6,
            "interval_start": list(range(6)),
            "departures": [10, 20, 30, 40, 0, 0],
        }
    )
    out = demand.add_own_lags(df, "departures").sort("interval_start")
    # own_lag_1h at t=4 = mean of t=0..3 = mean(10,20,30,40) = 25.
    assert out["own_lag_1h"].to_list()[4] == pytest.approx(25.0)
    # t=0 has no prior interval -> null.
    assert out["own_lag_1h"].to_list()[0] is None


def test_add_own_lags_week_lag_is_exact_shift():
    n = demand.OWN_LAG_WEEK_INTERVALS + 2
    df = pl.DataFrame(
        {
            "station_id": ["A"] * n,
            "interval_start": list(range(n)),
            "departures": list(range(n)),
        }
    )
    out = demand.add_own_lags(df, "departures").sort("interval_start")
    week = demand.OWN_LAG_WEEK_INTERVALS
    assert out["own_lag_1week"].to_list()[week] == 0
    assert out["own_lag_1week"].to_list()[week + 1] == 1


def test_add_own_lags_does_not_bridge_across_stations():
    df = pl.DataFrame(
        {
            "station_id": ["A", "A", "B", "B"],
            "interval_start": [0, 1, 0, 1],
            "departures": [100, 200, 5, 6],
        }
    )
    out = demand.add_own_lags(df, "departures").sort("station_id", "interval_start")
    # Station B's lag must not see station A's trailing values.
    b = out.filter(pl.col("station_id") == "B")
    assert b["own_lag_1h"].to_list()[0] is None


def test_add_neighbor_demand_sums_other_stations_same_zone_and_interval():
    df = pl.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "zone_agg": ["z1", "z1", "z2"],
            "interval_start": [0, 0, 0],
            "departures": [3, 4, 100],
        }
    )
    out = demand.add_neighbor_demand(df, "departures", "zone_agg").sort("station_id")
    # A's neighbor demand = zone z1 total (7) minus its own (3) = 4 (i.e. B's value).
    assert out.filter(pl.col("station_id") == "A")["neighbor_demand"].item() == 4
    assert out.filter(pl.col("station_id") == "B")["neighbor_demand"].item() == 3
    # C is alone in its zone -> neighbor demand is 0, not null.
    assert out.filter(pl.col("station_id") == "C")["neighbor_demand"].item() == 0


def test_calibrate_bucket_bias_drops_low_n_buckets():
    recovery = pl.DataFrame(
        {
            "stratum": ["overall", "run_length_bucket", "run_length_bucket"],
            "run_length_bucket": [None, 0, 1],
            "bias": [1.0, -2.5, 9.0],
            "mae": [1.0, 2.5, 9.0],
            "n": [100, 50, 5],  # bucket 1 has too few synthetic runs
        }
    )
    out = demand.calibrate_bucket_bias(recovery, min_n=20)
    assert out == {0: -2.5}


class _FakeGBT:
    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def _minimal_direction_df(spec_prefix: str) -> pl.DataFrame:
    n = 3
    base = {
        "station_id": ["A", "B", "C"],
        "interval_start": [0, 1, 2],
        "hour_of_week": [0, 1, 2],
        "month": [1, 1, 1],
        "temp_c": [10.0, 10.0, 10.0],
        "precip_mm": [0.0, 0.0, 0.0],
        "precip_lag1h": [0.0, 0.0, 0.0],
        "precip_lag2h": [0.0, 0.0, 0.0],
        "wind_kph": [5.0, 5.0, 5.0],
        "humidity_pct": [50, 50, 50],
        "capacity": [20, 20, 20],
        "zone_agg": ["z1", "z1", "z1"],
        "is_holiday": [False, False, False],
        "departures": [5, 20, 20],
        "arrivals": [5, 20, 20],
        "is_bike_empty": [True, False, True],
        "is_dock_full": [True, False, True],
        "bike_empty_run_len": [1, 0, 6],
        "dock_full_run_len": [1, 0, 6],
        f"{spec_prefix}_own_lag_1h": [1.0, 1.0, 1.0],
        f"{spec_prefix}_own_lag_1week": [1.0, 1.0, 1.0],
        f"{spec_prefix}_neighbor_demand": [1.0, 1.0, 1.0],
    }
    return pl.DataFrame(base)


def _fake_fitted(bucket_bias: dict) -> demand.FittedDirection:
    # Encoding tables/global means are irrelevant to _FakeGBT (ignores X
    # content and returns a constant), but must be well-formed so
    # _with_model_columns' join + fillna machinery runs without error.
    return demand.FittedDirection(
        spec=demand.DEPARTURES,
        gbt=_FakeGBT(20.0),
        glm=None,
        station_enc=pl.DataFrame({"station_id": ["A", "B", "C"], "station_id_target_enc": [0.0, 0.0, 0.0]}),
        station_global_mean=0.0,
        zone_enc=pl.DataFrame({"zone_agg": ["z1"], "zone_agg_target_enc": [0.0]}),
        zone_global_mean=0.0,
        bucket_bias=bucket_bias,
    )


def test_impute_unmet_demand_applies_bucket_bias_correction_and_zeros_uncensored():
    df = _minimal_direction_df("dep")
    fitted = _fake_fitted({0: -3.0, 2: 5.0})
    params = CensoringParams(run_length_bucket_edges=[1, 4])
    out = demand.impute_unmet_demand(fitted, df, params).sort("interval_start")

    # Row 0: run_len=1 -> bucket 0, raw_unmet=max(0,20-5)=15, corrected=15-(-3)=18, censored=True.
    # Row 1: uncensored -> forced to 0 regardless of raw prediction.
    # Row 2: run_len=6 -> bucket 2, raw_unmet=max(0,20-20)=0, corrected=0-5=-5 -> clipped to 0.
    assert out["unmet_demand"].to_list() == pytest.approx([18.0, 0.0, 0.0])


def test_impute_unmet_demand_with_no_bias_correction_is_plain_max_zero():
    df = _minimal_direction_df("dep")
    fitted = _fake_fitted({})
    params = CensoringParams(run_length_bucket_edges=[1, 4])
    out = demand.impute_unmet_demand(fitted, df, params).sort("interval_start")
    assert out["unmet_demand"].to_list() == pytest.approx([15.0, 0.0, 0.0])


def test_bucket_bias_expr_defaults_to_zero_for_unlisted_buckets():
    df = pl.DataFrame({"run_length_bucket": [0, 1, 2]})
    expr = demand._bucket_bias_expr({0: 5.0})
    out = df.with_columns(expr.alias("bias"))
    assert out["bias"].to_list() == [5.0, 0.0, 0.0]


def _synthetic_full_panel(n_stations: int = 4, n_intervals: int = 200):
    """A small but schema-complete synthetic panel/inventory/weather so
    build_unmet_demand_table can run the REAL fit/calibrate/impute pipeline
    end to end, not a stubbed model -- fitting sklearn models on ~800 rows
    is near-instant, so this stays a fast unit test while still exercising
    the actual code path (per CLAUDE.md: every model function gets a test
    with a known-answer synthetic fixture)."""
    import random

    rng = random.Random(0)
    start = dt.datetime(2025, 1, 6, 0, 0)  # a Monday
    times = [start + dt.timedelta(minutes=15 * i) for i in range(n_intervals)]

    panel_rows = []
    inv_rows = []
    for s in range(n_stations):
        station_id = f"S{s}"
        zone = f"z{s % 2}"
        capacity = 20 + s
        for i, t in enumerate(times):
            panel_rows.append(
                {
                    "station_id": station_id,
                    "interval_start": t,
                    "departures": rng.randint(0, 5),
                    "arrivals": rng.randint(0, 5),
                    "capacity": capacity,
                    "zone_h3": f"h3_{zone}",
                    "zone_agg": zone,
                    "hour": t.hour,
                    "dow": t.weekday(),
                    "month": t.month,
                    "is_holiday": False,
                    "hour_of_week": t.weekday() * 24 + t.hour,
                    "temp_c": 10.0,
                    "precip_mm": 0.0,
                    "wind_kph": 5.0,
                    "humidity_pct": 50,
                }
            )
            # S0 gets one 6-interval bike-empty run; S1 gets one 4-interval dock-full run.
            inv_rows.append(
                {
                    "station_id": station_id,
                    "interval_start": t,
                    "is_bike_empty": station_id == "S0" and 90 <= i <= 95,
                    "is_dock_full": station_id == "S1" and 100 <= i <= 103,
                }
            )

    panel = pl.DataFrame(panel_rows)
    inventory = pl.DataFrame(inv_rows)
    weather = pl.DataFrame(
        {
            "timestamp": [start + dt.timedelta(hours=h) for h in range(60)],
            "temp_c": [10.0] * 60,
            "precip_mm": [0.0] * 60,
            "wind_kph": [5.0] * 60,
            "humidity_pct": [50] * 60,
        }
    )
    return panel, inventory, weather


def test_build_unmet_demand_table_end_to_end_schema_and_invariants():
    panel, inventory, weather = _synthetic_full_panel()
    df = demand.build_features(panel, inventory, weather)
    params = CensoringParams(run_length_bucket_edges=[1, 4])
    stale = pl.DataFrame(
        {"station_id": ["S0", "S1", "S2", "S3"], "is_capacity_stale": [False, False, False, False]}
    )

    unmet = demand.build_unmet_demand_table(df, params, stale)

    for col in demand.SHARED_OUTPUT_COLS + ["dep_D_hat", "dep_gross_unmet", "arr_D_hat", "arr_gross_unmet"]:
        assert col in unmet.columns, f"missing column {col}"
    assert unmet.height == df.height

    # gross_unmet must be exactly 0 wherever the corresponding side wasn't censored.
    assert (unmet.filter(~pl.col("is_bike_empty"))["dep_gross_unmet"] == 0).all()
    assert (unmet.filter(~pl.col("is_dock_full"))["arr_gross_unmet"] == 0).all()
    # never negative.
    assert (unmet["dep_gross_unmet"] >= 0).all()
    assert (unmet["arr_gross_unmet"] >= 0).all()
    # departures and arrivals gross_unmet are never conflated into one column.
    assert "gross_unmet" not in unmet.columns
