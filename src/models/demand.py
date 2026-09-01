"""Latent demand model D(s,t), per SPEC.md §3 and the RUNBOOK/DECISIONS Phase
5 design-fork discussion. Y = min(D, availability) is observed; this module
fits D on strictly-uncensored intervals and imputes unmet demand into
censored ones.

Two independent models are fit -- departures (censored by is_bike_empty) and
arrivals (censored by is_dock_full) -- because they're different problems
with different reliability (see the capacity-staleness discussion below and
in censoring.py).

Estimator: sklearn HistGradientBoostingRegressor(loss="poisson") as the
primary model, sklearn PoissonRegressor as an interpretable linear baseline.
Both ship with the project's existing scikit-learn==1.6.1 -- no new
dependency, despite RUNBOOK's aspirational setup list naming LightGBM/
statsmodels (which were never actually added to requirements.txt).

station_id (2,428 levels) and zone_agg (544 levels) are both fed in as
TARGET-ENCODED numeric features (per-group mean of the training target),
not as sklearn's native categorical_features="from_dtype" support -- tried
first, but HistGradientBoostingRegressor hard-caps categorical cardinality
at 255 (`ValueError: ... cardinality <= 255`, checked directly against
sklearn 1.6.1, not assumed from docs), and both exceed that.

SPEC.md §3 step 3's imputation formula scales predicted demand by
stockout_minutes / interval_minutes. That ratio isn't available --
inventory.parquet's minutes_empty/minutes_full are a binary 0-or-15
re-encoding of is_bike_empty/is_dock_full (see inventory.py's "Coarse
per-bucket proxy" comment), not a continuous within-interval measurement.
This module substitutes a run-length-bucket bias correction instead,
calibrated empirically via the artificial-censoring recovery test
(censoring.py) rather than an assumed fractional scaling: own-station lag
features are themselves built from observed Y, so during a multi-interval
drought they systematically understate recent demand, biasing D_hat
downward exactly when the drought (run length) is longest. The recovery
test measures that bias per bucket on synthetically-censored, KNOWN-true-D
data, and impute_unmet_demand() applies the learned correction to real
censored intervals.

---

MEMORY-SAFETY RESTRUCTURE (see ~/.claude/plans/lucky-
leaping-zebra.md): a first attempt to run this on the real ~80M-row panel
thrashed for 8+ hours on a 16GB machine and was killed before writing any
output -- the whole-panel-in-memory, four-full-pandas-copies design below
this note describes what changed and why.

Only two things need a station's FULL chronological history: own_lag_1h/
own_lag_1week (add_own_lags) and run-length (censoring.add_run_length).
Both are computed ONCE, globally, on narrow columns only (Stage 0:
build_lag_features_artifact), producing a small parquet artifact that every
other stage joins against. Nothing else in the feature set has cross-
interval lookback, so everything else is chunked by real calendar month
(month_key_expr -- never the raw Int8 `month` column, which collides
Dec-2024 with Dec-2025 in the real panel).

Model FIT is global (one FittedDirection per direction, frozen and pickled)
because a single consistent D_hat scale and one train/holdout split
(RUNBOOK's "last 3 months by time") require it -- refitting per month would
silently produce a different bucket-bias correction every month. Target
encoding is computed from the FULL uncensored train population (cheap,
streamed month-by-month as running sums); the GBT/GLM fit itself runs on a
bounded stratified sample (config/params.yaml's fit_sample_size) so the
pandas conversion + sklearn .fit() memory is bounded regardless of how many
months of training data exist.

IMPUTE runs per month against the frozen FittedDirection, checkpointed to
data/processed/unmet_demand/{YYYY-MM}.parquet so a kill costs one month, not
a re-run of everything.
"""
from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

import models.censoring as censoring
import utils.checkpoint as checkpoint
import utils.progress as progress
from models.inventory import prepare_panel

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = REPO_ROOT / "data" / "processed" / "panel.parquet"
INVENTORY_PATH = REPO_ROOT / "data" / "processed" / "inventory.parquet"
WEATHER_PATH = REPO_ROOT / "data" / "interim" / "weather.parquet"
PARAMS_PATH = REPO_ROOT / "config" / "params.yaml"

# Legacy single-file path (small-fixture / manual full-panel path only --
# see build_unmet_demand_table). The production chunked path writes/reads
# UNMET_DEMAND_DIR instead; consolidate_monthly() can materialize this path
# on demand for convenience, but nothing downstream requires it to exist.
UNMET_DEMAND_PATH = REPO_ROOT / "data" / "processed" / "unmet_demand.parquet"
UNMET_DEMAND_DIR = REPO_ROOT / "data" / "processed" / "unmet_demand"

LAG_FEATURES_PATH = REPO_ROOT / "data" / "processed" / "demand_lag_features.parquet"
DEMAND_MODEL_DIR = REPO_ROOT / "data" / "processed" / "demand_models"

# Columns carried straight through from the feature table into the
# persisted unmet-demand artifact -- everything src/models/substitution.py
# and src/viz/heatmap.py need that isn't direction-specific model output.
SHARED_OUTPUT_COLS = [
    "station_id",
    "interval_start",
    "zone_h3",
    "zone_agg",
    "capacity",
    "hour",
    "dow",
    "hour_of_week",
    "departures",
    "is_bike_empty",
    "bike_empty_run_len",
    "arrivals",
    "is_dock_full",
    "dock_full_run_len",
]

OWN_LAG_HOUR_INTERVALS = 4  # 1h = 4 x 15-min intervals
OWN_LAG_WEEK_INTERVALS = 7 * 24 * 4  # same hour, 7 days back

STATION_ENC_COL = "station_id_target_enc"
ZONE_ENC_COL = "zone_agg_target_enc"

# Same feature set for both models -- target encoding (see module docstring)
# means there's no high-cardinality categorical left to handle differently
# between the tree model and the linear baseline.
FEATURES = [
    "hour_of_week",
    "month",
    "temp_c",
    "precip_mm",
    "precip_lag1h",
    "precip_lag2h",
    "wind_kph",
    "humidity_pct",
    "capacity",
    "own_lag_1h",
    "own_lag_1week",
    "neighbor_demand",
    STATION_ENC_COL,
    ZONE_ENC_COL,
    "is_holiday_num",
]


@dataclass
class DirectionSpec:
    name: str
    target_col: str
    censor_col: str
    run_len_col: str


DEPARTURES = DirectionSpec("departures", "departures", "is_bike_empty", "bike_empty_run_len")
ARRIVALS = DirectionSpec("arrivals", "arrivals", "is_dock_full", "dock_full_run_len")

# Raw (pre-encoding) feature-source columns needed to reconstruct FEATURES
# for a row given only (station_id, interval_start) -- used to join model
# inputs back onto synthetic-censoring rows, which otherwise only carry
# station_id/interval_start/true_D/censored_Y/synthetic_run_length.
RAW_SOURCE_COLS = [
    "zone_agg",
    "is_holiday",
    "hour_of_week",
    "month",
    "temp_c",
    "precip_mm",
    "precip_lag1h",
    "precip_lag2h",
    "wind_kph",
    "humidity_pct",
    "capacity",
    "own_lag_1h",
    "own_lag_1week",
    "neighbor_demand",
]


@dataclass
class FittedDirection:
    spec: DirectionSpec
    gbt: HistGradientBoostingRegressor
    glm: PoissonRegressor
    station_enc: pl.DataFrame  # station_id, station_id_target_enc
    station_global_mean: float
    zone_enc: pl.DataFrame  # zone_agg, zone_agg_target_enc
    zone_global_mean: float
    glm_scaler: StandardScaler = None
    bucket_bias: dict[int, float] = field(default_factory=dict)


@dataclass
class PipelineParams:
    """Memory-safety restructure knobs -- see config/params.yaml's
    demand_model section for the full rationale on each value."""

    holdout_start_date: str
    fit_sample_size: int
    fit_sample_seed: int


def load_pipeline_params(path: Path = PARAMS_PATH) -> PipelineParams:
    cfg = yaml.safe_load(path.read_text())["demand_model"]
    return PipelineParams(
        holdout_start_date=cfg["holdout_start_date"],
        fit_sample_size=int(cfg["fit_sample_size"]),
        fit_sample_seed=int(cfg["fit_sample_seed"]),
    )


# ---------------------------------------------------------------------------
# Feature assembly (small-fixture / manual full-panel path)
# ---------------------------------------------------------------------------


def compute_weather_lag_table(weather: pl.DataFrame) -> pl.DataFrame:
    """1h/2h precip lags -- SPEC.md §3 names lagged precip explicitly as the
    largest demand shifter. weather.parquet is tiny (~8.7K hourly rows), so
    this global computation is cheap and done ONCE; both the small-fixture
    path (add_weather_lags) and the per-month chunked path
    (build_month_features) reuse its output via _join_weather_lag_table
    instead of recomputing the same shift on every chunk."""
    return (
        weather.sort("timestamp")
        .with_columns(
            pl.col("precip_mm").shift(1).alias("precip_lag1h"),
            pl.col("precip_mm").shift(2).alias("precip_lag2h"),
        )
        .select("timestamp", "precip_lag1h", "precip_lag2h")
        .rename({"timestamp": "weather_hour"})
    )


def _join_weather_lag_table(df: pl.DataFrame, weather_lag_table: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("interval_start").dt.truncate("1h").alias("weather_hour"))
        .join(weather_lag_table, on="weather_hour", how="left")
        .drop("weather_hour")
    )


def add_weather_lags(df: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """Same public contract as before the restructure -- joins 1h/2h precip
    lags onto df from raw weather. Internally a thin wrapper around
    compute_weather_lag_table + _join_weather_lag_table."""
    return _join_weather_lag_table(df, compute_weather_lag_table(weather))


def add_own_lags(df: pl.DataFrame, value_col: str) -> pl.DataFrame:
    """own_lag_1h: rolling mean of the PRIOR 4 intervals (t-4..t-1), not
    including t itself. own_lag_1week: same-hour value exactly 7 days back.
    Both use observed Y, not de-censored demand -- legitimate autoregressive
    inputs, but during a persistent stockout these will understate true
    recent demand; that's the exact feedback the run-length-bucket bias
    correction (see module docstring) exists to counteract downstream, not
    something fixed here.

    Needs a station's FULL chronological history to be correct (the .over()
    shift/rolling ops), so it's only ever called globally (Stage 0's
    build_lag_features_artifact), never on a month-chunked frame."""
    df = df.sort("station_id", "interval_start")
    df = df.with_columns(pl.col(value_col).shift(1).over("station_id").alias("_shifted"))
    df = df.with_columns(
        pl.col("_shifted").rolling_mean(window_size=OWN_LAG_HOUR_INTERVALS, min_samples=1).over("station_id").alias("own_lag_1h")
    ).drop("_shifted")
    return df.with_columns(
        pl.col(value_col).shift(OWN_LAG_WEEK_INTERVALS).over("station_id").alias("own_lag_1week")
    )


def add_neighbor_demand(df: pl.DataFrame, value_col: str, zone_col: str = "zone_agg") -> pl.DataFrame:
    """Sum of the SAME value_col across other stations in the same zone at
    the same interval -- proxies substitution pressure / local demand
    shocks, per SPEC.md §3's "demand at neighbors" feature. Same-instant
    only, no cross-interval lookback -- safe to compute per month chunk.
    Stations with null zone (158 GBFS-unmatched stations, see panel.py) get
    null neighbor_demand; left as-is rather than imputed, since GBT handles
    missing values natively and a fabricated 0 would misleadingly claim
    "no neighbor demand" for exactly the stations with no known neighbors."""
    zone_totals = df.group_by([zone_col, "interval_start"]).agg(pl.col(value_col).sum().alias("_zone_total"))
    df = df.join(zone_totals, on=[zone_col, "interval_start"], how="left")
    return df.with_columns((pl.col("_zone_total") - pl.col(value_col)).alias("neighbor_demand")).drop("_zone_total")


def build_features(panel: pl.DataFrame, inventory: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """Assembles the full Phase 5 feature table in one shot. Small-fixture /
    manual full-panel path ONLY -- calls add_own_lags/censoring.add_run_length
    on the whole frame passed in, which is exactly the whole-panel-at-once
    pattern the memory-safety restructure moves off the production path (see
    module docstring). tests/test_demand.py exercises this directly on a
    small synthetic panel; do not call this with the real ~80M-row panel."""
    df = panel.join(
        inventory.select("station_id", "interval_start", "is_bike_empty", "is_dock_full"),
        on=["station_id", "interval_start"],
        how="inner",
    )
    df = add_weather_lags(df, weather)
    df = add_own_lags(df, "departures")
    df = df.rename({"own_lag_1h": "dep_own_lag_1h", "own_lag_1week": "dep_own_lag_1week"})
    df = add_own_lags(df, "arrivals")
    df = df.rename({"own_lag_1h": "arr_own_lag_1h", "own_lag_1week": "arr_own_lag_1week"})
    df = add_neighbor_demand(df, "departures", "zone_agg")
    df = df.rename({"neighbor_demand": "dep_neighbor_demand"})
    df = add_neighbor_demand(df, "arrivals", "zone_agg")
    df = df.rename({"neighbor_demand": "arr_neighbor_demand"})
    df = censoring.add_run_length(df, "is_bike_empty", "bike_empty_run_len")
    df = censoring.add_run_length(df, "is_dock_full", "dock_full_run_len")
    return df


def _direction_columns(df: pl.DataFrame, spec: DirectionSpec) -> pl.DataFrame:
    """Renames the direction-specific own_lag/neighbor_demand columns built
    by build_features()/build_month_features() to the generic names
    FEATURES expects, so one fit/predict/impute codepath serves both
    directions."""
    prefix = "dep" if spec.name == "departures" else "arr"
    return df.with_columns(
        pl.col(f"{prefix}_own_lag_1h").alias("own_lag_1h"),
        pl.col(f"{prefix}_own_lag_1week").alias("own_lag_1week"),
        pl.col(f"{prefix}_neighbor_demand").alias("neighbor_demand"),
    )


# ---------------------------------------------------------------------------
# Model frames
# ---------------------------------------------------------------------------


def _pl_to_pd(df: pl.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Column-by-column via Series.to_numpy() rather than DataFrame.to_pandas()
    -- pyarrow isn't installed (requirements.txt doesn't have it, matching
    the project's pattern of not adding deps beyond what's actually used),
    and to_pandas() hard-requires it. to_numpy() doesn't."""
    return pd.DataFrame({c: df[c].to_numpy() for c in cols})


def compute_target_encoding(df: pl.DataFrame, target_col: str, group_col: str) -> tuple[pl.DataFrame, float]:
    """Per-group mean of target_col, computed on df -- callers must pass the
    uncensored TRAINING subset only, never censored/held-out rows, or the
    encoding leaks information the model shouldn't have at fit time.
    Returns (table with columns [group_col, f'{group_col}_target_enc'],
    global mean, used to fill groups unseen in training). Small-fixture path
    only -- the production path streams this incrementally per month via
    compute_encodings_and_sample/_finalize_encoding instead, so it never
    needs the full uncensored population resident in memory at once."""
    global_mean = float(df[target_col].mean())
    enc = df.group_by(group_col).agg(pl.col(target_col).mean().alias(f"{group_col}_target_enc"))
    return enc, global_mean


def apply_target_encoding(df: pl.DataFrame, enc: pl.DataFrame, global_mean: float, group_col: str) -> pl.DataFrame:
    col = f"{group_col}_target_enc"
    if col in df.columns:
        df = df.drop(col)
    df = df.join(enc, on=group_col, how="left")
    return df.with_columns(pl.col(col).fill_null(global_mean))


def _with_model_columns(fitted: FittedDirection, df: pl.DataFrame) -> pl.DataFrame:
    """Applies this direction's fitted target encodings and the
    is_holiday->numeric cast, so to_gbt_frame/to_glm_frame can just select
    FEATURES. Must be called on any df before those two functions."""
    df = apply_target_encoding(df, fitted.station_enc, fitted.station_global_mean, "station_id")
    df = apply_target_encoding(df, fitted.zone_enc, fitted.zone_global_mean, "zone_agg")
    return df.with_columns(pl.col("is_holiday").cast(pl.Int8).alias("is_holiday_num"))


def to_gbt_frame(df: pl.DataFrame) -> pd.DataFrame:
    return _pl_to_pd(df, FEATURES)


def to_glm_frame(df: pl.DataFrame) -> pd.DataFrame:
    """PoissonRegressor can't accept missing values (GBT handles them
    natively and doesn't need this) -- numeric NaNs filled with 0 as a
    documented simplification for a baseline/sanity-check model, not the
    primary estimator."""
    return _pl_to_pd(df, FEATURES).fillna(0.0)


def _scale_glm_frame(fitted: FittedDirection, pdf: pd.DataFrame, refit_scaler: bool = False) -> np.ndarray:
    """PoissonRegressor's IRLS solver is scale-sensitive -- FEATURES mixes
    wildly different ranges (capacity ~0-100, hour_of_week 0-167,
    precip_mm ~0-50), which produced divide-by-zero/overflow warnings and
    an unreliable fit when benchmarked unscaled. GBT is split-based and
    doesn't need this; only the GLM baseline does. The scaler is fit once
    on training data and reused at predict time via fitted.glm_scaler."""
    if refit_scaler:
        fitted.glm_scaler = StandardScaler().fit(pdf)
    return fitted.glm_scaler.transform(pdf)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def _fit_models(fitted: FittedDirection, train: pl.DataFrame) -> FittedDirection:
    """Shared GBT+GLM fit body. `train` must already be uncensored, direction-
    columned (_direction_columns), and target-encoded (_with_model_columns)
    -- both fit_direction (small-fixture path, encodes from its own input)
    and fit_direction_bounded (production path, encodings computed
    separately on the full population) prepare `train` this way before
    calling this."""
    y = train[fitted.spec.target_col].to_numpy().astype(float)

    X_gbt = to_gbt_frame(train)
    fitted.gbt = HistGradientBoostingRegressor(loss="poisson", random_state=0, max_iter=200).fit(X_gbt, y)

    X_glm = _scale_glm_frame(fitted, to_glm_frame(train), refit_scaler=True)
    fitted.glm = PoissonRegressor(max_iter=1000).fit(X_glm, y)

    return fitted


def fit_direction(df: pl.DataFrame, spec: DirectionSpec) -> FittedDirection:
    """Fits on strictly-uncensored intervals only (censor_col == False), per
    SPEC.md §3 step 2 -- no soft weighting of censored rows into training;
    the coarse binary flag gives no reliable basis to partially trust a
    censored Y as ground truth (see censoring.py module docstring).

    Small-fixture / manual full-panel path -- encodes AND fits from
    whatever `df` it's given. The production path (get_or_fit_direction)
    uses fit_direction_bounded instead, with encodings computed separately
    on the full population and fitting done on a bounded sample; see module
    docstring."""
    train = _direction_columns(df, spec).filter(~pl.col(spec.censor_col))

    station_enc, station_global_mean = compute_target_encoding(train, spec.target_col, "station_id")
    zone_enc, zone_global_mean = compute_target_encoding(train, spec.target_col, "zone_agg")
    fitted = FittedDirection(
        spec=spec,
        gbt=None,
        glm=None,
        station_enc=station_enc,
        station_global_mean=station_global_mean,
        zone_enc=zone_enc,
        zone_global_mean=zone_global_mean,
    )
    train = _with_model_columns(fitted, train)
    return _fit_models(fitted, train)


def predict_gbt(fitted: FittedDirection, df: pl.DataFrame) -> np.ndarray:
    df = _with_model_columns(fitted, _direction_columns(df, fitted.spec))
    return fitted.gbt.predict(to_gbt_frame(df))


def predict_glm(fitted: FittedDirection, df: pl.DataFrame) -> np.ndarray:
    df = _with_model_columns(fitted, _direction_columns(df, fitted.spec))
    X_glm = _scale_glm_frame(fitted, to_glm_frame(df), refit_scaler=False)
    return fitted.glm.predict(X_glm)


# ---------------------------------------------------------------------------
# Calibration (artificial-censoring recovery test -> bucket bias correction)
# ---------------------------------------------------------------------------


def calibrate_direction(
    fitted: FittedDirection,
    df: pl.DataFrame,
    params: censoring.CensoringParams,
    run_lengths: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12),
    n_runs_per_length: int = 300,
    stale_flags: pl.DataFrame | None = None,
    rng_seed: int = 0,
) -> tuple[FittedDirection, pl.DataFrame]:
    """SPEC.md §3's artificial-censoring recovery test, extended: censors
    contiguous runs (not scattered points, since run length is inherently
    sequential) of known-uncensored data, predicts D_hat from the model on
    the row's OTHER (unmodified) features, computes recovery error against
    the known true gap, and calibrates a per-run-length-bucket bias
    correction from the result. stale_flags (station_id, is_capacity_stale)
    should only be passed for the arrival-side direction -- capacity
    staleness doesn't implicate departure-side censoring (see
    censoring.py's module docstring).

    Production path passes df restricted to the holdout period
    (PipelineParams.holdout_start_date onward) -- real data the GBT never
    trained on, not the training sample. This function itself doesn't know
    or care which period df covers; that restriction is the caller's job
    (get_or_fit_direction)."""
    df = _direction_columns(df, fitted.spec)
    uncensored = df.filter(~pl.col(fitted.spec.censor_col))

    synthetic = censoring.make_synthetic_censoring_runs(
        uncensored,
        value_col=fitted.spec.target_col,
        run_lengths=list(run_lengths),
        n_runs_per_length=n_runs_per_length,
        rng_seed=rng_seed,
    )

    joined = synthetic.join(
        df.select(["station_id", "interval_start"] + RAW_SOURCE_COLS),
        on=["station_id", "interval_start"],
        how="left",
    )
    joined = _with_model_columns(fitted, joined)
    d_hat = fitted.gbt.predict(to_gbt_frame(joined))
    raw_unmet = np.maximum(0.0, d_hat - joined["censored_Y"].to_numpy())

    recovery = censoring.evaluate_recovery(
        synthetic, pl.Series(raw_unmet), edges=params.run_length_bucket_edges, stale_flags=stale_flags
    )
    bucket_bias = calibrate_bucket_bias(recovery)
    fitted.bucket_bias = bucket_bias
    return fitted, recovery


def calibrate_direction_matched(
    fitted: FittedDirection,
    df: pl.DataFrame,
    params: censoring.CensoringParams,
    run_lengths: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12),
    n_runs_per_length: int = 300,
    stale_flags: pl.DataFrame | None = None,
    rng_seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Re-runs the recovery test with the synthetic sample stratum-matched to
    the GENUINELY-censored population on predicted-demand decile,
    hour-of-week, and capacity band (censoring.py's
    make_synthetic_censoring_runs_matched), instead of drawn uniformly from
    uncensored intervals.

    Why this exists as a separate function rather than a flag on
    calibrate_direction: real censoring is not a random sample of
    conditions -- stations run empty because demand outran supply, so real
    stockouts concentrate in the upper predicted-demand tail. Unmatched
    sampling draws mostly from typical, mid-demand uncensored moments (that's
    what most uncensored intervals look like), so a good unmatched-recovery
    score shows the model interpolates well but is silent on whether it
    extrapolates well into the demand regime where real censoring actually
    happens. This function is the honest version of that question.

    Does NOT touch fitted.bucket_bias -- matched-sample cell sizes are
    smaller and uneven (some target strata may have thin or zero pool
    support, by design; see censoring.py's coverage diagnostic), so this is
    for reporting/comparison, not recalibration. Returns (recovery,
    coverage); coverage is censoring.py's per-(run_length, stratum) fill
    report -- pass it through summarize_coverage_by_decile() for the
    headline "how much of the top demand deciles could even be matched"
    number."""
    df = _direction_columns(df, fitted.spec)
    uncensored = df.filter(~pl.col(fitted.spec.censor_col))
    censored = df.filter(pl.col(fitted.spec.censor_col))

    uncensored = uncensored.with_columns(pl.Series("D_hat", predict_gbt(fitted, uncensored)))
    censored = censored.with_columns(pl.Series("D_hat", predict_gbt(fitted, censored)))

    synthetic, coverage = censoring.make_synthetic_censoring_runs_matched(
        uncensored,
        censored,
        value_col=fitted.spec.target_col,
        d_hat_col="D_hat",
        run_lengths=list(run_lengths),
        n_runs_per_length=n_runs_per_length,
        rng_seed=rng_seed,
    )

    if synthetic.height == 0:
        return pl.DataFrame(), coverage

    joined = synthetic.join(
        df.select(["station_id", "interval_start"] + RAW_SOURCE_COLS),
        on=["station_id", "interval_start"],
        how="left",
    )
    joined = _with_model_columns(fitted, joined)
    d_hat = fitted.gbt.predict(to_gbt_frame(joined))
    raw_unmet = np.maximum(0.0, d_hat - joined["censored_Y"].to_numpy())

    recovery = censoring.evaluate_recovery(
        synthetic, pl.Series(raw_unmet), edges=params.run_length_bucket_edges, stale_flags=stale_flags
    )
    return recovery, coverage


def calibrate_bucket_bias(recovery_result: pl.DataFrame, min_n: int = 20) -> dict[int, float]:
    """{bucket: bias} from evaluate_recovery's 'run_length_bucket' stratum.
    Buckets with fewer than min_n synthetic runs are dropped (no
    correction, bias=0) rather than trusting a noisy estimate off a small
    sample."""
    rows = recovery_result.filter(pl.col("stratum") == "run_length_bucket")
    return {
        int(r["run_length_bucket"]): float(r["bias"])
        for r in rows.iter_rows(named=True)
        if r["n"] >= min_n and r["run_length_bucket"] is not None
    }


def _bucket_bias_expr(bucket_bias: dict[int, float]) -> pl.Expr:
    expr = pl.lit(0.0)
    for bucket, bias in bucket_bias.items():
        expr = pl.when(pl.col("run_length_bucket") == bucket).then(pl.lit(bias)).otherwise(expr)
    return expr


def compute_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else float("nan")


def predict_vs_actual_holdout(fitted: FittedDirection, holdout_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, d_hat, interval_start) on held-out UNCENSORED intervals only --
    censored Y is a lower bound on true D, not usable as ground truth.
    Factored out of held_out_wmape so Phase 9's policy_compare.py
    (SPEC.md §8's bootstrap axis (a): demand-model-residual uncertainty)
    can reuse the exact same holdout-evaluation logic instead of
    re-deriving it against a private function in a different module.
    interval_start is included because that module aggregates to daily
    totals before computing a residual ratio (per-row ratios are
    dominated by zero-departure intervals at 15-min granularity -- see its
    docstring), which held_out_wmape itself doesn't need but also doesn't
    lose anything by carrying."""
    df = _direction_columns(holdout_df, fitted.spec).filter(~pl.col(fitted.spec.censor_col))
    d_hat = predict_gbt(fitted, df)
    y = df[fitted.spec.target_col].to_numpy().astype(float)
    interval_start = df["interval_start"].to_numpy()
    return y, d_hat, interval_start


def held_out_wmape(fitted: FittedDirection, holdout_df: pl.DataFrame, n_deciles: int = 10) -> pl.DataFrame:
    """WMAPE on held-out UNCENSORED intervals only (RUNBOOK.md Phase 5's
    deliverable: "WMAPE ... on held-out uncensored intervals, overall and by
    demand decile"). Returns one row per stratum ("overall" +
    "decile_0".."decile_{n_deciles-1}", by D_hat)."""
    y, d_hat, _interval_start = predict_vs_actual_holdout(fitted, holdout_df)

    overall = pl.DataFrame({"stratum": ["overall"], "wmape": [compute_wmape(y, d_hat)], "n": [len(y)]})

    decile_edges = np.quantile(d_hat, np.linspace(0, 1, n_deciles + 1)[1:-1])
    decile = np.searchsorted(decile_edges, d_hat, side="right")
    tmp = pl.DataFrame({"decile": decile, "abs_err": np.abs(y - d_hat), "abs_y": np.abs(y)})
    by_decile = (
        tmp.group_by("decile")
        .agg(
            (pl.col("abs_err").sum() / pl.col("abs_y").sum()).alias("wmape"),
            pl.len().alias("n"),
        )
        .sort("decile")
        .with_columns((pl.lit("decile_") + pl.col("decile").cast(pl.Utf8)).alias("stratum"))
        .select("stratum", "wmape", "n")
    )
    return pl.concat([overall, by_decile], how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# Imputation on real (not synthetic) censored intervals
# ---------------------------------------------------------------------------


def impute_unmet_demand(fitted: FittedDirection, df: pl.DataFrame, params: censoring.CensoringParams) -> pl.DataFrame:
    """unmet_demand = max(0, D_hat - Y) for every row, bias-corrected by the
    run-length bucket the row's REAL censored run falls into (using the
    calibration learned in calibrate_direction, not a fabricated
    stockout-minutes fraction -- see module docstring), then zeroed for
    uncensored rows (Y is already trusted to equal D there)."""
    dcols = _direction_columns(df, fitted.spec)
    d_hat = predict_gbt(fitted, df)
    y = dcols[fitted.spec.target_col].to_numpy().astype(float)
    raw_unmet = np.maximum(0.0, d_hat - y)

    out = dcols.with_columns(
        pl.Series("D_hat", d_hat),
        pl.Series("raw_unmet", raw_unmet),
        censoring.bucket_run_length(pl.col(fitted.spec.run_len_col), params.run_length_bucket_edges).alias(
            "run_length_bucket"
        ),
    )
    bias_expr = _bucket_bias_expr(fitted.bucket_bias)
    out = out.with_columns((pl.col("raw_unmet") - bias_expr).clip(lower_bound=0.0).alias("corrected_unmet"))
    return out.with_columns(
        pl.when(pl.col(fitted.spec.censor_col)).then(pl.col("corrected_unmet")).otherwise(0.0).alias("unmet_demand")
    )


def label_sampling(recovery: pl.DataFrame, label: str) -> pl.DataFrame:
    if recovery.height == 0:
        return recovery
    return recovery.with_columns(pl.lit(label).alias("sampling"))


# ---------------------------------------------------------------------------
# Small-fixture / manual full-panel driver (unchanged -- NOT the production
# path; see module docstring and get_or_fit_direction/run_month below)
# ---------------------------------------------------------------------------


def build_direction_unmet_table(
    df: pl.DataFrame,
    spec: DirectionSpec,
    params: censoring.CensoringParams,
    stale_flags: pl.DataFrame | None = None,
) -> tuple[FittedDirection, pl.DataFrame]:
    """Fits + unmatched-calibrates + imputes gross unmet demand for one
    direction, all from one `df` (SPEC.md §3 steps 2-3). Small-fixture path
    only (exercised by tests/test_demand.py) -- the production path never
    calls this, since fitting per chunk would produce inconsistent
    bucket-bias corrections across chunks (see module docstring)."""
    fitted = fit_direction(df, spec)
    fitted, _recovery = calibrate_direction(fitted, df, params, stale_flags=stale_flags)
    imputed = impute_unmet_demand(fitted, df, params)

    prefix = "dep" if spec.name == "departures" else "arr"
    table = imputed.select(
        "station_id",
        "interval_start",
        pl.col("D_hat").alias(f"{prefix}_D_hat"),
        pl.col("unmet_demand").alias(f"{prefix}_gross_unmet"),
    )
    return fitted, table


def build_unmet_demand_table(
    df: pl.DataFrame, params: censoring.CensoringParams, stale: pl.DataFrame
) -> pl.DataFrame:
    """One row per (station_id, interval_start) carrying both directions'
    gross unmet demand side by side, never summed (departures and arrivals
    are different quantities -- see module docstring and
    src/models/substitution.py). Small-fixture path only; production output
    is produced per month by impute_month/run_month instead."""
    _fitted_dep, dep_table = build_direction_unmet_table(df, DEPARTURES, params)
    _fitted_arr, arr_table = build_direction_unmet_table(df, ARRIVALS, params, stale_flags=stale)

    shared = df.select(SHARED_OUTPUT_COLS)
    return shared.join(dep_table, on=["station_id", "interval_start"]).join(
        arr_table, on=["station_id", "interval_start"]
    )


def run_recovery_validation(df: pl.DataFrame, params: censoring.CensoringParams, stale: pl.DataFrame) -> None:
    """The matched-vs-unmatched artificial-censoring recovery comparison.
    Manual validation exercise, not part of any automated pipeline. WARNING:
    like fit_direction/build_direction_unmet_table, this fits on whatever
    `df` it's given in one shot -- if ever invoked with the real full-year
    panel (not a small fixture or an already-bounded sample), it inherits
    the same OOM risk the memory-safety restructure moved off the
    production path. Point it at the same bounded sample production fitting
    uses (compute_encodings_and_sample's output), not a full-year frame."""
    for spec, stale_arg in ((DEPARTURES, None), (ARRIVALS, stale)):
        print(f"\n[demand] === {spec.name} ===")
        fitted = fit_direction(df, spec)
        fitted, recovery_unmatched = calibrate_direction(fitted, df, params, stale_flags=stale_arg)
        print(f"[demand] {spec.name} bucket bias correction (from UNMATCHED calibration): {fitted.bucket_bias}")

        # Matched sampling spreads its draws across up to (n_deciles x
        # n_capacity_bands x 168 hours) = ~6,720 strata, vs. the unmatched
        # draw's single flat pool -- at the unmatched default (300), most
        # per-cell desired counts round down to 0. Checked directly: at
        # 5,000 the per-cell rounding stops dominating (departures showed
        # fill_rate=1.0 in every decile on a 60-day subset) at essentially
        # the same wall-clock cost as 300, since runtime is dominated by
        # fixed per-stratum overhead, not draw size.
        recovery_matched, coverage = calibrate_direction_matched(
            fitted, df, params, n_runs_per_length=5000, stale_flags=stale_arg
        )

        print(f"\n[demand] {spec.name}: unmatched vs matched recovery, side by side")
        comparison = pl.concat(
            [label_sampling(recovery_unmatched, "unmatched"), label_sampling(recovery_matched, "matched")],
            how="diagonal_relaxed",
        )
        print(
            comparison.filter(pl.col("stratum").is_in(["overall", "run_length_bucket"])).sort(
                ["stratum", "run_length_bucket", "sampling"], nulls_last=True
            )
        )

        print(f"\n[demand] {spec.name}: matched-sampling coverage by predicted-demand decile")
        print(censoring.summarize_coverage_by_decile(coverage))

        imputed = impute_unmet_demand(fitted, df, params)
        total_unmet = imputed["unmet_demand"].sum()
        n_censored = imputed.filter(pl.col(spec.censor_col)).height
        print(f"\n[demand] {spec.name}: {n_censored:,} censored intervals, total imputed unmet demand = {total_unmet:,.0f}")


# ---------------------------------------------------------------------------
# Stage 0: global lag/run-length pre-pass (narrow columns only)
# ---------------------------------------------------------------------------


def build_lag_features_artifact(panel: pl.DataFrame, inventory: pl.DataFrame) -> pl.DataFrame:
    """The ONLY features needing a station's full chronological history --
    run once, globally, on narrow columns only (station_id, interval_start,
    departures, arrivals, is_bike_empty, is_dock_full), so month-chunked
    processing never needs a lookback buffer. Same join order and .over()
    semantics as build_features(), so the output is exactly what the
    unchunked pipeline would have produced for these columns, not an
    approximation -- just isolated from the wide feature/weather/encoding
    columns that made the full frame dangerous."""
    df = panel.select("station_id", "interval_start", "departures", "arrivals").join(
        inventory.select("station_id", "interval_start", "is_bike_empty", "is_dock_full"),
        on=["station_id", "interval_start"],
        how="inner",
    )
    df = add_own_lags(df, "departures").rename({"own_lag_1h": "dep_own_lag_1h", "own_lag_1week": "dep_own_lag_1week"})
    df = add_own_lags(df, "arrivals").rename({"own_lag_1h": "arr_own_lag_1h", "own_lag_1week": "arr_own_lag_1week"})
    df = censoring.add_run_length(df, "is_bike_empty", "bike_empty_run_len")
    df = censoring.add_run_length(df, "is_dock_full", "dock_full_run_len")
    return df.select(
        "station_id",
        "interval_start",
        "dep_own_lag_1h",
        "dep_own_lag_1week",
        "arr_own_lag_1h",
        "arr_own_lag_1week",
        "bike_empty_run_len",
        "dock_full_run_len",
    )


def ensure_lag_features_artifact(force: bool = False) -> pl.LazyFrame:
    """Loads LAG_FEATURES_PATH if it exists and not force; else builds it
    (one eager pass, narrow columns, ~76M rows x 8 cols) and checkpoints it
    via an atomic write. Returns a LazyFrame either way, so per-month
    callers can filter+collect just their slice."""
    if LAG_FEATURES_PATH.exists() and not force:
        print(f"[demand] Stage 0: reusing cached lag-features artifact at {LAG_FEATURES_PATH}")
        return pl.scan_parquet(LAG_FEATURES_PATH)

    t0 = time.monotonic()
    panel = pl.read_parquet(PANEL_PATH, columns=["station_id", "interval_start", "departures", "arrivals"])
    inventory = pl.read_parquet(
        INVENTORY_PATH, columns=["station_id", "interval_start", "is_bike_empty", "is_dock_full"]
    )
    artifact = build_lag_features_artifact(panel, inventory)
    del panel, inventory
    checkpoint.write_checkpoint(artifact, LAG_FEATURES_PATH)
    elapsed = time.monotonic() - t0
    print(
        f"[demand] Stage 0: built lag-features artifact, {artifact.height:,} rows, "
        f"{elapsed:.1f}s, peak RSS {progress.peak_rss_mb():.0f} MB -> {LAG_FEATURES_PATH}"
    )
    return pl.scan_parquet(LAG_FEATURES_PATH)


def compute_stale_flags() -> pl.DataFrame:
    """Capacity-staleness flag (censoring.compute_capacity_staleness) needs
    each station's single-interval peak throughput across the WHOLE year --
    like Stage 0's lag features, this can't be month-chunked. Narrow columns
    only (station_id, interval_start, capacity, departures, arrivals --
    prepare_panel needs interval_start too, for its week_start column and
    final sort), reusing inventory.prepare_panel's usable-population filter,
    the same population Stage 0 and build_month_features both restrict to
    via the inner join with inventory.parquet."""
    narrow = pl.read_parquet(
        PANEL_PATH, columns=["station_id", "interval_start", "capacity", "departures", "arrivals"]
    )
    usable = prepare_panel(narrow)
    return censoring.compute_capacity_staleness(usable)


# ---------------------------------------------------------------------------
# Per-month feature assembly (production path)
# ---------------------------------------------------------------------------


def all_month_keys() -> list[str]:
    """Every real (interval_start-derived) YYYY-MM key present in the panel
    -- never the raw Int8 `month` column, which collides Dec-2024 with
    Dec-2025 in the real data (confirmed: 569 vs 6.58M rows)."""
    keys = (
        pl.scan_parquet(PANEL_PATH)
        .select(checkpoint.month_key_expr().alias("month_key"))
        .unique()
        .sort("month_key")
        .collect()
    )
    return keys["month_key"].to_list()


def month_bounds(month_key: str) -> tuple[datetime, datetime]:
    start = datetime.strptime(month_key, "%Y-%m")
    end = datetime(start.year + 1, 1, 1) if start.month == 12 else datetime(start.year, start.month + 1, 1)
    return start, end


def build_month_features(month_key: str, weather_lag_table: pl.DataFrame, lag_features_lf: pl.LazyFrame) -> pl.DataFrame:
    """Memory-bounded, per-month replacement for build_features() on the
    production path: scans (not eager-reads) panel/inventory filtered to
    this month, joins the precomputed weather-lag table and Stage-0
    lag/run-length artifact (both filtered to this month too), and computes
    add_neighbor_demand fresh (safe -- same-instant only, no lookback).
    Output has the same columns build_features() would produce for these
    rows."""
    start, end = month_bounds(month_key)
    panel_m = pl.scan_parquet(PANEL_PATH).filter(pl.col("interval_start").is_between(start, end, closed="left"))
    inv_m = (
        pl.scan_parquet(INVENTORY_PATH)
        .filter(pl.col("interval_start").is_between(start, end, closed="left"))
        .select("station_id", "interval_start", "is_bike_empty", "is_dock_full")
    )
    df = panel_m.join(inv_m, on=["station_id", "interval_start"], how="inner").collect()

    df = _join_weather_lag_table(df, weather_lag_table)
    df = add_neighbor_demand(df, "departures", "zone_agg").rename({"neighbor_demand": "dep_neighbor_demand"})
    df = add_neighbor_demand(df, "arrivals", "zone_agg").rename({"neighbor_demand": "arr_neighbor_demand"})

    lag_m = lag_features_lf.filter(pl.col("interval_start").is_between(start, end, closed="left")).collect()
    df = df.join(lag_m, on=["station_id", "interval_start"], how="left")

    return df.with_columns(checkpoint.month_key_expr().alias("month_key"))


def build_range_features(month_keys: list[str], weather_lag_table: pl.DataFrame, lag_features_lf: pl.LazyFrame) -> pl.DataFrame:
    """Concatenates build_month_features() over a handful of months --
    used for the holdout period (typically 3 months, RUNBOOK's "last 3
    months by time"), which is small enough to hold as one wide frame at
    once. Do not call this with more than a few months; use the per-month
    loop (get_or_fit_direction / run_month) for anything month-count-scaled
    to a full year."""
    return pl.concat(
        [build_month_features(mk, weather_lag_table, lag_features_lf) for mk in month_keys], how="vertical"
    )


# ---------------------------------------------------------------------------
# Fit stage (production path): global, once per direction, bounded memory
# ---------------------------------------------------------------------------


def _finalize_encoding(parts: pl.DataFrame, group_col: str) -> tuple[pl.DataFrame, float]:
    """Combines per-month (sum, n) parts (from compute_encodings_and_sample)
    into the same [group_col, f'{group_col}_target_enc'] + global_mean
    contract compute_target_encoding returns, without ever holding the full
    uncensored population in memory at once."""
    agg = parts.group_by(group_col).agg(pl.col("_sum").sum().alias("_sum"), pl.col("_n").sum().alias("_n"))
    global_mean = float(agg["_sum"].sum() / agg["_n"].sum())
    enc = agg.with_columns((pl.col("_sum") / pl.col("_n")).alias(f"{group_col}_target_enc")).select(
        group_col, f"{group_col}_target_enc"
    )
    return enc, global_mean


def compute_encodings_and_sample(
    train_months: list[str],
    spec: DirectionSpec,
    weather_lag_table: pl.DataFrame,
    lag_features_lf: pl.LazyFrame,
    pipeline_params: PipelineParams,
) -> tuple[pl.DataFrame, float, pl.DataFrame, float, pl.DataFrame]:
    """One pass over the TRAIN months (never the holdout months), one month
    at a time: accumulates per-station/per-zone (sum, n) for target
    encoding (so the encoding reflects the FULL uncensored train
    population, not just the fit sample) and draws this month's share of
    the bounded fit sample (pipeline_params.fit_sample_size split evenly
    across train months). Never holds more than one month's wide feature
    frame in memory at a time. Returns (station_enc, station_global_mean,
    zone_enc, zone_global_mean, sample)."""
    per_month_target = max(1, pipeline_params.fit_sample_size // max(1, len(train_months)))
    station_parts, zone_parts, sample_parts = [], [], []

    for month_key in train_months:
        t0 = time.monotonic()
        month_df = build_month_features(month_key, weather_lag_table, lag_features_lf)
        month_df = _direction_columns(month_df, spec)
        uncensored = month_df.filter(~pl.col(spec.censor_col))

        station_parts.append(
            uncensored.group_by("station_id").agg(pl.col(spec.target_col).sum().alias("_sum"), pl.len().alias("_n"))
        )
        zone_parts.append(
            uncensored.group_by("zone_agg").agg(pl.col(spec.target_col).sum().alias("_sum"), pl.len().alias("_n"))
        )
        n = min(per_month_target, uncensored.height)
        sample_parts.append(uncensored.sample(n=n, seed=pipeline_params.fit_sample_seed))
        progress.log_month(month_key, uncensored.height, time.monotonic() - t0, extra=f"{spec.name} encode+sample pass")

    station_enc, station_gm = _finalize_encoding(pl.concat(station_parts), "station_id")
    zone_enc, zone_gm = _finalize_encoding(pl.concat(zone_parts), "zone_agg")
    sample = pl.concat(sample_parts, how="diagonal_relaxed")
    print(
        f"[demand] {spec.name}: fit sample {sample.height:,} rows "
        f"(target {pipeline_params.fit_sample_size:,}) across {len(train_months)} train months"
    )
    return station_enc, station_gm, zone_enc, zone_gm, sample


def fit_direction_bounded(
    train_sample: pl.DataFrame,
    spec: DirectionSpec,
    station_enc: pl.DataFrame,
    station_global_mean: float,
    zone_enc: pl.DataFrame,
    zone_global_mean: float,
) -> FittedDirection:
    """Production fit path: train_sample is already uncensored, direction-
    columned, and bounded (compute_encodings_and_sample's output);
    station_enc/zone_enc are already computed on the FULL uncensored train
    population, not recomputed from the sample -- see config/params.yaml's
    fit_sample_size docstring for why. Shares _fit_models with fit_direction
    so both paths fit identically once encodings are attached."""
    fitted = FittedDirection(
        spec=spec,
        gbt=None,
        glm=None,
        station_enc=station_enc,
        station_global_mean=station_global_mean,
        zone_enc=zone_enc,
        zone_global_mean=zone_global_mean,
    )
    train = _with_model_columns(fitted, train_sample)
    return _fit_models(fitted, train)


def save_fitted(fitted: FittedDirection, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(fitted, f)
    tmp_path.replace(path)


def load_fitted(path: Path) -> FittedDirection:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_or_fit_direction(
    spec: DirectionSpec,
    censoring_params: censoring.CensoringParams,
    pipeline_params: PipelineParams,
    weather_lag_table: pl.DataFrame,
    lag_features_lf: pl.LazyFrame,
    stale_flags: pl.DataFrame | None,
    refit: bool,
) -> FittedDirection:
    """Checkpoint-aware: loads DEMAND_MODEL_DIR/{name}.pkl if present and not
    refit; else runs build-sample -> fit -> calibrate-on-holdout -> save for
    THIS direction only. Must be called sequentially per direction (not with
    both directions' pools live at once) to stay in the 16GB budget."""
    path = DEMAND_MODEL_DIR / f"{spec.name}.pkl"
    if path.exists() and not refit:
        print(f"[demand] {spec.name}: loading cached fit from {path}")
        return load_fitted(path)

    t0 = time.monotonic()
    months = all_month_keys()
    holdout_prefix = pipeline_params.holdout_start_date[:7]
    train_months = [m for m in months if m < holdout_prefix]
    holdout_months = [m for m in months if m >= holdout_prefix]
    print(f"[demand] {spec.name}: {len(train_months)} train months, {len(holdout_months)} holdout months (>= {holdout_prefix})")

    station_enc, station_gm, zone_enc, zone_gm, sample = compute_encodings_and_sample(
        train_months, spec, weather_lag_table, lag_features_lf, pipeline_params
    )
    fitted = fit_direction_bounded(sample, spec, station_enc, station_gm, zone_enc, zone_gm)
    del sample
    print(f"[demand] {spec.name}: fit done, peak RSS {progress.peak_rss_mb():.0f} MB")

    holdout_df = build_range_features(holdout_months, weather_lag_table, lag_features_lf)
    fitted, recovery = calibrate_direction(fitted, holdout_df, censoring_params, stale_flags=stale_flags)
    wmape_table = held_out_wmape(fitted, holdout_df)
    del holdout_df

    elapsed = time.monotonic() - t0
    print(f"[demand] {spec.name}: fit+calibrate total {elapsed:.1f}s, peak RSS {progress.peak_rss_mb():.0f} MB")
    print(f"[demand] {spec.name}: held-out WMAPE:\n{wmape_table}")
    print(f"[demand] {spec.name}: bucket bias correction: {fitted.bucket_bias}")
    print(f"[demand] {spec.name}: recovery table:\n{recovery}")

    save_fitted(fitted, path)
    return fitted


# ---------------------------------------------------------------------------
# Per-month impute stage (production path): chunked, resumable
# ---------------------------------------------------------------------------


def impute_direction_table(fitted: FittedDirection, df: pl.DataFrame, params: censoring.CensoringParams) -> pl.DataFrame:
    """Impute-only counterpart to build_direction_unmet_table -- takes an
    already fit+calibrated FittedDirection (frozen, from
    get_or_fit_direction) instead of fitting one from df. A model is never
    refit per chunk on the production path."""
    imputed = impute_unmet_demand(fitted, df, params)
    prefix = "dep" if fitted.spec.name == "departures" else "arr"
    return imputed.select(
        "station_id",
        "interval_start",
        pl.col("D_hat").alias(f"{prefix}_D_hat"),
        pl.col("unmet_demand").alias(f"{prefix}_gross_unmet"),
    )


def impute_month(
    month_key: str,
    fitted_dep: FittedDirection,
    fitted_arr: FittedDirection,
    censoring_params: censoring.CensoringParams,
    weather_lag_table: pl.DataFrame,
    lag_features_lf: pl.LazyFrame,
) -> pl.DataFrame:
    df = build_month_features(month_key, weather_lag_table, lag_features_lf)
    dep_table = impute_direction_table(fitted_dep, df, censoring_params)
    arr_table = impute_direction_table(fitted_arr, df, censoring_params)
    shared = df.select(SHARED_OUTPUT_COLS)
    return shared.join(dep_table, on=["station_id", "interval_start"]).join(
        arr_table, on=["station_id", "interval_start"]
    )


def run_month(
    month_key: str,
    fitted_dep: FittedDirection,
    fitted_arr: FittedDirection,
    censoring_params: censoring.CensoringParams,
    weather_lag_table: pl.DataFrame,
    lag_features_lf: pl.LazyFrame,
) -> None:
    t0 = time.monotonic()
    unmet = impute_month(month_key, fitted_dep, fitted_arr, censoring_params, weather_lag_table, lag_features_lf)
    checkpoint.write_checkpoint(unmet, checkpoint.checkpoint_path(UNMET_DEMAND_DIR, month_key))
    progress.log_month(month_key, unmet.height, time.monotonic() - t0)


def consolidate_monthly() -> None:
    """Optional: merges UNMET_DEMAND_DIR's per-month checkpoints into one
    UNMET_DEMAND_PATH file via a streaming sink (low memory regardless of
    total size). Downstream consumers (substitution.py, heatmap.py) read
    the monthly directory directly and never need this to have run."""
    UNMET_DEMAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    pl.scan_parquet(UNMET_DEMAND_DIR / "*.parquet").sink_parquet(UNMET_DEMAND_PATH)
    n = pl.scan_parquet(UNMET_DEMAND_PATH).select(pl.len()).collect().item()
    print(f"[demand] consolidated {n:,} rows -> {UNMET_DEMAND_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5/6 demand model: fit once, impute per month.")
    p.add_argument("--month", help="process exactly one month, e.g. 2025-03 (the required single-month dry run)")
    p.add_argument("--all", action="store_true", help="process every month in the panel, skipping already-checkpointed ones")
    p.add_argument("--refit", action="store_true", help="force re-fit/re-calibrate even if a cached model exists")
    p.add_argument("--force", action="store_true", help="reprocess a month's checkpoint even if it already exists")
    p.add_argument("--consolidate", action="store_true", help="merge monthly checkpoints into one unmet_demand.parquet")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    censoring_params = censoring.load_params()
    pipeline_params = load_pipeline_params()

    print("[demand] Stage 0: lag/run-length pre-pass...")
    lag_features_lf = ensure_lag_features_artifact()

    weather = pl.read_parquet(WEATHER_PATH)
    weather_lag_table = compute_weather_lag_table(weather)

    print("[demand] capacity-staleness pre-pass...")
    stale = compute_stale_flags()
    n_stale = stale.filter(pl.col("is_capacity_stale")).height
    print(f"[demand] capacity-staleness pool: {n_stale:,} / {stale.height:,} usable stations flagged")

    fitted_dep = get_or_fit_direction(
        DEPARTURES, censoring_params, pipeline_params, weather_lag_table, lag_features_lf, None, args.refit
    )
    fitted_arr = get_or_fit_direction(
        ARRIVALS, censoring_params, pipeline_params, weather_lag_table, lag_features_lf, stale, args.refit
    )

    if args.month:
        months_to_run = [args.month]
    elif args.all:
        months_to_run = all_month_keys()
    else:
        months_to_run = []
        print("[demand] no --month or --all given; fit-only run complete.")

    for month_key in months_to_run:
        if checkpoint.is_checkpointed(UNMET_DEMAND_DIR, month_key) and not args.force:
            print(f"[demand] {month_key}: checkpoint exists, skipping (use --force to reprocess)")
            continue
        run_month(month_key, fitted_dep, fitted_arr, censoring_params, weather_lag_table, lag_features_lf)

    if args.consolidate:
        consolidate_monthly()


if __name__ == "__main__":
    main()
