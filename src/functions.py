"""Funções reutilizáveis do experimento de forecasting de temperatura."""

from __future__ import annotations

import os
import random
import warnings
import logging
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK", "true")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


SEED = 42
TARGET = "meantemp"
FEATURES = [
    "temp_lag_1", "temp_lag_2", "temp_lag_3", "temp_lag_7",
    "temp_lag_14", "temp_lag_30", "temp_roll_mean_3",
    "temp_roll_std_3", "temp_roll_mean_7", "temp_roll_std_7",
    "temp_roll_mean_14", "temp_roll_std_14", "temp_roll_mean_30",
    "temp_roll_std_30", "doy_sin", "doy_cos", "trend_days",
]
FOLDS = [
    ("2016-Q1", "2016-01-01", "2016-03-31"),
    ("2016-Q2", "2016-04-01", "2016-06-30"),
    ("2016-Q3", "2016-07-01", "2016-09-30"),
    ("2016-Q4", "2016-10-01", "2016-12-31"),
]


@dataclass
class ExperimentResult:
    data: pd.DataFrame
    cv_detail: pd.DataFrame
    cv_summary: pd.DataFrame
    test_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    importance: pd.DataFrame
    ablation: pd.DataFrame
    arima_order: tuple[int, int, int]
    sarima_order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_official_splits(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    development = pd.read_csv(data_dir / "DailyDelhiClimateTrain.csv", parse_dates=["date"])
    test = pd.read_csv(data_dir / "DailyDelhiClimateTest.csv", parse_dates=["date"])
    # A data 2017-01-01 aparece com valores conflitantes nos dois arquivos.
    # A versão do teste é preservada; o desenvolvimento termina em 2016-12-31.
    development = development[development["date"] < test["date"].min()].copy()
    development["official_split"] = "development"
    test["official_split"] = "test"
    return development.reset_index(drop=True), test.reset_index(drop=True)


def make_features(development: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat([development, test], ignore_index=True).sort_values("date").reset_index(drop=True)
    expected = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
    if frame["date"].duplicated().any() or not np.array_equal(frame["date"].to_numpy(), expected.to_numpy()):
        raise ValueError("A série deve ser diária, contínua e sem duplicidades.")
    for lag in (1, 2, 3, 7, 14, 30):
        frame[f"temp_lag_{lag}"] = frame[TARGET].shift(lag)
    past = frame[TARGET].shift(1)
    for window in (3, 7, 14, 30):
        frame[f"temp_roll_mean_{window}"] = past.rolling(window).mean()
        frame[f"temp_roll_std_{window}"] = past.rolling(window).std()
    day = frame["date"].dt.dayofyear
    frame["doy_sin"] = np.sin(2 * np.pi * day / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * day / 365.25)
    frame["trend_days"] = (frame["date"] - frame["date"].min()).dt.days
    return frame.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def model_catalog() -> dict[str, object]:
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            max_features=0.8,
            n_jobs=-1,
            random_state=SEED,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=2.0,
            verbosity=-1,
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def error_metrics(y_true: np.ndarray, pred: np.ndarray, mase_scale: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(pred)) / 2
    smape = 100 * np.mean(np.divide(np.abs(y_true - pred), denominator, out=np.zeros_like(denominator), where=denominator != 0))
    return {
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": mean_squared_error(y_true, pred) ** 0.5,
        "MASE": mean_absolute_error(y_true, pred) / mase_scale,
        "sMAPE_%": smape,
    }


def select_statistical_orders(y: pd.Series) -> tuple[tuple, tuple, tuple]:
    """Seleciona ordens por AIC usando somente dados anteriores aos folds."""
    arima_candidates = [(1, 0, 1), (2, 0, 1), (1, 1, 1), (2, 1, 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scored_arima = []
        for order in arima_candidates:
            try:
                fit = ARIMA(y.to_numpy(), order=order, trend="t" if order[1] == 1 else "ct").fit()
                scored_arima.append((fit.aic, order))
            except Exception:
                continue
        arima_order = min(scored_arima)[1]

        seasonal_candidates = [(1, 0, 0, 7), (0, 0, 1, 7), (1, 0, 1, 7)]
        scored_sarima = []
        for seasonal_order in seasonal_candidates:
            try:
                fit = SARIMAX(
                    y.to_numpy(), order=arima_order, seasonal_order=seasonal_order,
                    trend="t" if arima_order[1] == 1 else "ct",
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False, maxiter=100)
                scored_sarima.append((fit.aic, seasonal_order))
            except Exception:
                continue
        seasonal_order = min(scored_sarima)[1]
    return arima_order, arima_order, seasonal_order


def rolling_statistical_forecast(
    train_y: pd.Series,
    evaluation_y: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Prevê D+1 e atualiza o estado com o realizado, sem reestimar parâmetros."""
    trend = "t" if order[1] == 1 else "ct"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if seasonal_order is None:
            result = ARIMA(train_y.to_numpy(), order=order, trend=trend).fit()
        else:
            result = SARIMAX(
                train_y.to_numpy(), order=order, seasonal_order=seasonal_order,
                trend=trend, enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=150)
        predictions = []
        for actual in evaluation_y.to_numpy():
            predictions.append(float(np.asarray(result.forecast(1))[0]))
            result = result.extend([actual])
    return np.asarray(predictions)


def _build_lstm(lookback: int, seed: int):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    model = Sequential([Input((lookback, 1)), LSTM(32), Dropout(0.20), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    return model


def lstm_forecast(
    history: pd.DataFrame,
    train_mask: pd.Series,
    evaluation_mask: pd.Series,
    lookback: int = 30,
    seed: int = SEED,
) -> tuple[np.ndarray, int]:
    """Early stopping interno e refit no treino externo pelo número de épocas escolhido."""
    from tensorflow.keras.callbacks import EarlyStopping

    values = history[TARGET].to_numpy(dtype=float)
    train_positions = np.flatnonzero(train_mask.to_numpy())
    eval_positions = np.flatnonzero(evaluation_mask.to_numpy())
    train_positions = train_positions[train_positions >= lookback]
    if len(train_positions) < 180:
        raise ValueError("Histórico insuficiente para a LSTM.")

    inner_cut = max(60, int(len(train_positions) * 0.85))
    fit_positions, inner_positions = train_positions[:inner_cut], train_positions[inner_cut:]
    scaler = MinMaxScaler().fit(values[: train_positions[inner_cut - 1] + 1].reshape(-1, 1))
    scaled = scaler.transform(values.reshape(-1, 1)).ravel()

    def sequences(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray([scaled[pos - lookback:pos] for pos in positions])[..., None]
        y = scaled[positions]
        return x, y

    x_fit, y_fit = sequences(fit_positions)
    x_inner, y_inner = sequences(inner_positions)
    probe = _build_lstm(lookback, seed)
    history_fit = probe.fit(
        x_fit, y_fit, validation_data=(x_inner, y_inner), epochs=60,
        batch_size=32, shuffle=False, verbose=0,
        callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
    )
    best_epoch = int(np.argmin(history_fit.history["val_loss"]) + 1)

    final_scaler = MinMaxScaler().fit(values[train_positions].reshape(-1, 1))
    final_scaled = final_scaler.transform(values.reshape(-1, 1)).ravel()

    def final_sequences(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray([final_scaled[pos - lookback:pos] for pos in positions])[..., None]
        return x, final_scaled[positions]

    x_train, y_train = final_sequences(train_positions)
    x_eval, _ = final_sequences(eval_positions)
    model = _build_lstm(lookback, seed)
    model.fit(x_train, y_train, epochs=best_epoch, batch_size=32, shuffle=False, verbose=0)
    pred_scaled = model.predict(x_eval, verbose=0)
    pred = final_scaler.inverse_transform(pred_scaled).ravel()
    return pred, best_epoch


def _evaluate_period(
    frame: pd.DataFrame,
    train_mask: pd.Series,
    eval_mask: pd.Series,
    arima_order: tuple,
    sarima_order: tuple,
    seasonal_order: tuple,
) -> tuple[list[dict], dict[str, np.ndarray], int]:
    train, evaluation = frame[train_mask], frame[eval_mask]
    scale = train[TARGET].diff().abs().dropna().mean()
    predictions: dict[str, np.ndarray] = {
        "Persistência D-1": evaluation["temp_lag_1"].to_numpy(),
        "Sazonal ingênuo D-7": evaluation["temp_lag_7"].to_numpy(),
        "ARIMA": rolling_statistical_forecast(train[TARGET], evaluation[TARGET], arima_order),
        "SARIMA": rolling_statistical_forecast(train[TARGET], evaluation[TARGET], sarima_order, seasonal_order),
    }
    for name, model in model_catalog().items():
        fitted = clone(model).fit(train[FEATURES], train[TARGET])
        predictions[name] = fitted.predict(evaluation[FEATURES])
    predictions["LSTM"], best_epoch = lstm_forecast(frame, train_mask, eval_mask)
    rows = [{"model": name, **error_metrics(evaluation[TARGET], pred, scale)} for name, pred in predictions.items()]
    return rows, predictions, best_epoch


def interpret_tree_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[(frame["official_split"] == "development") & (frame["date"] < "2016-10-01")]
    valid = frame[(frame["date"] >= "2016-10-01") & (frame["date"] <= "2016-12-31")]
    scale = train[TARGET].diff().abs().dropna().mean()
    importance_rows, ablation_rows = [], []
    variants = {
        "todas": FEATURES,
        "sem lag 1": [feature for feature in FEATURES if feature != "temp_lag_1"],
        "somente lag 1": ["temp_lag_1"],
    }
    for name, estimator in model_catalog().items():
        full = clone(estimator).fit(train[FEATURES], train[TARGET])
        permutation = permutation_importance(
            full, valid[FEATURES], valid[TARGET], scoring="neg_mean_absolute_error",
            n_repeats=20, random_state=SEED, n_jobs=-1,
        )
        for feature, mean, std in zip(FEATURES, permutation.importances_mean, permutation.importances_std):
            importance_rows.append({"model": name, "feature": feature, "MAE_increase": mean, "std": std})
        for variant, columns in variants.items():
            fitted = clone(estimator).fit(train[columns], train[TARGET])
            pred = fitted.predict(valid[columns])
            ablation_rows.append({"model": name, "features": variant, **error_metrics(valid[TARGET], pred, scale)})
    importance = pd.DataFrame(importance_rows).sort_values(["model", "MAE_increase"], ascending=[True, False])
    ablation = pd.DataFrame(ablation_rows).sort_values(["model", "MAE"])
    return importance.reset_index(drop=True), ablation.reset_index(drop=True)


def run_experiment(data_dir: str | Path = "data") -> ExperimentResult:
    set_seed()
    development, test = load_official_splits(data_dir)
    frame = make_features(development, test)
    pre_validation = frame[(frame["official_split"] == "development") & (frame["date"] < "2016-01-01")]
    arima_order, sarima_order, seasonal_order = select_statistical_orders(pre_validation[TARGET])

    cv_rows = []
    for fold, start, end in FOLDS:
        train_mask = (frame["official_split"] == "development") & (frame["date"] < start)
        eval_mask = (frame["date"] >= start) & (frame["date"] <= end)
        rows, _, epoch = _evaluate_period(frame, train_mask, eval_mask, arima_order, sarima_order, seasonal_order)
        for row in rows:
            cv_rows.append({"fold": fold, "best_epoch": epoch if row["model"] == "LSTM" else np.nan, **row})
    cv_detail = pd.DataFrame(cv_rows)
    summary = cv_detail.groupby("model")[["MAE", "RMSE", "MASE", "sMAPE_%"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    cv_summary = summary.reset_index().sort_values("MAE_mean").reset_index(drop=True)

    train_mask = frame["official_split"] == "development"
    test_mask = frame["official_split"] == "test"
    test_rows, test_pred, epoch = _evaluate_period(frame, train_mask, test_mask, arima_order, sarima_order, seasonal_order)
    test_metrics = pd.DataFrame(test_rows).sort_values("MAE").reset_index(drop=True)
    baseline_mae = test_metrics.loc[test_metrics["model"] == "Persistência D-1", "MAE"].iloc[0]
    test_metrics["skill_vs_persistence_%"] = 100 * (1 - test_metrics["MAE"] / baseline_mae)
    test_metrics["best_epoch"] = np.where(test_metrics["model"] == "LSTM", epoch, np.nan)
    predictions = frame.loc[test_mask, ["date", TARGET]].copy()
    for name, values in test_pred.items():
        predictions[name] = values

    importance, ablation = interpret_tree_models(frame)
    return ExperimentResult(
        frame, cv_detail, cv_summary, test_metrics, predictions,
        importance, ablation, arima_order, sarima_order, seasonal_order,
    )


def stationarity_row(series, name):
    values = pd.Series(series).dropna()
    adf = adfuller(values, autolag='AIC')
    kpss_result = kpss(values, regression='c', nlags='auto')
    return {
        'série': name,
        'ADF p-valor': adf[1],
        'KPSS p-valor': kpss_result[1],
        'evidência conjunta de estacionaridade': adf[1] < 0.05 and kpss_result[1] > 0.05,
    }


def block_bootstrap_comparison(predictions, baseline='Persistência D-1', block=7, n_boot=5000):
    rng = np.random.default_rng(SEED)
    y = predictions[TARGET].to_numpy()
    base_loss = np.abs(y - predictions[baseline].to_numpy())
    n = len(y)
    rows = []
    for model in [c for c in predictions.columns if c not in {'date', TARGET, baseline}]:
        model_loss = np.abs(y - predictions[model].to_numpy())
        daily_gain = base_loss - model_loss  # positivo = modelo melhora o baseline
        boot = []
        for _ in range(n_boot):
            starts = rng.integers(0, n, size=int(np.ceil(n / block)))
            idx = np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]
            boot.append(daily_gain[idx].mean())
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append({
            'modelo': model,
            'ganho médio de MAE (°C)': daily_gain.mean(),
            'IC 95% inferior': low,
            'IC 95% superior': high,
            'ganho robusto?': low > 0,
        })
    return pd.DataFrame(rows).sort_values('ganho médio de MAE (°C)', ascending=False)
