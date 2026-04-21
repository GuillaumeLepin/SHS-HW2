"""
HW2-2 Part 3: SARIMA on the Airline Passengers dataset.
Loads the dataset via statsmodels (AirPassengers), inspects trend/seasonality,
runs ADF tests on the original and differenced series, identifies SARIMA
orders from ACF/PACF, fits a manual SARIMA, compares to ARIMA and to
auto-SARIMA (pmdarima if available, otherwise a small grid search on AIC),
and evaluates forecast performance on a 12-month hold-out.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.datasets import get_rdataset
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Part 1.2: Explore the dataset
# ---------------------------------------------------------------------------
def load_series():
    data = get_rdataset("AirPassengers").data
    # `time` is a decimal year; convert to a proper monthly DatetimeIndex.
    idx = pd.date_range(start="1949-01", periods=len(data), freq="MS")
    series = pd.Series(data["value"].values.astype(float),
                       index=idx, name="passengers")
    print(series.head())
    print(f"\nseries length = {len(series)}  "
          f"({series.index.min().date()} -> {series.index.max().date()})")
    return series


def plot_original(series):
    fig, ax = plt.subplots(figsize=(10, 4))
    series.plot(ax=ax)
    ax.set_title("Monthly airline passengers (1949-1960)")
    ax.set_ylabel("passengers (thousands)")
    fig.tight_layout()
    fig.savefig(HERE / "part3_original.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 1.3: Stationarity testing
# ---------------------------------------------------------------------------
def adf_report(series, name):
    res = adfuller(series.dropna(), autolag="AIC")
    print(f"\nADF test — {name}")
    print(f"  statistic = {res[0]:.4f}")
    print(f"  p-value   = {res[1]:.4f}")
    print(f"  lags used = {res[2]}")
    print(f"  n obs     = {res[3]}")
    print("  critical values:")
    for k, v in res[4].items():
        print(f"    {k:>5s}: {v:.4f}")
    print(f"  => {'stationary' if res[1] < 0.05 else 'NON-stationary'} at 5%")
    return res[1]


def make_stationary(series):
    # Log transform stabilizes variance (variance grows with the level).
    log_series = np.log(series)
    # First difference removes trend.
    diff1 = log_series.diff()
    # Seasonal difference (period 12) removes yearly seasonality.
    diff_seasonal = log_series.diff(12)
    # Combined: first difference of seasonally-differenced series.
    diff_both = log_series.diff(12).diff()
    return log_series, diff1, diff_seasonal, diff_both


def plot_differencing(series, log_series, diff1, diff_seasonal, diff_both):
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(series); axes[0].set_title("original")
    axes[1].plot(log_series); axes[1].set_title("log(series)")
    axes[2].plot(diff1); axes[2].set_title("log diff (d=1)")
    axes[3].plot(diff_seasonal); axes[3].set_title("seasonal diff (D=1, s=12)")
    axes[4].plot(diff_both)
    axes[4].set_title("seasonal + first diff (d=1, D=1, s=12)")
    for ax in axes:
        ax.axhline(0, color="grey", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(HERE / "part3_differencing.png", dpi=120)
    plt.close(fig)


def plot_acf_pacf(series, name, lags=36):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF — {name}")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF — {name}")
    fig.tight_layout()
    fig.savefig(HERE / f"part3_acf_pacf_{name}.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 2: Fit SARIMA
# ---------------------------------------------------------------------------
def fit_sarima(train, order, seasonal_order):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    return model.fit(disp=False)


def fit_arima(train, order):
    return ARIMA(train, order=order).fit()


def grid_search_sarima(train, s=12):
    """Exhaustive small grid for (p,d,q)(P,D,Q,s) — minimizes AIC."""
    best = None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                for P in range(2):
                    for D in range(2):
                        for Q in range(2):
                            try:
                                r = fit_sarima(train, (p, d, q),
                                               (P, D, Q, s))
                                if best is None or r.aic < best[0]:
                                    best = (r.aic, (p, d, q),
                                            (P, D, Q, s), r)
                            except Exception:
                                pass
    return best


# ---------------------------------------------------------------------------
# Part 3: Forecast & evaluate
# ---------------------------------------------------------------------------
def evaluate_forecast(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n=== {name} ===")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  MAE  = {mae:.3f}")
    print(f"  MAPE = {mape:.2f}%")
    return {"name": name, "rmse": rmse, "mae": mae, "mape": mape}


def plot_forecast(train, test, forecast, conf_int, name, out_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index, train, label="train")
    ax.plot(test.index, test, label="actual (test)", color="black")
    ax.plot(forecast.index, forecast, label=f"forecast ({name})",
            color="red")
    if conf_int is not None:
        ax.fill_between(forecast.index,
                        conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                        color="red", alpha=0.15, label="95% CI")
    ax.set_title(f"Forecast vs actual — {name}")
    ax.set_ylabel("passengers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    series = load_series()
    plot_original(series)

    # --- Stationarity ---
    adf_report(series, "original")
    log_series, diff1, diff_seasonal, diff_both = make_stationary(series)
    adf_report(log_series, "log")
    adf_report(diff1.dropna(), "log diff1")
    adf_report(diff_seasonal.dropna(), "log seasonal-diff (s=12)")
    adf_report(diff_both.dropna(), "log seasonal + first diff")

    plot_differencing(series, log_series, diff1, diff_seasonal, diff_both)

    # --- ACF / PACF for order identification ---
    plot_acf_pacf(diff_both, "diff_both")
    plot_acf_pacf(series, "original")

    # --- Split into train / test (last 12 months = 1960 held out) ---
    train, test = series.iloc[:-12], series.iloc[-12:]

    # --- Manual SARIMA: based on ACF/PACF of the d=1, D=1 series on
    # log-passengers, a common choice for this dataset is (1,1,1)(0,1,1,12).
    # We fit on the original scale to keep reporting simple; SARIMAX handles
    # the differencing internally.
    manual_order = (1, 1, 1)
    manual_seasonal = (0, 1, 1, 12)
    manual = fit_sarima(train, manual_order, manual_seasonal)
    print("\n", manual.summary())

    fc = manual.get_forecast(steps=12)
    manual_fc = fc.predicted_mean
    manual_ci = fc.conf_int(alpha=0.05)
    plot_forecast(train, test, manual_fc, manual_ci,
                  f"SARIMA{manual_order}{manual_seasonal}",
                  HERE / "part3_forecast_manual.png")
    manual_metrics = evaluate_forecast(test.values, manual_fc.values,
                                       f"manual SARIMA{manual_order}"
                                       f"{manual_seasonal}")

    # --- Plain ARIMA (non-seasonal) for comparison ---
    arima_order = (2, 1, 2)
    ar = fit_arima(train, arima_order)
    arima_fc = ar.forecast(steps=12)
    plot_forecast(train, test, arima_fc, None,
                  f"ARIMA{arima_order}",
                  HERE / "part3_forecast_arima.png")
    arima_metrics = evaluate_forecast(test.values, arima_fc.values,
                                      f"ARIMA{arima_order}")

    # --- Auto-SARIMA ---
    try:
        import pmdarima as pm
        auto = pm.auto_arima(train, seasonal=True, m=12,
                             trace=False, error_action="ignore",
                             suppress_warnings=True,
                             stepwise=True)
        print(f"\npmdarima auto_arima picked: {auto.order} x "
              f"{auto.seasonal_order}")
        auto_fc, auto_ci = auto.predict(n_periods=12, return_conf_int=True)
        auto_fc = pd.Series(auto_fc, index=test.index)
        auto_ci = pd.DataFrame(auto_ci, index=test.index,
                               columns=["lower", "upper"])
        auto_name = f"auto-SARIMA{auto.order}{auto.seasonal_order}"
    except Exception as e:
        print(f"\npmdarima not available ({e}); "
              "falling back to grid search by AIC.")
        aic, o, so, fit_auto = grid_search_sarima(train)
        print(f"grid-search best: SARIMA{o}{so}  (AIC={aic:.2f})")
        fc = fit_auto.get_forecast(steps=12)
        auto_fc = fc.predicted_mean
        auto_ci = fc.conf_int(alpha=0.05)
        auto_name = f"auto-SARIMA{o}{so}"

    plot_forecast(train, test, auto_fc, auto_ci, auto_name,
                  HERE / "part3_forecast_auto.png")
    auto_metrics = evaluate_forecast(test.values, auto_fc.values, auto_name)

    # --- Summary table ---
    summary = pd.DataFrame([manual_metrics, arima_metrics, auto_metrics])
    print("\n=== Comparison ===")
    print(summary.to_string(index=False))
    summary.to_csv(HERE / "part3_metrics.csv", index=False)


if __name__ == "__main__":
    main()
