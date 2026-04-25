import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.datasets import get_rdataset
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

HERE = Path(__file__).parent


def load_data():
    raw = get_rdataset("AirPassengers").data
    # build monthly datetime index
    idx    = pd.date_range(start="1949-01", periods=len(raw), freq="MS")
    series = pd.Series(raw["value"].values.astype(float), index=idx, name="passengers")
    print("loaded:", len(series), "observations")
    print(series.head(10))
    return series


def run_adf(series, name):
    res  = adfuller(series.dropna(), autolag="AIC")
    stat = res[0]
    pval = res[1]
    lags = res[2]
    crit = res[4]
    print(f"\nADF test ({name})")
    print(f"  stat   = {stat:.4f}")
    print(f"  pvalue = {pval:.4f}")
    print(f"  lags   = {lags}")
    print(f"  crit values: 1%={crit['1%']:.3f}  5%={crit['5%']:.3f}  10%={crit['10%']:.3f}")
    if pval < 0.05:
        print("  => stationary")
    else:
        print("  => NOT stationary")
    return pval


def plot_acf_pacf(series, label, nlags=36):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf( series.dropna(), lags=nlags, ax=axes[0])
    plot_pacf(series.dropna(), lags=nlags, ax=axes[1], method="ywm")
    axes[0].set_title("ACF - " + label)
    axes[1].set_title("PACF - " + label)
    plt.tight_layout()
    plt.savefig(HERE / f"part3_acf_pacf_{label}.png", dpi=120)
    plt.close()


def get_forecast_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{name}")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  MAE  = {mae:.3f}")
    print(f"  MAPE = {mape:.2f}%")
    return {"name": name, "rmse": rmse, "mae": mae, "mape": mape}


def plot_forecast(train, test, fc, ci, label, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index, train, label="train")
    ax.plot(test.index, test,   label="actual", color="black")
    ax.plot(fc.index, fc,       label="forecast", color="red", linestyle="--")
    if ci is not None:
        ax.fill_between(fc.index, ci.iloc[:, 0], ci.iloc[:, 1],
                        color="red", alpha=0.2, label="95% CI")
    ax.set_title("Forecast - " + label)
    ax.set_ylabel("passengers")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":

    series = load_data()

    # plot original series
    fig, ax = plt.subplots(figsize=(10, 4))
    series.plot(ax=ax)
    ax.set_title("Monthly airline passengers 1949-1960")
    ax.set_ylabel("passengers (thousands)")
    plt.tight_layout()
    plt.savefig(HERE / "part3_original.png", dpi=120)
    plt.close()

    # --- stationarity analysis ---
    run_adf(series, "original")

    log_s     = np.log(series)
    diff1     = log_s.diff()
    diff_s12  = log_s.diff(12)
    diff_both = log_s.diff(12).diff()

    run_adf(log_s,             "log")
    run_adf(diff1.dropna(),    "log diff1")
    run_adf(diff_s12.dropna(), "log seasonal diff")
    run_adf(diff_both.dropna(),"log seasonal+first diff")

    # plot all transformations
    fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    axes[0].plot(series);     axes[0].set_title("original")
    axes[1].plot(log_s);      axes[1].set_title("log")
    axes[2].plot(diff1);      axes[2].set_title("log diff (d=1)")
    axes[3].plot(diff_s12);   axes[3].set_title("seasonal diff (D=1, s=12)")
    axes[4].plot(diff_both);  axes[4].set_title("seasonal + first diff")
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.5)
    plt.tight_layout()
    plt.savefig(HERE / "part3_differencing.png", dpi=120)
    plt.close()

    # acf/pacf plots
    plot_acf_pacf(diff_both, "diff_both")
    plot_acf_pacf(series, "original")

    # train/test split - hold out last 12 months (year 1960)
    train = series.iloc[:-12]
    test  = series.iloc[-12:]
    print(f"\ntrain: {len(train)}  test: {len(test)}")

    # ===================== manual SARIMA =====================
    # based on acf/pacf: (1,1,1)(0,1,1,12)
    print("\nfitting manual SARIMA(1,1,1)(0,1,1,12)...")
    m1 = SARIMAX(train, order=(1,1,1), seasonal_order=(0,1,1,12),
                 enforce_stationarity=False, enforce_invertibility=False)
    res1 = m1.fit(disp=False)
    print(res1.summary())

    fc1      = res1.get_forecast(steps=12)
    fc1_mean = fc1.predicted_mean
    fc1_ci   = fc1.conf_int(alpha=0.05)

    plot_forecast(train, test, fc1_mean, fc1_ci,
                  "SARIMA(1,1,1)(0,1,1,12)", HERE / "part3_forecast_manual.png")
    metrics1 = get_forecast_metrics(test.values, fc1_mean.values,
                                    "manual SARIMA(1, 1, 1)(0, 1, 1, 12)")

    # ===================== plain ARIMA =====================
    print("\nfitting ARIMA(2,1,2) (no seasonality)...")
    m2   = ARIMA(train, order=(2,1,2))
    res2 = m2.fit()
    fc2  = res2.forecast(steps=12)

    plot_forecast(train, test, fc2, None,
                  "ARIMA(2,1,2)", HERE / "part3_forecast_arima.png")
    metrics2 = get_forecast_metrics(test.values, fc2.values, "ARIMA(2, 1, 2)")

    # ===================== auto SARIMA =====================
    try:
        import pmdarima as pm
        print("\nrunning auto_arima...")
        auto = pm.auto_arima(train, seasonal=True, m=12,
                             stepwise=True, trace=False,
                             error_action="ignore", suppress_warnings=True)
        print("selected:", auto.order, auto.seasonal_order)
        fc3_vals, fc3_ci_vals = auto.predict(n_periods=12, return_conf_int=True)
        fc3    = pd.Series(fc3_vals, index=test.index)
        fc3_ci = pd.DataFrame(fc3_ci_vals, index=test.index,
                              columns=["lower", "upper"])
        auto_name = f"auto-SARIMA{auto.order}{auto.seasonal_order}"
    except Exception as e:
        print("pmdarima not found, doing grid search instead:", e)
        best_aic = np.inf
        best_fit = None
        best_o   = None
        best_so  = None
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    for P in range(2):
                        for D in range(2):
                            for Q in range(2):
                                try:
                                    tmp = SARIMAX(train,
                                                  order=(p,d,q),
                                                  seasonal_order=(P,D,Q,12),
                                                  enforce_stationarity=False,
                                                  enforce_invertibility=False).fit(disp=False)
                                    if tmp.aic < best_aic:
                                        best_aic = tmp.aic
                                        best_fit = tmp
                                        best_o   = (p,d,q)
                                        best_so  = (P,D,Q,12)
                                except:
                                    pass
        print(f"best by AIC: SARIMA{best_o}{best_so}  aic={best_aic:.2f}")
        fc3_raw = best_fit.get_forecast(steps=12)
        fc3     = fc3_raw.predicted_mean
        fc3_ci  = fc3_raw.conf_int(alpha=0.05)
        auto_name = f"auto-SARIMA{best_o}{best_so}"

    plot_forecast(train, test, fc3, fc3_ci,
                  auto_name, HERE / "part3_forecast_auto.png")
    metrics3 = get_forecast_metrics(test.values, fc3.values, auto_name)

    # summary
    results = pd.DataFrame([metrics1, metrics2, metrics3])
    print("\n=== comparison ===")
    print(results.to_string(index=False))
    results.to_csv(HERE / "part3_metrics.csv", index=False)
