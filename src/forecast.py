"""Day-ahead forecast of GB national demand: baseline vs gradient-boosted trees.

Setting: a forecast for every half-hour of day D is issued around midday on D-1.
The latest demand that is certainly known at that point for *every* target half-hour
is two days before the target, so demand lags are restricted to 2 days and 7 days.
"""
from pathlib import Path

import holidays
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from data import load_all
from style import AQUA, BLUE, INK, ORANGE, setup

OUT = Path(__file__).resolve().parents[1] / "results"
TEST_START = pd.Timestamp("2025-09-19", tz="UTC")   # final 12 months held out
VAL_START = pd.Timestamp("2025-06-19", tz="UTC")    # last 3 months of training, for tuning
LAG_2D, LAG_7D = 96, 336                            # in half-hours

CALENDAR = ["half_hour", "day_of_week", "is_holiday", "is_christmas", "doy_sin", "doy_cos"]
LAGS = ["demand_lag_2d", "demand_lag_7d"]
WEATHER = ["temperature", "temperature_24h", "solar_radiation", "solar_potential", "wind_speed"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    local = df.index.tz_convert("Europe/London")
    X["half_hour"] = local.hour * 2 + local.minute // 30
    X["day_of_week"] = local.dayofweek
    hols = holidays.UK(subdiv="ENG", years=range(2022, 2028))
    X["is_holiday"] = np.array([d in hols for d in local.date], dtype=int)
    X["is_christmas"] = (((local.month == 12) & (local.day >= 24)) |
                         ((local.month == 1) & (local.day <= 2))).astype(int)
    doy = local.dayofyear
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

    X["demand_lag_2d"] = df["demand"].shift(LAG_2D)
    X["demand_lag_7d"] = df["demand"].shift(LAG_7D)

    X["temperature"] = df["temperature"]
    X["temperature_24h"] = df["temperature"].rolling(48, min_periods=1).mean()  # thermal inertia
    X["solar_radiation"] = df["solar_radiation"]
    # rooftop solar output scales with sunshine x installed capacity (capacity grows over time)
    X["solar_potential"] = df["solar_radiation"] * df["solar_capacity"] / 1000
    X["wind_speed"] = df["wind_speed"]
    return X


def seasonal_average(X, y, train, test) -> pd.Series:
    """Climatology baseline: mean training demand for the same half-hour, day type
    (working day vs weekend or bank holiday) and calendar month."""
    month = pd.Series(X.index.tz_convert("Europe/London").month, index=X.index)
    day_off = (X["day_of_week"] >= 5) | (X["is_holiday"] == 1)
    keys = [X["half_hour"], day_off, month]
    table = y[train].groupby([k[train] for k in keys]).mean()
    lookup = list(zip(*(k[test] for k in keys)))
    return pd.Series([table[key] for key in lookup], index=X.index[test])


def metrics(y, yhat) -> dict:
    err = y - yhat
    return {"MAE (MW)": np.abs(err).mean(),
            "RMSE (MW)": np.sqrt((err ** 2).mean()),
            "MAPE (%)": 100 * (np.abs(err) / y).mean()}


def fit(X, y, max_iter):
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, categorical_features=["day_of_week"],
        random_state=0).fit(X, y)


def tune_iterations(X, y, features) -> int:
    """Pick the number of boosting rounds on the last 3 months of training data."""
    tr, va = X.index < VAL_START, X.index >= VAL_START
    m = fit(X.loc[tr, features], y[tr], 1500)
    maes = [np.abs(y[va] - p).mean() for p in m.staged_predict(X.loc[va, features])]
    return int(np.argmin(maes)) + 1


def main():
    setup()
    OUT.mkdir(exist_ok=True)
    df = load_all()
    X = make_features(df)
    y = df["demand"]
    ok = X.notna().all(axis=1)
    X, y = X[ok], y[ok]

    train, test = X.index < TEST_START, X.index >= TEST_START
    print(f"train {X.index[train].min():%Y-%m-%d} to {X.index[train].max():%Y-%m-%d} ({train.sum()} rows)")
    print(f"test  {X.index[test].min():%Y-%m-%d} to {X.index[test].max():%Y-%m-%d} ({test.sum()} rows)")

    preds = {"Baseline: same half-hour last week": X.loc[test, "demand_lag_7d"],
             "Baseline: seasonal average": seasonal_average(X, y, train, test)}
    feature_sets = {"Model without weather": CALENDAR + LAGS,
                    "Model with weather": CALENDAR + LAGS + WEATHER}
    models = {}
    for name, feats in feature_sets.items():
        n_iter = tune_iterations(X[train], y[train], feats)
        models[name] = fit(X.loc[train, feats], y[train], n_iter)
        preds[name] = pd.Series(models[name].predict(X.loc[test, feats]), index=X.index[test])
        print(f"{name}: {n_iter} boosting rounds")

    yt = y[test]
    table = pd.DataFrame({k: metrics(yt, v) for k, v in preds.items()}).T.round(1)
    for col, ref in [("vs weekly baseline", 0), ("vs seasonal average", 1)]:
        base_mae = table.iloc[ref]["MAE (MW)"]
        table[col] = (100 * (table["MAE (MW)"] / base_mae - 1)).round(0).astype(int).astype(str) + "%"
        table.loc[table.index[:2], col] = "-"
    print("\n", table.to_string())
    table.to_csv(OUT / "metrics.csv")

    # Errors split by day type
    hol = X.loc[test, "is_holiday"].astype(bool) | X.loc[test, "is_christmas"].astype(bool)
    best = preds["Model with weather"]
    split = pd.DataFrame({
        "Baseline MAE": [np.abs(yt - preds[list(preds)[0]])[~hol].mean(), np.abs(yt - preds[list(preds)[0]])[hol].mean()],
        "Model MAE": [np.abs(yt - best)[~hol].mean(), np.abs(yt - best)[hol].mean()]},
        index=["Normal days", "Bank holidays and Christmas"]).round(0)
    print("\n", split.to_string())
    split.to_csv(OUT / "metrics_by_day_type.csv")

    # Week after each clock change: the lags are fixed UTC shifts, so for up to 7 days
    # after a change they point at a different local clock time than the target
    offset = np.asarray(yt.index.tz_convert("Europe/London").map(lambda t: t.utcoffset()))
    changes = yt.index[1:][offset[1:] != offset[:-1]]
    after = np.zeros(len(yt), dtype=bool)
    for c in changes:
        after |= (yt.index >= c) & (yt.index < c + pd.Timedelta(days=7))
    base_err, model_err = np.abs(yt - preds[list(preds)[0]]), np.abs(yt - best)
    clock = pd.DataFrame({
        "Baseline MAE": [base_err[after].mean(), base_err[~after].mean()],
        "Model MAE": [model_err[after].mean(), model_err[~after].mean()]},
        index=["Week after a clock change", "All other days"]).round(0)
    print(f"\nclock changes in test period: {[f'{c:%Y-%m-%d}' for c in changes]}")
    print(clock.to_string())
    clock.to_csv(OUT / "metrics_clock_change.csv")

    # Monthly MAE: baseline vs model
    local_month = yt.index.tz_convert("Europe/London").tz_localize(None).to_period("M")
    monthly = pd.DataFrame({
        "Baseline": np.abs(yt - preds[list(preds)[0]]).groupby(local_month).mean(),
        "Model with weather": np.abs(yt - best).groupby(local_month).mean()}) / 1000
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xpos = np.arange(len(monthly))
    ax.bar(xpos - 0.2, monthly["Baseline"], 0.38, color=BLUE, label="Baseline: same half-hour last week")
    ax.bar(xpos + 0.2, monthly["Model with weather"], 0.38, color=ORANGE, label="Gradient-boosted model with weather")
    ax.set_xticks(xpos, [p.strftime("%b\n%y") for p in monthly.index])
    ax.set_ylabel("Mean absolute error (GW)")
    ax.set_title("Day-ahead error by month, held-out year (Sep 2025 to Sep 2026)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "mae_by_month.png", dpi=150)

    # Example winter week
    wk = (yt.index >= pd.Timestamp("2026-01-12", tz="UTC")) & (yt.index < pd.Timestamp("2026-01-19", tz="UTC"))
    t = yt.index[wk].tz_convert("Europe/London")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(t, yt[wk] / 1000, color=INK, lw=2, label="Actual")
    ax.plot(t, preds[list(preds)[0]][wk] / 1000, color=BLUE, lw=1.5, label="Baseline: same half-hour last week")
    ax.plot(t, best[wk] / 1000, color=ORANGE, lw=1.5, label="Model with weather")
    ax.set_ylabel("National demand (GW)")
    ax.set_title("Example held-out week, 12 to 18 January 2026")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=t.tz))
    fig.tight_layout()
    fig.savefig(OUT / "example_week.png", dpi=150)

    # Permutation importance on a sample of the test year
    feats = feature_sets["Model with weather"]
    sample = X.loc[test, feats].sample(4000, random_state=0)
    imp = permutation_importance(models["Model with weather"], sample, yt.loc[sample.index],
                                 scoring="neg_mean_absolute_error", n_repeats=5, random_state=0)
    importance = pd.Series(imp.importances_mean, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(importance.index, importance / 1000, color=AQUA)
    ax.set_xlabel("Increase in MAE when the feature is shuffled (GW)")
    ax.set_title("Which inputs the model relies on (permutation importance)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    print("\nimportance (MW):\n", importance.sort_values(ascending=False).round(0).to_string())


if __name__ == "__main__":
    main()
