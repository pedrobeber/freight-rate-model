"""Cleaning + feature engineering shared by experiments and the final model."""
import numpy as np
import pandas as pd

EQUIP = ["Dry Van", "Reefer", "Flatbed"]
# columns that depend on time / the market: built for the date experiments, NOT fed to the final model
TIME_COLS = ["mi_day", "qs_day", "mi_day_7", "mi_chg_7", "market_index", "dom", "dow", "is_weekend",
             "market_missing", "quote_signal"]
OUTLIER_LOG_THRESHOLD = 0.5  # |log(actual/expected)| above this = outlier (normal loads: sd 0.036; outliers: 0.77 and beyond)


def load(path):
    return pd.read_csv(path, parse_dates=["date"])


def daily_market(*frames):
    """Per-date mean market_index / quote_signal from ALL feature rows (no labels used).
    Fixes missing market_index and gives a denoised, forward-available time signal."""
    allf = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    d = allf.groupby("date").agg(mi_day=("market_index", "mean"), qs_day=("quote_signal", "mean"))
    d["mi_day_7"] = d.mi_day.rolling(7, min_periods=1, center=True).mean()
    d["mi_chg_7"] = d.mi_day - d.mi_day.shift(7).bfill()
    return d


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 7918 * np.arcsin(np.sqrt(a))  # miles


def build(df, daily, weight_median):
    X = pd.DataFrame(index=df.index)
    X["distance"] = df.distance
    X["log_dist"] = np.log(df.distance)
    # data quality: negative weights are sign flips -> abs; missing -> median + flag
    X["weight_missing"] = df.weight.isna().astype(int)
    X["weight"] = df.weight.abs().fillna(weight_median)
    for e in EQUIP:
        X["eq_" + e.replace(" ", "")] = (df.equipment == e).astype(int)
    d = daily.reindex(df.date).reset_index(drop=True).set_index(df.index)
    X["market_index"] = df.market_index.fillna(d.mi_day)
    X["market_missing"] = df.market_index.isna().astype(int)
    X["quote_signal"] = df.quote_signal
    X[["mi_day", "qs_day", "mi_day_7", "mi_chg_7"]] = d[["mi_day", "qs_day", "mi_day_7", "mi_chg_7"]]
    # date: cyclical/seasonal info that generalises (no raw trend -> would not extrapolate to Nov/Dec)
    X["dow"] = df.date.dt.dayofweek
    X["is_weekend"] = (X.dow >= 5).astype(int)
    X["dom"] = df.date.dt.day
    # geography (works for unseen cities, unlike one-hot)
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[c] = df[c]
    X["dlat"] = df.delivery_lat - df.pickup_lat
    X["dlon"] = df.delivery_lon - df.pickup_lon
    X["bearing"] = np.arctan2(X.dlon, X.dlat)
    X["circuity"] = df.distance / haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon).clip(lower=1)
    X["weight_x_dist"] = X.weight * df.distance / 1e6
    return X


# ---------------------------------------------------------------- feature-set variants (used by the SVR tuning)
BASE_COLS = None  # filled lazily: all non-time columns of build()


def add_extras(X):
    """Extra engineered columns on top of build(): nonlinear transforms and interactions."""
    X = X.copy()
    X["log_weight"] = np.log(X.weight.clip(lower=1))
    X["weight_sq"] = (X.weight / 1e4) ** 2
    X["sqrt_dist"] = np.sqrt(X.distance)
    X["dist_sq"] = (X.distance / 1e3) ** 2
    X["haversine"] = haversine(X.pickup_lat, X.pickup_lon, X.delivery_lat, X.delivery_lon)
    for e in EQUIP:
        c = "eq_" + e.replace(" ", "")
        X[c + "_x_logdist"] = X[c] * X.log_dist
    X["log_mi7"] = np.log(X.mi_day_7)
    X["dow_sin"] = np.sin(2 * np.pi * X.dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * X.dow / 7)
    return X


def feature_set(X, name):
    """Select the columns of a feature set from an add_extras() frame."""
    base = [c for c in X.columns if c not in TIME_COLS and c not in EXTRA_COLS]
    if name == "base":
        return X[base]
    if name == "lean":
        return X[["distance", "log_dist", "weight", "eq_DryVan", "eq_Reefer", "eq_Flatbed", "pickup_lat", "pickup_lon",
                  "delivery_lat", "delivery_lon", "circuity"]]
    if name == "rich":
        return X[base + RICH_COLS]
    if name == "time":
        return X[base + TIME_EXTRA]
    if name == "rich+time":
        return X[base + RICH_COLS + TIME_EXTRA]
    raise ValueError(name)


RICH_COLS = ["log_weight", "weight_sq", "sqrt_dist", "dist_sq", "haversine",
             "eq_DryVan_x_logdist", "eq_Reefer_x_logdist", "eq_Flatbed_x_logdist"]
TIME_EXTRA = ["log_mi7", "dow_sin", "dow_cos"]
EXTRA_COLS = RICH_COLS + TIME_EXTRA


def outlier_mask(df, daily, threshold=OUTLIER_LOG_THRESHOLD, seed=0):
    """Two-sided outlier flag for LABELED rows. Expected rate = out-of-fold gradient-boosting prediction (each row is predicted
    by a model that never saw it, trained without the obvious extremes). Returns (mask, log residual); mask True = outlier."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    idx = df.index; df = df.reset_index(drop=True)
    rpm = (df.posted_rate / df.distance).values
    X = build(df, daily, df.weight.abs().median()); X = X[[c for c in X.columns if c not in TIME_COLS]]
    y = np.log(rpm); ok = (rpm > 1) & (rpm < 4); oof = np.zeros(len(df))
    for a, b in KFold(5, shuffle=True, random_state=seed).split(X):
        a = a[ok[a]]
        oof[b] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0).fit(X.iloc[a], y[a]).predict(X.iloc[b])
    res = y - oof
    return pd.Series(np.abs(res) > threshold, index=idx), pd.Series(res, index=idx)
