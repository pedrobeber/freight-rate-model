"""Cleaning and feature engineering shared by the experiments and the final model."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

EQUIP = ["Dry Van", "Reefer", "Flatbed"]

# Columns that depend on time or on the market. They are built for the date experiments
# but are not fed to the final model.
TIME_COLS = [
    "mi_day", "qs_day", "mi_day_7", "mi_chg_7", "market_index", "dom", "dow",
    "is_weekend", "market_missing", "quote_signal",
]
DAILY_COLS = ["mi_day", "qs_day", "mi_day_7", "mi_chg_7"]
COORDINATE_COLS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]

# A load is an outlier when |log(actual / expected)| exceeds this threshold.
# Normal loads have a standard deviation of 0.036, outliers start at 0.77.
OUTLIER_LOG_THRESHOLD = 0.5

EARTH_DIAMETER_MILES = 7918

# Inputs of the final model.
MODEL_COLS = [
    "distance", "log_dist", "weight", "eq_DryVan", "eq_Reefer", "eq_Flatbed",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "circuity",
]


def load(path):
    return pd.read_csv(path, parse_dates=["date"])


def equipment_column(equipment):
    return "eq_" + equipment.replace(" ", "")


def daily_market(*frames):
    """Per-date mean of market_index and quote_signal over all feature rows (no labels).

    It fills missing market_index values and gives a smoothed time signal.
    """
    columns = ["date", "market_index", "quote_signal"]
    all_rows = pd.concat([frame[columns] for frame in frames])
    daily = all_rows.groupby("date").agg(
        mi_day=("market_index", "mean"),
        qs_day=("quote_signal", "mean"),
    )
    daily["mi_day_7"] = daily.mi_day.rolling(7, min_periods=1, center=True).mean()
    daily["mi_chg_7"] = daily.mi_day - daily.mi_day.shift(7).bfill()
    return daily


def haversine(lat1, lon1, lat2, lon2):
    """Straight-line distance in miles between two coordinates."""
    radians = np.pi / 180
    a = (
        np.sin((lat2 - lat1) * radians / 2) ** 2
        + np.cos(lat1 * radians) * np.cos(lat2 * radians) * np.sin((lon2 - lon1) * radians / 2) ** 2
    )
    return EARTH_DIAMETER_MILES * np.arcsin(np.sqrt(a))


def build(df, daily, weight_median):
    """All candidate inputs of a load table, with the data-quality fixes applied."""
    X = pd.DataFrame(index=df.index)
    X["distance"] = df.distance
    X["log_dist"] = np.log(df.distance)

    # Weight: negative values are sign flips, missing values get the median plus a flag.
    X["weight_missing"] = df.weight.isna().astype(int)
    X["weight"] = df.weight.abs().fillna(weight_median)

    for equipment in EQUIP:
        X[equipment_column(equipment)] = (df.equipment == equipment).astype(int)

    # Market columns: a missing market_index is filled with the average of the same date.
    day = daily.reindex(df.date).reset_index(drop=True).set_index(df.index)
    X["market_index"] = df.market_index.fillna(day.mi_day)
    X["market_missing"] = df.market_index.isna().astype(int)
    X["quote_signal"] = df.quote_signal
    X[DAILY_COLS] = day[DAILY_COLS]

    X["dow"] = df.date.dt.dayofweek
    X["is_weekend"] = (X.dow >= 5).astype(int)
    X["dom"] = df.date.dt.day

    # Geography is described by coordinates, so cities unseen in training still work.
    for column in COORDINATE_COLS:
        X[column] = df[column]
    X["dlat"] = df.delivery_lat - df.pickup_lat
    X["dlon"] = df.delivery_lon - df.pickup_lon
    X["bearing"] = np.arctan2(X.dlon, X.dlat)
    straight_line = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    X["circuity"] = df.distance / straight_line.clip(lower=1)
    X["weight_x_dist"] = X.weight * df.distance / 1e6
    return X


def model_inputs(df, daily, weight_median):
    """The inputs of the final model: distance, weight, equipment, coordinates and circuity."""
    return build(df, daily, weight_median)[MODEL_COLS]


def outlier_mask(df, daily, threshold=OUTLIER_LOG_THRESHOLD, seed=0):
    """Two-sided outlier flag for labeled rows.

    The expected rate of each row is an out-of-fold gradient boosting prediction, so every
    row is predicted by a model that never saw it. The models are trained without the
    obvious extremes. Returns (mask, log residual), where mask is True for outliers.
    """
    original_index = df.index
    df = df.reset_index(drop=True)
    rate_per_mile = (df.posted_rate / df.distance).values

    X = build(df, daily, df.weight.abs().median())
    X = X[[c for c in X.columns if c not in TIME_COLS]]
    log_target = np.log(rate_per_mile)
    plausible = (rate_per_mile > 1) & (rate_per_mile < 4)

    expected = np.zeros(len(df))
    for fit_idx, score_idx in KFold(5, shuffle=True, random_state=seed).split(X):
        fit_idx = fit_idx[plausible[fit_idx]]
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0)
        model.fit(X.iloc[fit_idx], log_target[fit_idx])
        expected[score_idx] = model.predict(X.iloc[score_idx])

    residual = log_target - expected
    return (
        pd.Series(np.abs(residual) > threshold, index=original_index),
        pd.Series(residual, index=original_index),
    )
