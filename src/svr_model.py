"""SVR rate model with a configurable kernel, target transform, scaler and feature set.

A day-of-week effect can be added on top of the prediction.

Configuration keys:
    kernel, C, gamma, eps_sd (epsilon in standard deviations of the transformed target),
    degree, coef0, target (log_rpm | log_rate | rpm | sqrt_rpm),
    features (base | lean | rich | time | rich+time), scaler.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler,
)
from sklearn.svm import SVR

import features as F

SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "quantile": lambda: QuantileTransformer(n_quantiles=200, output_distribution="normal"),
    "power": PowerTransformer,
}


# Target transforms. Each entry is a pair of functions:
# (rate, distance) -> target and (prediction, distance) -> rate.
def _log_rpm(rate, distance):
    return np.log(rate / distance)


def _log_rpm_inverse(prediction, distance):
    return np.exp(prediction) * distance


def _log_rate(rate, distance):
    return np.log(rate)


def _log_rate_inverse(prediction, distance):
    return np.exp(prediction)


def _rpm(rate, distance):
    return rate / distance


def _rpm_inverse(prediction, distance):
    return np.clip(prediction, 0.2, None) * distance


def _sqrt_rpm(rate, distance):
    return np.sqrt(rate / distance)


def _sqrt_rpm_inverse(prediction, distance):
    return np.clip(prediction, 0.45, None) ** 2 * distance


TARGETS = {
    "log_rpm": (_log_rpm, _log_rpm_inverse),
    "log_rate": (_log_rate, _log_rate_inverse),
    "rpm": (_rpm, _rpm_inverse),
    "sqrt_rpm": (_sqrt_rpm, _sqrt_rpm_inverse),
}

ORIGINAL = dict(
    kernel="rbf", C=1.0, gamma=0.02, eps_sd=0.13, degree=None, coef0=None,
    target="log_rpm", features="base", scaler="standard",
)


def make_svr(config, target_sd):
    """Scaler and SVR in one pipeline. Epsilon is given in standard deviations of the target."""
    svr_params = dict(
        kernel=config["kernel"],
        C=config["C"],
        epsilon=config["eps_sd"] * target_sd,
        max_iter=200000,
        cache_size=500,
    )
    if config["kernel"] != "linear":
        svr_params["gamma"] = config["gamma"]
    if config["kernel"] == "poly":
        svr_params.update(degree=int(config["degree"]), coef0=float(config["coef0"]))
    if config["kernel"] == "sigmoid":
        svr_params.update(coef0=float(config["coef0"]))
    return make_pipeline(SCALERS[config["scaler"]](), SVR(**svr_params))


class SVRRateModel:
    def __init__(self, config=None, use_dow=True, max_rows=None, seed=0):
        self.config = dict(config or ORIGINAL)
        self.use_dow = use_dow
        self.max_rows = max_rows
        self.seed = seed

    def _inputs(self, df):
        X = F.add_extras(F.build(df.reset_index(drop=True), self.daily, self.weight_median))
        return F.feature_set(X, self.config["features"])

    def fit(self, df, daily):
        self.daily = daily
        self.weight_median = df.weight.abs().median()
        df = df.reset_index(drop=True)
        X = self._inputs(df)
        self.columns = list(X.columns)

        to_target, self.to_rate = TARGETS[self.config["target"]]
        y = to_target(df.posted_rate.values, df.distance.values)
        self.target_mean, self.target_sd = y.mean(), y.std()

        rows = np.arange(len(X))
        if self.max_rows and len(X) > self.max_rows:
            rows = np.random.default_rng(self.seed).choice(rows, self.max_rows, replace=False)
        standardized = (y[rows] - self.target_mean) / self.target_sd
        self.model = make_svr(self.config, self.target_sd).fit(X.iloc[rows], standardized)

        self.dow = self._weekday_effect(df, X)
        return self

    def _weekday_effect(self, df, X):
        """Average weekday residual relative to a 15-day moving average, in log terms."""
        fitted = self._rate(X, df.distance.values)
        daily_residual = pd.Series(
            np.log(df.posted_rate.values) - np.log(fitted),
            index=pd.DatetimeIndex(df.date.values),
        ).groupby(level=0).mean()
        trend = daily_residual.rolling(15, center=True, min_periods=5).mean()
        return (daily_residual - trend).groupby(daily_residual.index.dayofweek).mean()

    def _rate(self, X, distance):
        prediction = self.model.predict(X) * self.target_sd + self.target_mean
        return np.clip(self.to_rate(prediction, distance), 1, None)

    def predict(self, df, use_dow=None):
        df = df.reset_index(drop=True)
        rate = self._rate(self._inputs(df), df.distance.values)
        apply_dow = self.use_dow if use_dow is None else use_dow
        if apply_dow:
            rate = rate * np.exp(self.dow.reindex(df.date.dt.dayofweek).values)
        return rate
