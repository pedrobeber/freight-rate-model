"""RBF support vector regression of the rate per mile, with an optional day-of-week adjustment.

The model predicts sqrt(rate per mile) from distance, weight, equipment, city coordinates and
circuity. The prediction is multiplied by the distance to get the rate in dollars.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

import config
import features as F

MIN_SQRT_RATE_PER_MILE = 0.45
MIN_RATE = 1


class SVRRateModel:
    def __init__(self, settings=None, use_dow=True):
        self.settings = dict(settings or config.SVR_SETTINGS)
        self.use_dow = use_dow

    def _inputs(self, df):
        return F.model_inputs(df.reset_index(drop=True), self.daily, self.weight_median)

    def fit(self, df, daily):
        self.daily = daily
        self.weight_median = df.weight.abs().median()
        df = df.reset_index(drop=True)
        X = self._inputs(df)

        target = np.sqrt(df.posted_rate.values / df.distance.values)
        self.target_mean, self.target_sd = target.mean(), target.std()
        standardized = (target - self.target_mean) / self.target_sd

        svr = SVR(
            kernel="rbf",
            C=self.settings["C"],
            gamma=self.settings["gamma"],
            epsilon=self.settings["epsilon_sd"] * self.target_sd,
            max_iter=200000,
            cache_size=500,
        )
        self.model = make_pipeline(RobustScaler(), svr).fit(X, standardized)

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
        sqrt_rate_per_mile = self.model.predict(X) * self.target_sd + self.target_mean
        rate = np.clip(sqrt_rate_per_mile, MIN_SQRT_RATE_PER_MILE, None) ** 2 * distance
        return np.clip(rate, MIN_RATE, None)

    def predict(self, df, use_dow=None):
        df = df.reset_index(drop=True)
        rate = self._rate(self._inputs(df), df.distance.values)
        apply_dow = self.use_dow if use_dow is None else use_dow
        if apply_dow:
            rate = rate * np.exp(self.dow.reindex(df.date.dt.dayofweek).values)
        return rate
