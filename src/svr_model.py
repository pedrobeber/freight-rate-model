"""SVR rate model: configurable kernel / target transform / scaler / feature set, plus a day-of-week effect.

cfg keys: kernel, C, gamma, eps_sd (epsilon in standard deviations of the transformed target), degree, coef0,
          target (log_rpm | log_rate | rpm | sqrt_rpm), features (base | lean | rich | time | rich+time), scaler.
"""
import numpy as np, pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, PowerTransformer
import features as F

SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler,
           "quantile": lambda: QuantileTransformer(n_quantiles=200, output_distribution="normal"), "power": PowerTransformer}
# target transforms: (rate, distance) -> target ,  (prediction, distance) -> rate
TARGETS = {"log_rpm": (lambda r, d: np.log(r / d), lambda p, d: np.exp(p) * d),
           "log_rate": (lambda r, d: np.log(r), lambda p, d: np.exp(p)),
           "rpm": (lambda r, d: r / d, lambda p, d: np.clip(p, .2, None) * d),
           "sqrt_rpm": (lambda r, d: np.sqrt(r / d), lambda p, d: np.clip(p, .45, None) ** 2 * d)}
ORIGINAL = dict(kernel="rbf", C=1.0, gamma=0.02, eps_sd=0.13, degree=None, coef0=None, target="log_rpm", features="base", scaler="standard")


def make_svr(cfg, y_sd):
    kw = dict(kernel=cfg["kernel"], C=cfg["C"], epsilon=cfg["eps_sd"] * y_sd, max_iter=200000, cache_size=500)
    if cfg["kernel"] != "linear": kw["gamma"] = cfg["gamma"]
    if cfg["kernel"] == "poly": kw.update(degree=int(cfg["degree"]), coef0=float(cfg["coef0"]))
    if cfg["kernel"] == "sigmoid": kw.update(coef0=float(cfg["coef0"]))
    return make_pipeline(SCALERS[cfg["scaler"]](), SVR(**kw))


class SVRRateModel:
    def __init__(self, cfg=None, use_dow=True, max_rows=None, seed=0):
        self.cfg, self.use_dow, self.max_rows, self.seed = dict(cfg or ORIGINAL), use_dow, max_rows, seed

    def _X(self, df):
        X = F.add_extras(F.build(df.reset_index(drop=True), self.daily, self.wmed))
        return F.feature_set(X, self.cfg["features"])

    def fit(self, df, daily):
        self.daily, self.wmed = daily, df.weight.abs().median()
        df = df.reset_index(drop=True); X = self._X(df); self.cols = list(X.columns)
        fwd, self.inv = TARGETS[self.cfg["target"]]
        y = fwd(df.posted_rate.values, df.distance.values); self.mu, self.sd = y.mean(), y.std()
        idx = np.arange(len(X))
        if self.max_rows and len(X) > self.max_rows: idx = np.random.default_rng(self.seed).choice(idx, self.max_rows, replace=False)
        self.model = make_svr(self.cfg, self.sd).fit(X.iloc[idx], (y[idx] - self.mu) / self.sd)
        p = self._rate(X, df.distance.values)
        res = pd.Series(np.log(df.posted_rate.values) - np.log(p), index=pd.DatetimeIndex(df.date.values)).groupby(level=0).mean()
        dev = res - res.rolling(15, center=True, min_periods=5).mean()
        self.dow = dev.groupby(res.index.dayofweek).mean()
        return self

    def _rate(self, X, dist):
        return np.clip(self.inv(self.model.predict(X) * self.sd + self.mu, dist), 1, None)

    def predict(self, df, use_dow=None):
        df = df.reset_index(drop=True); p = self._rate(self._X(df), df.distance.values)
        if self.use_dow if use_dow is None else use_dow:
            p = p * np.exp(self.dow.reindex(df.date.dt.dayofweek).values)
        return p
