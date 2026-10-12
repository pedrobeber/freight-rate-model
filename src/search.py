"""Helpers shared by the SVR search scripts: dev data, rolling-origin scoring and sampling.

They live in a module of their own, not in a script, so that joblib worker processes can import them.
"""
import time
import warnings
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
import features as F
from svr_model import SCALERS, TARGETS, make_svr

warnings.filterwarnings("ignore")
config.export_src_path()

DATE_FREE_SETS = ["base", "lean", "rich"]
ALL_SETS = DATE_FREE_SETS + ["time", "rich+time"]
CONFIG_KEYS = ["kernel", "C", "gamma", "eps_sd", "degree", "coef0", "target", "features", "scaler"]
RESULT_COLUMNS = ["cap", "devCV_MAPE", "devCV_MAE", "devCV_MSE", "devCV_MAPE_dm", "secs", "stage"]
N_JOBS = 10

ORIGINAL_CONFIG = dict(
    kernel="rbf", C=1.0, gamma=0.02, eps_sd=0.13, degree=None, coef0=None,
    target="log_rpm", features="base", scaler="standard",
)

# Per mode: feature sets searched, selection criterion, random seed, output tag.
MODES = {
    "all": dict(sets=ALL_SETS, criterion="devCV_MAPE", seed=42, tag="all_sets"),
    "notime": dict(sets=DATE_FREE_SETS, criterion="devCV_MAPE", seed=7, tag="notime_rawcv"),
    "notime_dm": dict(sets=DATE_FREE_SETS, criterion="devCV_MAPE_dm", seed=11, tag="notime"),
}


@lru_cache(maxsize=1)
def dev_context():
    """Clean dev rows and their candidate inputs. Loaded once per process."""
    train = F.load(config.TRAIN_FILE)
    valid = F.load(config.VALIDATION_FILE)
    daily = F.daily_market(train, valid)
    is_outlier, _ = F.outlier_mask(train, daily)

    clean = train[~is_outlier].reset_index(drop=True)
    clean["rpm"] = clean.posted_rate / clean.distance
    dev = clean[clean.date < config.TEST_START].reset_index(drop=True)
    inputs = F.add_extras(F.build(dev, daily, dev.weight.abs().median()))
    return SimpleNamespace(
        clean=clean,
        dev=dev,
        daily=daily,
        inputs=inputs,
        distance=dev.distance.values,
        rate=dev.posted_rate.values,
        dates=dev.date.values,
    )


def fold_rows(dates, start, end, cap):
    """Rows to train on (everything before start, capped) and rows to score (start to end)."""
    train_rows = np.where(dates < np.datetime64(start))[0]
    score_rows = np.where((dates >= np.datetime64(start)) & (dates <= np.datetime64(end)))[0]
    if len(train_rows) > cap:
        train_rows = np.random.default_rng(0).choice(train_rows, cap, replace=False)
    return train_rows, score_rows


def score_fold(model_config, features, ctx, train_rows, score_rows):
    """MAPE, MAE, log MSE and day-demeaned MAPE of one fold."""
    to_target, to_rate = TARGETS[model_config["target"]]
    y = to_target(ctx.rate[train_rows], ctx.distance[train_rows])
    mean, sd = y.mean(), y.std()
    model = make_svr(model_config, sd).fit(features[train_rows], (y - mean) / sd)

    prediction = model.predict(features[score_rows]) * sd + mean
    predicted = np.clip(to_rate(prediction, ctx.distance[score_rows]), 1, None)
    actual = ctx.rate[score_rows]
    error = predicted - actual

    # Day-demeaned error: remove the market level of each day from the residuals.
    log_ratio = pd.Series(np.log(actual / predicted))
    day_level = log_ratio.groupby(ctx.dates[score_rows]).transform("mean").values
    error_demeaned = predicted * np.exp(day_level) - actual

    return (
        np.mean(np.abs(error) / actual) * 100,
        np.mean(np.abs(error)),
        np.mean((np.log(predicted) - np.log(actual)) ** 2),
        np.mean(np.abs(error_demeaned) / actual) * 100,
    )


def evaluate(model_config, cap):
    """Rolling-origin dev-CV of one configuration, training on at most `cap` rows per fold."""
    ctx = dev_context()
    features = F.feature_set(ctx.inputs, model_config["features"]).values
    folds = []
    for start, end in config.ROLLING_FOLDS:
        train_rows, score_rows = fold_rows(ctx.dates, start, end, cap)
        folds.append(score_fold(model_config, features, ctx, train_rows, score_rows))

    mape, mae, mse, mape_dm = np.mean(folds, axis=0)
    return dict(
        model_config, cap=cap,
        devCV_MAPE=mape, devCV_MAE=mae, devCV_MSE=mse, devCV_MAPE_dm=mape_dm,
    )


def timed(model_config, cap):
    start = time.time()
    result = evaluate(model_config, cap)
    result["secs"] = time.time() - start
    return result


def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_config(rng, feature_sets):
    kernel = rng.choice(["rbf", "rbf", "rbf", "rbf", "poly", "poly", "linear", "sigmoid"])
    sampled = dict(kernel=kernel)
    sampled["C"] = log_uniform(rng, 0.3, 60)
    sampled["eps_sd"] = log_uniform(rng, 0.03, 0.6)
    uses_gamma = kernel in ("rbf", "poly", "sigmoid")
    sampled["gamma"] = log_uniform(rng, 0.003, 0.4) if uses_gamma else None
    sampled["degree"] = int(rng.choice([2, 3])) if kernel == "poly" else None
    sampled["coef0"] = float(rng.choice([0, 1])) if kernel in ("poly", "sigmoid") else None
    sampled["target"] = str(rng.choice(list(TARGETS)))
    sampled["features"] = str(rng.choice(feature_sets))
    sampled["scaler"] = str(rng.choice(list(SCALERS)))
    return sampled


def perturb(model_config, rng, feature_sets):
    """A local variation of a configuration, for the refinement stage."""
    varied = dict(model_config)
    varied["C"] *= float(np.exp(rng.normal(0, 0.5)))
    varied["eps_sd"] *= float(np.exp(rng.normal(0, 0.3)))
    if varied["gamma"] is not None and not pd.isna(varied["gamma"]):
        varied["gamma"] *= float(np.exp(rng.normal(0, 0.5)))
    if rng.random() < 0.25:
        varied["scaler"] = str(rng.choice(list(SCALERS)))
    if rng.random() < 0.25:
        varied["features"] = str(rng.choice(feature_sets))
    return varied


def run_stage(configs, cap):
    return pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(timed)(c, cap) for c in configs))


def config_from_row(row):
    """Back from a results row to a plain configuration dictionary."""
    result = {}
    for key in CONFIG_KEYS:
        value = row[key]
        if pd.isna(value):
            result[key] = None
        else:
            result[key] = value.item() if hasattr(value, "item") else value
    if result["degree"] is not None:
        result["degree"] = int(result["degree"])
    return result


@lru_cache(maxsize=1)
def clean_split():
    """Clean dev rows (Jan-Aug), clean test rows (Sep-Oct) and the daily market table."""
    ctx = dev_context()
    test = ctx.clean[ctx.clean.date >= config.TEST_START]
    return ctx.dev, test, ctx.daily
