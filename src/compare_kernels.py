"""SVR kernel comparison.

Each kernel gets its own random search of C, gamma and epsilon (and coef0) on the dev months
(Jan-Aug), with the target, scaler and inputs fixed to the final configuration so that only the
kernel changes.

Stage 1: random search on 6,000-row subsamples.
Stage 2: the best 4 configurations per kernel on 16,000 rows.
Then the winner of each kernel is fitted on all dev rows and scored once on the held-out
Sep-Oct months. The selection criterion is the day-demeaned dev-CV MAPE.

Usage: python src/compare_kernels.py [N_PER_KERNEL]
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
import search
from evaluation import metrics, mse_log
from svr_model import SVRRateModel

warnings.filterwarnings("ignore")

KERNELS = [("linear", None), ("poly", 2), ("poly", 3), ("rbf", None), ("sigmoid", None)]
FIXED_KEYS = ("target", "features", "scaler")
SEED = 5


def fixed_settings():
    final_config = json.load(open(config.FINAL_CONFIG_FILE))
    return {key: value for key, value in final_config.items() if key in FIXED_KEYS}


def sample_config(kernel, degree, rng, fixed):
    sampled = dict(kernel=kernel, degree=degree, **fixed)
    sampled["C"] = search.log_uniform(rng, 0.3, 100)
    sampled["eps_sd"] = search.log_uniform(rng, 0.05, 0.5)
    sampled["gamma"] = search.log_uniform(rng, 0.003, 0.3) if kernel != "linear" else None
    sampled["coef0"] = float(rng.choice([0, 1])) if kernel in ("poly", "sigmoid") else None
    return sampled


def kernel_name(results):
    degree = results.degree.fillna(0).astype(int).astype(str)
    return np.where(results.kernel == "poly", "poly deg " + degree, results.kernel)


def score_on_test_months(row):
    """Fit one kernel winner on all dev rows and score it on Sep-Oct."""
    dev, test, daily = search.clean_split()

    started = time.time()
    model = SVRRateModel(search.config_from_row(row), use_dow=False).fit(dev, daily)
    fit_seconds = time.time() - started

    predicted = model.predict(test)
    actual = test.posted_rate.values
    return dict(
        kernel=row["name"], C=row.C, gamma=row.gamma, eps_sd=row.eps_sd, coef0=row.coef0,
        devCV_MAPE=row.devCV_MAPE, devCV_MAPE_dm=row.devCV_MAPE_dm,
        test_MSE_log=mse_log(actual, predicted), fit_seconds=fit_seconds,
        **{"test_" + name: value for name, value in metrics(actual, predicted).items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("n_per_kernel", type=int, nargs="?", default=24)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    fixed = fixed_settings()
    configs = [
        sample_config(kernel, degree, rng, fixed)
        for kernel, degree in KERNELS
        for _ in range(args.n_per_kernel)
    ]

    stage1 = search.run_stage(configs, 6000)
    stage1["stage"] = 1
    stage1["name"] = kernel_name(stage1)

    drop = search.RESULT_COLUMNS + ["name"]
    top = []
    for _, group in stage1.groupby("name"):
        best_four = group.sort_values("devCV_MAPE_dm").head(4)
        top += [row.drop(drop).to_dict() for _, row in best_four.iterrows()]

    stage2 = search.run_stage(top, 16000)
    stage2["stage"] = 2
    stage2["name"] = kernel_name(stage2)
    pd.concat([stage1, stage2]).to_csv(config.RESULTS_DIR / "svr_kernel_search.csv", index=False)

    winners = stage2.sort_values("devCV_MAPE_dm").groupby("name").head(1)
    rows = [row for _, row in winners.iterrows()]
    scores = pd.DataFrame(Parallel(n_jobs=5)(delayed(score_on_test_months)(row) for row in rows))
    scores.to_csv(config.RESULTS_DIR / "svr_kernel_test.csv", index=False)
    print(scores.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
