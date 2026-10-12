"""Score the selected SVR configurations once on the held-out Sep-Oct months.

Each candidate is fitted on all dev rows (Jan-Aug). The candidates are:
    - the original settings
    - the dev-CV winner over all feature sets
    - the dev-CV winner over date-independent sets (selected on raw MAPE, and on day-demeaned MAPE)
    - the all-sets winner with its date-dependent columns replaced by the base columns (ablation)

Run the three `tune_svr.py` modes first, since the winners are read from scorer_results/.
"""
import json
import warnings

import pandas as pd
from joblib import Parallel, delayed

import config
import search
from evaluation import metrics, mse_log
from svr_model import ORIGINAL, SVRRateModel

warnings.filterwarnings("ignore")
config.export_src_path()


def load_best_config(tag):
    return json.load(open(config.RESULTS_DIR / f"svr_best_config_{tag}.json"))


def candidates():
    all_sets_winner = load_best_config("all_sets")
    return {
        "original (rbf, C=1, gamma=0.02, log rate/mile, base, standard)": ORIGINAL,
        "dev-CV winner, all feature sets (raw MAPE)": all_sets_winner,
        "dev-CV winner, date-independent sets (raw MAPE)": load_best_config("notime_rawcv"),
        "dev-CV winner, date-independent sets (day-demeaned MAPE)": load_best_config("notime"),
        "ablation: all-sets winner with base columns": dict(all_sets_winner, features="base"),
    }


def score_candidate(name, model_config):
    dev, test, daily = search.clean_split()
    model = SVRRateModel(model_config, use_dow=False).fit(dev, daily)
    predicted = model.predict(test)
    actual = test.posted_rate.values
    return dict(
        candidate=name,
        kernel=model_config["kernel"],
        features=model_config["features"],
        target=model_config["target"],
        scaler=model_config["scaler"],
        **metrics(actual, predicted),
        MSE_log=mse_log(actual, predicted),
    )


def main():
    jobs = (delayed(score_candidate)(name, cfg) for name, cfg in candidates().items())
    results = pd.DataFrame(Parallel(n_jobs=5)(jobs))
    results.to_csv(config.RESULTS_DIR / "svr_test_check.csv", index=False)
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
