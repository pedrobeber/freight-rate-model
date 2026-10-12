"""SVR search on the dev months only (Jan-Aug): kernel, C, gamma, epsilon, target transform,
feature set and scaler.

Stage 1: random search on a small subsample.
Stage 2: the best configurations re-scored on a larger sample.
Stage 3: local refinement around the best one.

Selection uses rolling-origin dev-CV only. The test months (Sep-Oct) are not touched here.

Usage: python src/tune_svr.py N MODE
    all          all feature sets, including the date-dependent ones (raw dev-CV MAPE)
    notime       date-independent feature sets only, selected on raw dev-CV MAPE
    notime_dm    date-independent feature sets only, selected on the day-demeaned MAPE, which is
                 the error left after removing each day's average residual (the market level
                 that no date-independent input can explain). This selects the final config.
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

import config
from search import (
    DATE_FREE_SETS, MODES, ORIGINAL_CONFIG, RESULT_COLUMNS,
    config_from_row, perturb, run_stage, sample_config,
)

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("n_random", type=int, nargs="?", default=200)
    parser.add_argument("mode", choices=MODES, nargs="?", default="notime")
    args = parser.parse_args()

    mode = MODES[args.mode]
    feature_sets, criterion = mode["sets"], mode["criterion"]
    rng = np.random.default_rng(mode["seed"])
    shown = ["kernel", "C", "gamma", "eps_sd", "target", "features", "scaler",
             "devCV_MAPE", "devCV_MAPE_dm", "devCV_MSE"]

    # Stage 1: random search on 6,000-row subsamples.
    configs = [ORIGINAL_CONFIG] + [sample_config(rng, feature_sets) for _ in range(args.n_random)]
    started = time.time()
    stage1 = run_stage(configs, 6000)
    stage1["stage"] = 1
    if args.mode == "notime":
        # Reuse the date-independent configurations already scored by the `all` search.
        previous = pd.read_csv(config.RESULTS_DIR / "svr_tuning_all_sets.csv")
        previous = previous[(previous.stage == 1) & previous.features.isin(DATE_FREE_SETS)]
        stage1 = pd.concat([previous, stage1], ignore_index=True)
    print(f"stage 1: {len(stage1)} configs in {time.time() - started:.0f}s")
    print(stage1.sort_values(criterion).head(10)[shown].round(4).to_string(), flush=True)

    # Stage 2: the best 16 re-scored on 16,000 rows.
    top = stage1.sort_values(criterion).head(16).drop(columns=RESULT_COLUMNS).to_dict("records")
    started = time.time()
    stage2 = run_stage([ORIGINAL_CONFIG] + top, 16000)
    stage2["stage"] = 2
    print(f"stage 2 in {time.time() - started:.0f}s")
    print(stage2.sort_values(criterion).head(8)[shown].round(4).to_string(), flush=True)

    # Stage 3: 30 local variations of the best configuration.
    best = stage2.sort_values(criterion).iloc[0].drop(RESULT_COLUMNS)
    refinements = [perturb(best.to_dict(), rng, feature_sets) for _ in range(30)]
    started = time.time()
    stage3 = run_stage(refinements, 16000)
    stage3["stage"] = 3
    print(f"stage 3 in {time.time() - started:.0f}s")

    results = pd.concat([stage1, stage2, stage3])
    results.to_csv(config.RESULTS_DIR / f"svr_tuning_{mode['tag']}.csv", index=False)

    final = results[results.cap == 16000].sort_values(criterion)
    print(final.head(8)[shown + ["stage"]].round(4).to_string())
    winner = config_from_row(final.iloc[0])
    with open(config.RESULTS_DIR / f"svr_best_config_{mode['tag']}.json", "w") as handle:
        json.dump(winner, handle, indent=1)

    original_scores = stage2[stage2.stage == 2].iloc[0]
    print("original settings at 16k rows:",
          original_scores[["devCV_MAPE", "devCV_MAPE_dm", "devCV_MAE", "devCV_MSE"]].round(4).to_dict())


if __name__ == "__main__":
    main()
