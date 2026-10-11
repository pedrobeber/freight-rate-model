"""Score the selected SVR configurations once on the held-out Sep-Oct months (fit on all DEV rows, Jan-Aug).
Candidates: original settings; dev-CV winner over ALL feature sets; dev-CV winner over date-independent sets (selected on raw
MAPE, and on the day-demeaned MAPE); and the all-sets winner with its date-dependent columns replaced by the base columns."""
import json, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
import features as F
from svr_model import SVRRateModel, ORIGINAL
from evaluation import metrics, mse_log
warnings.filterwarnings("ignore")
tr, va = F.load("train-test.csv"), F.load("validation.csv"); daily = F.daily_market(tr, va)
mask, _ = F.outlier_mask(tr, daily); cl = tr[~mask]
dev, test = cl[cl.date < "2025-09-01"], cl[cl.date >= "2025-09-01"]
load = lambda tag: json.load(open(f"scorer_results/svr_best_config_{tag}.json"))
cands = {"original (rbf, C=1, gamma=0.02, log rate/mile, base, standard)": ORIGINAL,
         "dev-CV winner, all feature sets (raw MAPE)": load("all_sets"),
         "dev-CV winner, date-independent sets (raw MAPE)": load("notime_rawcv"),
         "dev-CV winner, date-independent sets (day-demeaned MAPE)": load("notime")}
cands["ablation: all-sets winner with base columns"] = dict(cands["dev-CV winner, all feature sets (raw MAPE)"], features="base")


def run(name, cfg):
    m = SVRRateModel(cfg, use_dow=False).fit(dev, daily); p = m.predict(test)
    return dict(candidate=name, kernel=cfg["kernel"], features=cfg["features"], target=cfg["target"], scaler=cfg["scaler"], **metrics(test.posted_rate.values, p), MSE_log=mse_log(test.posted_rate.values, p))


if __name__ == "__main__":
    out = pd.DataFrame(Parallel(n_jobs=5)(delayed(run)(n, c) for n, c in cands.items())); out.to_csv("scorer_results/svr_test_check.csv", index=False)
    print(out.round(4).to_string(index=False))
