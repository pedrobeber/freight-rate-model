"""SVR kernel comparison. Each kernel gets its own random search of C / gamma / epsilon (/ coef0) on DEV (Jan-Aug) only, with target,
scaler and inputs fixed to the final configuration so that only the kernel changes. Stage 1: 6,000-row subsample; stage 2: best 4 per kernel on
16,000 rows. The winner of each kernel is then fitted on all DEV rows and scored ONCE on the held-out Sep-Oct months.
Selection criterion: day-demeaned dev-CV MAPE (see the notebook). Usage: PYTHONPATH=src python src/svr_kernels.py [N_per_kernel]"""
import sys, json, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
import svr_tuning as T
from svr_model import SVRRateModel
from evaluation import metrics, mse_log
warnings.filterwarnings("ignore")

KERNELS = [("linear", None), ("poly", 2), ("poly", 3), ("rbf", None), ("sigmoid", None)]
FIXED = {k: v for k, v in json.load(open("scorer_results/svr_final_config.json")).items() if k in ("target", "features", "scaler")}


def sample(kernel, degree, rng):
    c = dict(kernel=kernel, degree=degree, **FIXED)
    c["C"] = float(np.exp(rng.uniform(np.log(.3), np.log(100)))); c["eps_sd"] = float(np.exp(rng.uniform(np.log(.05), np.log(.5))))
    c["gamma"] = float(np.exp(rng.uniform(np.log(.003), np.log(.3)))) if kernel != "linear" else None
    c["coef0"] = float(rng.choice([0, 1])) if kernel in ("poly", "sigmoid") else None
    return c


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24; rng = np.random.default_rng(5)
    cfgs = [sample(k, d, rng) for k, d in KERNELS for _ in range(n)]
    s1 = pd.DataFrame(Parallel(n_jobs=10)(delayed(T.timed)(c, 6000) for c in cfgs)); s1["stage"] = 1
    s1["name"] = np.where(s1.kernel == "poly", "poly deg " + s1.degree.astype(str).str[0], s1.kernel)
    drop = ["cap", "devCV_MAPE", "devCV_MAE", "devCV_MSE", "devCV_MAPE_dm", "secs", "stage", "name"]
    top = [r.drop(drop).to_dict() for _, g in s1.groupby("name") for _, r in g.sort_values("devCV_MAPE_dm").head(4).iterrows()]
    s2 = pd.DataFrame(Parallel(n_jobs=10)(delayed(T.timed)(c, 16000) for c in top)); s2["stage"] = 2
    s2["name"] = np.where(s2.kernel == "poly", "poly deg " + s2.degree.astype(int).astype(str), s2.kernel)
    pd.concat([s1, s2]).to_csv("scorer_results/svr_kernel_search.csv", index=False)
    best = s2.sort_values("devCV_MAPE_dm").groupby("name").head(1)
    cl = T.cl; dev, test = cl[cl.date < "2025-09-01"], cl[cl.date >= "2025-09-01"]

    def run(r):
        cfg = {k: (None if pd.isna(r[k]) else r[k]) for k in ["kernel", "C", "gamma", "eps_sd", "degree", "coef0", "target", "features", "scaler"]}
        if cfg["degree"] is not None: cfg["degree"] = int(cfg["degree"])
        import time; t0 = time.time(); m = SVRRateModel(cfg, use_dow=False).fit(dev, T.daily); fit_s = time.time() - t0
        p = m.predict(test); y = test.posted_rate.values
        return dict(kernel=r["name"], C=r.C, gamma=r.gamma, eps_sd=r.eps_sd, coef0=r.coef0, devCV_MAPE=r.devCV_MAPE, devCV_MAPE_dm=r.devCV_MAPE_dm,
                    test_MSE_log=mse_log(y, p), fit_seconds=fit_s, **{"test_" + k: v for k, v in metrics(y, p).items()})
    out = pd.DataFrame(Parallel(n_jobs=5)(delayed(run)(r) for _, r in best.iterrows())); out.to_csv("scorer_results/svr_kernel_test.csv", index=False)
    print(out.round(4).to_string(index=False))
