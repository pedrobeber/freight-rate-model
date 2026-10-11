"""Broad SVR search on DEV only (Jan-Aug): kernels, C/gamma/epsilon, target transforms, feature sets, scalers.
Stage 1: random search on a small subsample.  Stage 2: re-score the best on a larger sample.
Stage 3: local refinement around the best.  Selection uses rolling-origin dev-CV MAPE only; TEST (Sep-Oct) is not touched here.
Usage: python src/svr_tuning.py N all      search over all feature sets (including the date-dependent ones)
       python src/svr_tuning.py N notime   date-independent feature sets only, selected on raw dev-CV MAPE (reuses stage-1 of `all`)
       python src/svr_tuning.py N notime_dm   date-independent feature sets only, selected on the day-demeaned dev-CV MAPE: error after
                                           removing each day's average residual (the market level no date-independent input can explain)"""
import sys, time, json, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
import features as F
from svr_model import SCALERS, TARGETS, make_svr
warnings.filterwarnings("ignore")

tr, va = F.load("train-test.csv"), F.load("validation.csv"); daily = F.daily_market(tr, va)
mask, _ = F.outlier_mask(tr, daily); cl = tr[~mask].reset_index(drop=True)
cl["rpm"] = cl.posted_rate / cl.distance
dev = cl[cl.date < "2025-09-01"].reset_index(drop=True)
XE = F.add_extras(F.build(dev, daily, dev.weight.abs().median()))
dist, rate, rpm, dt = dev.distance.values, dev.posted_rate.values, dev.rpm.values, dev.date.values
FOLDS = [("2025-03-01", "2025-04-30"), ("2025-05-01", "2025-06-30"), ("2025-07-01", "2025-08-31")]
NO_TIME_SETS = ["base", "lean", "rich"]                 # inputs that do not depend on the date
ALL_SETS = NO_TIME_SETS + ["time", "rich+time"]          # "time" sets add market level and day of week
FEATURE_SETS = ALL_SETS


def evaluate(cfg, cap):
    fs = F.feature_set(XE, cfg["features"]).values; fwd, inv = TARGETS[cfg["target"]]; out = []
    for lo, hi in FOLDS:
        ia = np.where(dt < np.datetime64(lo))[0]; ib = np.where((dt >= np.datetime64(lo)) & (dt <= np.datetime64(hi)))[0]
        if len(ia) > cap: ia = np.random.default_rng(0).choice(ia, cap, replace=False)
        y = fwd(rate[ia], dist[ia]); mu, sd = y.mean(), y.std()
        m = make_svr(cfg, sd).fit(fs[ia], (y - mu) / sd)
        p = np.clip(inv(m.predict(fs[ib]) * sd + mu, dist[ib]), 1, None)
        e = p - rate[ib]; lr = pd.Series(np.log(rate[ib] / p)); day_mean = lr.groupby(dt[ib]).transform("mean").values   # market level of the day
        e_dm = p * np.exp(day_mean) - rate[ib]
        out.append((np.mean(np.abs(e) / rate[ib]) * 100, np.mean(np.abs(e)), np.mean((np.log(p) - np.log(rate[ib])) ** 2), np.mean(np.abs(e_dm) / rate[ib]) * 100))
    o = np.mean(out, axis=0); return dict(cfg, cap=cap, devCV_MAPE=o[0], devCV_MAE=o[1], devCV_MSE=o[2], devCV_MAPE_dm=o[3])


def timed(cfg, cap):
    t0 = time.time(); r = evaluate(cfg, cap); r["secs"] = time.time() - t0; return r


def sample(rng):
    k = rng.choice(["rbf", "rbf", "rbf", "rbf", "poly", "poly", "linear", "sigmoid"]); c = dict(kernel=k)
    c["C"] = float(np.exp(rng.uniform(np.log(.3), np.log(60)))); c["eps_sd"] = float(np.exp(rng.uniform(np.log(.03), np.log(.6))))
    c["gamma"] = float(np.exp(rng.uniform(np.log(.003), np.log(.4)))) if k in ("rbf", "poly", "sigmoid") else None
    c["degree"] = int(rng.choice([2, 3])) if k == "poly" else None; c["coef0"] = float(rng.choice([0, 1])) if k in ("poly", "sigmoid") else None
    c["target"] = str(rng.choice(list(TARGETS))); c["features"] = str(rng.choice(FEATURE_SETS)); c["scaler"] = str(rng.choice(list(SCALERS)))
    return c


def perturb(cfg, rng):
    c = dict(cfg)
    c["C"] *= float(np.exp(rng.normal(0, .5))); c["eps_sd"] *= float(np.exp(rng.normal(0, .3)))
    if c["gamma"] is not None and not pd.isna(c["gamma"]): c["gamma"] *= float(np.exp(rng.normal(0, .5)))
    if rng.random() < .25: c["scaler"] = str(rng.choice(list(SCALERS)))
    if rng.random() < .25: c["features"] = str(rng.choice(FEATURE_SETS))
    return c


if __name__ == "__main__":
    N1 = int(sys.argv[1]) if len(sys.argv) > 1 else 200; mode = sys.argv[2] if len(sys.argv) > 2 else "notime"; NJ = 10
    FEATURE_SETS = ALL_SETS if mode == "all" else NO_TIME_SETS; CRIT = "devCV_MAPE_dm" if mode == "notime_dm" else "devCV_MAPE"
    rng = np.random.default_rng({"all": 42, "notime": 7, "notime_dm": 11}[mode])
    base = dict(kernel="rbf", C=1.0, gamma=0.02, eps_sd=0.13, degree=None, coef0=None, target="log_rpm", features="base", scaler="standard")
    cfgs = [base] + [sample(rng) for _ in range(N1)]
    t0 = time.time(); s1 = pd.DataFrame(Parallel(n_jobs=NJ)(delayed(timed)(c, 6000) for c in cfgs)); s1["stage"] = 1
    if mode == "notime":                       # add the date-independent configurations already scored by the `all` search
        prev = pd.read_csv("scorer_results/svr_tuning_all_sets.csv"); prev = prev[(prev.stage == 1) & prev.features.isin(NO_TIME_SETS)]
        s1 = pd.concat([prev, s1], ignore_index=True)
    show = ["kernel", "C", "gamma", "eps_sd", "target", "features", "scaler", "devCV_MAPE", "devCV_MAPE_dm", "devCV_MSE"]
    print(f"stage 1: {len(s1)} configs in {time.time()-t0:.0f}s"); print(s1.sort_values(CRIT).head(10)[show].round(4).to_string(), flush=True)
    drop = ["cap", "devCV_MAPE", "devCV_MAE", "devCV_MSE", "devCV_MAPE_dm", "secs", "stage"]
    top = s1.sort_values(CRIT).head(16).drop(columns=drop).to_dict("records")
    t0 = time.time(); s2 = pd.DataFrame(Parallel(n_jobs=NJ)(delayed(timed)(c, 16000) for c in [base] + top)); s2["stage"] = 2
    print(f"stage 2 in {time.time()-t0:.0f}s"); print(s2.sort_values(CRIT).head(8)[show].round(4).to_string(), flush=True)
    best = s2.sort_values(CRIT).iloc[0].drop(drop).to_dict()
    ref = [perturb(best, rng) for _ in range(30)]
    t0 = time.time(); s3 = pd.DataFrame(Parallel(n_jobs=NJ)(delayed(timed)(c, 16000) for c in ref)); s3["stage"] = 3
    print(f"stage 3 in {time.time()-t0:.0f}s")
    tag = {"all": "all_sets", "notime": "notime_rawcv", "notime_dm": "notime"}[mode]
    allr = pd.concat([s1, s2, s3]); allr.to_csv(f"scorer_results/svr_tuning_{tag}.csv", index=False)
    fin = allr[allr.cap == 16000].sort_values(CRIT); print(fin.head(8)[show + ["stage"]].round(4).to_string())
    win = fin.iloc[0]
    cfg = {k: (None if pd.isna(win[k]) else (win[k].item() if hasattr(win[k], "item") else win[k])) for k in ["kernel", "C", "gamma", "eps_sd", "degree", "coef0", "target", "features", "scaler"]}
    if cfg["degree"] is not None: cfg["degree"] = int(cfg["degree"])
    json.dump(cfg, open(f"scorer_results/svr_best_config_{tag}.json", "w"), indent=1)
    print("original settings at 16k rows:", s2[s2.stage == 2].iloc[0][["devCV_MAPE", "devCV_MAPE_dm", "devCV_MAE", "devCV_MSE"]].round(4).to_dict())
