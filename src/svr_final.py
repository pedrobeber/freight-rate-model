"""Fit the final SVR on ALL labeled clean rows and write the submission files.
Config: scorer_results/svr_final_config.json (dev-CV winner of src/svr_tuning.py notime_dm, see the notebook)."""
import json, warnings, numpy as np, pandas as pd
import features as F
from svr_model import SVRRateModel
from evaluation import metrics
warnings.filterwarnings("ignore")
cfg = json.load(open("scorer_results/svr_final_config.json")); print("config:", cfg)
tr, va = F.load("train-test.csv"), F.load("validation.csv"); daily = F.daily_market(tr, va)
mask, _ = F.outlier_mask(tr, daily); print("outliers removed:", int(mask.sum()), "of", len(tr)); clean = tr[~mask]

# one honest check: fit on Jan-Aug, score the held-out Sep-Oct months (clean rows)
dev, test = clean[clean.date < "2025-09-01"], clean[clean.date >= "2025-09-01"]
chk = SVRRateModel(cfg).fit(dev, daily); print("Sep-Oct check (fit on Jan-Aug):", {k: round(v, 3) for k, v in metrics(test.posted_rate.values, chk.predict(test)).items()})

final = SVRRateModel(cfg).fit(clean, daily); print("final model fitted on", len(clean), "rows")
tmpl = pd.read_csv("validation-predictions-template.csv")
va["predicted_rate"] = final.predict(va).round(2)
out = tmpl[["load_id"]].merge(va[["load_id", "predicted_rate"]], on="load_id", how="left")
assert out.predicted_rate.notna().all() and (out.predicted_rate > 0).all() and len(out) == 12000
out.to_csv("validation_predictions.csv", index=False)

dec = pd.read_csv("december-chart-inputs.csv", parse_dates=["date"])
city = {}
for side in ["pickup", "delivery"]:
    for c, la, lo in tr[[side, f"{side}_lat", f"{side}_lon"]].drop_duplicates(side).itertuples(index=False): city.setdefault(c, (la, lo))
d = dec.copy()
for side, c in [("pickup", "Lexington"), ("delivery", "Fort Wayne")]: d[f"{side}_lat"], d[f"{side}_lon"] = city[c]
d["market_index"], d["quote_signal"], d["posted_rate"] = np.nan, 2.05, np.nan
dec["predicted_rate"] = final.predict(d).round(2); dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
dec.to_csv("december-chart-inputs.csv", index=False)
print(out.predicted_rate.describe().round(1).to_dict()); print(dec.predicted_rate.describe().round(2).to_dict())
