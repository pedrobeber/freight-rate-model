"""Fit the final SVR on all labeled clean rows and write the submission files.

The model settings are in src/config.py.

Outputs (in predictions/):
    validation_predictions.csv   predicted rate for the 12,000 loads to predict
    december-chart-inputs.csv    the fixed December load with its predicted rate
"""
import warnings

import numpy as np
import pandas as pd

import config
import features as F
from evaluation import metrics
from svr_model import SVRRateModel

warnings.filterwarnings("ignore")

# Column values that the December file does not carry. The final model does not use them.
DECEMBER_QUOTE_SIGNAL = 2.05


def city_coordinates(train):
    """First known (latitude, longitude) of every city, as pickup or as delivery."""
    coordinates = {}
    for side in ["pickup", "delivery"]:
        columns = [side, f"{side}_lat", f"{side}_lon"]
        for city, lat, lon in train[columns].drop_duplicates(side).itertuples(index=False):
            coordinates.setdefault(city, (lat, lon))
    return coordinates


def check_on_test_months(clean, daily):
    """Fit on Jan-Aug and score the held-out Sep-Oct months, as a last sanity check."""
    dev = clean[clean.date < config.TEST_START]
    test = clean[clean.date >= config.TEST_START]
    model = SVRRateModel().fit(dev, daily)
    scores = metrics(test.posted_rate.values, model.predict(test))
    print("Sep-Oct check (fit on Jan-Aug):", {k: round(float(v), 3) for k, v in scores.items()})


def write_validation_predictions(model, valid):
    template = pd.read_csv(config.TEMPLATE_FILE)
    valid = valid.copy()
    valid["predicted_rate"] = model.predict(valid).round(2)
    output = template[["load_id"]].merge(valid[["load_id", "predicted_rate"]], on="load_id", how="left")

    assert output.predicted_rate.notna().all()
    assert (output.predicted_rate > 0).all()
    assert len(output) == 12000

    output.to_csv(config.VALIDATION_PREDICTIONS_FILE, index=False)
    return output


def write_december_predictions(model, train):
    december = pd.read_csv(config.DECEMBER_INPUT_FILE, parse_dates=["date"])
    coordinates = city_coordinates(train)

    loads = december.copy()
    for side, city in [("pickup", "Lexington"), ("delivery", "Fort Wayne")]:
        loads[f"{side}_lat"], loads[f"{side}_lon"] = coordinates[city]
    loads["market_index"] = np.nan
    loads["quote_signal"] = DECEMBER_QUOTE_SIGNAL
    loads["posted_rate"] = np.nan

    december["predicted_rate"] = model.predict(loads).round(2)
    december["date"] = december["date"].dt.strftime("%Y-%m-%d")
    december.to_csv(config.DECEMBER_PREDICTIONS_FILE, index=False)
    return december


def main():
    config.PREDICTIONS_DIR.mkdir(exist_ok=True)
    print("settings:", config.SVR_SETTINGS)

    train = F.load(config.TRAIN_FILE)
    valid = F.load(config.VALIDATION_FILE)
    daily = F.daily_market(train, valid)

    is_outlier, _ = F.outlier_mask(train, daily)
    print("outliers removed:", int(is_outlier.sum()), "of", len(train))
    clean = train[~is_outlier]

    check_on_test_months(clean, daily)

    model = SVRRateModel().fit(clean, daily)
    print("final model fitted on", len(clean), "rows")

    validation = write_validation_predictions(model, valid)
    december = write_december_predictions(model, train)
    print(validation.predicted_rate.describe().round(1).to_dict())
    print(december.predicted_rate.describe().round(2).to_dict())


if __name__ == "__main__":
    main()
