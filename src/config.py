"""Paths and constants shared by all scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "scorer_results"
PREDICTIONS_DIR = ROOT / "predictions"

TRAIN_FILE = DATA_DIR / "train-test.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEMPLATE_FILE = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_INPUT_FILE = DATA_DIR / "december-chart-inputs.orig.csv"

VALIDATION_PREDICTIONS_FILE = PREDICTIONS_DIR / "validation_predictions.csv"
DECEMBER_PREDICTIONS_FILE = PREDICTIONS_DIR / "december-chart-inputs.csv"

# Time split: Jan-Aug is used to fit and compare, Sep-Oct is held out.
TEST_START = "2025-09-01"

# RBF support vector regression. Epsilon is given in standard deviations of the target.
SVR_SETTINGS = dict(C=45.4862082634092, gamma=0.03125996297001373, epsilon_sd=0.19872959276495822)
