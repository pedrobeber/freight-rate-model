"""Paths and constants shared by all scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "scorer_results"
PREDICTIONS_DIR = ROOT / "predictions"

TRAIN_FILE = DATA_DIR / "train-test.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEMPLATE_FILE = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_INPUT_FILE = DATA_DIR / "december-chart-inputs.orig.csv"

VALIDATION_PREDICTIONS_FILE = PREDICTIONS_DIR / "validation_predictions.csv"
DECEMBER_PREDICTIONS_FILE = PREDICTIONS_DIR / "december-chart-inputs.csv"
FINAL_CONFIG_FILE = RESULTS_DIR / "svr_final_config.json"

# Time split: Jan-Aug is used to fit and select, Sep-Oct is held out.
TEST_START = "2025-09-01"

# Rolling-origin folds on Jan-Aug: each fold trains on everything before its start date.
ROLLING_FOLDS = [
    ("2025-03-01", "2025-04-30"),
    ("2025-05-01", "2025-06-30"),
    ("2025-07-01", "2025-08-31"),
]


def export_src_path():
    """Make src/ importable in joblib worker processes, which do not inherit sys.path."""
    paths = [str(SRC_DIR), os.environ.get("PYTHONPATH", "")]
    os.environ["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
