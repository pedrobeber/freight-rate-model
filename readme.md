# Freight rate prediction

Prediction of the posted rate of 12,000 loads for Nov-Dec 2025 (`data/validation.csv`), learning from
48,000 labeled loads from Jan-Oct 2025 (`data/train-test.csv`).

## Structure

```
data/                 inputs: training loads, loads to predict, prediction template, December inputs
src/                  code
  config.py             paths and shared constants
  features.py           data cleaning, outlier rule and model inputs
  svr_model.py          the SVR rate model
  evaluation.py         error metrics
  train_final.py        fits the final model and writes the predictions
  tune_svr.py           hyperparameter search (dev months only)
  compare_kernels.py    kernel comparison
  check_test_months.py  scores the tuning candidates on the held-out months
predictions/          output: validation_predictions.csv and the filled December file
scorer_results/       output of score.py (chart) and of the tuning scripts, plus the final settings
notebooks/report.ipynb  report: data checks, split, five-model comparison, December chart
reports/report.pdf      written report
score.py              provided scorer
```

## Run

```bash
pip install -r requirements.txt

# 1. fit the final model and write predictions/validation_predictions.csv
#    and predictions/december-chart-inputs.csv (about 6 minutes)
python src/train_final.py

# 2. validate both files and create scorer_results/candidate_december.png
python score.py --predictions predictions/validation_predictions.csv \
                --december-predictions predictions/december-chart-inputs.csv

# 3. open the report
jupyter notebook notebooks/report.ipynb
```

Optional, to reproduce the hyperparameter search (about 15 minutes each, 10 cores):

```bash
python src/tune_svr.py 250 all          # all feature sets, raw dev-CV MAPE
python src/tune_svr.py 100 notime       # date-independent sets, raw dev-CV MAPE
python src/tune_svr.py 200 notime_dm    # date-independent sets, day-demeaned MAPE (selects the final settings)
python src/check_test_months.py         # scores the candidates once on the held-out Sep-Oct months
python src/compare_kernels.py           # kernel comparison
```

## Method

- **Model:** RBF support vector regression (settings in `scorer_results/svr_final_config.json`) on the square
  root of the rate per mile, using distance, equipment, weight and city coordinates. Cities are described by
  coordinates instead of one-hot columns, because the Nov-Dec loads include 8 cities unseen in training.
  A day-of-week adjustment is applied to the prediction.
- **Data fixes:** missing weights are filled with the median, negative weights are made positive and missing
  `market_index` values are filled with the average of the same date. 677 labeled loads (1.4%) with rates far
  above or below the usual rate for that kind of load are left out of training.
- **Validation:** by time. Jan-Aug is used to fit and select (rolling-origin cross-validation, each fold
  trains on all earlier months and scores the next two), and Sep-Oct is held out. The final model is refit on
  all labeled clean rows.

`data/december-chart-inputs.orig.csv` is the untouched December input file.
