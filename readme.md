# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link

## Solution (run instructions)

```bash
pip install -r requirements.txt
PYTHONWARNINGS=ignore PYTHONPATH=src python src/svr_final.py    # fits the final SVR on all clean labeled rows; writes
                                                                # validation_predictions.csv and fills december-chart-inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
# optional, to reproduce the hyperparameter search (about 15 min each, 10 cores):
PYTHONPATH=src python src/svr_tuning.py 250 all          # all feature sets, raw dev-CV MAPE
PYTHONPATH=src python src/svr_tuning.py 100 notime       # date-independent sets, raw dev-CV MAPE
PYTHONPATH=src python src/svr_tuning.py 200 notime_dm    # date-independent sets, day-demeaned dev-CV MAPE (selects the final config)
PYTHONPATH=src python src/svr_test_check.py              # scores the candidates once on the held-out Sep-Oct months
jupyter notebook report.ipynb                            # report: data checks, split, model comparison, December chart
```

Model: RBF SVR (settings in `scorer_results/svr_final_config.json`) on the square root of the rate per mile, using route length,
equipment, weight and city coordinates (no one-hot cities, because validation has 8 unseen cities), plus a day-of-week effect.
Outliers (677 labeled rows, 1.4%, far above or far below the expected rate) are removed from training. Validation is by time:
Jan-Aug for selection, Sep-Oct held out. Code: `src/features.py` (cleaning, outlier rule, inputs), `src/svr_model.py` (model),
`src/svr_final.py` (training and predictions), `src/svr_tuning.py` and `src/svr_test_check.py` (hyperparameter search and its
held-out check), `src/evaluation.py` (metrics). `december-chart-inputs.orig.csv` is the untouched input.
