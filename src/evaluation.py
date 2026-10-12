"""Error metrics. In every function, y is the true rate and p the predicted rate in dollars."""
import numpy as np


def metrics(y, p):
    error = p - y
    return {
        "MAE": np.abs(error).mean(),
        "MAPE": np.mean(np.abs(error) / y) * 100,
        "MedAPE": np.median(np.abs(error) / y) * 100,
        "RMSE": np.sqrt(np.mean(error ** 2)),
    }

