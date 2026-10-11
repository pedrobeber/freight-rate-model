"""Error metrics used everywhere. y = true rate in dollars, p = predicted rate in dollars."""
import numpy as np


def metrics(y, p):
    e = p - y
    return dict(MAE=np.abs(e).mean(), MAPE=np.mean(np.abs(e) / y) * 100,
                MedAPE=np.median(np.abs(e) / y) * 100, RMSE=np.sqrt(np.mean(e ** 2)))


def mse_log(y, p):
    """Loss the models are trained on: MSE of log(rate per mile). y and p are rates in dollars."""
    return float(np.mean((np.log(p) - np.log(y)) ** 2))
