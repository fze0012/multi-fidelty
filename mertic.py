import torch
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def mertic(func_test, func_predict):
    mae = mean_absolute_error(func_test, func_predict)
    mse = mean_squared_error(func_test, func_predict)
    rmse = sqrt(mean_squared_error(func_test, func_predict))
    r2_Score = r2_score(func_test, func_predict)
    arde = ((torch.sum(torch.from_numpy(func_test - func_predict))) / (
        torch.sum(torch.from_numpy(func_test)))).detach().numpy()
    print("mae:", mae, "mse:", mse, "rmse:", rmse)

