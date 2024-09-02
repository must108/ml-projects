import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

teams = pd.read_csv("./teams.csv")

train, test = train_test_split(teams, test_size=.2, random_state=1)


def ridge_fit(train, preds, target, alpha):
    X = train[preds].copy()
    y = train[[target]].copy()

    x_mean = X.mean()
    x_std = X.std()

    X = (X - x_mean) / x_std
    X["intercept"] = 1
    X = X[["intercept"] + preds]

    penalty = alpha * np.identity(X.shape[1])
    penalty[0][0] = 0

    B = np.linalg.inv(X.T @ X + penalty) @ X.T @ y
    B.index = ["intercept", "athletes", "events"]
    return B, x_mean, x_std


def ridge_predict(test, preds, x_mean, x_std, B):
    # set up test set for X
    test_X = test[preds]
    test_X = (test_X - x_mean) / x_std  # use train data's mean and std
    test_X["intercept"] = 1
    test_X = test_X[["intercept"] + preds]

    predictions = test_X @ B
    return predictions


preds = ["athletes", "events"]
target = "medals"
alpha = 2

errors = []
alphas = [10**i for i in range(-2, 4)]

# to find the best alpha value
for alpha in alphas:
    B, x_mean, x_std = ridge_fit(train, preds, target, alpha)
    predictions = ridge_predict(test, preds, x_mean, x_std, B)
    errors.append(mean_absolute_error(test[[target]], predictions))
