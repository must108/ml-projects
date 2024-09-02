import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

teams = pd.read_csv("./teams.csv")

train, test = train_test_split(teams, test_size=.2, random_state=1)

preds = ["athletes", "events"]
target = "medals"
alpha = 2

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
B.index = ["intercept"] + preds

test_X = test[preds]
test_X = (test_X - x_mean) / x_std
test_X["intercept"] = 1
test_X = test_X[["intercept"] + preds]

predictions = test_X @ B
