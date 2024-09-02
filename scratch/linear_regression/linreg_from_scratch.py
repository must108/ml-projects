import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

teams = pd.read_csv("./teams.csv")

# set X and Y matricies to respective columns
X = teams[["athletes", "prev_medals"]].copy()
y = teams[["medals"]].copy()

X["intercept"] = 1
X = X[["intercept", "athletes", "prev_medals"]]

X_T = X.transpose()  # transpose of the matrix

# formula for calculating coefficients for linear regression
B = np.linalg.inv(X_T @ X) @ X_T @ y
B.index = X.columns  # so the dataframe is properly labelled
# inverse of the XTX matrix (@ is a matrix multiplier)

predictions = X @ B  # represents y = XB, X matrix times coefficient matrix

# calculate sum of squared residuals
SSR = ((y - predictions) ** 2).sum()
SST = ((y - y.mean()) ** 2).sum()

R2 = 1 - (SSR/SST)
# was at 0.87, so its a pretty good fit!

# test with sklearn

lr = LinearRegression()

X2 = teams[["athletes", "prev_medals"]]
y2 = teams[["medals"]]

lr.fit(X2, y2)
print(lr.intercept_)
print(lr.coef_)
# same as our algorithm
