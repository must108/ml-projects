# ridge regression utilizes the loss function

# alpha (a hyperparameter) is used to optimize model parameters
# similar to picking k in a knn

# the param alpha controls model complexity
# if alpha = 0, OLS is performed (can lead to overfitting)
# very high alpha leads to underfitting

from sklearn.linear_model import Ridge
scores = []
X_train = [[1, 1]]
y_train = [[1, 1]]
X_test = 1
y_test = 1

for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    ridge = Ridge(alpha = alpha)
    # # ridge.fit(X_train, y_train)
    # y_pred = ridge.predict(X_test)
    # scores.append(ridge.score(X_test, y_test))

# lasso regression

from sklearn.linear_model import Lasso
scores = []

for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    lasso = Lasso(alpha = alpha)
    # lasso.fit(X_train, y_train)
    # lasso_pred = lasso.predict(X_test)
    # scores.append(lasso.score(X_test, y_test))

# in both types of regression, scores drop
# as the alpha value gets higher and higher

# lasso can shrink coefficients of less important variables to 0,
# allowing for more important features to be selected

# lasso implementation

import pandas as pd
import matplotlib.pyplot as plt

diabetes_df = pd.read_csv('regression/diabetes_clean.csv')

X = diabetes_df.drop("glucose", axis = 1).values
y = diabetes_df["glucose"].values
names = diabetes_df.drop("glucose", axis = 1).columns
lasso = Lasso(alpha = 0.1)
lasso_coef = lasso.fit(X, y).coef_

plt.bar(names, lasso_coef)
plt.xticks(rotation = 45)
plt.show()

# lasso is good for seeing which variable can be the most important
