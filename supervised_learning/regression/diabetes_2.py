from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

diabetes_df = pd.read_csv('supervised_learning/regression/diabetes_clean.csv')
X = diabetes_df.drop("glucose", axis = 1).values
y = diabetes_df["glucose"].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size = 0.3, random_state = 42)

# R-squared quantifies the variance in target values
# high R-squared values datasets are generally closer to the line of best fit
# in comparison, low R-squared values datasets are generally farther from the line of best fit
# R-squared can be computed with .score with X_test and y_test

# mean squared error can also be used to test a linear regression model's performance
# like this:

y_pred = 1
mean_squared_error(y_test, y_pred, squared = False)

# gives the average error of the model