import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
diabetes_df = pd.read_csv("supervised_learning/regression/diabetes_clean.csv")

X = diabetes_df.drop("glucose", axis = 1).values
y = diabetes_df["glucose"].values
# X is all other variables but glucose.
# Y is only glucose

print(diabetes_df.head())

X_bmi = X[:, 6]
# as bmi is the 6th column, do this to slice all other columns
X_bmi = X_bmi.reshape(-1, 1)

# plt.scatter(X_bmi, y)
# plt.ylabel("Blood Glucose (mg/dl)")
# plt.xlabel("Body Mass Index")
# plt.show()

# ^ initial scatterplot

reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)

plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions, color="red")
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()

# calculator residuals with the residual sum of squares
# multiple linear regression models have multiple features


