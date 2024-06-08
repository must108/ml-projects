
# we need to preprocess our data before we can build models.

# we need to convert text values into numeric values for scikit-learn

import pandas as pd
sales_df = pd.read_csv('supervised_learning/preprocessing_data/sales.csv')
sales_dummies = pd.get_dummies(sales_df["influencer"], drop_first=True,
                               dtype=int)

sales_dummies = pd.concat([sales_df, sales_dummies], axis = 1)
sales_dummies = sales_dummies.drop("influencer", axis = 1)

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

X = sales_dummies.drop("sales", axis = 1).values
y = sales_dummies["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

kf = KFold(n_splits=5, shuffle = True, random_state=42)
linreg = LinearRegression()
linreg_cv = cross_val_score(linreg, X_train, y_train, cv = kf,
                            scoring="neg_mean_squared_error")

print(np.sqrt(-linreg_cv))