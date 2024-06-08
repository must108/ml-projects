from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

sales_df = pd.read_csv('supervised_learning/preprocessing_data/sales.csv')
# sales_df = sales_df.dropna(subset=["influencer"])

sales_df["influencer"] = np.where(sales_df["influencer"] == "Micro", 1, 0)
X = sales_df.drop("influencer", axis = 1).values
y = sales_df['influencer'].values

steps = [
    ("imputation", SimpleImputer()),
    ("logistic_regression", LogisticRegression())
]

pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state = 42)
pipeline.fit(X_train, y_train)
print(pipeline.score(X_test, y_test))

# pipeline is an object that can perform a series of transformations
# and build a model within the same workflow
