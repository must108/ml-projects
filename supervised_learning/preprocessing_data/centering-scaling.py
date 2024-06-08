
import pandas as pd

sales_df = pd.read_csv('supervised_learning/preprocessing_data/sales.csv')

# print(sales_df[["tv", "radio", "social_media"]].describe())

# check ranges of feature variables^
# knn uses distance explicitly to make predictions
# we want all our features to have a similar scale!
# we can do this by euither normalizing our data or standardizing it (scaling and centering)

# ways to scale data:

# standardization:
# subtract the mean and divide by variance - all features will be centered around zero and have a
# variance of one.
# you can also subtract the minimum and divide by the range
# minimum zero and maximum one

# normalization:
# we can also center (normalize) our data: so the data ranges from -1 to 1

# scaling in scikit-learn:

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = sales_df.drop("influencer", axis = 1).values
y = sales_df["influencer"].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                test_size = 0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(np.mean(X), np.std(X))
print(np.mean(X_train_scaled), np.std(X_train_scaled))

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps)

parameters = {"knn__n_neighbors": np.arange(1, 50)}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                            random_state=21)

cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)

y_pred = cv.predict(X_test)
print(cv.score(X_test, y_test))

##### vs unscaled data! #####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                random_state=21)
knn_unscaled = KNeighborsClassifier(n_neighbors=6).fit(X_train, y_train)
print(knn_unscaled.score(X_test, y_test))