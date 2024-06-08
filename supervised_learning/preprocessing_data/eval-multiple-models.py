# some guiding principles to decide on a model

# size of the dataset:
# fewer features generally means a simpler model
# artificial neural networks, for example, require a lot of data to perform well.

# interpretability:
# linear regression, for example, is easier to explain
# model coefficients in linear regression is easier to interpret

# flexibility:
# by making fewer assumptions about the data, the accuracy of the model may improve
# for example, knn doesn't assume any linear relationship, leading to more accurate predictions.


# you can easily compare multiple models in scikit-learn.
# many of the same methods are available to be used on all models

# regression models can be evaluated with:
# the root mean-squared error
# and the r-squared value.

# classification models can be evaluated with:
# accuracy
# confusion matrix
# precision, recall, and f1-score
# roc auc curve

# a typical approach is to train models and evaluate their performance without
# any hyperparameter tuning.

# SOME models are affected by scaling (standardization/normalization)
# for example: knn, linear regression, lasso regression,
# ridge regression, logistic regression, and artificial neural networks.

# scale your data before evaluating models!

# model evaluation example below. evaluates knn, logistic regression, and decision tree.

import pandas as pd

sales_df = pd.read_csv("supervised_learning/preprocessing_data/sales.csv")

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

X = sales_df.drop("influencer", axis = 1).values
y = sales_df["influencer"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

results = []

for model in models.values():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv = kf)
    results.append(cv_results)

plt.boxplot(results, labels = models.keys())
plt.show()

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print("{} Test Set Accuracy: {}".format(name, test_score))