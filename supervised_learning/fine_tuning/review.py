from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from sklearn.linear_model import Ridge

sales_df = pd.read_csv(
    'supervised_learning/fine_tuning/telecom_churn_clean.csv'
    )
X = sales_df[["total_day_charge", "total_eve_charge"]].values
y = sales_df["churn"].values

knn = KNeighborsClassifier(n_neighbors=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=42)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# accuracy - number of correct predictions / total number of predictions
# essentially accuracy is the fraction of predictions that the model
# got right.

# precision - number of true positive predictions / true positive
# + false positive
# essentially the proportion of true positives / all LABELLED positives
# gets the percentage of correctness essentially.
# if a model has a precision of 0.5, this means
# if a model predicts positive, it is correct 50% of the time.

# recall - number of true positives / true positive + false negative
# proportion of true positives to all positives (regardless of their label.
# this is why false positives are included).

# f1-score: mean of precision and recall.
# check this statistic to find a reasonably well performing model

# logistic regression is good for classification problems

logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

y_pred_probs = logreg.predict_proba(X_test)[:, 1]
print(y_pred_probs[0])

fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
# false pos rate, true pos rate, thresholds
# roc curve shows the performance of a classification model at 
# different thresholds


plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Logistic Regression ROC Curve")
plt.show()

# 1 for false positive, 0 for true positive
# ROC AUC score can be used to further assess the model

print(roc_auc_score(y_test, y_pred_probs))

# hyperparameter tuning

# params such as n_neighbors and alpha are hyperparameters
# choosing the correct hyperparameters is crucial

# use cross-validation to avoid overfitting the test set

kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {"alpha": np.arange(0.0001, 1, 10),
              "solver": ["sag", "lsqr"]}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv = kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)

# randomizedsearchcv is not as intensive computationally

kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'alpha': np.arange(0.0001, 1, 10),
              "solver": ['sag', 'lsqr']}
ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter=2)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
print(ridge_cv.score(X_test, y_test))

# similar to gridsearch but less intensive