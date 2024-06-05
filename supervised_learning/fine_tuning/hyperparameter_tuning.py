
# gridsearchcv 

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge
import numpy as np

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

param_grid = {"alpha": np.arange(0.0001, 1, 10),
              "solver": ["sag", "lsqr"]}

ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv = kf)
ridge_cv.fit(X_train, y_train)

ridge_cv.best_params_
ridge_cv.best_score_

# this method doesnt scale as well however

# randomizedsearchcv

from sklearn.model_selection import RandomizedSearchCV

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
param_grid = {'alpha': np.arange(0.0001, 1, 10),
              "solver": ['sag', 'lsqr']}
ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv = kf, n_iter = 2)
ridge_cv.fit(X_train, y_train)

test_score = ridge_cv.score(X_test, y_test)