import pandas as pd

music_df = pd.read_csv(
    'supervised_learning/preprocessing_data/music_clean.csv')
music_dummies = pd.get_dummies(music_df["genre"], drop_first = True)

# print(music_dummies.head())

# use .shape to get the shape of a pandas dataframe

music_dummies = pd.concat([music_df, music_dummies], axis = 1)
music_dummies = music_dummies.drop("genre", axis = 1)

print(music_dummies.columns)

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = music_dummies.drop("popularity", axis = 1).values
y = music_dummies["popularity"].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                    test_size = 0.2, random_state = 42)

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
linreg = LinearRegression()
linreg_cv = cross_val_score(linreg, X_train, y_train, cv = kf,
                            scoring = "neg_mean_squared_error")