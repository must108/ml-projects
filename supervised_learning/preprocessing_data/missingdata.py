
# when there is no value for a feature in a particular row - missing data
# or it is corrupt, or whatever

import pandas as pd
import numpy as np
school_df = pd.read_csv('supervised_learning/preprocessing_data/school.csv')
print(school_df.isna().sum().sort_values()) # see missing values

# school_df = school_df.dropna(subset=["Marks", "Address"])
# ^ should only be used when missing data is less than 5% of all data

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

X_cat = school_df["City"].values.reshape(-1, 1)
X_num = school_df.drop(["City", "Marks"], axis = 1).values
y = school_df["Marks"].values

X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y, test_size = 0.2,
                                                            random_state = 12)

X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size = 0.2,
                                                            random_state = 12)

imp_cat = SimpleImputer(strategy="most_frequent")
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_Cat = imp_cat.transform(X_test_cat)

imp_num = SimpleImputer()
X_train_num = imp_cat.fit_transform(X_train_num)
X_test_num = imp_cat.transform(X_test_num)

X_train = np.append(X_train_num, X_train_cat, axis = 1)
X_test = np.append(X_test_num, X_test_cat, axis = 1)

# imputers are known as transformers
# SimpleImputer() can be used to fil in null values in a dataset