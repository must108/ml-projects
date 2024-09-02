# linear regression program from scratch!

import pandas as pd
import numpy as np

teams = pd.read_csv("./teams.csv")

# set X and y
X = teams[["athletes", "prev_medals"]].copy()
y = teams[["medals"]].copy()

X["intercept"] = 1
X = X[["intercept", "athletes", "prev_medals"]]

X_T = X.transpose()

B = np.linalg.inv(X_T @ X) @ X_T @ y
B.index = X.columns

predictions = X @ B

SSR = ((y - predictions) ** 2).sum()
SST = ((y - y.mean()) ** 2).sum()

R2 = 1 - (SSR/SST)

intercept = B.loc["intercept", "medals"]
athlete_slope = B.loc["athletes", "medals"]
prev_medals_slope = B.loc["prev_medals", "medals"]

athletes = int(input("How many athletes? "))
prev_medals = int(input("How many previous medals? "))

num_medals = (intercept +
              (athlete_slope * athletes) +
              (prev_medals_slope * prev_medals))

print(
    f"The estimated number of medals this team would win is: {num_medals}"
)

print(
    f"The accuracy of this value is: {R2.loc["medals"]}"
)
