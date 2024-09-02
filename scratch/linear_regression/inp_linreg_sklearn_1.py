# the other program, implemented with sklearn built in

import pandas as pd
from sklearn.linear_model import LinearRegression

teams = pd.read_csv("./teams.csv")
model = LinearRegression()

X = teams[["athletes", "prev_medals"]]
y = teams[["medals"]]

model.fit(X, y)
R2 = model.score(X, y)

athletes = int(input("How many athletes? "))
prev_medals = int(input("How many previous medals? "))

print(model.predict([[athletes, prev_medals]]))

result = model.predict([[athletes, prev_medals]])[0][0]

print(f"Estimated number of medals: {result}")
print(f"Accuracy of our number: {R2}")
