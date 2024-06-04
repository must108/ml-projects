from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression 

kf = KFold(n_splits=6, shuffle=True, random_state=42)
reg = LinearRegression()
X = 1
y = 1

cv_results = cross_val_score(reg, X, y, cv = kf)

# quantile calculates 95% conf int