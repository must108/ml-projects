# All required libraries are imported here for you.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

crops = pd.read_csv("soil_measures.csv")

print(crops.isna().sum().sort_values())
print(crops["crop"].unique())

X = crops.drop("crop", axis=1)
y = crops["crop"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

features_dict = {}

for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(multi_class="multinomial", solver='lbfgs', max_iter=200)
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    feature_performance = metrics.f1_score(y_test, y_pred, average="weighted")
    features_dict[feature] = feature_performance
    print(f"F1-score for {feature}: {feature_performance}")
    
best_predictive_feature = { max(features_dict, key=features_dict.get) 
                           : features_dict[max(features_dict, key=features_dict.get)] }
print(best_predictive_feature)