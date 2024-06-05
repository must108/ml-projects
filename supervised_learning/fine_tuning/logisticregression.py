from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y 
                    test_size = 0.3, random_state = 42)

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

y_pred_probs = logreg.predict_proba(X_test)[:, 1]
