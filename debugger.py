from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def analyze_model(model, df):
    result = {}

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_acc = model.score(X_train, y_train)
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    result["accuracy"] = test_acc
    result["overfitting"] = train_acc - test_acc > 0.2
    result["underfitting"] = test_acc < 0.5

    return result