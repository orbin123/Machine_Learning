import joblib 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_model():
    iris = load_iris()
    X = iris.data 
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)

    print('Acc', acc_score)
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    model_filename='models/model.joblib'
    joblib.dump(model, model_filename)

    print('File size:', joblib.load(model_filename).__sizeof__())

    return model_filename

if __name__ == '__main__':
    train_and_save_model()