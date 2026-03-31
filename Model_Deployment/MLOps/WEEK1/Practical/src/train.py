import mlflow 
import yaml 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle 

with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Load data 
train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")

X_train = train_data.drop("target", axis=1)
y_train = train_data["target"]
X_test = test_data.drop("target", axis=1)
y_test = test_data["target"]

mlflow.set_experiment("churn-predictionsl")

with mlflow.start_run():
    mlflow.log_parameters("learning_rate", params["train"]["learning_rate"])
    mlflow.log_param("n_estimators", params["train"]["n_estimators"])
    mlflow.log_param("max_depth", params["train"]["max_depth"])

    model = RandomForestClassifier(
        n_estimators=params["train"]["n_estimators"],
        max_depth=params["train"]["max_depth"],
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate 
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    mlflow.log_metric("accuracy", accuracy)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    mlflow.sklearn.log_model(model, "model")