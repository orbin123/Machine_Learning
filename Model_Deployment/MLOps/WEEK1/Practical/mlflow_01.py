# MLflow Tracking 
import mlflow 

mlflow.set_experiment("customer-churn-experiment")

with mlflow.start_run():
    # Log Parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 50)
    mlflow.log_param("model_type", "random_forest")

    # Train your model

    # Log Metrics 
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("f1_score", 0.89)
    
    # Log the model itself
    mlflow.sklearn.log_model(model, "model")

# Load model
loaded_model = mlflow.sklearn.load_model("runs:/<run_id>/my_model")

# Make predictions
predictions = loaded_model.predict(new_data)

