"""
Model Training: Build, train, and evaluate the ML pipeline.
"""

import os
import time
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from extract import create_spark_session


# Configuration 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "telco_churn_clean")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "churn_model")
SEED = 42

FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
    "TenureGroup", "AvgMonthlySpendRatio"
]
TARGET_COLUMN = "Churn"


def load_data(spark):
    """Load the processed Parquet dataset."""
    print(f"\n[DATA] Loading from: {DATA_PATH}")
    df = spark.read.parquet(DATA_PATH)
    print(f"[DATA] Loaded {df.count()} rows, {len(df.columns)} columns")
    return df


def build_lr_pipeline():
    """Build a Logistic Regression pipeline."""
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features",
        handleInvalid="skip"
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol=TARGET_COLUMN,
        predictionCol="prediction",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.8
    )

    pipeline = Pipeline(stages=[assembler, lr])
    return pipeline


def build_rf_pipeline():
    """Build a Random Forest pipeline."""
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features",
        handleInvalid="skip"
    )

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol=TARGET_COLUMN,
        predictionCol="prediction",
        numTrees=100,
        maxDepth=5,
        seed=SEED
    )

    pipeline = Pipeline(stages=[assembler, rf])
    return pipeline


def evaluate_model(predictions, model_name="Model"):
    """
    Evaluate model predictions using multiple metrics.

    Args:
        predictions: DataFrame with prediction and label columns
        model_name: Name for logging
    """
    print(f"\n{'=' * 50}")
    print(f"  EVALUATION: {model_name}")
    print(f"{'=' * 50}")

    # Accuracy
    accuracy_eval = MulticlassClassificationEvaluator(
        labelCol=TARGET_COLUMN,
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = accuracy_eval.evaluate(predictions)

    # Precision
    precision_eval = MulticlassClassificationEvaluator(
        labelCol=TARGET_COLUMN,
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    precision = precision_eval.evaluate(predictions)

    # Recall
    recall_eval = MulticlassClassificationEvaluator(
        labelCol=TARGET_COLUMN,
        predictionCol="prediction",
        metricName="weightedRecall"
    )
    recall = recall_eval.evaluate(predictions)

    # F1 Score
    f1_eval = MulticlassClassificationEvaluator(
        labelCol=TARGET_COLUMN,
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = f1_eval.evaluate(predictions)

    # AUC-ROC
    auc_eval = BinaryClassificationEvaluator(
        labelCol=TARGET_COLUMN,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc = auc_eval.evaluate(predictions)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"{'=' * 50}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


def show_predictions(predictions, n=20):
    """Display predicted vs actual values."""
    print(f"\n[PREDICTIONS] Showing top {n} predictions:")
    predictions.select(
        TARGET_COLUMN, "prediction", "probability"
    ).show(n, truncate=False)

    # Confusion matrix summary
    print("[PREDICTIONS] Confusion Matrix Summary:")
    predictions.groupBy(TARGET_COLUMN, "prediction").count().orderBy(
        TARGET_COLUMN, "prediction"
    ).show()


def run_hyperparameter_tuning(pipeline, train_df):
    """
    Perform hyperparameter tuning using grid search and cross-validation.

    Args:
        pipeline: ML Pipeline to tune
        train_df: Training DataFrame

    Returns:
        Best model from cross-validation
    """
    print("\n[TUNING] Starting hyperparameter tuning...")
    print("[TUNING] This may take several minutes.\n")

    # Get the Logistic Regression stage from the pipeline
    lr = pipeline.getStages()[-1]

    # Define the parameter grid
    param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .addGrid(lr.maxIter, [50, 100]) \
        .build()

    print(f"[TUNING] Grid size: {len(param_grid)} parameter combinations")

    # Define the evaluator
    evaluator = BinaryClassificationEvaluator(
        labelCol=TARGET_COLUMN,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    # Create CrossValidator
    cross_validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        seed=SEED
    )

    # Fit the cross-validator
    cv_model = cross_validator.fit(train_df)

    # Report results
    print(f"[TUNING] Best AUC-ROC: {max(cv_model.avgMetrics):.4f}")
    print(f"[TUNING] All fold averages: {[f'{m:.4f}' for m in cv_model.avgMetrics]}")

    return cv_model.bestModel


def save_model(model, path):
    """Save the trained model to disk."""
    print(f"\n[SAVE] Saving model to: {path}")
    model.write().overwrite().save(path)
    print(f"[SAVE] Model saved successfully")


def run_ml_pipeline():
    """Execute the complete ML pipeline."""
    print("=" * 60)
    print("  TELCO CUSTOMER CHURN - ML PIPELINE")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Create Spark Session
    spark = create_spark_session("TelcoChurnML")

    # Step 2: Load Processed Data 
    df = load_data(spark)

    # Step 3: Train/Test Split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=SEED)
    print(f"\n[SPLIT] Training set: {train_df.count()} rows")
    print(f"[SPLIT] Test set:     {test_df.count()} rows")

    # Step 4: Train Logistic Regression 
    print("\n--- Logistic Regression ---")
    lr_pipeline = build_lr_pipeline()
    lr_model = lr_pipeline.fit(train_df)
    lr_predictions = lr_model.transform(test_df)
    lr_metrics = evaluate_model(lr_predictions, "Logistic Regression")
    show_predictions(lr_predictions)

    # Step 5: Train Random Forest
    print("\n--- Random Forest ---")
    rf_pipeline = build_rf_pipeline()
    rf_model = rf_pipeline.fit(train_df)
    rf_predictions = rf_model.transform(test_df)
    rf_metrics = evaluate_model(rf_predictions, "Random Forest")

    # Step 6: Compare Models
    print("\n" + "=" * 50)
    print("  MODEL COMPARISON")
    print("=" * 50)
    print(f"  {'Metric':<12} {'Log. Reg.':<12} {'Random Forest':<12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        print(f"  {metric:<12} {lr_metrics[metric]:<12.4f} {rf_metrics[metric]:<12.4f}")
    print("=" * 50)

    # Step 7: Select Best Model
    if lr_metrics["auc"] >= rf_metrics["auc"]:
        best_model = lr_model
        best_name = "Logistic Regression"
    else:
        best_model = rf_model
        best_name = "Random Forest"

    print(f"\n[RESULT] Best model: {best_name}")

    # Step 8: Hyperparameter Tuning on Best Pipeline
    if best_name == "Logistic Regression":
        print("\n--- Hyperparameter Tuning (Logistic Regression) ---")
        tuned_model = run_hyperparameter_tuning(build_lr_pipeline(), train_df)
        tuned_predictions = tuned_model.transform(test_df)
        evaluate_model(tuned_predictions, "Tuned Logistic Regression")
        best_model = tuned_model

    # Step 9: Save Best Model
    save_model(best_model, MODEL_SAVE_PATH)

    # Summary 
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  ML PIPELINE COMPLETE in {elapsed:.2f} seconds")
    print(f"  Model saved to: {MODEL_SAVE_PATH}")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    run_ml_pipeline()