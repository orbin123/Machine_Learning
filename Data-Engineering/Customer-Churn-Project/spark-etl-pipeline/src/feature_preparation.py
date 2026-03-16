"""Prepare feature vector for ML models."""

from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler


# Feature columns used for training
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
    "TenureGroup", "AvgMonthlySpendRatio"
]

TARGET_COLUMN = "Churn"


def get_vector_assembler() -> VectorAssembler:
    """Create a VectorAssembler for feature columns."""
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features",
        handleInvalid="skip"
    )
    return assembler


def prepare_features(df: DataFrame) -> DataFrame:
    """Combine feature columns into a single vector."""
    assembler = get_vector_assembler()
    df_assembled = assembler.transform(df)

    print(f"[FEATURES] {len(FEATURE_COLUMNS)} columns assembled into 'features'")
    print(f"[FEATURES] Columns: {FEATURE_COLUMNS}")

    return df_assembled


if __name__ == "__main__":
    from ml_data_loader import create_spark_session, load_processed_data

    spark = create_spark_session()

    df = load_processed_data(spark, "data/processed/telco_churn_clean")
    df_prepared = prepare_features(df)

    df_prepared.select("features", TARGET_COLUMN).show(5, truncate=False)

    spark.stop()