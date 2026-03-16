"""
ML Data Loader: Load the cleaned Parquet dataset for machine learning.
"""
import os
from extract import create_spark_session
from pyspark.sql import SparkSession, DataFrame


def load_processed_data(spark: SparkSession, path: str) -> DataFrame:
    print(f"[ML DATA] Loading processed data from: {path}")
    df = spark.read.parquet(path)
    row_count = df.count()
    col_count = len(df.columns)
    print(f"[ML DATA] Loaded {row_count} rows and {col_count} columns")
    print(f"[ML DATA] Columns: {df.columns}")
    return df


if __name__ == "__main__":
    spark = create_spark_session("TelcoChurnML")

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "telco_churn_clean")

    df = load_processed_data(spark, PROCESSED_PATH)
    df.printSchema()
    df.show(5)

    print("\n[ML DATA] Target variable distribution:")
    df.groupBy("Churn").count().show()

    spark.stop()