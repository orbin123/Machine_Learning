"""Runs the ETL pipeline for the Telco churn project."""

import time
import os
from extract import create_spark_session, extract_all
from transform import run_transformations
from load import load_to_parquet, verify_output


# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Telco-Customer-Churn.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "supplementary_data.db")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "telco_churn_clean")


def run_etl_pipeline():
    """Execute the ETL pipeline."""

    print("Starting Telco Customer Churn ETL pipeline")

    start_time = time.time()

    # Step 1: Create Spark session
    spark = create_spark_session("TelcoChurnETL")

    # Step 2: Extract data
    df_raw = extract_all(spark, CSV_PATH, DB_PATH)

    # Check source distribution
    print("[PIPELINE] Source breakdown:")
    df_raw.select("source").groupBy("source").count().show()

    # Step 3: Drop source column
    df_raw = df_raw.drop("source")

    # Step 4: Transform data
    df_transformed = run_transformations(df_raw)
    df_transformed.cache()

    # Step 5: Load data
    load_to_parquet(df_transformed, OUTPUT_PATH)

    # Step 6: Verify output
    verify_output(spark, OUTPUT_PATH)

    # Pipeline summary
    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed:.2f} seconds")
    print(f"Output path: {OUTPUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    run_etl_pipeline()