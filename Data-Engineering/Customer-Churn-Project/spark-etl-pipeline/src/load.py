"""Load transformed data to storage."""

from pyspark.sql import DataFrame


def load_to_parquet(df: DataFrame, output_path: str) -> None:
    """Write DataFrame to Parquet."""
    
    print(f"[LOAD] Writing {df.count()} rows to {output_path}")

    # Repartition before writing
    df.repartition(4).write.mode("overwrite").parquet(output_path)

    print("[LOAD] Write complete")
    print(f"[LOAD] Output path: {output_path}")


def verify_output(spark, output_path: str) -> None:
    """Read and verify saved data."""
    
    print("\n[LOAD] Verifying output")

    df = spark.read.parquet(output_path)

    print(f"[LOAD] {df.count()} rows, {len(df.columns)} columns")
    df.printSchema()
    df.show(5)