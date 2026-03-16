"""
Transform Phase: Clean, encode, and engineer features.
"""


from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, trim, regexp_replace
)
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer




def remove_duplicates(df: DataFrame) -> DataFrame:
    """Remove duplicate rows."""
    before = df.count()
    df = df.dropDuplicates()
    after = df.count()
    print(f'[TRANSFORM] Removed {before - after} duplicate rows')
    return df




def fix_total_charges(df: DataFrame) -> DataFrame:
    """Convert TotalCharges from string to double and handle blanks."""
    df = df.withColumn(
        "TotalCharges",
        when(trim(col("TotalCharges")) == "", None)
        .otherwise(col("TotalCharges").cast(DoubleType()))
    )
    print('[TRANSFORM] Converted TotalCharges to DoubleType')
    return df




def handle_missing_values(df: DataFrame) -> DataFrame:
    """Fill missing TotalCharges with 0.0 (new customers)."""
    missing_count = df.filter(col("TotalCharges").isNull()).count()
    print(f'[TRANSFORM] Found {missing_count} missing TotalCharges values')
    df = df.fillna({"TotalCharges": 0.0})
    print('[TRANSFORM] Filled missing TotalCharges with 0.0')
    return df




def drop_unnecessary_columns(df: DataFrame) -> DataFrame:
    """Drop columns not needed for ML."""
    df = df.drop("customerID")
    print('[TRANSFORM] Dropped customerID column')
    return df




def encode_binary_columns(df: DataFrame) -> DataFrame:
    """Encode Yes/No and Male/Female columns to 1/0."""
    binary_cols = [
        "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "Churn"
    ]
    for c in binary_cols:
        df = df.withColumn(
            c,
            when(col(c) == "Yes", 1).otherwise(0)
        )


    df = df.withColumn(
        "gender",
        when(col("gender") == "Male", 1).otherwise(0)
    )


    print(f'[TRANSFORM] Encoded binary columns: {binary_cols + ["gender"]}')
    return df




def encode_categorical_columns(df: DataFrame) -> DataFrame:
    """Use StringIndexer for multi-class categorical columns."""
    categorical_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]


    for c in categorical_cols:
        indexer = StringIndexer(
            inputCol=c,
            outputCol=f"{c}_indexed",
            handleInvalid="keep"
        )
        df = indexer.fit(df).transform(df)
        df = df.drop(c).withColumnRenamed(f'{c}_indexed', c)


    print(f'[TRANSFORM] Indexed categorical columns: {categorical_cols}')
    return df




def engineer_features(df: DataFrame) -> DataFrame:
    """Create new derived features."""


    # Tenure Group: bucket tenure into categories
    df = df.withColumn(
        "TenureGroup",
        when(col('tenure') <= 12, 0)
        .when(col('tenure') <= 24, 1)
        .when(col('tenure') <= 48, 2)
        .otherwise(3)
    )


    # Monthly to Total ratio
    df = df.withColumn(
        "AvgMonthlySpendRatio",
        when(col('TotalCharges') > 0,
             col('MonthlyCharges') / col('TotalCharges'))
        .otherwise(0.0)
    )


    print('[TRANSFORM] Engineered features: TenureGroup, AvgMonthlySpendRatio')
    return df




def run_transformations(df: DataFrame) -> DataFrame:
    """Execute all transformations in sequence."""
    print('\n=== TRANSFORM PHASE ===')
    df = remove_duplicates(df)
    df = fix_total_charges(df)
    df = handle_missing_values(df)
    df = drop_unnecessary_columns(df)
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)
    df = engineer_features(df)
    print(f'[TRANSFORM] Final schema has {len(df.columns)} columns')
    print('=== TRANSFORM PHASE COMPLETE ===\n')
    return df
