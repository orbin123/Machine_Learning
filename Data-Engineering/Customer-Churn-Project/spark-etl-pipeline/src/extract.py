"""Extract step for the Telco churn ETL pipeline."""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)


# Dataset schema
TELCO_SCHEMA = StructType([
    StructField("customerID", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("SeniorCitizen", IntegerType(), True),
    StructField("Partner", StringType(), True),
    StructField("Dependents", StringType(), True),
    StructField("tenure", IntegerType(), True),
    StructField("PhoneService", StringType(), True),
    StructField("MultipleLines", StringType(), True),
    StructField("InternetService", StringType(), True),
    StructField("OnlineSecurity", StringType(), True),
    StructField("OnlineBackup", StringType(), True),
    StructField("DeviceProtection", StringType(), True),
    StructField("TechSupport", StringType(), True),
    StructField("StreamingTV", StringType(), True),
    StructField("StreamingMovies", StringType(), True),
    StructField("Contract", StringType(), True),
    StructField("PaperlessBilling", StringType(), True),
    StructField("PaymentMethod", StringType(), True),
    StructField("MonthlyCharges", DoubleType(), True),
    StructField("TotalCharges", StringType(), True),
    StructField("Churn", StringType(), True),
])


# Create Spark session
def create_spark_session(app_name: str = "TelcoChurnETL") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.jars.packages", "org.xerial:sqlite-jdbc:3.46.0.0")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# Extract data from CSV
def extract_csv(spark: SparkSession, file_path: str) -> DataFrame:
    print(f"[EXTRACT-CSV] Reading {file_path}")

    df = (
        spark.read
        .option("header", "true")
        .schema(TELCO_SCHEMA)
        .csv(file_path)
    )

    row_count = df.count()
    print(f"[EXTRACT-CSV] {row_count} rows loaded")

    df = df.withColumn("source", lit("csv"))
    return df


# Extract data from SQLite
def extract_sql(spark: SparkSession, db_path: str,
                table: str = "supplementary_records") -> DataFrame:

    jdbc_url = f"jdbc:sqlite:{db_path}"
    print(f"[EXTRACT-SQL] Reading table {table}")

    df = (
        spark.read
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table)
        .option("driver", "org.sqlite.JDBC")
        .load()
    )

    row_count = df.count()
    print(f"[EXTRACT-SQL] {row_count} rows loaded")

    for field in TELCO_SCHEMA.fields:
        df = df.withColumn(field.name, df[field.name].cast(field.dataType))

    df = df.withColumn("source", lit("sql"))
    return df


# Validate schemas
def validate_schemas(df_csv: DataFrame, df_sql: DataFrame) -> None:

    schema_csv = {f.name: f.dataType for f in df_csv.schema.fields if f.name != "source"}
    schema_sql = {f.name: f.dataType for f in df_sql.schema.fields if f.name != "source"}

    if schema_csv != schema_sql:
        mismatches = []
        all_cols = set(schema_csv.keys()) | set(schema_sql.keys())

        for col in sorted(all_cols):
            csv_type = schema_csv.get(col, "MISSING")
            sql_type = schema_sql.get(col, "MISSING")

            if csv_type != sql_type:
                mismatches.append(f"{col}: csv={csv_type} sql={sql_type}")

        raise ValueError(
            "Schema mismatch between CSV and SQL sources:\n"
            + "\n".join(mismatches)
        )

    print(f"[VALIDATE] {len(schema_csv)} columns verified")


# Combine both sources
def extract_all(spark: SparkSession, csv_path: str,
                db_path: str) -> DataFrame:

    print("\nStarting extract phase")

    df_csv = extract_csv(spark, csv_path)
    df_sql = extract_sql(spark, db_path)

    validate_schemas(df_csv, df_sql)

    col_order = [f.name for f in TELCO_SCHEMA.fields] + ["source"]

    df_combined = df_csv.select(col_order).union(df_sql.select(col_order))

    total = df_combined.count()
    csv_count = df_combined.filter(df_combined.source == "csv").count()
    sql_count = df_combined.filter(df_combined.source == "sql").count()

    print(f"[EXTRACT] Combined rows: {total} (csv={csv_count}, sql={sql_count})")

    return df_combined


if __name__ == "__main__":

    CSV_PATH = "File:///Users/orbin/Documents/GitHub/Machine_Learning/Data-Engineering/Customer-Churn-Project/spark-etl-pipeline/data/raw/Telco-Customer-Churn.csv"
    DB_PATH = "/Users/orbin/Documents/GitHub/Machine_Learning/Data-Engineering/Customer-Churn-Project/spark-etl-pipeline/data/raw/supplementary_data.db"

    spark = create_spark_session()

    df = extract_all(spark, CSV_PATH, DB_PATH)

    print("Sample CSV rows")
    df.filter(df.source == "csv").show(3, truncate=False)

    print("Sample SQL rows")
    df.filter(df.source == "sql").show(3, truncate=False)

    df.printSchema()

    spark.stop()