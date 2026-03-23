from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col, when, count
from pyspark.ml.feature import MinMaxScaler, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

# -------------------------------
# 1. Create Spark Session
# -------------------------------
spark = SparkSession.builder \
    .appName("HousingETL") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", 8) \
    .getOrCreate()

# -------------------------------
# 2. Define Schema
# -------------------------------
schema = StructType([
    StructField("price", IntegerType(), True),
    StructField("area", IntegerType(), True),
    StructField("bedrooms", IntegerType(), True),
    StructField("bathrooms", IntegerType(), True),
    StructField("stories", IntegerType(), True),
    StructField("mainroad", StringType(), True),
    StructField("guestroom", StringType(), True),
    StructField("basement", StringType(), True),
    StructField("hotwaterheating", StringType(), True),
    StructField("airconditioning", StringType(), True),
    StructField("parking", IntegerType(), True),
    StructField("prefarea", StringType(), True),
    StructField("furnishingstatus", StringType(), True),
])

# -------------------------------
# 3. Read Data
# -------------------------------
df_raw = spark.read.csv(
    "file:///Users/orbin/Documents/GitHub/Machine_Learning/Data-Engineering/Customer-Churn-Project/test/Housing.csv",
    header=True,
    schema=schema
)

print(f"Rows: {df_raw.count()}, Columns: {len(df_raw.columns)}")
df_raw.show(5, truncate=False)

# -------------------------------
# 4. Remove Duplicates
# -------------------------------
df = df_raw.dropDuplicates()

# -------------------------------
# 5. Handle Missing Values
# -------------------------------
numeric_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
cat_cols = ["mainroad", "guestroom", "basement", "hotwaterheating",
            "airconditioning", "prefarea", "furnishingstatus"]

# Fill numeric with median
medians = {
    c: df.approxQuantile(c, [0.5], 0.01)[0]
    for c in numeric_cols
}
df = df.na.fill(medians)

# Fill categorical with mode
for c in cat_cols:
    mode_val = df.groupBy(c).count().orderBy(col("count").desc()).first()[0]
    df = df.na.fill({c: mode_val})

# -------------------------------
# 6. Handle Outliers (IQR)
# -------------------------------
def cap_outliers(df, column):
    q1, q3 = df.approxQuantile(column, [0.25, 0.75], 0.01)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return df.withColumn(
        column,
        when(col(column) < lower, lower)
        .when(col(column) > upper, upper)
        .otherwise(col(column))
    )

for c in numeric_cols:
    df = cap_outliers(df, c)

# -------------------------------
# 7. Feature Engineering
# -------------------------------

# Binary encoding
binary_cols = ["mainroad", "guestroom", "basement",
               "hotwaterheating", "airconditioning", "prefarea"]

for c in binary_cols:
    df = df.withColumn(c, when(col(c) == "yes", 1).otherwise(0))

# String Indexing
indexer = StringIndexer(
    inputCol="furnishingstatus",
    outputCol="furnishing_index"
)
df = indexer.fit(df).transform(df).drop("furnishingstatus")

# Safe feature creation
df = df.withColumn(
    "price_per_sqft",
    when(col("area") != 0, col("price") / col("area")).otherwise(0)
)

df = df.withColumn(
    "total_rooms",
    col("bedrooms") + col("bathrooms")
)

# -------------------------------
# 8. Feature Scaling Pipeline
# -------------------------------
feature_cols = [
    "area", "bedrooms", "bathrooms", "stories", "mainroad",
    "guestroom", "basement", "hotwaterheating", "airconditioning",
    "parking", "prefarea", "furnishing_index",
    "price_per_sqft", "total_rooms"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

minmax = MinMaxScaler(
    inputCol="features_raw",
    outputCol="features_normalized"
)

stdscaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=True,
    withStd=True
)

pipeline = Pipeline(stages=[assembler, minmax, stdscaler])
df = pipeline.fit(df).transform(df)

# -------------------------------
# 9. Clean Data Before Writing
# -------------------------------
df = df.dropna()

# -------------------------------
# 10. Write to Parquet
# -------------------------------
output_path = "file:///Users/orbin/output/housing_transformed"

df.select(
    "price", "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "parking", "prefarea", "furnishing_index",
    "price_per_sqft", "total_rooms",
    "features_raw", "features_normalized", "features_scaled"
) \
.write.mode("overwrite") \
.parquet(output_path)

# -------------------------------
# 11. Verify Output
# -------------------------------
df_verify = spark.read.parquet(output_path)
print(f"Verified: {df_verify.count()} rows, {len(df_verify.columns)} columns")

spark.stop()