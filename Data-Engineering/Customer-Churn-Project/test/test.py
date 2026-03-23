from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when 

spark = SparkSession.builder.appName("etl").getOrCreate()

df = spark.read.csv("housing.csv", header=True, inferSchema=True)

# Handling duplicates, missing, outliers

df = df.dropDuplicates()

mean_area = df.selectExpr("avg(area)").first()[0]
mean_bed = df.selectExpr("avg(bedrooms)").first()[0]
mode_loc = df.groupBy("location").count().orderBy("count", ascending=False).first()[0]

df = df.fill.na(
    {
        "area": mean_area,
        "bedrooms": mean_bed,
        "location": "mode_loc"
    }
)

q_low, q_high = df.approxQuantile("price", [0.05, 0.95], 0.01)

df = df.withColumn("price", when(col("price")<q_low, q_low).when(col("price") > q_high), q_high)

# Feature engineer
df = df.withColumn("price_per_area", col("price") / col("area"))

# load
df.save.parquet("output/data")

spark.stop()