from pyspark.sql import SparkSession 
from pyspark.sql.functions import col, when, window

spark = SparkSession.builder \
    .appName("StructuredStreaming") \
    .master("local[2]") \
    .getOrCreate()

# Step 1: Create a simulated Streaming source 
stream_df = spark.readStream \
    .format("rate") \
    .option("rowPerSecond", 2) \
    .load()

# Step 2: STATELESS Operation - filter + add a column
processed = stream_df \
    .filter(col("value") % 2 == 0) \
    .withColumn("category", when(col("value") < 10, "low").otherwise("high"))

# Step 3: STATEFUL aggregation - count per category
agg = processed \
    .groupBy("category") \
    .count()

# Step 4: Start the stream
query = agg.writeStream \
    .outputMode("complete") \
    .format("console") \
    .trigger(processingTime="5 seconds") \
    .start()

print("Streaming started. Watch the console output.")
print("Open http://localhost:4040 for the Streaming tab.")
print("Press Ctrl+C to stop.\n")

# Wait until manually stopped
query.awaitTermination()