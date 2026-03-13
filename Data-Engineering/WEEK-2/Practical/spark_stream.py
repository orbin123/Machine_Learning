from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when 

spark = SparkSession.builder.appName("ExamStream").master("local[*]").getOrCreate()

stream = spark.readStream.format("rate").option("rowsPerSecond", 2).load()

filtered =stream.filter(col("value") % 2 == 0) \
    .withColumn("size", when(col("value") < 10, "small").otherwise("big"))

agg = filtered.groupBy("size").count()

query = agg.writeStream \
    .outputMode("complete") \
    .format("console") \
    .trigger(processingTime="5 seconds") \
    .start()

query.awaitTermination()