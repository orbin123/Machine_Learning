from pyspark.sql import SparkSession
import time 

spark = SparkSession.builder \
    .appName("RDDOperations") \
    .master("local[2]") \
    .getOrCreate()

sc = spark.sparkContext

# Step 1: Create RDD from raw log strings
logs = [
    "ERROR 2024-01-10 disk failure on node-1",
    "WARN 2024-01-10 high memory usage on node-2",
    "ERROR 2024-01-11 network timeout on node-3",
    "INFO 2024-01-11 backup completed on node-1",
    "ERROR 2024-01-12 disk failure on node-2",
    "WARN 2024-01-12 cpu spike on node-1",
    "INFO 2024-01-12 health check passed on node-3",
]

rdd = sc.parallelize(logs, 2)

# Step 2: NARROW transformation — filter
errors_rdd = rdd.filter(lambda line: line.startswith("ERROR"))

# Step 3: NARROW transformation - map 
pairs_rdd = rdd.map(lambda line: (line.split(" ")[0], 1))

# Step 4: WIDE transformation - reduceByKey
counts_rdd = pairs_rdd.reduceByKey(lambda a, b: a + b)

print("Errors only (narrow filter): ",errors_rdd.collect())

print("\nLog level counts (wide reduceByKey): ", counts_rdd.collect())

print("\nTotal log lines: ", rdd.count())

time.sleep(120)

spark.stop()