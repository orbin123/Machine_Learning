from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Spark SQL
spark = SparkSession.builder \
    .appName("examApp") \
    .master("local[*]") \
    .getOrCreate()

sc = spark.sparkContext

data = [
    (1, 25, "Mumbai", 3000.0, 1),
    (2, 40, "Delhi", 7000.0, 0),
    (3, 35, "Mumbai", 5000.0, 1),
    (4, 28, "Chennai", 2000.0, 0),
    (5, 55, "Delhi", 9000.0, 1),
]

df = spark.createDataFrame(data, ["user_id", "age", "city", "monthly_spend", "is_active"])
df.show()

df.createOrReplaceTempView("users")

spark.sql("""
    SELECT * FROM users
    WHERE is_active = 1
    ORDER BY monthly_spend DESC
""").show()

result = spark.sql("""
    SELECT city, COUNT(*) AS cnt, AVG(monthly_spend) AS avg_spend
    FROM users
    GROUP BY city
    ORDER BY avg_spend DESC
""")

result.show()

# Spark DataFrame
result2 = result.withColumn("tag", when(col("avg_spend")> 4000, "High").otherwise("Low"))
result.show()


# Spark RDD
logs = [
    "ERROR disk full",
    "INFO started",
    "ERROR timeout",
    "WARN low memory",
    "ERROR crash",
]
rdd = sc.parallelize(logs) 

errors = rdd.filter(lambda x: x.startswith("ERROR"))

pairs = rdd.map(lambda x: (x.split(" ")[0], 1))
count = pairs.reduceByKey(lambda a, b: a + b)

print(errors.collect())
print(count.collect())


import time 
time.sleep(120)
spark.stop()