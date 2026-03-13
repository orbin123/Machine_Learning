from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import time 


spark = SparkSession.builder \
    .appName("MyFirstSparkApp") \
    .master("local[2]") \
    .getOrCreate()

# Step 1: Read DataFrame from CSV file
df = spark.read.csv("file:////Users/orbin/Documents/GitHub/Machine_Learning/Data-Engineering/WEEK-2/Mini-Project/users.csv", header=True, inferSchema=True)

df.show()
df.printSchema()

# Step 2: Register as a temporary view
df.createOrReplaceTempView("users")

# Step 3: SQL Query 1 — Filtering + Ordering
query1 = """
    SELECT user_id, city, monthly_spend
    FROM users
    WHERE is_active = 1 AND monthly_spend > 5000
    ORDER BY monthly_spend DESC
"""

spark.sql(query1).show()

# Step 4: SQL Query 2 — Grouping + Aggregation
query2 =  """
    SELECT  city,
        COUNT(*) AS user_count,
        ROUND(AVG(monthly_spend), 2) AS avg_spend
    FROM users
    GROUP BY city
"""

result = spark.sql(query2)
result.show()

# Step 5:  Convert SQL result back to Dataframe, apply transformation
transformed = result.withColumn(
    "spend_tier",
    when(col("avg_spend") > 7000, "High").otherwise("Low")
)
transformed.show()

time.sleep(120)

spark.stop()