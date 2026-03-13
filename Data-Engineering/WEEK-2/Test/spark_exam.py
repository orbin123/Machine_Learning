from pyspark.sql import SparkSession

spark = SparkSession

sc = spark.sparkContext

df = sc.parallelize("file:\\\/Users/orbin/Documents/GitHub/Machine_Learning/Data-Engineering/WEEK-2/Test/users.csv")
df.show()