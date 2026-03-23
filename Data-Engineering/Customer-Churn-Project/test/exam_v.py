from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder \
    .appName("ETL") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .getOrCreate()

df = spark.read.csv("file:///Users/orbin/Documents/GitHub/Machine_Learning/Data-Engineering/Customer-Churn-Project/test/housing.csv", header=True, inferSchema=True)

df = df.dropDuplicates()

mean_area = df.selectExpr("avg(area)").first()[0]
mean_bed = df.selectExpr("avg(bedrooms)").first()[0]
mode_loc = df.groupBy("location").count().orderBy("count", ascending=False).first()[0]

df = df.na.fill({
    'area': mean_area,
    'bedrooms': mean_bed,
    'location':mode_loc
})

q_low, q_high = df.approxQuantile("price", [0.5, 0.95], 0.01)
df = df.withColumn("price",
                   when(col("price") < q_low, q_low)
                   .when(col("price")> q_high, q_high)
                   .otherwise(col("price")))

df = df.withColumn("price_per_area", col("price")/col("area"))

df.write.mode("overwrite").parquet("file:///Users/orbin/Documents/GitHub/Machine_Learning/Data-Engineering/Customer-Churn-Project/test/output/data")
print("ETL Completed")


indexer = StringIndexer(inputCol="location", outputCol="loc_idx")
assembler = VectorAssembler(
    inputCols = ["area", "bedrooms", "loc_idx", "price_per_area"],
    outputCol="features"
)
scaler = StandardScaler(inputCol="features", outputCol="scaled")
rf = RandomForestRegressor(labelCol="price", featuresCol="scaled")
pipeline = Pipeline(stages=[indexer, assembler, scaler, rf])

train, test = df.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(train)
preds = model.transform(test)

evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")

print("RMSE: ", evaluator.setMetricName("rmse").evaluate(preds))
print("R2", evaluator.setMetricName("r2").evaluate(preds))

paramgrid=ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20]) \
    .addGrid(rf.maxDepth, [3, 5]) \
    .build()

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramgrid,
    evaluator=evaluator,
    numFolds=2
)

cv_model = cv.fit(train)

cv_model.bestModel.write().overwrite().save("file:///Users/orbin/Documents/GitHub/Machine_Learning/Data-Engineering/Customer-Churn-Project/test/output/model")

loaded=PipelineModel.load("file:///Users/orbin/Documents/GitHub/Machine_Learning/Data-Engineering/Customer-Churn-Project/test/output/model/")
loaded.transform(test).select("area", "price", "prediction").show()

spark.stop()