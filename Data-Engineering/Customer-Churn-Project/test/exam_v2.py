from spark.sql import SparkSession
from spark.sql.functions import col, when 

spark = SparkSession.builder.appName('etl').getOrCreate()

df = spark.read.csv("housing.csv", header=True, inferSchema=True)

# Delete Duplicate
df = df.dropDuplicates()

# Fill missing values
mean_area = df.selectExpr("avg(area)").first()[0]
mean_bed = df.selectExpr("avg(bedrooms)").first()[0]
mode_loc = df.groupBy("location").count().orderBy("count", ascending=False).first()[0]

df = df.na.fill({
    "area": mean_area,
    "bedrooms": mean_bed,
    "location": mode_loc
})

# Handle Outliers
q_low, q_high = df.approxQuantile("price", [0.05, 0.95], 0.01)

df = df.withColumn(
    "price",
    when(col("price") < q_low, q_low)
    .when(col("price" > q_high, q_high)
    .otherwise(col("price")))
)

# feature engineering
df = df.withColumn("price_per_area", col("price")/col("area"))

df.write.mode("overwrite").parquet("output/data")

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

df = spark.read.parquet("output/data")

indexer = StringIndexer(inputCol="location", outputCol="loc_idx")
assembler = VectorAssembler(
    inputsCols = ["area", "bedrooms", "loc_idx", "price_per_area"],
    outputCol = "features"
)
scaler = StandardScaler(inputCol="features", outputCol="scaled")
rf = RandomForestRegressor(labelCol="price", featuresCol="scaled")

pipeline = Pipeline(stages=[indexer, assembler, scaler, rf])

train, test = df.randomSplit([0.8, 0.2 ], seed=42)

model = pipeline.fit(train)
preds = model.transform(test)

evaluator = RandomForestRegressor(labelCol="price", predictionCol="prediction")

print("RMSE: ", evaluator.setMetricName("rmse").evaluate(preds))
print("r2", evaluator.setMetricName("r2").evaluate(preds))

param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20]) \
    .addGrid(rf.maxDepth, [3, 5]) \
    .build()

cv = CrossValidator(
    estimator = pipeline, 
    estimatroParamMaps = param_grid, 
    evaluator = evaluator,
    numFolds=2
)

spark.stop()