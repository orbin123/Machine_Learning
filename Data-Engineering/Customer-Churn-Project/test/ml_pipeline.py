from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import (
    StringIndexer, VectorAssembler, StandardScaler
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import (
    RandomForestRegressor, GBTRegressor, LinearRegression
)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder \
    .appName("SparkML") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", 8) \
    .getOrCreate()

df = spark.read.csv("file:///Users/orbin/Documents/GitHub/Machine_Learning/Data-Engineering/Customer-Churn-Project/test/Housing.csv", header=True, inferSchema=True)

binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"] 

for c in binary_cols:
    df = df.withColumn(c, when(col(c)=="yes", 1).otherwise(0))

furnishing_indexer = StringIndexer(
    inputCol="furnishingstatus",
    outputCol = "furnishing_index",
    handleInvalid = "keep"
)

df = df.withColumn("price_per_sqft", col("price") / col("area"))
df = df.withColumn("total_rooms", col("bedrooms") + col("bathrooms"))

feature_cols = ["area", "bedrooms", "bathrooms", "stories", "mainroad",
                "guestroom", "basement", "hotwaterheating", "airconditioning",
                "parking", "prefarea", "price_per_sqft", "total_rooms"]

assembler = VectorAssembler(inputCols = feature_cols, outputCol="features_raw")

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True, 
    withStd = True
)

train, test = df.randomSplit([0.8, 0.2], seed=42)
print(f"train: {train.count()} rows, Test: {test.count()} rows")

rf = RandomForestRegressor(
    labelCol="price",
    featureCol="features",
    numTrees=100,
    maxDepth=10, 
    seed=42
)

pipeline_rf =Pipeline(stages=[furnishing_indexer, assembler, scaler, rf])

model_rf = pipeline_rf.fit(train)

predictions_rf = model_rf.transform(test)
predictions_rf.select("price", "predicition").show(10)

# Evaluation
evaluator_rmse = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="rmse"
)
evaluator_mae = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="mae"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="price", 
)

rmse=evaluator_rmse.evaluate(predictions_rf)
mae=evaluator_mae.evaluate(predictions_rf)
r2=evaluator_r2.evaluate(predictions_rf)

print(f"RMSE: {rmse:,.0f}")
print(f"MAE: {mae:,.0f}")
print(f"R2: {r2:,.0f}")

rf_model=model_rf.stages[-1]
importances=rf_model.featureImportances
print("\nFeature Importances")
for i, name in enumerate(feature_cols):
    print(f"{name}: {importances[i]:.4f}")

rf_tuning = RandomForestRegressor(
    labelCol="price", featureCol="features", seed=42
)

pipeline_tune=Pipeline(stages=[furnishing_indexer, assembler, scaler, rf_tuning])

paramGrid = ParamGridBuilder() \
    .addGrid(rf_tuning.numTrees, [50, 100, 200]) \
    .addGrid(rf_tuning.maxDepth, [5, 10, 15]) \
    .addGrid(rf_tuning.minInstancesPerNode, [1, 5]) \
    .build()

cv = CrossValidator(
    estimator=pipeline_tune,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator_rmse,
    numFolds=3,
    parallelism=4,
    seed=42
)

cv_model=cv.fit(train)
best_model=cv_model.bestModel

best_predictions=best_model.transform(test)
best_rmse=evaluator_rmse.evaluate(best_predictions)
best_r2=evaluator_r2.evaluate(best_predictions)

print(f"RMSE: {best_rmse}")
print(f"R2: {best_r2}")

best_rf = best_model.stages[-1]
print(f"numTrees: {best_rf.getNumTrees}")
print(f"maxDepth: {best_rf.getOrDefault('maxDepth')}")

# Save
best_model.save("output/housing_model")


loaded_model = PipelineModel.load("output/housing_model")

new_data=test.drop("price")
new_predictions = loaded_model.transform(test)

new_predictions.select(
    "area", "bedrooms", "bathrooms", "stories", "parking",
    "prediction"
).coalesce(1) \
    .write.model("overwrite") \
    .parquet("output/housing_predictions")

spark.stop()