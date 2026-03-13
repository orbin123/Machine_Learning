from pyspark.sql import  SparkSession
from pyspark.ml import Pipeline 
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression

spark =  SparkSession.builder.appName("ExamML").master("local[*]").getOrCreate()

data = [
    (25, "Mumbai", 3000.0, 1),
    (40, "Delhi", 7000.0, 0),
    (35, "Mumbai", 5000.0, 1),
    (28, "Chennai", 2000.0, 0),
    (55, "Delhi", 9000.0, 1),
    (22, "Chennai", 1500.0, 0),
]

df = spark.createDataFrame(data, ["age", "city", "spend", "label"])

indexer = StringIndexer(inputCol="city", outputCol="city_idx")

assembler = VectorAssembler(inputCols=["age", "city_idx", "spend"], outputCol="features")

lr = LogisticRegression(maxIter=5)

pipeline = Pipeline(stages=[indexer, assembler, lr])

model = pipeline.fit(df)
predictions = model.transform(df)
predictions.select("age", "city", "label", "prediction").show()

spark.stop()