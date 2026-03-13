from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder \
    .appName("MLlibPipeline") \
    .master("local[2]") \
    .getOrCreate()

# Step 1: Read dataset from CSV - is_active is out binary target 
df = spark.read.csv("file:////Users/orbin/Documents/GitHub/Machine_Learning/Data-Engineering/WEEK-2/Mini-Project/users.csv", header=True, inferSchema=True)

# Rename target column to "label" - Spark ML convention
df = df.withColumnRenamed("is_active", "label")

df.show()

# Step 2: Stage 1 - StringIndexer for "city"
city_indexer = StringIndexer(inputCol="city", outputCol="city_index")

# Step 3: Stage 2 - VectorAssembler 
assembler = VectorAssembler(
    inputCols = ["age", "city_index", "monthly_spend"],
    outputCol="features"
)

# Step 4: Stage 3 - Logistic Regression 
lr = LogisticRegression(maxIter=10)

# Step 5: Build the Pipeline (3 stages in order)
pipeline = Pipeline(stages=[city_indexer, assembler, lr])

# Step 5: fit the pipeline on the data
model = pipeline.fit(df)

# Step 7: Generate Prediction on the same data
predictions = model.transform(df)

predictions.select("user_id", "city", "features", "label", "prediction").show(truncate=False)

spark.stop()