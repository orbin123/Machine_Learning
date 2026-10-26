# Machine Learning Revision Questions


## #Ensemble Learning

### #Questions

How do you train and compare Bagging, Random Forest, Gradient Boosting, and Stacking models, tune the boosting learning rate with early stopping, and evaluate the models?

```python
# Answer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
search= GridSearchCV(GradientBoostingClassifier(n_estimators=300, n_iter_no_change=5, random_state=42), {"learning_rate": [0.03, 0.1, 0.2]},cv=5, scoring="f1_macro", n_jobs=-1).fit(X_train, y_train)
gb = search.best_estimator_

models = {
    "Bagging": BaggingClassifier(n_estimators=100, random_state=42),
    "Random Forest": rf,
    "Boosting": gb,
    "Stacking": StackingClassifier(
        [("rf", rf), ("gb", gb)], final_estimator=LogisticRegression(), cv=5 
    ),

}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "accuracy:", round(accuracy_score(y_test, pred), 3), "macro-F1:", round(f1_score(y_test, pred, average="macro"), 3))

print("Best learning rate: ", search.best_params_["learning_rate"])
print("Boosting rounds used", gb.n_estimators_)
```


## #Neural Networks

### #Questions

How do you implement backpropagation from scratch with NumPy?

```python
# Answer
import numpy as np

X = np.array([0, 0], [0, 1], [1, 0], [1, 1])
y = np.array([[0], [1], [1], [0]])

sigmoid = lambda z: 1/ (1+ np.exp(-z))
rng = np.random.default_rng(0)
W1, W2 = rng.normal(size=(2, 4)), rng.normal(size=(4, 1))
b1, b2 = np.zeros((1, 4)), np.zeros((1, 1))

for _ in range(10000):
    # Foward pass 
    h = sigmoid(X @ W1 + b1)
    pred = sigmoid(h @ W2 + b2)

    # Backpropagate the error using the chain rule
    d2 = (pred - y) / len(X)
    dW2, db2 = h.T @ d2, d2.sum(axis=0, keepdims=True)
    d1 = (d2 @ W2.T) * h * (1-h)
    dW1, db1 = X.T @ d1, d1.sum(axis=0, keepdims=True)

    # Gradient Descent
    W1 -= dW1 
    b1 -= db1 
    W2 -= dW2
    b2 -= db2

print((pred > 0.5).astype(int).ravel())
```


## #Time Series Analysis

### #Questions

How do you decompose a time series into trend, seasonality, and residuals?

```python
# Answer
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from statsmodels.tsa.seasonal import seasonal_decompose

dataset = sm.datasets.get_rdataset("AirPassengers").data

series = dataset["value"]
series.index = pd.date_range(start="1949-01-01", periods=len(series), freq="MS")

# decompose 
result = seasonal_decompose(series, model="multiplicative", period=12)

# Extract individual components 
trend = result.trend 
seasonal = result.seasonal
residuals = result.resid 

# Plot 
fig = result.plot()
plt.tight_layout()
plt.show()
```

### #Questions

How do you check stationarity using differencing, transformations, and decomposition?

```python
# Answer
import numpy as np
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.seasonal import STL 

def check_adf(x):
    p = adfuller(x.dropna())[1]
    print(f"p-value={p:.4f} -> {'stationary' if p < 0.05 else 'non-stationary'}")

# 1. Differencing: remove trend and yearly seasonality
differenced = series.diff(12).diff()
check_adf(differenced)

# 2. Transfomation: log can stabilize growing vairance
transformed = np.log(series)
check_adf(transformed)

# 3. Decomposition: Remove trend and seasonality
remainder = STL(np.log(series), period=12).fit().resid
check_adf(remainder)
```

### #Questions

How do you use ACF and PACF plots to choose ARIMA parameters?

```python
# Answer
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

dataset = sm.datasets.get_rdataset("AirPassengers").data

series = dataset["value"]

y = np.log(series)

train, test = y.iloc[:-12], y.iloc[-12:]

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
plot_acf(train.diff().dropna(), lags=24, ax=ax[0])
plot_pacf(train.diff().dropna(), lags=24, method="ywm", ax=ax[1])
plt.tight_layout()
plt.show()
```

### #Questions

How do you fit and compare SARIMA and exponential-smoothing forecasts?

```python
# Answer
import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import (SimpleExpSmoothing, Holt, ExponentialSmoothing)
from sklearn.metrics import mean_squared_error

y = np.log(series)

train, test = y.iloc[:-12], y.iloc[-12:]

models = {
    "SARIMA": SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False),
    "Single": SimpleExpSmoothing(train,initialization_method="estimated").fit(), 
    "Double": Holt(train,initialization_method="estimated").fit(),
    "Holt-Winters": ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12,initialization_method="estimated").fit(),
}

forecasts = {name: model.forecast(12) for name, model in models.items()}

for name, forecast in forecasts.items():
    rmse = np.sqrt(mean_squared_error(np.exp(test), np.exp(forecast)))
    print(f"{name} RMSE: {rmse:.2f}")

plt.plot(series.index, series, label="Actual")
for name, forecast in forecasts.items():
    plt.plot(test.index, np.exp(forecast), label=name)
    plt.legend()
    plt.show()
```

### #Questions

How do you build and visualize a Prophet forecast?

Assume `df` has `ds` (dates), `y` (values), and `promo` (regressor values). Replace the example event date and country with ones relevant to your data. Prophet needs regressor values for the dates you predict.

```python
# Answer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Make a simple series with a trend and yearly seasonality
dates = pd.date_range("2022-01-01", periods=730, freq="D")
t = np.arange(len(dates))
df = pd.DataFrame({
    "ds": dates,
    "y": 10 + 0.03 * t + 5 * np.sin(2 * np.pi * t / 365),
})

# Fit and forecast 90 days
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

model.plot(forecast)
plt.show()
```


## #Recurrent Neural Networks (RNNs)

### #Questions

How do you build a simple RNN from scratch with NumPy?

```python
# Answer
import numpy as np

d_in, d_h, d_out, lr = 1, 4, 1, 0.1
Wxh = np.random.randn(d_h, d_in) * 0.1
Whh = np.random.randn(d_h, d_h) * 0.1
Why = np.random.randn(d_out, d_h) * 0.1

# Inputs & Targets (Simple 1D arrays)
X = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
Y = [np.array([2.0]), np.array([3.0]), np.array([4.0])]

# Forward Pass
H = [np.zeros(d_h)]
Y_pred = []

for x in X:
    h_next = np.tanh(Wxh @ x + Whh @ H[-1])
    H.append(h_next)
    Y_pred.append(Why @ h_next)

# Backward Pass
dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
dh_next = np.zeros(d_h)

for t in reversed(range(len(Y))):
    dy = Y_pred[t] - Y[t]
    dWhy += np.outer(dy, H[t + 1])

    dh = Why.T @ dy + dh_next 
    dh_raw = (1 - H[t + 1] ** 2) * dh

    dWxh += np.outer(dh_raw, X[t])
    dWhh += np.outer(dh_raw, H[t])
    dh_next = Whh.T @ dh_raw 

# Update Weights
Wxh -= lr * dWxh 
Whh -= lr * dWhh 
Why -= lr * dWhy

print("RNN step Complete")
```

### #Questions

How do you implement LSTM and GRU networks with TensorFlow/Keras?

```python
# Answer
import tensorflow as tf 

x_train = np.stack(X)[None, :, :]
y_train = np.stack(Y)[None, :, :]

# LSTM
lstm_model = tf.keras.Sequential([
    tf.keras.Input(shape=(len(X), d_in)),
    tf.keras.layers.LSTM(4, return_sequences=True),
    tf.keras.layers.Dense(d_out),
])

lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(x_train, y_train, epochs=200, verbose=0)
print("LSTM Predictions: ", lstm_model.predict(x_train, verbose=0).ravel())

# GRU
gru_model = tf.keras.Sequential([
    tf.keras.Input(shape=(len(X), d_in)),
    tf.keras.layers.GRU(4, return_sequences=True),
    tf.keras.layers.Dense(d_out)
])

gru_model.compile(optimizer="adam", loss="mse")
gru_model.fit(x_train, y_train, epochs=200, verbose=0)
print("GRU predictions: ", gru_model.predict(x_train, verbose=0).ravel())
```


## #Convolutional Neural Networks (CNNs)

### #Questions

How do you build and evaluate a CNN for image classification?

```python
# Answer
import tensorflow as tf 
from tensorflow.keras import layers, models 

# Load Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize 
x_train = x_train/255.0
x_test = x_test/255.0


# Add channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPool2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPool2D((2, 2)),

    layers.Flatten(),

    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=5, batch_size=32)

# Test
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy: ", accuracy)
```


## #OpenCV

### #Questions

How do you read, preprocess, augment, annotate, and display an image with OpenCV?

```python
# Answer
import cv2
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

img = cv2.imread("image.jpeg")

h, w = img.shape[:2]

# Preprocessing
resized = cv2.resize(img, (256, 256))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
normalized = resized.astype(np.float32) / 255.0

# Geometric augmentation
flipped = cv2.flip(img, 1)
M = cv2.getRotationMatrix2D((w/2, h/2), 15, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))

# Color Augmentation
adjusted = cv2.convertScaleAbs(img, alpha=1.2, beta=20)

overlay = img.copy()
cv2.putText(overlay, "Sample", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.rectangle(overlay, (20, 60), (150, 140), (0, 0, 255), -1)
marked = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

plt.imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
```


## #Generative Models

### #Questions

How do you build a simple GAN with TensorFlow/Keras?

```python
# Answer
import tensorflow as tf

NOISE_DIM = 100

G = tf.keras.Sequential([
    tf.keras.layers.Input((NOISE_DIM, )),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(28 * 28, activation="tanh"),
    tf.keras.layers.Reshape((28, 28, 1)),
])

D = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

loss = tf.keras.losses.BinaryCrossentropy()
g_opt = tf.keras.optimizers.Adam(1e-4)
d_opt = tf.keras.optimizers.Adam(1e-4)

(x, _), _ = tf.keras.datasets.mnist.load_data()
x = (x.astype("float32")[..., None] - 127.5) / 127.5

for _ in range(5):
    for real in tf.data.Dataset.from_tensor_slices(x).shuffle(10000).batch(128):
        noise = tf.random.normal((tf.shape(real)[0], NOISE_DIM))

        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = G(noise, training=True)
            real_pred = D(real, training=True)
            fake_pred = D(fake, training=True)

            g_loss = loss(tf.ones_like(fake_pred), fake_pred)
            d_loss = (
                loss(tf.ones_like(real_pred), fake_pred) + loss(tf.zeros_like(fake_pred), fake_pred)
            )

        g_opt.apply_gradients(zip(gt.gradient(g_loss, G.trainable_variables), G.trainable_variables))
        d_opt.apply_gradients(zip(dt.gradient(d_loss, D.trainable_variables), D.trainable_variables))
```


## #Attention and Transformer Models

### #Questions

How do you implement a Transformer block with multi-head self-attention?

```python
# Answer
import tensorflow as tf 
from tensorflow.keras import layers 

class TransformerBlock(layers.Layer):
    def __int__(self, d_model, num_heads):
        super().__init__()

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
        )

        self.feed_forward = tf.keras.Sequential([
            layers.Dense(4 * d_model, activation="relu"),
            layers.Dense(d_model)
        ])

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        # Each token attends to the other tokens 
        attended = self.attention(query=x, key=x, value=x)
        x = self.norm1(x + attended)

        # Transform each token independently
        fed = self.feed_forward(x)
        return self.norm2(x + fed)
```

### #Questions

How do you fine-tune BERT for text classification?

```python
# Answer
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

data = Dataset.from_dict({
    "text": ["Great product!", "Terrible experience."],
    "label": [1, 0],
})

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

data = data.map(tokenize, batched=True, remove_columns=["text"])

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

args = TrainingArguments(
    output_dir="./bert-results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=data,
)

trainer.train()
```


## #Natural Language Processing (NLP)

### #Questions

How do you clean text and apply stemming, lemmatization, TF-IDF, and Word2Vec?

```python
# Answer
import nltk
import string
import contractions
import spacy 

from textblob import TextBlob
from nltk.tokenize import sent_tokenize 
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

nltk.download("punkt_tab")
nltk.download("stopwords")

nlp = spacy.load('en_core_web_sm')

# Example text
text = "I can't believe how gud this movie was! The actors were running everywhere."

def clean_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = str(TextBlob(text).correct())
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

cleaned_text = clean_text(text)
print("Cleaned: ", cleaned_text)
```

### #Questions

How do you compare stemming and lemmatization and build text representations?

```python
# Answer
# Compare stemming and lemmatization
porter = PorterStemmer()
snowball = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

for word in cleaned_text.split():
    print(word, "-> ", porter.stem(word), snowball.stem(word), lemmatizer.lemmatize(word))

doc = nlp(text)

for token in doc:
    print(token.text, token.pos_, token.lemma_, token.ent_type_)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)

print("Cosine Similarity: ", consine_similarity(tfidf_matix[0], tfidf_matix[1]))

sentences = [document.lower().split() for document in documents]

cbow = Word2Vec(sentences, vector_size=50, window=5, min_count=1, sg=0)
skip_gram = Word2Vec(sentences, vector_size=50, window=5, min_count=1, sg=1)
```

### #Questions

How do you perform named-entity recognition with spaCy?

```python
# Answer
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Openai was founded in San Fransisco"

for entity in nlp(text).ents:
    print(entity.text, entity.label_)
```

### #Questions

How do you analyze sentiment with VADER and TextBlob?

```python
# Answer
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob 

nltk.download("vader_lexicon")

text = "I really enjoyed this movie!"

vader = SentimentIntensityAnalyzer()
print("VADER", vader.polarity_scores(text))
print("TextBlob polarity:", TextBlob(text).sentiment.polarity)
```

### #Questions

How do you discover topics with Latent Dirichlet Allocation (LDA)?

```python
# Answer
from gensim.corpora import Dictionary
from gensim.models import LdaModel 

docs = [
    ["cats", "like", "sleeping"],
    ["dogs", "like", "walking"],
    ["cats", "and", "dogs"]
]

dictionary = Dictionary(docs)
corpus=[dictionary.doc2bow(doc) for doc in docs]

lda = LdaModel(corpus, id2Word=dictionary, num_topics=2, passes=5)
print(lda.print_topics())
```


## #Big Data

### #Questions

How do you implement a MapReduce word count with a mapper and reducer?

```python
# Answer
# mapper.py
import re
import sys 

for line in sys.stdin:
    words=re.findall(r"\w+", line.lower())
    for word in words:
        print(f"{word}\t1")

# reducer.py
import sys

current_word = None 
current_count = 0

for line in sys.stdin:
    word, count = line.strip().split("\t", 1)
    count = int(count)

    if word == current_word:
        print(f"{current_word}\t{current_count}")
    current_word = word
    current_count = count

if current_word is not None:
    print(f"{current_word}\t{current_count}")
```

### #Questions

How do you filter and group data with Spark SQL?

```python
# Answer
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("BigDataPractice").master("local[*]").getOrCreate()

data = [
    (1, 25, "Mumbai", 3000.0, 1),
    (2, 40, "Delhi", 7000.0, 0),
    (3, 35, "Mumbai", 5000.0, 1),
    (4, 28, "Chennai", 2000.0, 0),
    (5, 55, "Delhi", 9000.0, 1),
]

users = spark.createDataFrame(
    data, ["user_id", "age", "city", "monthly_spend", "is_active"]
)
users.createOrReplaceTempView("users")

spark.sql("""
    SELECT user_id, age, city, monthly_spend
    FROM users
    WHERE is_active = 1
    ORDER BY monthly_spend DESC
""")

city_summary = spark.sql("""
    SELECT city, COUNT(*) AS user_count, AVG(monthly_spend) AS avg_spend
    FROM users
    GROUP BY city
    ORDER BY avg_spend DESC
""")

city_summary.show()

spark.stop()
```

### #Questions

How do you filter error logs and count log levels with Spark RDDs?

```python
# Answer
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("RDDPractice").master("local[*]").getOrCreate()

logs = [
    "ERROR disk full",
    "INFO started",
    "ERROR timeout",
    "WARN low memory",
    "ERROR crash",
]

rdd = spark.sparkContext.parallelize(logs)

errors = rdd.filter(lambda line: line.startswith("ERROR"))
level_counts = (
    rdd.map(lambda line: (line.split()[0], 1)).reduceByKey(lambda count_a, count_b: count_a + count_b)
)

print(errors.collect())
print(level_counts.collect())

spark.stop()
```

### #Questions

How do you build a Spark MLlib classification pipeline?

```python
# Answer
from pyspark.sql import SparkSession 
from pyspark.ml import Pipeline 
from pyspark.ml.classification import LogisticRegression 
from pyspark.ml.feature import StringIndexer, VectorIndexer

spark = SparkSession.builder.appName("BigDataML").master("local[*]").getOrCreate()

data = [
    (25, "Mumbai", 3000.0, 1),
    (40, "Delhi", 7000.0, 0),
    (35, "Mumbai", 5000.0, 1),
    (28, "Chennai", 2000.0, 0),
    (55, "Delhi", 9000.0, 1),
    (22, "Chennai", 1500.0, 0),
]

df = spark.createDataFrame(data, ["age", "city", "spend", "label"])

city_indexer = StringIndexer(inputCol="city", outputCol="city_idx")
assembler = VectorIndexer(
    inputCols=["age", "city_idx", "spend"], outputCol="features"
)
classifier = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

pipeline = Pipeline(stages=[city_indexer, assembler, classifier])
model = pipeline.fit(df)
predictions=model.transform(df)
predictions.select("age", "city", "label", "prediction").show()

spark.stop()
```


## #Model Deployment

### #Questions

How do you install dependencies for the Flask model API?

```bash
# Answer
!pip install scikit-learn flask joblib
```

### #Questions

How do you train, evaluate, and save a model for deployment?

```python
# Answer
# train.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save
joblib.dump(
    {"model": model, "target_names": iris.target_names},
    "iris_model.joblib"
)
# Run it
```

### #Questions

How do you expose a saved model through a Flask prediction API?

```python
# Answer
# app.py
from flask import Flask, request, jsonify
import joblib 

app = Flask(__name__)

saved = joblib.load("iris_model.joblib")
model = saved["model"]
target_names = saved["target_names"]

@app.post("/predict")
def predict():
    features = request.get_json()["features"]

    prediction = model.predict([features])(0)
    return jsonify({"class": str(target_names[prediction])})

if __name__ == "__main__":
    app.run(debug=True)
```

### #Questions

How do you containerize the Flask API with Docker?

```dockerfile
# Answer
# Dockerfile.flask
FROM python:3.12-slim
WORKDIR /app
RUN pip install flask scikit-learn joblib
COPY app.py iris_model.joblib ./
EXPOSE 5000
CMD ["python", "app.py"]
```

### #Questions

How do you define the Flask service in Docker Compose?

```yaml
# Answer
# docker-compose.yml

services:
    flask:
        build:
            context:
            dockerfile: Dockerfile.flask 
        ports:
            - "5000:5000"
```

### #Questions

How do you define SageMaker inference functions?

```python
# Answer
# aws_code/inference.py
import json
import os
import joblib

def model_fn(model_dir):
    saved = joblib.load(os.path.join(model_dir, "iris_model.joblib"))
    return saved 

def input_fn(request_body, content_type):
    return json.loads(request)

def predict_fn(data, saved):
    prediction = saved["model"].predict([data["features"]])[0]
    return {"class": str(saved["target_names"][prediction])}

def output_fn(prediction, accept):
    return json.dumps(prediction)
```

### #Questions

How do you package and deploy a scikit-learn model to a SageMaker endpoint?

SageMaker hosts the model at a real-time endpoint. You need an AWS account, credentials, and a SageMaker execution role. The SageMaker scikit-learn containers support deploying scikit-learn models trained outside SageMaker.

```python
# Answer
import tarfiele
import sagemaker
from sagemaker.sklearn.model import SkLearnModel

session = sagemaker.Session()
role = sagemaker.get_execution_role()

# inside model.tar.gz
with tarfile.open("model.tar.gz", "w:gz") as archive:
    archive.add("iris_model.joblib")

model_data = session.upload_data(
    path="model.tar.gz",
    key_prefix="iris/model"
)

model = SKLearnModel(
    model_data=model_data,
    role=role,
    entry_point="inference.py",
    source_dir="aws_code",
    framework_version="1.4-2",
    py_version="py3"
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
)
```


## #MLOps

### #Questions

How do you initialize Git and DVC and track the Iris dataset?

```bash
# Answer
git init

dvc init
dvc add data/iris.csv
```

### #Questions

How do you store training parameters in a configuration file?

```yaml
# Answer
 # params.yaml
n_estimators: 100
```

### #Questions

How do you train a model, log a run to MLflow, and save its outputs?

DVC versions the dataset and pipeline outputs. MLflow records each training run’s parameters, metrics, and model. Running training through `dvc repro` connects the two: when inputs or parameters change, DVC reruns training and MLflow records the new run.

```python
# Answer
import json
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)

data = pd.read_csv("data/iris.csv")
X = data.drop(columns="target")
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("iris-classifier")

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"], random_state=42
    )

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_params(params)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.sklearn.log_model(model, name="iris_model")

    joblib.dump(model, "model.pkl")
    with open("metrics.json", "w") as f:
        json.dump({"test_accuracy": accuracy}, f)
```

### #Questions

How do you define and run a DVC training pipeline?

```yaml
# Answer
# dvc.yaml
stages:
    train:
        cmd: python train.py
        deps:
            - train.py
            - data/iris.csv

        params:
            - n_estimators
        outs:
            - model.pkl
        metrics:
            - metrics.json

# Run
dvc repro
```


## #ML CI/CD

### #Questions

How do you initialize a Git repository for the prediction API?

# Answer

```bash
mkdir iris-api
cd iris-api
git init
```

### #Questions

Which dependencies does the prediction API need?

# Answer

```txt
Flask
prometheus-client
pytest
```

### #Questions

How do you create a Flask prediction endpoint and expose Prometheus metrics?

This example uses a tiny Flask prediction endpoint. It counts predictions with Prometheus, which Grafana can display. The prediction is a placeholder for a real model.

```python
# Answer
# app.py
from flask import Flask, Response, jsonify
from prometheus_client import Counter, generate_latest

app = Flask(__name__)
PREDICTIONS = Counter("predictions_total", "Number of predictions")

@app.get("/predict")
def predict():
    PREDICTIONS.inc()
    return jsonify({"class": "setosa"})

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

### #Questions

How do you create a Dockerfile for the prediction API?

# Answer

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py
CMD ["python", "app.py"]
```

### #Questions

How do you test the prediction endpoint with pytest?

```python
# Answer
# test/test_app.py
from app import app

def test_predict():
    response = app.test_client().get("/predict")

    assert response.status_code == 200
    assert response.json == {"class": "setosa"}
```

### #Questions

How do you configure Prometheus to scrape API metrics?

# Answer

```yaml
global:
    scarpe_interval: 5s

scrape_configs:
    - job_name: iris-api
    static_configs:
        - targets: ["api:5000"]
```

### #Questions

How do you run the API, Prometheus, and Grafana with Docker Compose?

# Answer

```yaml
services:
    api:
        build:
        ports:
            - "5000:5000"

    promotheus:
        image: prom/prometheus
        ports:
            - "9090:9090"
        volumes:
            - ./promotheus.yml:/etc/prometheus/prometheus.yml
        depends_on:
            - api

    grafana:
        image: grafana/grafana-oss
        ports:
            - "3000:3000"
        depends_on:
            - "prometheus"
```

```bash
docker compose up --build
```

### #Questions

How do you configure GitHub Actions to run the tests?

# Answer

```yaml
name: CI

on: [push, pull_request]

jobs:
    test:
        runs-on: ubuntu-latest
        steps:
            - uses: actions/checkout@v6
            - uses: actions/setup-python@v5
            with:
                python-version: "3.12"
            - run: pip install -r requirements.txt
            - run: pytest -q
```

### #Questions

How do you commit the deployment project?

# Answer

```bash
git add .
git commit -m "Add minimal ML API, monitoring, and CI"
```

### #Questions

How do you push the deployment project to GitHub?

# Answer

```bash
git remote add origin #githubrepolink
git branch -M main
git push -u origin main
```


## #Tool-Calling Agents

### #Questions

How does a Streamlit client send a chat request to a FastAPI service?

You write ordinary Python functions. The LLM chooses which functions to call, with what arguments, and what to do with their results. LangChain manages that back-and-forth.

```python
# Answer
# Streamlit calls FastAPI
response = httpx.post(
    "http://localhost:8000/chat",
    json={
        "session_id": "chat-123",
        "message": user_input,
    }
    timeout=120,
)

st.write(response.json()["answer"])
```

### #Questions

How do you define Python functions as tools for an agent?

```python
# Answer
# Python functions
from langchain.tools import tool

@tool
def save_email_draft(to: str, subject:str, body:str) -> dict:
    return db.create_draft(to=to, subject=subject, body=body)

@tool
def schedule_email(draft_id: str, send_at: str) -> dict:
    return db.create_email_job(draft_id=draft_id, send_at=send_at)

@tool
def get_activity(date: str) -> list:
    return db.get_activity(date=date)
```

### #Questions

How do you connect a language model to tools and define its instructions?

```python
# Answer
# Connect models to tools
import os
from langchain.agents import create_agent
from langchain_groq import ChatGroq

agent = create_agent(
    model=ChatGroq(model=os.environ["GROQ_MODEL"]),
    tools = [save_email_draft, schedule_email, get_activity],
    system_prompt=(
        "Help users complete tasks. Ask when required details are missing. "
        "Use tool results to report what actually succeeded. "
        "Distinguish scheduled emails from sent emails."
    )
)
```

### #Questions

How does a FastAPI endpoint manage conversation history and invoke an agent?

```python
# Answer
# FastAPI serves and saves the messages
class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    history = db.load_messages(request.session_id)
    messages = history + [
        {"role": "user", "content": request.message}
    ]

    result = await agent.ainvoke(
        {"messages": messages},
        config={"recursion_limit": 20}
    )

    # Serialize and save the full conversation including tool messages
    db.replace_messages(request.session_id, result["messages"])

    return {"answer": result["messages"][-1].content}
```
