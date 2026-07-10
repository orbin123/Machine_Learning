# Machine Learning — Practical & Coding Assessment Guide

> This is the *coding* companion to `theory.md`. It predicts the practical
> questions you are likely to face in coding rounds, lab exams, and viva
> practicals, and gives **production-quality Python solutions** with complexity
> notes, alternatives, variations, and follow-ups. Machine Learning is usually
> tested **inside a Jupyter notebook**, so most sections are structured as
> **notebook cell workflows** — type every cell out by hand at least once.

## How to use this guide

- For each problem: read the **Problem Statement**, attempt it *yourself first*, then compare with the solution.
- Study the **Approach** before the code — examiners care more about your reasoning than syntax.
- For notebook workflows, run **cell by cell** and read the explanation under each cell.
- Rehearse the **Follow-up Questions** — that's where vivas are decided.
- Always be ready to say *why* you chose a technique (link back to `theory.md`).

## Contents

- [Section 0 — Environment & Imports Cheat-Sheet](#section-0--environment--imports-cheat-sheet)
- [Section A — The Complete End-to-End ML Notebook Workflow](#section-a--the-complete-end-to-end-ml-notebook-workflow)
- [Section B — Exploratory Data Analysis (Notebook)](#section-b--exploratory-data-analysis-notebook)
- [Section C — Data Cleaning: Missing Values & Outliers](#section-c--data-cleaning-missing-values--outliers)
- [Section D — Feature Scaling](#section-d--feature-scaling)
- [Section E — Encoding & Feature Engineering](#section-e--encoding--feature-engineering)
- [Section F — Feature Selection](#section-f--feature-selection)
- [Section G — Statistics & Hypothesis Testing (Notebook)](#section-g--statistics--hypothesis-testing-notebook)
- [Section H — Unsupervised Learning: K-Means & PCA](#section-h--unsupervised-learning-k-means--pca)
- [Section I — Pure-Python Coding Questions (from scratch)](#section-i--pure-python-coding-questions-from-scratch)
- [Coding Questions Bank (Easy / Medium / Hard)](#coding-questions-bank)
- [Exam & Viva Survival Tips](#exam--viva-survival-tips)

> **Environment note:** Solutions use Python 3 with `numpy`, `pandas`,
> `matplotlib`, `seaborn`, `scikit-learn`, and `scipy`. Cells are written so you
> can paste them sequentially into a notebook. Where a problem is pure algorithm
> practice, a plain-Python (from-scratch) version is given so you can be tested
> "without sklearn."

---

# Section 0 — Environment & Imports Cheat-Sheet

Memorize this starter cell — examiners often expect you to set up quickly.

```python
# --- Core stack ---
import numpy as np
import pandas as pd

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- scikit-learn: preprocessing, models, evaluation ---
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             mean_squared_error, r2_score)

# --- Reproducibility & display ---
RANDOM_STATE = 42
pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")
```

**Why memorize this:** in a timed lab you should not lose minutes recalling import paths. Knowing that scalers live in `sklearn.preprocessing`, imputers in `sklearn.impute`, and metrics in `sklearn.metrics` is itself an assessed skill.

---

# Section A — The Complete End-to-End ML Notebook Workflow

> **This is the single most important section.** ML practical exams almost always
> ask you to "build a model on this dataset." The examiner is really grading
> whether you follow the full pipeline: load → EDA → clean → preprocess →
> train → tune → evaluate → interpret. Below is a **template you can adapt to any
> tabular dataset**, split into notebook cells with explanations.

## Practical Question 1 — Build a full classification pipeline

**Difficulty:** Medium
**Estimated Time:** 45–60 minutes
**Concepts Tested:** end-to-end workflow, train/test split discipline, pipelines, leakage prevention, evaluation, interpretation

**Problem Statement**
Given a tabular dataset with a binary target, build a complete, leakage-free ML pipeline that predicts the target, and report honest performance with interpretation.

**Approach (step-by-step)**
1. Load and inspect the data (shape, types, target balance).
2. EDA: distributions, missingness, correlations.
3. **Split first** (train/test) to prevent leakage.
4. Build a preprocessing `Pipeline` (impute → encode → scale) fit on train only.
5. Train a baseline model, then a stronger model.
6. Cross-validate and tune hyperparameters on the training set.
7. Evaluate once on the test set with appropriate metrics.
8. Interpret (feature importance, confusion matrix) and note deployment/monitoring.

### Cell 1 — Import libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")
```
*Explanation:* one import cell up front keeps the notebook clean. `ColumnTransformer` lets us apply different preprocessing to numeric vs categorical columns; `Pipeline` chains everything so preprocessing is refit correctly inside cross-validation (no leakage).

### Cell 2 — Load the dataset

```python
# Example uses the Titanic dataset shipped with seaborn; swap for your file.
df = sns.load_dataset("titanic")
# For a CSV in an exam:  df = pd.read_csv("data.csv")
print(df.shape)
df.head()
```
*Explanation:* Always print `shape` first — it anchors your sense of scale (rows × columns). `head()` shows the columns and a sample of values so you understand what you're working with.

### Cell 3 — Explore / understand the dataset

```python
df.info()                       # dtypes + non-null counts (reveals missingness)
display(df.describe(include="all").T)   # summary stats for all columns
print("\nTarget balance:\n", df["survived"].value_counts(normalize=True))
print("\nMissing per column:\n", df.isna().sum().sort_values(ascending=False))
```
*Explanation:* `info()` reveals dtypes and missing counts in one shot. `describe(include="all")` gives numeric summaries (mean/std/quartiles) plus categorical counts. **Checking target balance is essential** — if it's, say, 90/10, accuracy is a misleading metric and we'll need precision/recall/AUC.

### Cell 4 — Quick EDA visualizations

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(df["age"], kde=True, ax=axes[0]).set_title("Age distribution")
sns.boxplot(x="survived", y="fare", data=df, ax=axes[1]).set_title("Fare by survival")
sns.countplot(x="pclass", hue="survived", data=df, ax=axes[2]).set_title("Survival by class")
plt.tight_layout(); plt.show()
```
*Explanation:* A histogram shows the shape/skew of a numeric feature (age is right-skewed with missing values); a box plot compares a numeric feature across the target (higher fares survived more); a count plot exposes a categorical relationship (1st class survived more). This is where hypotheses about predictive features form.

### Cell 5 — Define features/target and split FIRST

```python
# Choose a manageable feature set for the demo
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = df[features].copy()
y = df["survived"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(X_train.shape, X_test.shape)
```
*Explanation:* **Splitting before any imputation/scaling is the golden rule** that prevents data leakage — test statistics must never influence training. `stratify=y` keeps the class ratio identical in train and test, important for imbalanced targets. `random_state` makes the split reproducible.

### Cell 6 — Data cleaning & preprocessing pipeline

```python
numeric_features = ["age", "sibsp", "parch", "fare"]
categorical_features = ["pclass", "sex", "embarked"]

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),   # median: robust to skew/outliers
    ("scale", StandardScaler()),                     # standardize for logistic regression
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),  # mode for categoricals
    ("onehot", OneHotEncoder(handle_unknown="ignore")),   # nominal -> one-hot, no false order
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features),
])
```
*Explanation:* Each column type gets the right treatment: numeric columns are **median-imputed** (robust to skew) then **standardized**; categorical columns are **mode-imputed** then **one-hot encoded** (`handle_unknown="ignore"` prevents crashes if a category appears only in test). Wrapping this in a `ColumnTransformer` means it's all **fit on training data only** when we call `.fit`, killing leakage.

### Cell 7 — Model training (baseline)

```python
clf = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
clf.fit(X_train, y_train)   # preprocessing + model fit together, on train only
```
*Explanation:* We start with **logistic regression** as an interpretable **baseline** — always beat a simple model before reaching for complex ones. Because preprocessing lives inside the same `Pipeline`, calling `.fit` learns imputation/scaling/encoding parameters from the training fold only, then trains the model on the transformed data.

### Cell 8 — Cross-validation & hyperparameter tuning

```python
rf = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
])

param_grid = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_leaf": [1, 5],
}
grid = GridSearchCV(rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 4))
```
*Explanation:* `GridSearchCV` with `cv=5` runs 5-fold cross-validation **inside the training set**, refitting the *entire pipeline* on each fold so preprocessing never sees the validation fold — the correct way to avoid leakage during tuning. We score with **ROC-AUC** (threshold-independent, good for imbalance) rather than accuracy. `model__` prefix reaches into the pipeline step named `model`.

### Cell 9 — Final evaluation on the untouched test set

```python
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("Test ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix"); plt.show()
```
*Explanation:* We touch the **test set only once**, at the very end, for an unbiased estimate. `classification_report` gives precision/recall/F1 per class (not just accuracy); the **confusion matrix** shows *where* errors happen (false positives vs false negatives), which matters when error costs differ. Reporting ROC-AUC alongside makes the evaluation threshold-independent.

### Cell 10 — Interpretation

```python
# Pull feature names out of the fitted ColumnTransformer
ohe = best_model.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
cat_names = ohe.get_feature_names_out(categorical_features)
feature_names = np.r_[numeric_features, cat_names]

importances = best_model.named_steps["model"].feature_importances_
imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

sns.barplot(x=imp.values, y=imp.index)
plt.title("Feature importances"); plt.xlabel("Importance"); plt.show()
```
*Explanation:* Interpretation turns a black box into insight. Random-forest **feature importances** show which features drive predictions (typically `sex`, `fare`, `pclass` for Titanic) — you must be able to *explain the model's story*, not just its score. In an exam, narrate: "Being female and in a higher class most increases survival probability, consistent with the EDA."

### Cell 11 — Deployment & monitoring notes (viva talking points)

```python
# import joblib
# joblib.dump(best_model, "titanic_model.joblib")   # persist the WHOLE pipeline
```
*Explanation:* Saving the **entire pipeline** (not just the model) guarantees identical preprocessing at inference — avoiding **train/serve skew**. In production you'd expose it behind an API, log predictions, and **monitor data/concept drift**, retraining when live performance decays. Examiners love when you mention monitoring — it shows you think beyond the notebook.

**Complexity note**
- Training cost is dominated by the model (Random Forest: ~O(n·log n·features·trees)); preprocessing is roughly O(n·features). The exam value here is *correct methodology*, not asymptotic tuning.

### Alternative Solution
For a **regression** target, swap: `StandardScaler` stays, use `LinearRegression`/`RandomForestRegressor`, and evaluate with **RMSE/MAE/R²** instead of accuracy/AUC. The skeleton is identical — which is the point of learning the template.

### Interview Variations
- Do it for a **regression** dataset (predict `fare`).
- Handle a **multiclass** target (`pclass`) — metrics become macro/micro averaged.
- Add **class_weight="balanced"** for an imbalanced target and compare.

### Common Follow-up Questions
- *Where exactly could leakage occur here?* Any preprocessing fit before the split, or tuning on the test set.
- *Why ROC-AUC over accuracy?* Threshold-independent and robust to imbalance.
- *Why put preprocessing in a Pipeline?* Correct CV, no leakage, reproducible inference.

---

# Section B — Exploratory Data Analysis (Notebook)

## Practical Question 2 — Perform EDA and summary statistics

**Difficulty:** Easy–Medium
**Estimated Time:** 25 minutes
**Concepts Tested:** summary statistics, distributions, skewness, correlation, visualization

**Problem Statement**
Given a dataset, produce a concise EDA: summary statistics, distribution shapes, missingness, and the correlation structure, and state what you learned.

**Approach**
1. Summary statistics (`describe`) and dtypes.
2. Univariate: histograms + skewness.
3. Bivariate: box plots and scatter plots vs the target.
4. Multivariate: correlation heatmap.
5. Write down insights.

### Cell 1 — Load & summary statistics

```python
import seaborn as sns, pandas as pd, numpy as np, matplotlib.pyplot as plt
df = sns.load_dataset("tips")     # bill, tip, sex, smoker, day, time, size
df.describe()                     # count, mean, std, min, quartiles, max
```
*Explanation:* `describe()` is your first numeric snapshot — central tendency (mean, 50%=median) and spread (std, IQR via 25%/75%). A large gap between `75%` and `max` hints at right-skew/outliers even before plotting.

### Cell 2 — Central tendency by hand (viva favourite)

```python
col = df["total_bill"]
print("Mean  :", round(col.mean(), 2))
print("Median:", round(col.median(), 2))
print("Mode  :", col.mode().iloc[0])
print("Std   :", round(col.std(), 2))
print("Variance:", round(col.var(), 2))
print("IQR   :", round(col.quantile(0.75) - col.quantile(0.25), 2))
print("Skew  :", round(col.skew(), 2))   # >0 right-skew, <0 left-skew
```
*Explanation:* Examiners often ask you to compute these explicitly. **Mean > median** here confirms a **right skew** (verified by positive `skew()`), which tells you the median is the more representative center and a log transform might help. Variance is in squared units; std is in original units (report std to humans).

### Cell 3 — Univariate distributions

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["total_bill"], kde=True, ax=ax[0]).set_title("total_bill (right-skewed)")
sns.histplot(np.log(df["total_bill"]), kde=True, ax=ax[1]).set_title("log(total_bill) (more symmetric)")
plt.tight_layout(); plt.show()
```
*Explanation:* The raw histogram shows a long right tail; the **log-transformed** version is more symmetric. This visually justifies transformation — a concrete, examinable demonstration of "how do you fix skew?"

### Cell 4 — Bivariate: box & scatter

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(x="day", y="total_bill", data=df, ax=ax[0]).set_title("Bill by day (spread + outliers)")
sns.scatterplot(x="total_bill", y="tip", data=df, ax=ax[1]).set_title("Tip vs bill (positive relationship)")
plt.tight_layout(); plt.show()
```
*Explanation:* The box plot compares distributions across a category and flags outliers (points past the whiskers = beyond 1.5·IQR). The scatter plot shows tip rises with bill — a **positive correlation** we'll quantify next.

### Cell 5 — Correlation heatmap

```python
corr = df.select_dtypes("number").corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation matrix"); plt.show()
```
*Explanation:* The heatmap reveals which numeric features move together. `total_bill`–`tip` shows strong positive correlation (~0.68). Diagonal is always 1. Use this to spot **predictive** features (correlated with target) and **redundant** ones (correlated with each other → multicollinearity).

### Cell 6 — Insight write-up (do this out loud in a viva)

```python
print("""
Insights:
1. total_bill is right-skewed -> use median; consider log transform.
2. tip increases with total_bill (r ~ 0.68) -> strong predictor.
3. Some high-bill outliers on weekends -> inspect, likely genuine.
4. size correlates modestly with both bill and tip.
""")
```
*Explanation:* EDA without a conclusion is incomplete. Stating insights (and their modelling implications) is exactly what graders reward.

### Alternative Solution
`df.profile_report()` from **ydata-profiling** auto-generates a full EDA report in one line — good to mention, but examiners usually want you to demonstrate the manual skills above.

### Interview Variations
- Compute **covariance** (`df.cov()`) and explain why correlation is preferred.
- Use a **pair plot** (`sns.pairplot(df, hue="time")`) for all pairwise relationships.
- Detect skew across all columns programmatically (`df.select_dtypes('number').skew()`).

### Common Follow-up Questions
- *Mean vs median — which and when?* Median for skewed/outlier data.
- *Correlation 0 means independent?* No — only no *linear* relationship.
- *What does a negative correlation look like on a scatter plot?* Downward cloud.

---

# Section C — Data Cleaning: Missing Values & Outliers

## Practical Question 3 — Handle missing values (multiple strategies)

**Difficulty:** Medium
**Estimated Time:** 25 minutes
**Concepts Tested:** missingness diagnosis, imputation (simple/KNN), missingness indicators, leakage

**Problem Statement**
A dataset has missing values in several columns. Diagnose the missingness, then apply and compare removal vs simple vs KNN imputation — without leaking test information.

**Approach**
1. Quantify missingness per column.
2. Decide drop vs impute per column.
3. Simple imputation (median/mode) inside a train-only fit.
4. KNN imputation for numeric columns.
5. Add a "was-missing" indicator.

### Cell 1 — Inspect missingness

```python
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
df = sns.load_dataset("titanic")[["age", "embarked", "fare", "deck", "survived"]].copy()

miss = df.isna().mean().sort_values(ascending=False)   # fraction missing
print(miss)
sns.heatmap(df.isna(), cbar=False)     # visual map of missingness
plt.title("Missingness map"); plt.show()
```
*Explanation:* `isna().mean()` gives the **fraction** missing per column. `deck` is ~77% missing (candidate to **drop**), `age` ~20% (candidate to **impute**), `embarked` ~0.2% (drop those 2 rows or mode-impute). The heatmap shows whether missingness clusters (a hint toward MAR/MNAR).

### Cell 2 — Split first, then decide per column

```python
from sklearn.model_selection import train_test_split
X = df.drop(columns="survived"); y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Drop the mostly-empty column
X_train = X_train.drop(columns="deck")
X_test  = X_test.drop(columns="deck")
```
*Explanation:* We **split before imputing** so imputation statistics come from training data only. `deck` is dropped because 77% missing leaves too little signal to impute reliably.

### Cell 3 — Simple imputation + missingness indicator (train-fit)

```python
from sklearn.impute import SimpleImputer

# Record the informative fact of missingness BEFORE filling
X_train["age_missing"] = X_train["age"].isna().astype(int)
X_test["age_missing"]  = X_test["age"].isna().astype(int)

num_imp = SimpleImputer(strategy="median")     # robust to age's skew
X_train[["age", "fare"]] = num_imp.fit_transform(X_train[["age", "fare"]])
X_test[["age", "fare"]]  = num_imp.transform(X_test[["age", "fare"]])   # transform only!

cat_imp = SimpleImputer(strategy="most_frequent")
X_train[["embarked"]] = cat_imp.fit_transform(X_train[["embarked"]])
X_test[["embarked"]]  = cat_imp.transform(X_test[["embarked"]])
```
*Explanation:* Note `fit_transform` on **train** but plain `transform` on **test** — the imputer learned the median/mode from train and merely applies it to test (no leakage). The **`age_missing` indicator** preserves the signal that a value *was* missing, which is often predictive (here, missing age correlated with lower survival).

### Cell 4 — KNN imputation (captures feature relationships)

```python
from sklearn.impute import KNNImputer
knn_imp = KNNImputer(n_neighbors=5)
# KNN uses distances -> scale first in a real pipeline; shown standalone here
num_cols = ["age", "fare"]
knn_filled_train = knn_imp.fit_transform(X_train[num_cols])
```
*Explanation:* `KNNImputer` fills a missing value using the average of its `k` most similar rows (by the other features) — better than a global median because it respects relationships, but slower and **scale-sensitive** (scale before using it). In a real pipeline, wrap scaler + KNNImputer together.

### Cell 5 — Compare strategies quickly

```python
print("Rows if we DROP any missing:", df.dropna().shape[0], "of", df.shape[0])
print("Median age (train):", round(X_train["age"].median(), 1))
```
*Explanation:* Dropping every row with a missing value here would discard a large fraction of data — showing *why* imputation is usually preferred over deletion.

**Complexity note**
- Simple imputation: O(n) per column. KNN imputation: O(n²·features) in the naive case (distance to all rows) — expensive on large data.

### Alternative Solution
**MICE** via `from sklearn.experimental import enable_iterative_imputer; from sklearn.impute import IterativeImputer` models each column as a regression on the others, iterating to convergence — the most principled option for MAR data.

### Interview Variations
- Impute, then compare model AUC across strategies.
- Handle a **categorical** column with a new "Missing" category instead of mode.
- Detect whether missingness correlates with the target (`df.groupby(df.age.isna())["survived"].mean()`).

### Common Follow-up Questions
- *Why median not mean for age?* Age is skewed; median is robust.
- *Why the missingness indicator?* Missingness itself can be predictive (MNAR/MAR).
- *Where's the leakage risk?* Calling `fit`/`fit_transform` on test data.

---

## Practical Question 4 — Detect and treat outliers (Z-score & IQR)

**Difficulty:** Medium
**Estimated Time:** 20 minutes
**Concepts Tested:** Z-score, IQR rule, capping/winsorizing, transformation

**Problem Statement**
Detect outliers in a numeric column using both the Z-score and IQR methods, then treat them (cap and/or transform) — deciding first whether they are errors or genuine.

**Approach**
1. Visualize with a box plot.
2. Flag outliers with Z-score (|z|>3) and IQR (1.5·IQR).
3. Compare counts (they differ on skewed data).
4. Treat: cap (winsorize) or log-transform.

### Cell 1 — Visualize

```python
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
df = sns.load_dataset("tips")
sns.boxplot(x=df["total_bill"]); plt.title("total_bill box plot"); plt.show()
```
*Explanation:* The box plot instantly shows candidate outliers as points beyond the right whisker (1.5·IQR above Q3), matching `total_bill`'s right skew.

### Cell 2 — Z-score method

```python
col = df["total_bill"]
z = (col - col.mean()) / col.std()
z_outliers = df[np.abs(z) > 3]
print("Z-score outliers:", len(z_outliers))
```
*Explanation:* Z-score flags points more than 3 standard deviations from the mean. It **assumes normality** and is itself distorted by the outliers (mean/std are affected), so on skewed data it often *under*-flags.

### Cell 3 — IQR method (robust)

```python
Q1, Q3 = col.quantile(0.25), col.quantile(0.75)
IQR = Q3 - Q1
low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
iqr_outliers = df[(col < low) | (col > high)]
print(f"IQR bounds: [{low:.2f}, {high:.2f}] -> outliers:", len(iqr_outliers))
```
*Explanation:* The IQR method uses **quartiles** (robust to extremes) and makes **no normality assumption**, so on skewed data it's more reliable than Z-score. This is the same rule box-plot whiskers use.

### Cell 4 — Treat: capping (winsorizing) & log transform

```python
# Capping / flooring: clip to the IQR bounds (keeps the row, limits leverage)
df["bill_capped"] = col.clip(lower=low, upper=high)

# Transform: log compresses the right tail (needs positive values)
df["bill_log"] = np.log1p(col)     # log1p handles zeros safely

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(col, kde=True, ax=ax[0]).set_title("original")
sns.histplot(df["bill_capped"], kde=True, ax=ax[1]).set_title("capped")
sns.histplot(df["bill_log"], kde=True, ax=ax[2]).set_title("log")
plt.tight_layout(); plt.show()
```
*Explanation:* **Capping** replaces extremes with the boundary value (winsorizing) — keeps sample size, reduces leverage. **Log transform** pulls in the long tail and often makes the distribution more symmetric, fixing skew *and* outlier influence at once. Choose based on whether you want to keep values (cap) or reshape the variable (log).

**Complexity note:** all O(n) — single passes / vectorized operations.

### Alternative Solution
`from scipy.stats import mstats; mstats.winsorize(col, limits=[0.01, 0.01])` caps the extreme 1% on each side. For **multivariate** outliers, use `IsolationForest` or Mahalanobis distance instead of per-column rules.

### Interview Variations
- Remove instead of cap and compare model performance.
- Detect outliers across all numeric columns at once.
- Use `RobustScaler` and explain why it suits outlier-heavy data.

### Common Follow-up Questions
- *Z-score vs IQR — which for skewed data?* IQR (robust, no normality assumption).
- *Should you always remove outliers?* No — in fraud/anomaly they're the signal.
- *Why does log need positive values?* log(0) and log(negatives) are undefined; use `log1p` / add a constant.

---

# Section D — Feature Scaling

## Practical Question 5 — Apply and compare scalers

**Difficulty:** Easy–Medium
**Estimated Time:** 20 minutes
**Concepts Tested:** min-max, standardization, robust scaling, leakage, when scaling matters

**Problem Statement**
Apply Min-Max, Standard, and Robust scaling to numeric features (fit on train only), compare their effect, and show scaling changes a distance-based model's result but not a tree's.

**Approach**
1. Split first.
2. Fit each scaler on train, transform both.
3. Compare resulting statistics.
4. Demonstrate KNN accuracy changes with scaling; a tree's does not.

### Cell 1 — Setup & split

```python
import numpy as np, pandas as pd, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

df = sns.load_dataset("penguins").dropna()
num = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = df[num]; y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
*Explanation:* `body_mass_g` (~3000–6000) dwarfs `bill_depth_mm` (~13–21). Without scaling, mass dominates any distance calculation. We split first so scalers learn parameters from train only.

### Cell 2 — Fit-on-train, transform-both

```python
scalers = {"minmax": MinMaxScaler(), "standard": StandardScaler(), "robust": RobustScaler()}
scaled = {}
for name, sc in scalers.items():
    Xtr = sc.fit_transform(X_train)     # learn params on TRAIN
    Xte = sc.transform(X_test)          # apply to TEST
    scaled[name] = (Xtr, Xte)
    print(f"\n{name}: train mean={Xtr.mean(axis=0).round(2)}  std={Xtr.std(axis=0).round(2)}")
```
*Explanation:* **StandardScaler** → mean≈0, std≈1. **MinMaxScaler** → each column in [0,1]. **RobustScaler** → centered by median, scaled by IQR (means won't be exactly 0). The critical pattern is `fit_transform(train)` then `transform(test)` — never fit on test.

### Cell 3 — Scaling changes KNN, not trees

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# KNN (distance-based) — sensitive to scaling
knn_raw = KNeighborsClassifier().fit(X_train, y_train)
knn_std = KNeighborsClassifier().fit(scaled["standard"][0], y_train)
print("KNN  raw acc :", round(accuracy_score(y_test, knn_raw.predict(X_test)), 3))
print("KNN  std acc :", round(accuracy_score(y_test, knn_std.predict(scaled['standard'][1])), 3))

# Decision Tree — scale-invariant
dt_raw = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
dt_std = DecisionTreeClassifier(random_state=42).fit(scaled["standard"][0], y_train)
print("Tree raw acc :", round(accuracy_score(y_test, dt_raw.predict(X_test)), 3))
print("Tree std acc :", round(accuracy_score(y_test, dt_std.predict(scaled['standard'][1])), 3))
```
*Explanation:* KNN accuracy typically **jumps** after scaling (distances stop being dominated by `body_mass_g`), while the decision tree's accuracy is **identical** with or without scaling — trees split on thresholds, immune to monotonic rescaling. This is the classic examinable demonstration of "which models need scaling."

**Complexity note:** scaling is O(n·features) — a couple of passes over the data.

### Alternative Solution
Do it inside a `Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())])` so scaling is refit per CV fold automatically — the leakage-safe production approach.

### Interview Variations
- Add an outlier and show min-max collapses the other points while robust doesn't.
- Scale before PCA and show components change.

### Common Follow-up Questions
- *Which models need scaling?* Distance (KNN, K-Means, SVM) and gradient-based (linear/logistic, NN); trees don't.
- *Min-max vs standard?* Min-max → bounded [0,1] but outlier-fragile; standard → centered, unbounded.
- *Why fit on train only?* Test stats leaking into scaling = leakage.

---

# Section E — Encoding & Feature Engineering

## Practical Question 6 — Encode categoricals and engineer features

**Difficulty:** Medium
**Estimated Time:** 30 minutes
**Concepts Tested:** label vs one-hot encoding, ordinal handling, feature creation, datetime features, dummy trap

**Problem Statement**
Encode categorical variables correctly (respecting nominal vs ordinal) and engineer new features from existing columns including a datetime.

**Approach**
1. Distinguish nominal vs ordinal columns.
2. One-hot encode nominal; ordinal-encode ordered categories.
3. Engineer ratio, binned, and datetime features.

### Cell 1 — Inspect categorical columns

```python
import pandas as pd, numpy as np, seaborn as sns
df = sns.load_dataset("diamonds").copy()
for c in ["cut", "color", "clarity"]:
    print(c, "->", list(df[c].unique()))
```
*Explanation:* `cut` (Fair<Good<Very Good<Premium<Ideal) and `clarity` are **ordinal** (have a natural order); `color` is nominal-ish. Knowing which is which determines the correct encoding — the crux of this question.

### Cell 2 — One-hot encoding (nominal)

```python
# pandas get_dummies is quick for EDA; drop_first avoids the dummy-variable trap
ohe = pd.get_dummies(df[["color"]], prefix="color", drop_first=True)
print(ohe.head())
print("Columns created:", ohe.shape[1])
```
*Explanation:* One-hot creates a binary column per category **without implying order** — correct for nominal features with linear/distance models. `drop_first=True` removes one redundant column to avoid the **dummy-variable trap** (perfect multicollinearity in linear regression).

### Cell 3 — Ordinal encoding (ordered categories)

```python
from sklearn.preprocessing import OrdinalEncoder
cut_order = [["Fair", "Good", "Very Good", "Premium", "Ideal"]]
oe = OrdinalEncoder(categories=cut_order)
df["cut_encoded"] = oe.fit_transform(df[["cut"]])
print(df[["cut", "cut_encoded"]].drop_duplicates().sort_values("cut_encoded"))
```
*Explanation:* For **ordinal** data we *want* the numeric order preserved, so we explicitly pass the ranking. This is the one case where label/ordinal integer encoding is appropriate for non-tree models — because Fair<...<Ideal genuinely is an order.

### Cell 4 — sklearn OneHotEncoder (production, leakage-safe)

```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
color_ohe = enc.fit_transform(df[["color"]])   # fit on train in a real split
print("Categories:", enc.categories_)
```
*Explanation:* Unlike `get_dummies`, `OneHotEncoder` is a **stateful transformer**: it *learns* the categories on train and applies them consistently to test, with `handle_unknown="ignore"` preventing crashes on unseen categories at inference. This is the leakage-safe choice inside a pipeline.

### Cell 5 — Engineer new features

```python
# Ratio feature
df["price_per_carat"] = df["price"] / df["carat"]
# Binning a continuous variable
df["carat_bin"] = pd.cut(df["carat"], bins=[0, 0.5, 1, 2, 5],
                         labels=["tiny", "small", "medium", "large"])
# Interaction
df["volume"] = df["x"] * df["y"] * df["z"]
print(df[["price_per_carat", "carat_bin", "volume"]].head())
```
*Explanation:* **Ratios** (`price_per_carat`) often expose signal a raw column hides. **Binning** turns a continuous variable into interpretable groups. **Interactions** (`volume`) combine features into something more meaningful. This is domain-knowledge-driven and frequently outperforms swapping algorithms.

### Cell 6 — Datetime feature extraction

```python
# Synthetic datetime example
s = pd.Series(pd.to_datetime(["2026-07-09 14:30", "2026-12-25 09:00", "2026-01-01 23:59"]))
feat = pd.DataFrame({
    "hour": s.dt.hour,
    "dayofweek": s.dt.dayofweek,          # 0=Mon
    "is_weekend": s.dt.dayofweek.isin([5, 6]).astype(int),
    "month": s.dt.month,
    "is_month_start": s.dt.is_month_start.astype(int),
})
print(feat)
```
*Explanation:* A raw timestamp is almost useless to a model; decomposing it into **hour, day-of-week, is_weekend, month** exposes cyclical/seasonal patterns. This is the single most common feature-engineering task in real ML and a frequent exam item.

**Complexity note:** all vectorized, O(n).

### Alternative Solution
For **high-cardinality** categoricals (e.g., 10k zip codes), use **target/mean encoding** with out-of-fold computation (via `category_encoders.TargetEncoder`) or **frequency encoding** instead of one-hot to avoid dimensionality explosion.

### Interview Variations
- Encode a high-cardinality column three ways and compare.
- Cyclically encode `hour` with sin/cos so 23:00 ≈ 00:00.
- Build all of this inside a `ColumnTransformer`.

### Common Follow-up Questions
- *Label encoding on nominal data for logistic regression — why bad?* Invents false order/spacing.
- *One-hot on 10k categories — problem?* Dimensionality explosion; use target/frequency encoding.
- *What is the dummy-variable trap?* Redundant column → perfect multicollinearity; drop one.

---

# Section F — Feature Selection

## Practical Question 7 — Select features (filter, wrapper, embedded)

**Difficulty:** Medium–Hard
**Estimated Time:** 30 minutes
**Concepts Tested:** variance threshold, correlation filtering, RFE, Lasso, tree importances

**Problem Statement**
Given many features, reduce them using one method from each family (filter, wrapper, embedded) and justify the final selection.

**Approach**
1. Filter: variance threshold + correlation-with-target + drop redundant.
2. Wrapper: Recursive Feature Elimination.
3. Embedded: Lasso (L1) and tree importances.
4. Compare selected sets.

### Cell 1 — Data

```python
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Features:", X.shape[1])   # 30 features
```
*Explanation:* 30 features — a good candidate for selection. We split first so all selection is fit on training data only (selecting on the full dataset is a subtle leakage).

### Cell 2 — Filter: variance threshold + correlation

```python
from sklearn.feature_selection import VarianceThreshold

# Drop near-constant features (little information)
vt = VarianceThreshold(threshold=0.0).fit(X_train)   # 0.0 drops only constants; raise to prune more
kept = X_train.columns[vt.get_support()]

# Correlation of each feature with the target
corr_with_target = X_train.corrwith(y_train).abs().sort_values(ascending=False)
print("Top 5 by |corr with target|:\n", corr_with_target.head())

# Redundancy: find highly correlated feature pairs to drop one of
corr = X_train.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
redundant = [c for c in upper.columns if any(upper[c] > 0.95)]
print("\nRedundant (|r|>0.95) candidates to drop:", len(redundant))
```
*Explanation:* Filter methods are **fast and model-agnostic**. Variance threshold removes uninformative near-constant features; correlation-with-target ranks relevance; the upper-triangle trick finds **redundant** feature pairs (|r|>0.95) so we drop one of each. This is the cheap first-pass funnel.

### Cell 3 — Wrapper: Recursive Feature Elimination

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe = RFE(LogisticRegression(max_iter=5000), n_features_to_select=10)
rfe.fit(X_train, y_train)
rfe_features = X_train.columns[rfe.support_]
print("RFE selected 10:\n", list(rfe_features))
```
*Explanation:* **RFE** repeatedly trains the model, drops the weakest feature by coefficient magnitude, and repeats until 10 remain. It's **model-aware** and captures interactions, but **expensive** (many refits) — the classic wrapper trade-off.

### Cell 4 — Embedded: Lasso (L1) selection

```python
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

Xtr_s = StandardScaler().fit_transform(X_train)   # L1 needs scaled features
lasso = LassoCV(cv=5, random_state=42).fit(Xtr_s, y_train)
selected = X_train.columns[lasso.coef_ != 0]
print(f"Lasso kept {len(selected)} of {X.shape[1]} features (zeroed the rest)")
```
*Explanation:* **Lasso's L1 penalty drives some coefficients to exactly zero**, performing selection *during* training — an embedded method. Features with non-zero coefficients are the selected ones. Note we **scale first** because the penalty is scale-sensitive.

### Cell 5 — Embedded: tree importances

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("Top 10 by RF importance:\n", imp.head(10))
```
*Explanation:* Tree ensembles rank features by how much they reduce impurity across all splits — another embedded method, robust to scale and capturing nonlinear interactions. Comparing Lasso's and the forest's picks (and the filter ranking) builds confidence in the final set.

**Complexity note:** filter O(n·features); RFE O(features × model-fit-cost); Lasso ~ coordinate descent, cheap; RF O(trees·n·log n).

### Alternative Solution
`SelectKBest(score_func=f_classif, k=10)` picks the top-k by ANOVA F-test — a fast univariate filter. **Mutual information** (`mutual_info_classif`) captures nonlinear relevance a correlation filter misses.

### Interview Variations
- Compare model AUC using each selected subset.
- Use `SelectFromModel(LassoCV())` to auto-threshold.
- Explain how you'd combine filter → embedded for 10,000 features.

### Common Follow-up Questions
- *Selection vs extraction (PCA)?* Selection keeps original features; PCA creates new combined ones.
- *Why does Lasso zero coefficients but Ridge doesn't?* L1's diamond geometry hits axis corners.
- *Why is filtering a good first step on huge feature sets?* Cheap, model-agnostic; then refine with embedded/wrapper.

---

# Section G — Statistics & Hypothesis Testing (Notebook)

## Practical Question 8 — Run a t-test and a chi-square test

**Difficulty:** Medium
**Estimated Time:** 25 minutes
**Concepts Tested:** hypothesis formulation, t-test, chi-square, p-value interpretation, effect size

**Problem Statement**
Test (1) whether the mean of a numeric variable differs between two groups (t-test), and (2) whether two categorical variables are associated (chi-square). Interpret the results correctly.

**Approach**
1. State H0/H1 and α.
2. Check assumptions.
3. Run the test; read the p-value.
4. Report decision *and* effect size / practical meaning.

### Cell 1 — Two-sample t-test

```python
import seaborn as sns
from scipy import stats

df = sns.load_dataset("tips")
male   = df[df["sex"] == "Male"]["total_bill"]
female = df[df["sex"] == "Female"]["total_bill"]

# H0: mean bill is equal for men and women.  H1: it differs.  alpha = 0.05
t_stat, p_val = stats.ttest_ind(male, female, equal_var=False)  # Welch's t-test
print(f"t = {t_stat:.3f}, p = {p_val:.4f}")
print("Reject H0" if p_val < 0.05 else "Fail to reject H0")
```
*Explanation:* We compare **two group means**, so a t-test is appropriate (`equal_var=False` = Welch's, which doesn't assume equal variances — safer default). The p-value is the probability of seeing a difference this large **if the true means were equal**. If p<0.05 we reject H0. Always state H0/H1 and α *before* looking at p.

### Cell 2 — Report effect size (don't stop at p)

```python
import numpy as np
# Cohen's d — standardized mean difference (practical significance)
pooled_sd = np.sqrt((male.var() + female.var()) / 2)
cohens_d = (male.mean() - female.mean()) / pooled_sd
print(f"Mean diff = {male.mean() - female.mean():.2f}, Cohen's d = {cohens_d:.3f}")
```
*Explanation:* A p-value alone doesn't tell you if the difference **matters**. **Cohen's d** (≈0.2 small, 0.5 medium, 0.8 large) quantifies *how big* the effect is. With large samples, tiny differences become "significant," so reporting effect size guards against confusing statistical with practical significance.

### Cell 3 — Chi-square test of independence

```python
# H0: smoker status and day are independent.  H1: they are associated.
contingency = pd.crosstab(df["smoker"], df["day"])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(contingency)
print(f"\nchi2 = {chi2:.3f}, p = {p:.4f}, dof = {dof}")
print("Association" if p < 0.05 else "No evidence of association")
```
*Explanation:* Both variables are **categorical**, so we use a **chi-square test of independence** on their contingency table. It compares observed vs expected counts under independence. A small p-value means the variables are associated (not independent). `chi2_contingency` also returns the expected counts you'd need to check the assumption that expected cell counts are ≥5.

### Cell 4 — Correct interpretation script (viva gold)

```python
print("""
Interpretation checklist:
- p < alpha  => reject H0 (statistically significant), NOT 'H0 is false with certainty'.
- p is P(data this extreme | H0 true), NOT P(H0 true).
- 'Fail to reject' != 'H0 proven'.
- Always pair p with effect size + confidence interval.
- Beware multiple comparisons (correct with Bonferroni/FDR).
""")
```
*Explanation:* Examiners deliberately probe p-value misconceptions. Reciting the correct interpretation crisply is high-yield marks.

**Complexity note:** trivial — these are closed-form statistical tests over the data, O(n).

### Alternative Solution
- **ANOVA** (`stats.f_oneway`) for comparing 3+ group means.
- **Mann-Whitney U** (`stats.mannwhitneyu`) — non-parametric alternative to the t-test when normality fails.
- Build a **confidence interval** for the mean difference to complement the p-value.

### Interview Variations
- Paired t-test for before/after measurements.
- One-tailed vs two-tailed and how it changes the p-value.
- Simulate to show what a p-value means under H0.

### Common Follow-up Questions
- *What is a p-value?* P(data ≥ extreme | H0), not P(H0 true).
- *Type I vs Type II error?* False positive (α) vs false negative (β); power = 1−β.
- *Significant but tiny effect — ship it?* Weigh practical significance; often no.

---

# Section H — Unsupervised Learning: K-Means & PCA

## Practical Question 9 — K-Means clustering with the elbow method

**Difficulty:** Medium
**Estimated Time:** 25 minutes
**Concepts Tested:** scaling before clustering, choosing K (elbow/silhouette), interpreting clusters

**Problem Statement**
Cluster customers into segments with K-Means. Choose K objectively and interpret the segments.

**Approach**
1. Scale features (K-Means uses Euclidean distance).
2. Elbow method (inertia vs K) + silhouette to choose K.
3. Fit final model; profile the clusters.

### Cell 1 — Scale (mandatory for K-Means)

```python
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = sns.load_dataset("penguins").dropna()
X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
Xs = StandardScaler().fit_transform(X)     # scale FIRST — else body_mass dominates
```
*Explanation:* K-Means minimizes Euclidean distance, so an unscaled `body_mass_g` (thousands) would dominate `bill_depth_mm` (teens) and clusters would form along mass alone. **Scaling is not optional here** — it's the most common K-Means mistake.

### Cell 2 — Elbow method + silhouette

```python
inertias, sils = [], []
Ks = range(2, 10)
for k in Ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    inertias.append(km.inertia_)                       # within-cluster SS
    sils.append(silhouette_score(Xs, km.labels_))      # cohesion vs separation

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(Ks, inertias, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("K")
ax[1].plot(Ks, sils, "o-"); ax[1].set_title("Silhouette"); ax[1].set_xlabel("K")
plt.tight_layout(); plt.show()
```
*Explanation:* **Inertia** always falls as K rises; the **elbow** (where the drop flattens) suggests a good K. The **silhouette score** (−1 to 1; higher = better separated) gives a second, more objective opinion. Choosing K where the elbow bends *and* silhouette peaks is the defensible answer.

### Cell 3 — Fit final model & profile clusters

```python
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xs)
df["cluster"] = km.labels_
print(df.groupby("cluster")[X.columns].mean().round(1))   # profile each segment
sns.scatterplot(x="flipper_length_mm", y="body_mass_g", hue="cluster", data=df)
plt.title("Clusters"); plt.show()
```
*Explanation:* `n_init=10` runs 10 random initializations and keeps the best (lowest inertia), mitigating K-Means' sensitivity to initialization and local optima. Profiling cluster means turns anonymous labels into a **story** ("cluster 2 = large penguins with long flippers"), which is what interpretation requires.

**Complexity note:** K-Means is O(n·K·iters·features) per run — fast and scalable.

### Alternative Solution
`from sklearn.cluster import DBSCAN` finds arbitrarily-shaped clusters and auto-detects outliers (no K needed) — good when clusters aren't spherical. **Gaussian Mixture Models** give soft (probabilistic) assignments.

### Interview Variations
- Use `k-means++` (default) vs random init and compare stability.
- Cluster then use the cluster label as a feature for a supervised model.

### Common Follow-up Questions
- *How do you choose K?* Elbow + silhouette + domain knowledge.
- *Why scale first?* Distance is dominated by large-range features otherwise.
- *Why different results per run?* Random init → local optima; fix with `n_init`/seed.

---

## Practical Question 10 — PCA for dimensionality reduction

**Difficulty:** Medium
**Estimated Time:** 20 minutes
**Concepts Tested:** standardization, explained variance, choosing components, visualization

**Problem Statement**
Reduce a high-dimensional dataset with PCA, decide how many components to keep, and visualize in 2D.

**Approach**
1. Standardize (PCA is variance-based).
2. Fit PCA; inspect explained-variance ratio.
3. Keep enough components for ~95% variance.
4. Project to 2D and plot.

### Cell 1 — Standardize & fit PCA

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np, matplotlib.pyplot as plt

data = load_breast_cancer()
Xs = StandardScaler().fit_transform(data.data)    # standardize FIRST
pca = PCA().fit(Xs)
```
*Explanation:* PCA finds directions of **maximum variance**, so features must be **standardized** first — otherwise a large-unit feature would dominate the first component purely because of its scale, not its importance. This mirrors the scaling lesson from K-Means.

### Cell 2 — Explained variance & choosing components

```python
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
n_95 = np.argmax(cum >= 0.95) + 1
print(f"Components for 95% variance: {n_95} of {Xs.shape[1]}")

plt.plot(range(1, len(cum) + 1), cum, "o-")
plt.axhline(0.95, ls="--", color="r"); plt.axvline(n_95, ls="--", color="g")
plt.xlabel("Number of components"); plt.ylabel("Cumulative explained variance")
plt.title("Scree / cumulative variance"); plt.show()
```
*Explanation:* The **explained-variance ratio** tells you how much information each component retains. We keep the smallest number of components reaching ~95% cumulative variance — here often ~10 of 30, a big reduction with little information loss. This is the standard, defensible way to choose the component count.

### Cell 3 — Project to 2D and visualize

```python
X2 = PCA(n_components=2).fit_transform(Xs)
plt.scatter(X2[:, 0], X2[:, 1], c=data.target, cmap="coolwarm", alpha=0.7)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Data in 2 principal components")
plt.colorbar(label="target"); plt.show()
```
*Explanation:* Projecting 30 features to the top 2 components lets us **visualize** class separation in 2D — the malignant/benign classes separate visibly, showing the leading components carry the discriminative signal. Remember: PC1/PC2 are **combinations** of original features, so they're not individually interpretable.

**Complexity note:** PCA via SVD is ~O(n·features²) — fine for moderate dimensionality.

### Alternative Solution
For **nonlinear** structure, PCA (linear) misses curvature — use **t-SNE** or **UMAP** for visualization (but they're for viz, not a reversible transform). PCA remains the go-to for de-noising and speeding up supervised models.

### Interview Variations
- Feed PCA components into a classifier and compare accuracy/speed vs raw features.
- Show that skipping standardization changes the components.

### Common Follow-up Questions
- *Selection vs extraction?* PCA extracts new features; it doesn't select originals.
- *Why standardize before PCA?* Variance (hence PCs) would otherwise track units.
- *How many components?* Enough for ~95% cumulative explained variance (or the scree elbow).

---

# Section I — Pure-Python Coding Questions (from scratch)

> Some assessments ban sklearn to test that you understand the **math**. Practice
> implementing these with only `numpy`.

## Practical Question 11 — Implement core ML math from scratch

**Difficulty:** Medium–Hard
**Estimated Time:** 30 minutes
**Concepts Tested:** the actual formulas behind the library calls

### Min-Max & Standardization

```python
import numpy as np

def min_max_scale(x):
    """Scale to [0, 1].  x' = (x - min) / (max - min)"""
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min())

def standardize(x):
    """Z-score: mean 0, std 1.  x' = (x - mean) / std"""
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()
```
*Line notes:* these are the exact formulas `MinMaxScaler`/`StandardScaler` apply. Being able to write them proves you understand *what* scaling does. Watch the divide-by-zero edge case when `max == min` (constant feature).

### Euclidean distance & a tiny KNN

```python
def euclidean(a, b):
    """sqrt of sum of squared differences."""
    return np.sqrt(np.sum((np.asarray(a) - np.asarray(b)) ** 2))

def knn_predict(X_train, y_train, x, k=3):
    """Classify x by majority vote of its k nearest neighbors."""
    dists = [euclidean(x, row) for row in X_train]         # O(n*d)
    idx = np.argsort(dists)[:k]                            # k closest
    votes = y_train[idx]
    values, counts = np.unique(votes, return_counts=True)
    return values[np.argmax(counts)]                        # majority label
```
*Line notes:* KNN is a **lazy learner** — no training, all work at prediction time (O(n·d) per query). This from-scratch version cements *why* scaling matters (distance) and why KNN is slow at inference.

### Pearson correlation & covariance

```python
def covariance(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.mean((x - x.mean()) * (y - y.mean()))       # population covariance

def pearson_corr(x, y):
    """cov(x,y) / (std_x * std_y)  -> in [-1, 1]"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return covariance(x, y) / (x.std() * y.std())
```
*Line notes:* shows correlation is **standardized covariance**. If asked "why is correlation unit-free?", the division by the two standard deviations is your answer.

### Train/test split from scratch

```python
def train_test_split_scratch(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)                # shuffle indices
    cut = int(n * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]
```
*Line notes:* shuffling before splitting avoids order bias; seeding makes it reproducible — the same guarantees sklearn's `train_test_split` gives.

### Accuracy, precision, recall from a confusion matrix

```python
def classification_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1)
```
*Line notes:* Precision = "of predicted positives, how many were right"; Recall = "of actual positives, how many did we catch." Knowing these by heart (and the TP/FP/FN/TN grid) is a guaranteed viva question, especially "why is accuracy misleading on imbalanced data?"

### Common Follow-up Questions
- *Why is KNN slow at prediction?* No training; computes all distances per query.
- *Precision vs recall — when do you prioritize each?* Recall when misses are costly (disease, fraud); precision when false alarms are costly (spam).
- *Derive why min-max can divide by zero.* Constant feature → max=min.

---

# Coding Questions Bank

> Rapid-fire predicted questions. For each, know the **one-liner approach** and be
> ready to expand. "Why asked" tells you what the examiner is really probing.

## Easy

1. **Load a CSV and print shape, dtypes, and missing counts.** *Why:* baseline data-handling fluency. `pd.read_csv`, `.info()`, `.isna().sum()`.
2. **Compute mean/median/mode/std of a column.** *Why:* summary-statistics basics and mean-vs-median judgment.
3. **Plot a histogram and a box plot for a feature.** *Why:* can you read distribution shape and spot outliers visually.
4. **One-hot encode a categorical column.** *Why:* do you know nominal encoding and `get_dummies`/`OneHotEncoder`.
5. **Split data into train/test with a fixed seed and stratification.** *Why:* leakage awareness and reproducibility.
6. **Standardize features with `StandardScaler`.** *Why:* scaling mechanics and fit/transform discipline.

## Medium

1. **Build a preprocessing `Pipeline` (impute → scale → encode) and fit a model.** *Why:* leakage-safe engineering, the single most important practical skill.
2. **Detect outliers with both Z-score and IQR and treat them.** *Why:* method trade-offs (robustness, normality assumption).
3. **Handle missing values with median + a missingness indicator, then compare to KNNImputer.** *Why:* imputation depth and understanding of MAR/MNAR.
4. **Evaluate a classifier with precision/recall/F1/ROC-AUC and a confusion matrix on imbalanced data.** *Why:* why accuracy is misleading; metric selection.
5. **K-Means with the elbow method + silhouette to pick K.** *Why:* unsupervised evaluation without labels.
6. **PCA to 95% variance, then train a model on the components.** *Why:* dimensionality reduction and the variance-vs-usefulness nuance.
7. **Run a t-test / chi-square and interpret the p-value correctly.** *Why:* statistical literacy and p-value misconceptions.
8. **Feature selection with Lasso and RFE; compare selected sets.** *Why:* embedded vs wrapper trade-offs.

## Hard

1. **End-to-end project on a messy dataset: EDA → cleaning → feature engineering → tuning → evaluation → interpretation.** *Why:* the whole workflow under time pressure; this is the capstone practical.
2. **Diagnose and fix a data-leakage bug in a given notebook** (e.g., scaling before split, target-encoded without out-of-fold). *Why:* leakage is the #1 real-world ML failure; spotting it signals seniority.
3. **Handle severe class imbalance** (SMOTE / class weights / threshold tuning) and justify the metric choice. *Why:* real datasets are imbalanced; naive accuracy fails.
4. **Implement gradient descent for linear regression from scratch** (loss, gradient, update loop). *Why:* proves you understand how "learning" actually happens.
5. **Given drift in production metrics, design a monitoring + retraining strategy.** *Why:* MLOps maturity beyond the notebook.
6. **Build a `ColumnTransformer` handling mixed numeric/categorical/text columns with different pipelines.** *Why:* realistic heterogeneous data engineering.

---

# Exam & Viva Survival Tips

**Before you touch the data**
- Say your plan out loud: *load → EDA → clean → preprocess → split → model → evaluate → interpret.* Examiners grade methodology.
- Always **print `df.shape` and `df.head()` first.** Know your data before modelling.

**The non-negotiables (cheap marks people lose)**
- **Split before preprocessing.** Fit scalers/imputers/encoders on **train only**; `transform` test. State this explicitly — it's the leakage trap they set.
- **Check target balance.** If imbalanced, use precision/recall/F1/ROC-AUC, *not* accuracy — and say why.
- **Median for skewed numeric imputation, mode for categoricals.**
- **Scale for distance/gradient models (KNN, K-Means, SVM, linear, NN); don't bother for trees.**
- **One-hot for nominal, ordinal-encode ordered categories.** Never label-encode nominal for linear models.
- **Set `random_state`** everywhere for reproducibility.

**Interpreting results**
- Never report *only* accuracy or *only* a p-value. Pair with confusion matrix / effect size / confidence interval.
- Explain the model's *story* (feature importances), not just its score.
- Distinguish **statistical** vs **practical** significance.

**Talking points that signal seniority**
- Data leakage (where it hides, how a `Pipeline` prevents it).
- Bias–variance / overfitting and how cross-validation detects it.
- Data & concept **drift** and a **monitoring + retraining** plan.
- Train/serve skew and why you persist the *whole pipeline*, not just the model.

**When stuck**
- State the trade-off and your reasoning — a well-justified "it depends, because…" beats a confident wrong absolute.
- Fall back to a **simple baseline** first; you can always improve on it.
- If a library call escapes you, write the **from-scratch** formula (Section I) — it shows deeper understanding.

> **Final advice:** type out the Section A workflow and the Section I from-scratch
> snippets by hand until they're muscle memory. In a timed lab, fluent
> methodology and correct leakage discipline win more marks than a fancy model.
