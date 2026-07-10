# Supervised Machine Learning — Practical & Coding Assessment Guide

> Hands-on preparation for coding rounds, practical lab exams, and Jupyter-notebook-based assessments.
> Scope follows the **Supervised ML** syllabus. Companion file: **`theory.md`** (concepts + interview Q&A).

## How to use this guide

Each part has two sections:

- **Part A — Coding Questions:** predicted interview/assessment problems, each with `Difficulty`, `Estimated Time`, `Concepts Tested`, a problem statement with example input/output, a step-by-step approach, a complete **production-quality Python implementation** (with time/space complexity), an alternative solution, interview variations, and follow-up questions. Many require implementing algorithms **from scratch** with NumPy — a common assessment format.
- **Part B — Notebook Workflow:** a complete end-to-end Jupyter workflow split into ordered **cells** (imports → load → EDA → cleaning → feature engineering → split → scale → train → tune → cross-validate → evaluate → interpret), with an explanation of *what and why* for every cell.

### Setup

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install xgboost lightgbm catboost imbalanced-learn shap   # for ensemble / imbalance / interpretation parts
```

All code is written to be runnable top-to-bottom. Where a Kaggle-style CSV is referenced, a scikit-learn built-in dataset stand-in is provided so the notebook runs without external downloads.

## Table of Contents

1. **Regression** — OLS from scratch, metrics from scratch, Ridge/Lasso/ElasticNet, Polynomial pipelines; *House Price Prediction* notebook.
2. **Classification** — sigmoid/logistic & KNN from scratch, Gini/entropy splits, SVM kernels, decision-boundary plots; *Heart Disease Prediction* notebook.
3. **Model Optimization** — scaling from scratch, leakage-free pipelines, SMOTE inside CV, Grid vs Randomized search, permutation/SHAP importance; leakage & imbalance notebooks.
4. **Ensemble Learning** — bagging from scratch, Random Forest + OOB, Voting/Stacking, XGBoost/LightGBM/CatBoost with early stopping; *Customer Churn Prediction* notebook.
5. **Model Evaluation** — confusion matrix / ROC / PR curves from scratch, threshold tuning, manual k-fold; *Model Evaluation & Comparison Playbook* notebook.

---


# Part 1 — Regression (Practical)

## Part A — Coding Questions

> These are the questions I would actually throw at you in a coding round or a lab exam for **Supervised ML — Regression**. Each one is written the way an interviewer thinks: a clean problem, an expected input/output, a reasoned approach, and *production-quality* code you can paste into a Jupyter cell and run. Read the "Approach" before the code — in a real interview you talk first, code second.

---

# Practical Question 1: Implement OLS Linear Regression from Scratch (Normal Equation)

**Difficulty:** Easy
**Estimated Time:** 15 min
**Concepts Tested:** Linear algebra behind regression, the closed-form normal equation, bias/intercept handling, NumPy vectorization

## Problem Statement
Implement Ordinary Least Squares (OLS) linear regression **without** using scikit-learn. Fit `y ≈ Xw + b` by solving the **normal equation** and expose `fit`/`predict` methods. You should support multiple features (multiple linear regression) automatically.

## Example Input
```python
import numpy as np
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])   # single feature
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])            # perfectly y = 2x
```

## Example Output
```
weights (slope): [2.0]
bias   (intercept): ~0.0
prediction for x=6: ~12.0
```

## Approach
1. OLS minimizes the residual sum of squares `||Xw − y||²`. The closed-form minimizer is the **normal equation**: `w = (XᵀX)⁻¹ Xᵀy`.
2. To learn the intercept, prepend a column of ones to `X` (the "bias trick"). Then the first weight *is* the intercept.
3. Prefer `np.linalg.lstsq` or `np.linalg.pinv` (pseudo-inverse) over a raw `inv` — if `XᵀX` is singular/ill-conditioned, `inv` blows up but `pinv`/`lstsq` degrade gracefully.
4. `predict` just does `X_aug @ theta`.

## Python Implementation
```python
import numpy as np

class LinearRegressionScratch:
    """OLS via the normal equation:  theta = (X^T X)^-1 X^T y."""

    def __init__(self, use_pinv: bool = True):
        # use_pinv=True -> robust to singular X^T X (rank-deficient / collinear features)
        self.use_pinv = use_pinv
        self.coef_ = None        # weights for the real features
        self.intercept_ = None   # bias term

    def _add_bias(self, X):
        # Prepend a column of 1s so the intercept is learned like any other weight.
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        X_aug = self._add_bias(X)                      # shape (n, d+1)

        if self.use_pinv:
            # pinv = Moore-Penrose pseudo-inverse; equals lstsq's least-norm solution.
            theta = np.linalg.pinv(X_aug) @ y
        else:
            # Direct normal equation; fast but fails if X_aug^T X_aug is singular.
            theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y

        self.intercept_ = theta[0]     # first entry corresponds to the ones-column
        self.coef_ = theta[1:]         # remaining entries are the feature weights
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

# --- demo ---
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
model = LinearRegressionScratch().fit(X, y)
print("slope:", model.coef_)            # [2.]
print("intercept:", model.intercept_)   # ~0.0
print("pred x=6:", model.predict([[6.0]]))  # ~12.0
```

**Complexity:** Building `XᵀX` is `O(n·d²)`, inverting it is `O(d³)`. So **time ≈ O(n·d² + d³)**, **space ≈ O(d²)**. This is why the normal equation is great for *few features* but the wrong tool when `d` is huge (use gradient descent then).

## Alternative Solution
Use `np.linalg.lstsq`, which solves the least-squares system directly and is the numerically safest one-liner:
```python
X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
intercept, coef = theta[0], theta[1:]
```

## Interview Variations
1. "Now add **L2 regularization**" → the normal equation becomes `theta = (XᵀX + λI)⁻¹ Xᵀy` (don't regularize the bias row → set that diagonal entry to 0). That *is* Ridge.
2. "Return the **residuals** and the **R²**" → compute `y - y_hat` and `1 - SS_res/SS_tot`.
3. "What if `d > n` (more features than samples)?" → `XᵀX` is singular; `pinv`/`lstsq` still return the minimum-norm solution.

## Common Follow-up Questions
- Why prefer `pinv`/`lstsq` over `inv`? (numerical stability, handles rank deficiency)
- When is the normal equation *worse* than gradient descent? (large `d`, because of the `O(d³)` inverse)
- What assumptions does OLS make? (linearity, independence, homoscedasticity, normally distributed errors for inference)

---

# Practical Question 2: Linear Regression via Batch Gradient Descent

**Difficulty:** Medium
**Estimated Time:** 20 min
**Concepts Tested:** Cost function (MSE), gradient derivation, learning rate, convergence, feature scaling necessity

## Problem Statement
Implement linear regression trained with **batch gradient descent** instead of a closed form. Track the MSE loss per iteration so you can plot a convergence curve. Explain why feature scaling matters here.

## Example Input
```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
true_w = np.array([3.0, -2.0, 0.5])
y = X @ true_w + 4.0 + rng.normal(scale=0.1, size=200)   # true bias = 4
```

## Example Output
```
learned weights ≈ [ 3.0, -2.0, 0.5 ]
learned bias    ≈ 4.0
loss decreases monotonically to ~0.01
```

## Approach
1. Cost: `J(w,b) = (1/n) Σ (ŷ − y)²` (MSE). This is convex → gradient descent finds the global minimum.
2. Gradients: `dJ/dw = (2/n) Xᵀ(ŷ − y)`, `dJ/db = (2/n) Σ(ŷ − y)`.
3. Update rule: `w ← w − α·dJ/dw`, `b ← b − α·dJ/db`, repeated for `n_iters`.
4. **Scale features first** — if features have wildly different ranges the loss surface is elongated and GD zig-zags / needs a tiny learning rate.

## Python Implementation
```python
import numpy as np

class LinearRegressionGD:
    def __init__(self, lr=0.05, n_iters=1000, tol=1e-8):
        self.lr = lr            # learning rate (step size)
        self.n_iters = n_iters
        self.tol = tol          # early-stop when loss barely changes
        self.w = None
        self.b = 0.0
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        self.w = np.zeros(d)     # init weights at 0 (fine for convex MSE)
        self.b = 0.0

        prev_loss = np.inf
        for _ in range(self.n_iters):
            y_hat = X @ self.w + self.b
            error = y_hat - y                      # residual vector, shape (n,)

            # Vectorized gradients (the 2/n factor comes from d/dw of mean squared error)
            grad_w = (2 / n) * (X.T @ error)
            grad_b = (2 / n) * np.sum(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            loss = np.mean(error ** 2)             # MSE
            self.loss_history_.append(loss)
            if abs(prev_loss - loss) < self.tol:   # converged
                break
            prev_loss = loss
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b

# --- demo ---
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = X @ np.array([3.0, -2.0, 0.5]) + 4.0 + rng.normal(scale=0.1, size=200)
gd = LinearRegressionGD(lr=0.1, n_iters=2000).fit(X, y)
print("w:", np.round(gd.w, 2), "b:", round(gd.b, 2))
print("final loss:", gd.loss_history_[-1])
```

**Complexity:** each iteration is `O(n·d)` (one matrix-vector product), so **time ≈ O(iters·n·d)**, **space ≈ O(d)**. Note it never inverts a matrix → scales to very large `d` where the normal equation dies.

## Alternative Solution
**Stochastic / mini-batch GD**: update on one sample (or a small batch) at a time. Cheaper per step, noisier path, better for huge datasets and online learning:
```python
idx = rng.permutation(n)
for start in range(0, n, batch_size):
    b_idx = idx[start:start+batch_size]
    Xb, yb = X[b_idx], y[b_idx]
    err = Xb @ self.w + self.b - yb
    self.w -= self.lr * (2/len(b_idx)) * (Xb.T @ err)
    self.b -= self.lr * (2/len(b_idx)) * err.sum()
```

## Interview Variations
1. "Loss diverges to NaN — fix it." → learning rate too high; lower `lr` or scale features.
2. "Add L2 penalty" → add `+ 2·λ·w` to `grad_w` (Ridge via GD).
3. "Plot the loss curve" → `plt.plot(gd.loss_history_)` and interpret the elbow.

## Common Follow-up Questions
- Why does GD need feature scaling but the normal equation doesn't (mathematically)?
- Batch vs mini-batch vs stochastic — trade-offs?
- How do you pick the learning rate? (grid search, LR schedules, watching the loss curve)

---

# Practical Question 3: Implement Regression Metrics from Scratch (MAE, MSE, RMSE, R², Adjusted R²)

**Difficulty:** Easy–Medium
**Estimated Time:** 15 min
**Concepts Tested:** What each metric measures, the R² formula, why adjusted R² exists, verifying against sklearn

## Problem Statement
Write pure-NumPy functions for MAE, MSE, RMSE, R², and Adjusted R². Then verify MAE/MSE/R² match `sklearn.metrics` on random data.

## Example Input
```python
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5,  0.0, 2.0, 8.0])
```

## Example Output
```
MAE  = 0.5
MSE  = 0.375
RMSE = 0.6123...
R2   = 0.9486...
```

## Approach
- **MAE** = mean(|y − ŷ|): robust to outliers, same units as y.
- **MSE** = mean((y − ŷ)²): penalizes big errors more; differentiable → used as training loss.
- **RMSE** = √MSE: back in y's units; the "typical" error size.
- **R²** = `1 − SS_res/SS_tot` where `SS_res = Σ(y − ŷ)²`, `SS_tot = Σ(y − ȳ)²`. Fraction of variance explained; 1 is perfect, 0 = no better than predicting the mean, can go negative.
- **Adjusted R²** = `1 − (1 − R²)·(n − 1)/(n − p − 1)`. Penalizes adding useless features (`p` = number of predictors).

## Python Implementation
```python
import numpy as np

def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2_score_scratch(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)          # unexplained variance
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) # total variance
    return 1 - ss_res / ss_tot

def adjusted_r2(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score_scratch(y_true, y_pred)
    # guard against division by zero when n - p - 1 <= 0
    denom = n - n_features - 1
    if denom <= 0:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / denom

# --- verify against sklearn ---
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
rng = np.random.default_rng(1)
yt = rng.normal(size=100)
yp = yt + rng.normal(scale=0.3, size=100)
assert np.isclose(mae(yt, yp), mean_absolute_error(yt, yp))
assert np.isclose(mse(yt, yp), mean_squared_error(yt, yp))
assert np.isclose(r2_score_scratch(yt, yp), r2_score(yt, yp))
print("All metrics match sklearn ✔")
```

**Complexity:** every metric is a single pass → **O(n)** time, **O(1)** extra space.

## Alternative Solution
`sklearn.metrics.root_mean_squared_error` exists in newer sklearn; older versions use `mean_squared_error(y, yhat, squared=False)`. Adjusted R² has no sklearn function — you always compute it by hand.

## Interview Variations
1. "Why can R² be negative?" → your model is worse than the horizontal line `ȳ` (SS_res > SS_tot).
2. "MAE vs RMSE — which for a dataset with outliers?" → MAE (RMSE over-weights the outliers).
3. "Add MAPE" → `mean(|(y − ŷ)/y|)·100`; warn about divide-by-zero when `y=0`.

## Common Follow-up Questions
- Which metric would you report to a business stakeholder and why? (RMSE/MAE — real units)
- When does high R² still mean a bad model? (overfitting; R² always rises as you add features → use adjusted R² or a validation set)
- Difference between R² on train vs test?

---

# Practical Question 4: Ridge vs Lasso — Coefficient Shrinkage Comparison

**Difficulty:** Medium
**Estimated Time:** 20 min
**Concepts Tested:** L1 vs L2 regularization, feature selection via Lasso, coefficient paths, why scaling is mandatory before regularizing

## Problem Statement
On a dataset with several **irrelevant / correlated** features, fit `LinearRegression`, `Ridge`, and `Lasso`. Compare their coefficients and show that **Lasso drives some coefficients to exactly 0** (feature selection) while **Ridge only shrinks them**. Always scale first.

## Example Input
Synthetic data with 10 features but only 3 truly informative.
```python
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=200, n_features=10, n_informative=3,
                       noise=10.0, random_state=42)
```

## Example Output
```
Feature | OLS    | Ridge  | Lasso
   0    |  0.12  |  0.10  |  0.00   <- Lasso zeroed a noise feature
   3    | 88.4   | 85.1   | 86.9    <- informative feature survives
...
Lasso set 6/10 coefficients to exactly 0
```

## Approach
1. **Scale features** with `StandardScaler` — L1/L2 penalties act on coefficient magnitudes, which are meaningless if features are on different scales.
2. Fit all three models on the *same* scaled data.
3. Stack coefficients side by side in a DataFrame.
4. Count Lasso zeros → that's implicit feature selection. Explain the geometry: L1's diamond constraint has corners on the axes → solutions land exactly on 0; L2's circular constraint doesn't.

## Python Implementation
```python
import numpy as np, pandas as pd
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline

X, y = make_regression(n_samples=200, n_features=10, n_informative=3,
                       noise=10.0, random_state=42)

def fit_coefs(model):
    # Pipeline scales THEN fits; scaler learns stats only from training data.
    pipe = make_pipeline(StandardScaler(), model).fit(X, y)
    return pipe[-1].coef_        # coefficients from the final estimator

coefs = pd.DataFrame({
    "OLS":   fit_coefs(LinearRegression()),
    "Ridge": fit_coefs(Ridge(alpha=10.0)),
    "Lasso": fit_coefs(Lasso(alpha=1.0)),
})
coefs.index.name = "feature"
print(coefs.round(2))
print("\nLasso zeroed:", int((coefs['Lasso'].abs() < 1e-8).sum()), "of 10 features")
```

**Complexity:** Ridge has a closed form (`O(n·d² + d³)`); Lasso is solved by coordinate descent iteratively (no closed form because L1 is non-differentiable at 0).

## Alternative Solution
Plot the **regularization path** to see coefficients shrink as alpha grows:
```python
import matplotlib.pyplot as plt
alphas = np.logspace(-2, 2, 50)
paths = np.array([fit_coefs(Lasso(alpha=a, max_iter=10000)) for a in alphas])
plt.plot(alphas, paths); plt.xscale("log")
plt.xlabel("alpha"); plt.ylabel("coefficient"); plt.title("Lasso path")
```

## Interview Variations
1. "Use **ElasticNet** instead" → mixes L1+L2 via `l1_ratio`; best when features are correlated (Lasso arbitrarily picks one of a correlated group, ElasticNet keeps the group).
2. "Increase alpha a lot — what happens?" → all coefficients → 0, model → predicting the mean, underfitting.
3. "Forgot to scale — what breaks?" → penalty unfairly hits large-scale features; selection becomes garbage.

## Common Follow-up Questions
- Why does L1 produce sparsity but L2 doesn't? (diamond corners vs smooth circle — the geometric argument)
- When Ridge over Lasso? (all features somewhat useful, correlated predictors, you want stability)
- What does alpha=0 reduce to? (plain OLS)

---

# Practical Question 5: Polynomial Regression with a scikit-learn Pipeline (Avoiding Overfitting)

**Difficulty:** Medium
**Estimated Time:** 20 min
**Concepts Tested:** PolynomialFeatures, Pipelines (no data leakage), bias-variance trade-off, choosing polynomial degree via validation

## Problem Statement
Fit polynomial regression on non-linear data using a **Pipeline** of `PolynomialFeatures → StandardScaler → LinearRegression`. Compare degrees 1, 3, and 15 to demonstrate underfitting vs good fit vs overfitting. Pick the best degree by cross-validated RMSE.

## Example Input
```python
import numpy as np
rng = np.random.default_rng(0)
X = np.sort(rng.uniform(-3, 3, size=(80, 1)), axis=0)
y = 0.5 * X.ravel()**3 - X.ravel()**2 + 2 + rng.normal(scale=2.0, size=80)
```

## Example Output
```
degree  1 : CV RMSE ≈ 5.1   (underfit)
degree  3 : CV RMSE ≈ 2.1   (best)
degree 15 : CV RMSE ≈ 6.8   (overfit, huge variance)
Best degree = 3
```

## Approach
1. `PolynomialFeatures(degree=d)` expands `[x]` → `[1, x, x², …, xᵈ]`, turning a *linear* model into a curve fitter (it's still linear *in the parameters*).
2. Wrap everything in a `Pipeline` so the transform is refit inside each CV fold → **no leakage**.
3. Sweep degrees; for each, take the cross-validated RMSE. Low degree = high bias (underfit); high degree = high variance (overfit). The sweet spot minimizes validation error.

## Python Implementation
```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
X = np.sort(rng.uniform(-3, 3, size=(80, 1)), axis=0)
y = 0.5 * X.ravel()**3 - X.ravel()**2 + 2 + rng.normal(scale=2.0, size=80)

def poly_cv_rmse(degree):
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale",  StandardScaler()),          # scale AFTER expansion (x^15 is huge)
        ("linreg", LinearRegression()),
    ])
    # neg RMSE because sklearn maximizes score; flip the sign back.
    scores = cross_val_score(pipe, X, y, cv=5,
                             scoring="neg_root_mean_squared_error")
    return -scores.mean()

for d in (1, 3, 15):
    print(f"degree {d:2d}: CV RMSE = {poly_cv_rmse(d):.2f}")

best = min(range(1, 16), key=poly_cv_rmse)
print("Best degree =", best)
```

**Complexity:** `PolynomialFeatures` of degree `d` on `f` features creates `~C(f+d, d)` columns → column count explodes combinatorially. Watch out for `d` large with many features.

## Alternative Solution
Regularize a *high-degree* polynomial instead of hunting for the perfect degree: `PolynomialFeatures(degree=15) → StandardScaler → Ridge(alpha=…)`. Ridge tames the wild high-order coefficients so a too-flexible model stops overfitting.

## Interview Variations
1. "Why scale *after* PolynomialFeatures?" → `x¹⁵` has an enormous range; scaling keeps the linear solver numerically stable.
2. "Plot the fitted curves for each degree" → shows underfit line, good cubic, wiggly overfit.
3. "Two input features, degree 2 — how many terms?" → `1, a, b, a², ab, b²` (interaction term `ab` included).

## Common Follow-up Questions
- Is polynomial regression a linear or non-linear model? (linear in parameters, non-linear in x)
- How do you detect overfitting here? (train RMSE ≪ validation RMSE)
- Why a Pipeline instead of transforming X once up front? (prevents the scaler/poly from seeing validation data → no leakage)

---

# Practical Question 6: Tune Ridge/Lasso `alpha` with Cross-Validation (GridSearchCV & the built-in CV estimators)

**Difficulty:** Hard
**Estimated Time:** 25 min
**Concepts Tested:** Hyperparameter tuning, k-fold CV, GridSearchCV mechanics, RidgeCV/LassoCV shortcuts, avoiding leakage inside a Pipeline

## Problem Statement
Find the best regularization strength `alpha` for Ridge using `GridSearchCV` over a log-spaced grid, inside a scaling Pipeline. Report the best alpha, the CV score, and test-set RMSE. Then show the faster built-in `RidgeCV`/`LassoCV` equivalent.

## Example Input
`fetch_california_housing` (real, ships with sklearn).

## Example Output
```
Best alpha : 10.0
Best CV RMSE: 0.72
Test RMSE  : 0.73
RidgeCV picked alpha = 10.0  (agrees)
```

## Approach
1. Split off a held-out **test set** first (final honesty check).
2. Build `Pipeline(StandardScaler, Ridge)` — scaling must live *inside* the pipeline so each CV fold scales using only its own training portion.
3. `GridSearchCV` over `ridge__alpha = np.logspace(-3, 3, 13)`, `cv=5`, scoring negative RMSE. It refits the best model on all training data automatically (`refit=True`).
4. Evaluate the refit best estimator on the untouched test set.
5. Show `RidgeCV`/`LassoCV` as the vectorized shortcut (efficient generalized CV).

## Python Implementation
```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import root_mean_squared_error   # sklearn >=1.4

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
param_grid = {"ridge__alpha": np.logspace(-3, 3, 13)}   # 0.001 ... 1000

grid = GridSearchCV(
    pipe, param_grid, cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,          # parallelize folds across cores
    refit=True,         # refit best model on full training set
)
grid.fit(X_tr, y_tr)

print("Best alpha :", grid.best_params_["ridge__alpha"])
print("Best CV RMSE:", round(-grid.best_score_, 3))
print("Test RMSE  :", round(root_mean_squared_error(y_te, grid.predict(X_te)), 3))

# --- faster built-in equivalent: RidgeCV does efficient LOO/GCV internally ---
ridgecv = Pipeline([
    ("scale", StandardScaler()),
    ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
]).fit(X_tr, y_tr)
print("RidgeCV alpha:", ridgecv[-1].alpha_)
```

**Complexity:** GridSearchCV fits `|grid| × k` models → here `13 × 5 = 65` fits. `RidgeCV` uses closed-form Generalized Cross-Validation and is dramatically cheaper for Ridge specifically.

## Alternative Solution
`RandomizedSearchCV` when the grid is large (samples random alphas from a distribution — cheaper, often as good). For Lasso, use `LassoCV` (coordinate descent along a warm-started alpha path — very fast).

## Interview Variations
1. "Tune ElasticNet" → grid over both `alpha` **and** `l1_ratio` (a 2-D grid), or use `ElasticNetCV`.
2. "Why is scaling inside the Pipeline, not before the split?" → scaling before splitting leaks test statistics into training → optimistic scores.
3. "Nested CV — why?" → outer loop for unbiased performance estimate, inner loop (GridSearchCV) for tuning; avoids optimistic bias from tuning-on-the-test-fold.

## Common Follow-up Questions
- What does `refit=True` do and why is it convenient?
- Why log-spaced alphas instead of linear? (regularization effect is multiplicative/scale-free)
- GridSearch vs Random vs Bayesian search — when each?

---

## Part B — End-to-End Notebook Workflow: House Price Prediction

> This is the full lab you'd be asked to reproduce in a notebook exam. Dataset: **`sklearn.datasets.fetch_california_housing`** — real, ships with scikit-learn, no download needed, target is the median house value (in units of \$100,000) for California districts. I'll stay consistent with this dataset the whole way. Run the cells top to bottom.

---

### Cell 1 — Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
np.random.seed(42)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
```

**What & why:** We front-load every import so the notebook is reproducible and no later cell fails on a missing name. `numpy/pandas` for data, `matplotlib/seaborn` for EDA plots, and the scikit-learn pieces we'll need: dataset loader, splitting/CV/tuning utilities, preprocessing, the four regression models, the Pipeline glue, and metrics. Setting a global seed makes the train/test split and any randomness reproducible — critical in an exam where the grader re-runs your notebook.

---

### Cell 2 — Load the Dataset

```python
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()                 # features + target in one DataFrame
df = df.rename(columns={"MedHouseVal": "Price"})
print("Shape:", df.shape)
df.head()
```

**What & why:** `fetch_california_housing(as_frame=True)` returns a `Bunch` whose `.frame` is a ready-made DataFrame with 8 numeric features (median income, house age, average rooms, etc.) plus the target `MedHouseVal`, which we rename to `Price` for readability. Working in a DataFrame (rather than raw NumPy) keeps column names attached, which we need for EDA and coefficient interpretation later. Always eyeball `.head()` and `.shape` first — 20,640 rows × 9 columns here.

---

### Cell 3 — Exploratory Data Analysis (EDA): Structure & Statistics

```python
print(df.info())
display(df.describe().T)

# Distribution of the target
plt.figure(figsize=(6, 4))
sns.histplot(df["Price"], bins=50, kde=True)
plt.title("Target distribution: Median House Value")
plt.show()
```

**What & why:** `.info()` confirms dtypes and that there are no obvious nulls; `.describe().T` surfaces scale differences between features (e.g. `MedInc` ~0–15 vs `Population` in the thousands) — a first hint that **scaling will be mandatory**. The target histogram reveals a right-skewed distribution with a **capped spike at 5.0** (values were clipped in the original data). Knowing the target is capped explains later prediction errors near the ceiling and is exactly the kind of dataset-awareness an examiner rewards.

---

### Cell 4 — EDA: Correlations & Relationships

```python
corr = df.corr(numeric_only=True)

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature correlation heatmap")
plt.show()

# Strongest linear driver of price?
print(corr["Price"].sort_values(ascending=False))
```

**What & why:** The correlation heatmap tells us which features move linearly with `Price` and which are **collinear with each other** (e.g. `AveRooms` and `AveBedrms` are highly correlated — a red flag for plain OLS coefficient stability, and a reason Ridge/ElasticNet may help). We see `MedInc` (median income) is by far the strongest positive correlate of price — a sanity check that matches real-world intuition. Spotting multicollinearity *now* motivates the regularized models we compare later.

---

### Cell 5 — Data Cleaning & Outlier Handling

```python
print("Missing values:\n", df.isnull().sum())          # confirm none
print("Duplicate rows:", df.duplicated().sum())

# The engineered ratio features can explode on tiny districts; cap extreme outliers.
for col in ["AveRooms", "AveBedrms", "AveOccup", "Population"]:
    hi = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=hi)                    # winsorize top 1%

print("After clipping:\n", df[["AveRooms", "AveOccup"]].describe().T)
```

**What & why:** This dataset has no missing values or duplicates (we *verify* rather than assume). The real issue is **extreme outliers** in ratio features — a district with very few households can produce an absurd `AveOccup` or `AveRooms`. We winsorize (clip) the top 1% so a handful of extreme rows don't dominate the squared-error loss and distort coefficients. We deliberately *don't* touch the target. In an exam, always show the missing/duplicate check explicitly even when the answer is zero — it demonstrates process.

---

### Cell 6 — Feature Engineering

```python
df["RoomsPerHousehold"]    = df["AveRooms"]  / df["AveOccup"].replace(0, np.nan)
df["BedroomsPerRoom"]      = df["AveBedrms"] / df["AveRooms"].replace(0, np.nan)
df["PopulationPerHouse"]   = df["Population"] / df["AveOccup"].replace(0, np.nan)
df = df.fillna(df.median(numeric_only=True))           # fill any NaN from divisions

X = df.drop(columns="Price")
y = df["Price"]
print("Engineered feature set:", list(X.columns))
```

**What & why:** Raw features are good, but **ratios often carry more signal** for price: bedrooms-per-room and rooms-per-household describe housing *quality/density* better than absolute counts. We guard every division against zero (`replace(0, np.nan)` then fill with the median) so we never emit `inf`. Thoughtful feature engineering is frequently the single biggest score-mover in a regression task and is exactly what interviewers probe — "what features would you create and why?".

---

### Cell 7 — Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train:", X_train.shape, " Test:", X_test.shape)
```

**What & why:** We carve off 20% as a **held-out test set that no model sees until the very end**. This is the golden rule against optimistic evaluation: every fitting/scaling/tuning decision happens on `X_train` only. `random_state=42` makes the split reproducible so the grader gets identical numbers. We split *before* scaling to prevent test-set statistics from leaking into the scaler.

---

### Cell 8 — Feature Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # LEARN mean/std on train only
X_test_scaled  = scaler.transform(X_test)        # APPLY same stats to test

# keep column names for later interpretation
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X.columns, index=X_test.index)
```

**What & why:** `StandardScaler` standardizes each feature to mean 0, std 1. This matters for two reasons: (1) **gradient-based and regularized models require it** — L1/L2 penalties compare coefficient magnitudes, which is only fair when features share a scale; (2) it makes coefficients roughly comparable for feature-importance reading. The crucial detail: we `fit_transform` on **train** and only `transform` the test set — fitting the scaler on test data would be leakage. (In production I'd wrap this in a Pipeline; here I keep it explicit so each step is visible.)

---

### Cell 9 — Train the Baseline: Linear Regression

```python
lin = LinearRegression().fit(X_train_scaled, y_train)
y_pred_lin = lin.predict(X_test_scaled)

print("Linear Regression")
print("  R2  :", round(r2_score(y_test, y_pred_lin), 3))
print("  RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred_lin)), 3))
print("  MAE :", round(mean_absolute_error(y_test, y_pred_lin), 3))
```

**What & why:** Always establish a **baseline** before reaching for fancier models — everything else must beat plain OLS to justify its complexity. Ordinary linear regression has no hyperparameters, so it's the honest reference point. We report R² (variance explained), RMSE (typical error in \$100k units), and MAE (robust average error) together, because no single metric tells the whole story.

---

### Cell 10 — Train Polynomial Regression (via Pipeline)

```python
poly_model = Pipeline([
    ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
    ("scale", StandardScaler()),
    ("lin",   LinearRegression()),
])
poly_model.fit(X_train, y_train)                 # note: raw X_train (pipeline scales)
y_pred_poly = poly_model.predict(X_test)

print("Polynomial (deg 2)")
print("  R2  :", round(r2_score(y_test, y_pred_poly), 3))
print("  RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred_poly)), 3))
```

**What & why:** We add degree-2 polynomial + interaction terms to capture **non-linear** relationships (e.g. price rising faster than linearly with income) that plain OLS misses. Because the expansion must happen before scaling and fitting *together*, we use a `Pipeline` and feed it the **raw** `X_train` — the pipeline scales internally. Degree 2 is a deliberate choice: high enough to add curvature, low enough to avoid the combinatorial feature explosion (and overfitting) of higher degrees.

---

### Cell 11 — Train the Regularized Models: Ridge, Lasso, ElasticNet

```python
ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)
lasso = Lasso(alpha=0.01, max_iter=10000).fit(X_train_scaled, y_train)
enet  = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000).fit(X_train_scaled, y_train)

for name, m in [("Ridge", ridge), ("Lasso", lasso), ("ElasticNet", enet)]:
    pred = m.predict(X_test_scaled)
    print(f"{name:11s}  R2={r2_score(y_test, pred):.3f} "
          f"RMSE={np.sqrt(mean_squared_error(y_test, pred)):.3f}")

print("\nLasso zeroed features:",
      list(X.columns[np.abs(lasso.coef_) < 1e-8]))
```

**What & why:** Three regularized variants on the **same scaled train data**: **Ridge** (L2) shrinks coefficients to combat the multicollinearity we saw in Cell 4; **Lasso** (L1) can zero out weak features → automatic feature selection; **ElasticNet** blends both (`l1_ratio=0.5`) — the safe choice when predictors are correlated, because Lasso alone arbitrarily drops one of a correlated pair. We print which features Lasso eliminated, turning the model into an interpretability tool. These `alpha` values are provisional — Cell 13 tunes them properly.

---

### Cell 12 — Evaluate & Compare All Models (Metrics Table)

```python
def evaluate(name, y_true, y_pred, n_features):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    n    = len(y_true)
    adj  = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)   # adjusted R2
    return {"Model": name,
            "MAE":  mean_absolute_error(y_true, y_pred),
            "RMSE": rmse, "R2": r2, "AdjR2": adj}

p = X_test_scaled.shape[1]
results = pd.DataFrame([
    evaluate("Linear",     y_test, y_pred_lin,  p),
    evaluate("Polynomial", y_test, y_pred_poly, p),
    evaluate("Ridge",      y_test, ridge.predict(X_test_scaled), p),
    evaluate("Lasso",      y_test, lasso.predict(X_test_scaled), p),
    evaluate("ElasticNet", y_test, enet.predict(X_test_scaled),  p),
]).sort_values("RMSE").reset_index(drop=True)

display(results)
```

**What & why:** A single **comparison table** is what a reviewer wants to see — MAE, RMSE, R², and Adjusted R² for every model, sorted by RMSE (lower is better). Adjusted R² is computed by hand (sklearn has no built-in) because it penalizes models that add features without genuinely improving fit — the fair way to compare the polynomial model (many features) against the linear one. Sorting makes the winner obvious at a glance and sets up the tuning step for the leading regularized model.

---

### Cell 13 — Hyperparameter Tuning: GridSearchCV for `alpha`

```python
ridge_pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
param_grid = {"ridge__alpha": np.logspace(-3, 3, 13)}   # 0.001 ... 1000

grid = GridSearchCV(
    ridge_pipe, param_grid, cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1, refit=True,
)
grid.fit(X_train, y_train)                              # raw X_train, pipeline scales

print("Best alpha  :", grid.best_params_["ridge__alpha"])
print("Best CV RMSE:", round(-grid.best_score_, 3))
best_model = grid.best_estimator_
```

**What & why:** We search a **log-spaced** alpha grid (regularization strength is multiplicative, so log spacing is the right geometry) using 5-fold cross-validation. Wrapping the scaler+Ridge in a Pipeline guarantees each fold scales using only its own training portion — **no leakage inside CV**. `scoring="neg_root_mean_squared_error"` optimizes RMSE (sklearn maximizes, hence the negative), `n_jobs=-1` parallelizes, and `refit=True` automatically retrains the best model on all training data. `grid.best_estimator_` is now our tuned, ready-to-use model.

---

### Cell 14 — Cross-Validation Stability Check

```python
cv_rmse = -cross_val_score(
    best_model, X_train, y_train, cv=5,
    scoring="neg_root_mean_squared_error"
)
print("Per-fold RMSE:", np.round(cv_rmse, 3))
print(f"Mean {cv_rmse.mean():.3f}  ±  {cv_rmse.std():.3f}")
```

**What & why:** A single test score can be luck; **k-fold CV on the tuned model** tells us how *stable* performance is across different data slices. We report mean ± std: a small std means the model generalizes consistently, a large one means it's sensitive to which rows it trains on (a warning sign). This distinguishes "we got a good number once" from "this model is reliably good" — exactly the rigor an interviewer looks for.

---

### Cell 15 — Feature Importance / Coefficient Interpretation

```python
coefs = pd.Series(best_model[-1].coef_, index=X.columns).sort_values()

plt.figure(figsize=(7, 5))
coefs.plot(kind="barh", color=np.where(coefs > 0, "steelblue", "salmon"))
plt.title(f"Ridge coefficients (alpha={grid.best_params_['ridge__alpha']})")
plt.xlabel("Coefficient (on standardized features)")
plt.tight_layout()
plt.show()

print(coefs.sort_values(ascending=False))
```

**What & why:** Because features were standardized, coefficient **magnitudes are directly comparable** → a legitimate feature-importance ranking for a linear model. The sign tells direction (positive = pushes price up), the magnitude tells strength. We expect `MedInc` to dominate positively, matching the EDA. This is the model's *explanation* — for a linear model, interpretability is a headline selling point, and being able to read coefficients back into business language ("a one-std increase in median income raises predicted value by X") is a common follow-up.

---

### Cell 16 — Final Prediction on New Data & Interpretation

```python
# Take a few real test rows and compare predicted vs actual
sample = X_test.iloc[:5]
pred   = best_model.predict(sample)

comparison = pd.DataFrame({
    "Predicted": pred,
    "Actual":    y_test.iloc[:5].values,
    "AbsError":  np.abs(pred - y_test.iloc[:5].values),
})
display(comparison.round(3))

# Predict for a brand-new hypothetical district (must match column order!)
new_district = X.iloc[[0]].copy()          # template with correct columns
new_district["MedInc"] = 8.5               # a wealthy district
print("\nPredicted median value ($100k):",
      round(best_model.predict(new_district)[0], 3))
```

**What & why:** The finale: use the tuned pipeline to predict on unseen rows and lay predictions next to actuals with the absolute error, so we can *see* where the model is accurate and where it struggles (often near the \$500k cap we spotted in Cell 3). Then we run a genuinely new hypothetical district to show the deployment-time call — feeding raw features into `best_model` (the pipeline scales internally). Key gotcha we handle: new input must have the **same columns in the same order** as training, which is why we clone a template row. This closes the loop from raw data to an actionable price estimate.

---

### Summary of the Workflow

| Stage | Cells | Purpose |
|-------|-------|---------|
| Setup & load | 1–2 | Imports, real dataset into a DataFrame |
| Understand | 3–4 | EDA: distributions, correlations, multicollinearity |
| Prepare | 5–8 | Clean, engineer ratios, split, scale (leak-free) |
| Model | 9–11 | Baseline → Polynomial → Ridge/Lasso/ElasticNet |
| Compare | 12 | MAE/RMSE/R²/AdjR² table |
| Tune & validate | 13–14 | GridSearchCV alpha + CV stability |
| Interpret & ship | 15–16 | Coefficients + final predictions |

**Mentor's closing note:** The single most-tested judgment in these labs is **no data leakage** — scale/expand *inside* the CV via Pipelines and fit transforms on train only. The second is **matching the metric to the goal** (RMSE for typical error in real units, MAE when outliers exist, Adjusted R² when comparing models with different feature counts). Nail those two and you're already ahead of most candidates.


---

# Part 2 — Classification (Practical)

# Practical Question 1: Implement the Sigmoid and Logistic Regression Prediction from Scratch

**Difficulty:** Easy
**Estimated Time:** 15 min
**Concepts Tested:** Sigmoid activation, linear combination (`w·x + b`), vectorization with NumPy, numerical stability, decision threshold.

## Problem Statement

Without using scikit-learn, implement the forward (prediction) path of a logistic regression classifier. You are given a trained weight vector `w`, a bias `b`, and a feature matrix `X`. Return:

1. The predicted **probabilities** `P(y=1 | x)` for each row.
2. The predicted **class labels** using a threshold of `0.5`.

Your sigmoid must be **numerically stable** (it should not overflow when the linear score is a large negative number).

## Example Input

```python
import numpy as np

X = np.array([[2.0, 1.0],
              [1.0, 3.0],
              [3.0, 3.0]])
w = np.array([0.5, -0.4])
b = 0.1
```

## Example Output

```
Probabilities: [0.6456563  0.24973989 0.450166  ]
Predicted labels: [1 0 0]
```

## Approach

Logistic regression computes a **linear score** `z = X·w + b`, then squashes it through the **sigmoid** `σ(z) = 1 / (1 + e^{-z})` to map any real number into `(0, 1)`. That value is interpreted as the probability of the positive class. We threshold at `0.5` to obtain a hard label.

The naive sigmoid `1 / (1 + np.exp(-z))` overflows when `z` is a large **negative** number, because `np.exp(-z)` becomes `inf`. The classic fix is a branchless, piecewise-stable formulation:

- For `z >= 0`: `σ(z) = 1 / (1 + e^{-z})` (here `-z <= 0`, so `e^{-z}` is bounded).
- For `z < 0`:  `σ(z) = e^{z} / (1 + e^{z})` (here `z < 0`, so `e^{z}` is bounded).

`np.where` combined with a clip gives us a clean, vectorized, overflow-free implementation.

## Python Implementation

```python
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid, applied element-wise.

    Uses the piecewise identity to avoid overflow of exp for large |z|:
        z >= 0 : 1 / (1 + exp(-z))
        z <  0 : exp(z) / (1 + exp(z))
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)

    pos = z >= 0
    neg = ~pos

    # Positive branch: exp(-z) is in (0, 1], no overflow.
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))

    # Negative branch: exp(z) is in (0, 1], no overflow.
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)

    return out


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Return P(y=1 | x) for each row of X."""
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    z = X @ w + b            # linear score, shape (n_samples,)
    return sigmoid(z)


def predict(X: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> np.ndarray:
    """Return hard 0/1 class labels."""
    probs = predict_proba(X, w, b)
    return (probs >= threshold).astype(int)


if __name__ == "__main__":
    X = np.array([[2.0, 1.0], [1.0, 3.0], [3.0, 3.0]])
    w = np.array([0.5, -0.4])
    b = 0.1
    print("Probabilities:", predict_proba(X, w, b))
    print("Predicted labels:", predict(X, w, b))
```

**Time Complexity:** `O(n · d)` — one dot product per sample over `d` features.
**Space Complexity:** `O(n)` for the probability vector (`X` and `w` are inputs).

## Alternative Solution

A one-line stable sigmoid using `np.clip` to bound the exponent before it ever overflows. Less mathematically elegant but perfectly serviceable in an interview:

```python
def sigmoid_clip(z):
    z = np.clip(z, -500, 500)   # exp(500) is finite in float64
    return 1.0 / (1.0 + np.exp(-z))
```

Another alternative avoids the linear-score / sigmoid split by using `scipy.special.expit`, which is already numerically stable and C-optimized:

```python
from scipy.special import expit
probs = expit(X @ w + b)
```

## Interview Variations

- "Now implement the **loss** (binary cross-entropy) and its gradient, then do one gradient-descent step." — extends this into training.
- "Generalize the threshold: return labels for an arbitrary decision threshold and explain how threshold choice trades precision for recall."
- "Do it for **multiclass** using softmax instead of sigmoid."

## Common Follow-up Questions

- *Why not just use `1/(1+exp(-z))`?* Because `exp(-z)` overflows to `inf` for large negative `z`, producing `nan`/warnings; the piecewise form keeps every exponent `<= 0`.
- *Why is the output a probability?* Because sigmoid maps `(-inf, inf) → (0, 1)` monotonically, and logistic regression is fit by maximum likelihood so its outputs are calibrated log-odds.
- *What does the bias term do geometrically?* It shifts the decision hyperplane away from the origin; without it the boundary is forced through `0`.

---

# Practical Question 2: Implement K-Nearest Neighbors from Scratch

**Difficulty:** Medium
**Estimated Time:** 25 min
**Concepts Tested:** Distance metrics (Euclidean/Manhattan), broadcasting, `argpartition` vs `argsort`, majority voting, tie-breaking, lazy learning.

## Problem Statement

Implement a KNN classifier as a class with `fit` and `predict` methods, **without** scikit-learn. Support both **Euclidean** and **Manhattan** distance. For each query point, find the `k` nearest training points and return the **majority class** among them. Break ties deterministically (e.g., by choosing the smallest class label).

## Example Input

```python
X_train = np.array([[1, 1], [1, 2], [2, 1],
                    [6, 6], [6, 5], [5, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1])
X_test  = np.array([[1.5, 1.5],
                    [5.5, 5.5]])
k = 3
```

## Example Output

```
Predictions: [0 1]
```

## Approach

KNN is a **lazy learner**: `fit` merely stores the training data. The work happens at `predict` time:

1. Compute the distance from each **test** point to every **training** point. With broadcasting we build an `(n_test, n_train)` distance matrix in one shot.
   - Euclidean: `sqrt(Σ (x_i - t_i)^2)`. (For ranking, the `sqrt` is monotonic and can be skipped, but we keep it for clarity.)
   - Manhattan: `Σ |x_i - t_i|`.
2. For each test row, take the indices of the `k` smallest distances. `np.argpartition` is `O(n)` versus `argsort`'s `O(n log n)` — a good detail to mention.
3. Gather those neighbors' labels and take the **mode** (majority vote). Ties are broken by smallest label via `np.bincount` + `argmax`.

## Python Implementation

```python
import numpy as np
from collections import Counter


class KNNClassifier:
    """A from-scratch K-Nearest Neighbors classifier."""

    def __init__(self, k: int = 3, metric: str = "euclidean"):
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        self.k = k
        self.metric = metric

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        # Lazy learner: just memorize the training set.
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        """Return an (n_test, n_train) matrix of distances via broadcasting."""
        # X[:, None, :] -> (n_test, 1, d); X_train[None] -> (1, n_train, d)
        diff = X[:, None, :] - self.X_train[None, :, :]
        if self.metric == "euclidean":
            return np.sqrt(np.sum(diff ** 2, axis=2))
        return np.sum(np.abs(diff), axis=2)          # manhattan

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        dists = self._distances(X)                   # (n_test, n_train)

        # Indices of the k smallest distances per row (unordered but correct set).
        knn_idx = np.argpartition(dists, kth=self.k - 1, axis=1)[:, :self.k]

        preds = np.empty(X.shape[0], dtype=self.y_train.dtype)
        for i, idx in enumerate(knn_idx):
            neighbor_labels = self.y_train[idx]
            # bincount + argmax => majority vote, ties broken by smallest label.
            preds[i] = np.bincount(neighbor_labels).argmax()
        return preds


if __name__ == "__main__":
    X_train = np.array([[1, 1], [1, 2], [2, 1], [6, 6], [6, 5], [5, 6]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[1.5, 1.5], [5.5, 5.5]])

    model = KNNClassifier(k=3, metric="euclidean").fit(X_train, y_train)
    print("Predictions:", model.predict(X_test))
```

**Time Complexity:** `O(n_test · n_train · d)` to build the distance matrix; voting adds `O(n_test · k)`. `argpartition` makes neighbor selection `O(n_train)` per query.
**Space Complexity:** `O(n_test · n_train)` for the distance matrix. For very large datasets, loop per test point to trade time for memory.

## Alternative Solution

Compute Euclidean distances **without** materializing the full 3-D `diff` tensor, using the identity `||a - b||² = ||a||² + ||b||² - 2 a·b`. This is far more memory-efficient and is what optimized libraries do:

```python
def euclidean_matrix(A, B):
    a2 = np.sum(A ** 2, axis=1)[:, None]      # (n_a, 1)
    b2 = np.sum(B ** 2, axis=1)[None, :]      # (1, n_b)
    d2 = a2 + b2 - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0))         # clamp tiny negatives from FP error
```

For voting, `Counter(neighbor_labels).most_common(1)[0][0]` is a readable non-NumPy alternative (but its tie-break is insertion-order, not smallest-label).

## Interview Variations

- "Add **distance-weighted** voting (closer neighbors count more, weight `= 1/d`)."
- "Support **non-integer / string** labels" — replace `bincount` with `Counter`.
- "Return the top-k neighbor **indices and distances**, not just the label."
- "Implement **KNN regression** (average the neighbors' targets instead of voting)."

## Common Follow-up Questions

- *Why must features be scaled first?* Distance is dominated by large-range features; without standardization, a feature in the thousands drowns out one in `[0,1]`.
- *How do you pick `k`?* Cross-validation; small `k` = low bias/high variance (noisy), large `k` = smoother but can underfit. Odd `k` avoids ties in binary problems.
- *Why is KNN slow at inference?* No training compression — every prediction scans all training data. KD-trees / Ball-trees / approximate NN (FAISS) speed this up in low-to-moderate dimensions.
- *Curse of dimensionality?* In high dimensions all points become roughly equidistant, so "nearest" loses meaning.

---

# Practical Question 3: Implement Gini Impurity, Entropy, and the Best Split of a Decision Tree

**Difficulty:** Medium
**Estimated Time:** 30 min
**Concepts Tested:** Impurity measures (Gini/Entropy), information gain, exhaustive split search over thresholds, greedy recursive partitioning.

## Problem Statement

Implement, from scratch:

1. `gini(y)` and `entropy(y)` for a set of class labels.
2. `best_split(X, y)` that scans **every feature** and **every candidate threshold** and returns the `(feature_index, threshold)` that **maximizes information gain** (equivalently, minimizes the weighted child impurity).

## Example Input

```python
X = np.array([[2.7], [1.3], [3.1], [0.5], [3.8], [1.1]])
y = np.array([1, 0, 1, 0, 1, 0])
```

## Example Output

```
Gini of parent: 0.5
Entropy of parent: 1.0
Best split -> feature 0, threshold 2.0, info_gain 1.0
```

*(Any threshold that cleanly separates the two groups, e.g. between 1.3 and 2.7, is optimal here.)*

## Approach

A decision tree grows greedily: at each node it picks the split that makes the **children as pure as possible**.

- **Gini impurity**: `1 - Σ p_c²` — probability of misclassifying a randomly labeled sample. Range `[0, 0.5]` for binary.
- **Entropy**: `-Σ p_c log₂ p_c` — expected information (bits). Range `[0, 1]` for binary.
- **Information Gain** = `impurity(parent) - weighted_avg(impurity(children))`, where the weights are the child sizes. We want the split with the **largest** gain.

For candidate thresholds, sort the unique values of a feature and test the **midpoints** between consecutive values (each such midpoint is a distinct partition of the data). Loop over every feature × every midpoint, compute the weighted child impurity, and keep the best.

## Python Implementation

```python
import numpy as np


def gini(y: np.ndarray) -> float:
    """Gini impurity of a label array."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1.0 - np.sum(probs ** 2)


def entropy(y: np.ndarray) -> float:
    """Shannon entropy (base 2) of a label array."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    # Filter zero-prob terms; 0*log0 := 0.
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def _weighted_impurity(y_left, y_right, criterion) -> float:
    """Size-weighted impurity of the two children."""
    n = len(y_left) + len(y_right)
    return (len(y_left) / n) * criterion(y_left) + \
           (len(y_right) / n) * criterion(y_right)


def best_split(X: np.ndarray, y: np.ndarray, criterion=gini):
    """Exhaustively find the (feature, threshold) with maximum information gain."""
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    parent_impurity = criterion(y)

    best = {"gain": -np.inf, "feature": None, "threshold": None}

    for feat in range(n_features):
        values = np.sort(np.unique(X[:, feat]))
        if len(values) < 2:
            continue
        # Candidate thresholds = midpoints between consecutive unique values.
        thresholds = (values[:-1] + values[1:]) / 2.0

        for thr in thresholds:
            left_mask = X[:, feat] <= thr
            y_left, y_right = y[left_mask], y[~left_mask]
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            gain = parent_impurity - _weighted_impurity(y_left, y_right, criterion)
            if gain > best["gain"]:
                best.update(gain=gain, feature=feat, threshold=thr)

    return best["feature"], best["threshold"], best["gain"]


if __name__ == "__main__":
    X = np.array([[2.7], [1.3], [3.1], [0.5], [3.8], [1.1]])
    y = np.array([1, 0, 1, 0, 1, 0])
    print("Gini of parent:", gini(y))
    print("Entropy of parent:", entropy(y))
    f, t, g = best_split(X, y, criterion=entropy)
    print(f"Best split -> feature {f}, threshold {t}, info_gain {g}")
```

**Time Complexity:** `O(d · n²)` naive (for each of `d` features, up to `n` thresholds, each costing `O(n)` to evaluate). Optimized trees sort once and update counts incrementally for `O(d · n log n)`.
**Space Complexity:** `O(n)` for the boolean masks.

## Alternative Solution

Incremental, single-pass evaluation per feature: **sort** the feature, then sweep the threshold from left to right while maintaining running class counts for the left/right partitions. This avoids recomputing `unique`/`bincount` at every threshold and is how CART is implemented efficiently:

```python
def best_split_fast(X, y):
    n, d = X.shape
    best = (-np.inf, None, None)
    for feat in range(d):
        order = np.argsort(X[:, feat])
        xs, ys = X[order, feat], y[order]
        classes = np.unique(y)
        left = np.zeros(len(classes)); right = np.bincount(ys, minlength=len(classes)).astype(float)
        for i in range(1, n):
            c = ys[i - 1]
            left[c] += 1; right[c] -= 1
            if xs[i] == xs[i - 1]:
                continue  # can't split between equal values
            nl, nr = i, n - i
            gl = 1 - ((left / nl) ** 2).sum()
            gr = 1 - ((right / nr) ** 2).sum()
            gain = -(nl * gl + nr * gr) / n   # maximize == minimize weighted impurity
            if gain > best[0]:
                best = (gain, feat, (xs[i] + xs[i - 1]) / 2)
    return best[1], best[2], best[0]
```

## Interview Variations

- "Wrap `best_split` in **recursion** to build a full tree with `max_depth` and `min_samples_split` stopping criteria."
- "Add **majority-class prediction** at the leaves and a `predict` method."
- "Handle **categorical features** (equality splits instead of `<= threshold`)."

## Common Follow-up Questions

- *Gini vs Entropy — which is better?* Practically almost identical; Gini is slightly cheaper (no logarithm) and is scikit-learn's default. Entropy can favor more balanced splits marginally.
- *Why do trees overfit?* Grown fully, they memorize training data (pure leaves). Control with `max_depth`, `min_samples_leaf`, or post-pruning (`ccp_alpha`).
- *Why don't trees need feature scaling?* Splits depend only on the **order** of values within a feature, not their magnitude.

---

# Practical Question 4: Train an SVM with Different Kernels and Compare

**Difficulty:** Medium
**Estimated Time:** 25 min
**Concepts Tested:** Support Vector Machines, the kernel trick (linear/poly/RBF), `C` and `gamma` hyperparameters, feature scaling, model comparison.

## Problem Statement

Using scikit-learn, train SVM classifiers with **linear**, **polynomial**, and **RBF** kernels on the same dataset. Report test accuracy and macro-F1 for each, print the number of support vectors, and identify which kernel wins. Ensure features are scaled (SVMs are distance/inner-product based and are sensitive to scale).

## Example Input

```python
# Synthetic non-linearly-separable data (two moons).
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
```

## Example Output

```
kernel=linear  acc=0.850  f1=0.849  n_support=[.. ..]
kernel=poly    acc=0.900  f1=0.900  n_support=[.. ..]
kernel=rbf     acc=0.958  f1=0.958  n_support=[.. ..]
Best kernel: rbf
```

*(Exact numbers depend on the random seed; RBF should win on moons.)*

## Approach

An SVM finds the **maximum-margin** hyperplane. For non-linearly-separable data, the **kernel trick** implicitly maps inputs into a higher-dimensional space where a linear separator exists — without ever computing that mapping explicitly.

- **linear**: `K(x, x') = x·x'` — fast, interpretable, best when data is (nearly) linearly separable.
- **poly**: `K(x, x') = (γ x·x' + r)^degree` — captures polynomial interactions.
- **rbf** (Gaussian): `K(x, x') = exp(-γ ||x - x'||²)` — very flexible, the default workhorse.

Always put a `StandardScaler` before the SVM in a `Pipeline` so scaling is fit on training folds only (no leakage). Compare kernels on a held-out test set with a consistent metric.

## Python Implementation

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


def compare_svm_kernels(X, y, random_state: int = 42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    kernels = {
        "linear": dict(kernel="linear", C=1.0),
        "poly":   dict(kernel="poly", degree=3, C=1.0, gamma="scale"),
        "rbf":    dict(kernel="rbf", C=1.0, gamma="scale"),
    }

    results = {}
    for name, params in kernels.items():
        # Scaler + SVM in one pipeline => scaling fit only on training data.
        model = make_pipeline(StandardScaler(), SVC(**params, random_state=random_state))
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="macro")
        n_support = model.named_steps["svc"].n_support_
        results[name] = acc
        print(f"kernel={name:7s} acc={acc:.3f} f1={f1:.3f} n_support={n_support}")

    best = max(results, key=results.get)
    print("Best kernel:", best)
    return results


if __name__ == "__main__":
    X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
    compare_svm_kernels(X, y)
```

**Time Complexity:** Training a kernel SVM is roughly `O(n²)`–`O(n³)` in the number of samples (it solves a quadratic program), which is why SVMs don't scale to millions of rows.
**Space Complexity:** `O(n²)` in the worst case for the kernel matrix.

## Alternative Solution

Instead of three separate fits, let `GridSearchCV` select the kernel **and** its hyperparameters together — cleaner and cross-validated:

```python
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), SVC())
grid = {
    "svc__kernel": ["linear", "rbf", "poly"],
    "svc__C": [0.1, 1, 10],
    "svc__gamma": ["scale", 0.1, 1],
}
search = GridSearchCV(pipe, grid, cv=5, scoring="f1_macro", n_jobs=-1)
search.fit(X, y)
print(search.best_params_, search.best_score_)
```

For large `n`, `LinearSVC` (liblinear, `O(n)`) replaces `SVC(kernel="linear")`, and `SGDClassifier(loss="hinge")` approximates a linear SVM online.

## Interview Variations

- "Sweep `C` and `gamma` and describe the **bias-variance** effect of each."
- "Plot the **decision boundary** for each kernel" (see Question 6).
- "Explain `decision_function` vs `predict_proba` for SVMs and enable probabilities (`probability=True`)."

## Common Follow-up Questions

- *What does `C` control?* Regularization / margin softness. Small `C` = wider margin, more tolerance for misclassification (higher bias). Large `C` = tries to classify every point correctly (higher variance/overfit).
- *What does `gamma` control in RBF?* The reach of each point's influence. Large `gamma` = tight, wiggly boundaries (overfit); small `gamma` = smooth, near-linear.
- *Why scale features for SVM?* The RBF kernel and margins depend on Euclidean distances/inner products; unscaled features distort them.
- *What are support vectors?* The training points on or inside the margin that define the boundary; only they matter for prediction.

---

# Practical Question 5: Evaluate a Classifier — Confusion Matrix and Metrics from Scratch

**Difficulty:** Medium
**Estimated Time:** 25 min
**Concepts Tested:** Confusion matrix (TP/FP/FN/TN), precision, recall, F1, accuracy, ROC-AUC intuition, threshold effects, class imbalance.

## Problem Statement

Given ground-truth binary labels `y_true` and predicted labels `y_pred`, compute — **without** scikit-learn — the confusion matrix and the four core metrics: accuracy, precision, recall, and F1. Then, given predicted **probabilities**, show how sweeping the decision threshold changes precision and recall.

## Example Input

```python
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
```

## Example Output

```
Confusion matrix:
              pred=0  pred=1
 actual=0        4       1
 actual=1        1       4
accuracy=0.800 precision=0.800 recall=0.800 f1=0.800
```

## Approach

For binary classification with positive class `1`:

- **TP**: predicted 1 and actually 1. **TN**: predicted 0 and actually 0.
- **FP**: predicted 1 but actually 0 (false alarm). **FN**: predicted 0 but actually 1 (miss).

Then:
- `accuracy  = (TP + TN) / total`
- `precision = TP / (TP + FP)` — of those we flagged positive, how many were right? (Cost of false alarms.)
- `recall    = TP / (TP + FN)` — of the actual positives, how many did we catch? (Cost of misses.)
- `F1        = 2 · P · R / (P + R)` — harmonic mean, balances the two.

Guard every denominator against zero. For threshold sweeping, convert probabilities to labels at each candidate threshold and recompute precision/recall — as the threshold rises, precision typically increases while recall falls.

## Python Implementation

```python
import numpy as np


def confusion_matrix_binary(y_true, y_pred, positive=1):
    """Return TP, FP, FN, TN for the given positive class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == positive) & (y_true == positive)))
    fp = int(np.sum((y_pred == positive) & (y_true != positive)))
    fn = int(np.sum((y_pred != positive) & (y_true == positive)))
    tn = int(np.sum((y_pred != positive) & (y_true != positive)))
    return tp, fp, fn, tn


def classification_metrics(y_true, y_pred, positive=1):
    tp, fp, fn, tn = confusion_matrix_binary(y_true, y_pred, positive)
    total = tp + fp + fn + tn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1}


def precision_recall_over_thresholds(y_true, y_proba, thresholds=None):
    """Show how precision/recall move as the decision threshold changes."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)
    rows = []
    for t in thresholds:
        y_pred = (np.asarray(y_proba) >= t).astype(int)
        m = classification_metrics(y_true, y_pred)
        rows.append((round(t, 2), round(m["precision"], 3), round(m["recall"], 3)))
    return rows


if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    m = classification_metrics(y_true, y_pred)
    print("Confusion matrix:")
    print(f"              pred=0  pred=1")
    print(f" actual=0    {m['tn']:5d}   {m['fp']:5d}")
    print(f" actual=1    {m['fn']:5d}   {m['tp']:5d}")
    print(f"accuracy={m['accuracy']:.3f} precision={m['precision']:.3f} "
          f"recall={m['recall']:.3f} f1={m['f1']:.3f}")
```

**Time Complexity:** `O(n)` — a few vectorized boolean reductions.
**Space Complexity:** `O(n)` for the boolean masks.

## Alternative Solution

Use scikit-learn to cross-check your from-scratch numbers — always worth doing in a lab exam:

```python
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, classification_report)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=3))
```

## Interview Variations

- "Extend to **multiclass** with macro / micro / weighted averaging."
- "Compute **specificity** (`TN/(TN+FP)`) and explain when it matters (e.g., screening)."
- "Implement **ROC-AUC** by sorting scores and integrating TPR vs FPR."

## Common Follow-up Questions

- *Why is accuracy misleading on imbalanced data?* A model predicting the majority class always can score 99% accuracy while catching zero positives. Use precision/recall/F1/AUC instead.
- *Precision vs recall — which to optimize?* Depends on cost asymmetry: spam filter favors **precision** (don't drop real mail); cancer screening favors **recall** (don't miss a case).
- *What's ROC-AUC intuitively?* The probability that the model ranks a random positive above a random negative; threshold-independent, `0.5` = random, `1.0` = perfect.
- *PR-AUC vs ROC-AUC?* With heavy class imbalance, PR-AUC is more informative because ROC can look optimistic when negatives dominate.

---

# Practical Question 6: Plot the Decision Boundary of a Classifier

**Difficulty:** Hard
**Estimated Time:** 30 min
**Concepts Tested:** Meshgrid construction, `predict` over a grid, `contourf`, comparing model geometry (linear vs non-linear boundaries), visualization hygiene.

## Problem Statement

Write a reusable function `plot_decision_boundary(model, X, y)` that, for any fitted 2-feature scikit-learn classifier, shades the regions the model assigns to each class and overlays the data points. Use it to visually contrast a Logistic Regression (linear boundary), a Decision Tree (axis-aligned boxes), and an RBF SVM (curved boundary) on the same dataset.

## Example Input

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
```

## Example Output

A figure with three panels (LogReg / Tree / RBF-SVM). The LogReg panel shows a straight dividing line; the Tree shows rectangular staircase regions; the SVM shows a smooth curved boundary hugging the two moons. *(Visual output — no console text.)*

## Approach

A decision boundary is where the classifier's prediction flips. To draw it in 2-D:

1. Build a dense **meshgrid** spanning the feature ranges (with a small margin).
2. Flatten the grid into an `(m, 2)` array of points and call `model.predict`.
3. Reshape predictions back to the grid and pass them to `plt.contourf` — each predicted class becomes a colored region.
4. Overlay the actual training points with `plt.scatter`, colored by true label.

Keep the grid resolution moderate (`~300×300`) — finer grids are slower and rarely add insight. Always fit models inside a scaling pipeline so the plotted geometry reflects the real model.

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def plot_decision_boundary(model, X, y, ax=None, title="", resolution=300):
    """Shade class regions of a fitted 2-feature classifier and overlay data."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    # 1. Grid spanning the feature space with a small margin.
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    # 2-3. Predict over the grid and reshape.
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    # 4. Filled contour for regions + scatter for actual points.
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm",
               edgecolors="k", s=25)
    ax.set_title(title)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    return ax


if __name__ == "__main__":
    X, y = make_moons(n_samples=300, noise=0.25, random_state=0)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression()),
        "Decision Tree (depth=5)": DecisionTreeClassifier(max_depth=5, random_state=0),
        "RBF SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale")),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X, y)
        plot_decision_boundary(model, X, y, ax=ax, title=name)
    plt.tight_layout()
    plt.savefig("decision_boundaries.png", dpi=120)  # or plt.show()
    print("Saved decision_boundaries.png")
```

**Time Complexity:** `O(resolution² · inference_cost)` — the grid has `resolution²` points and each is predicted. This dominates, so keep the resolution sane.
**Space Complexity:** `O(resolution²)` for the meshgrid arrays.

## Alternative Solution

Use scikit-learn's built-in helper, which handles the meshgrid for you and can shade probabilities instead of hard classes:

```python
from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(
    model, X, response_method="predict",  # or "predict_proba" / "decision_function"
    alpha=0.3, cmap="coolwarm",
)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
```

To visualize **confidence** rather than hard regions, use `decision_function` / `predict_proba` with `contourf` and a diverging colormap centered at the boundary.

## Interview Variations

- "Shade **probability** (confidence) with a smooth gradient instead of hard regions."
- "Plot boundaries for **>2 features** — you can't directly, so reduce with PCA to 2-D first and explain the caveat."
- "Show how `max_depth` (tree) or `gamma` (SVM) changes the boundary's complexity."

## Common Follow-up Questions

- *Why is the Logistic Regression boundary always straight?* It's linear in the input features (`w·x + b = 0` is a hyperplane); it can only curve if you add polynomial features.
- *Why does the tree produce rectangular regions?* Each split is a single-feature threshold (`x_j <= t`), so boundaries are axis-aligned steps.
- *What does a very wiggly SVM boundary tell you?* High `gamma` / low regularization — likely overfitting; validate on held-out data.

---

# PART B — End-to-End Notebook Workflow: Heart Disease Prediction

> **Goal:** a complete, runnable binary-classification pipeline. For guaranteed reproducibility with zero external files, we use scikit-learn's `load_breast_cancer` as a stand-in "diagnosis" dataset (569 patients, 30 numeric features, target = malignant/benign). The exact same workflow transfers to a real `heart.csv` — a commented `pd.read_csv` variant is shown in Cell 2 so you can swap it in. Run the cells top to bottom.

---

### Cell 1 — Imports and Global Setup

```python
# --- Core stack ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Modeling ---
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# --- Evaluation ---
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, ConfusionMatrixDisplay)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
print("Environment ready.")
```

**What & why:** We front-load every import so later cells never fail mid-run with a `NameError`. A single `RANDOM_STATE` constant is threaded through every split and model so results are reproducible — non-negotiable in a graded lab. `sns.set_theme` gives consistent, readable plots.

---

### Cell 2 — Load the Dataset

```python
# Option A (used here): built-in, self-contained, always runnable.
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
# The sklearn target is 1 = benign, 0 = malignant. We rename for clarity;
# treat "1 = positive/disease-like" consistently. Here target==1 is "benign".
df = df.rename(columns={"target": "target"})

# Option B (real heart dataset) — uncomment to use your own CSV instead:
# df = pd.read_csv("heart.csv")
# # Typical UCI heart columns: age, sex, cp, trestbps, chol, fbs, restecg,
# # thalach, exang, oldpeak, slope, ca, thal, target (1 = disease, 0 = no disease)

X_cols = [c for c in df.columns if c != "target"]
print("Shape:", df.shape)
df.head()
```

**What & why:** We keep a DataFrame (`as_frame=True`) so we can do EDA with named columns. The commented Option B shows exactly how to switch to a real `heart.csv` — the rest of the notebook is agnostic to which source you pick, as long as the target column is named `target`. Always print the shape immediately to confirm the load.

---

### Cell 3 — First Look: Structure, Types, and Target Balance

```python
print("=== info ===")
df.info()

print("\n=== target distribution ===")
print(df["target"].value_counts())
print(df["target"].value_counts(normalize=True).round(3))

# Visualize class balance — dictates whether accuracy is trustworthy.
plt.figure(figsize=(4, 3))
sns.countplot(x="target", data=df)
plt.title("Class balance")
plt.show()
```

**What & why:** `info()` reveals dtypes and non-null counts in one shot. The **target distribution** is the single most important early check: it decides whether accuracy is a fair metric and whether we need `stratify`, `class_weight`, or resampling. This dataset is moderately imbalanced (~63% / 37%), enough to justify stratified splitting and reporting F1/AUC.

---

### Cell 4 — Descriptive Statistics and Missing-Value Audit

```python
print("=== describe (numeric) ===")
display(df.describe().T[["mean", "std", "min", "max"]])

print("\n=== missing values per column ===")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values.")

# Sanity check for duplicate rows.
print("\nDuplicate rows:", df.duplicated().sum())
```

**What & why:** `describe().T` exposes wildly different feature scales (e.g., `mean area` in the thousands vs `mean smoothness` near 0.1) — direct evidence that **scaling is mandatory** for KNN/SVM/LogReg. We explicitly audit missing values and duplicates now so nothing surprises us during modeling. (This built-in set is clean; a real `heart.csv` often has `?`/`0` sentinels — handled in Cell 6.)

---

### Cell 5 — Exploratory Data Analysis (Correlations & Distributions)

```python
# Correlation heatmap on a manageable subset of the most informative features.
top_features = (df.corr()["target"].abs()
                  .sort_values(ascending=False)
                  .index[1:9])  # top 8 excluding target itself

plt.figure(figsize=(8, 6))
sns.heatmap(df[list(top_features) + ["target"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation of top features with target")
plt.tight_layout()
plt.show()

# Distribution of the single most correlated feature, split by class.
best_feat = top_features[0]
plt.figure(figsize=(6, 3))
sns.kdeplot(data=df, x=best_feat, hue="target", fill=True, common_norm=False)
plt.title(f"Distribution of '{best_feat}' by class")
plt.show()
```

**What & why:** The heatmap shows which features carry signal (high `|corr|` with target) and flags **multicollinearity** (feature pairs near ±1), which hurts linear-model interpretability. The class-conditional KDE of the top feature gives an intuitive sense of separability — if the two class curves barely overlap, even simple models will do well.

---

### Cell 6 — Missing-Value Handling (Robust, Leakage-Safe Pattern)

```python
# For this clean dataset this is largely a no-op, but we write the
# production pattern you'd use on a real heart.csv.

# 1. Replace common sentinel values with real NaN (e.g., '?' or impossible 0s).
df = df.replace("?", np.nan)

# 2. Coerce everything numeric (strings from a CSV become NaN if unparseable).
for col in X_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3. IMPORTANT: imputation is done INSIDE the pipeline / after the split
#    to avoid leaking test-set statistics. Here we just confirm cleanliness.
print("Remaining NaNs:", int(df.isnull().sum().sum()))
```

**What & why:** The cardinal rule: **never impute using statistics computed over the whole dataset before splitting** — that leaks test information into training. We only do label-independent cleaning (sentinel replacement, type coercion) here; the actual mean/median imputation belongs in a `SimpleImputer` inside the pipeline so it's fit on training folds only. For a real dataset with NaNs, add `SimpleImputer(strategy="median")` as the first pipeline step.

---

### Cell 7 — Feature/Target Split and Stratified Train-Test Split

```python
X = df[X_cols].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,            # preserve class ratio in both splits
    random_state=RANDOM_STATE,
)

print("Train:", X_train.shape, " Test:", X_test.shape)
print("Train target balance:", np.round(np.bincount(y_train) / len(y_train), 3))
print("Test  target balance:", np.round(np.bincount(y_test) / len(y_test), 3))
```

**What & why:** `stratify=y` guarantees the train and test sets carry the same class proportions — critical on imbalanced data so the test set actually contains enough positives to measure recall. We hold out 20% as a **final, untouched** test set; all tuning happens via cross-validation on the training portion only.

---

### Cell 8 — Feature Scaling (Fit on Train Only)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on TRAIN
X_test_scaled = scaler.transform(X_test)         # transform ONLY on TEST

print("Train mean ~0:", np.round(X_train_scaled.mean(axis=0)[:3], 3))
print("Train std  ~1:", np.round(X_train_scaled.std(axis=0)[:3], 3))
```

**What & why:** `StandardScaler` centers each feature to mean 0, std 1 — essential for distance-based (KNN, SVM) and gradient/regularized (LogReg) models so no single large-magnitude feature dominates. Note we `fit` **only on training data** and merely `transform` the test set: the test set must be scaled using training statistics to avoid leakage. (In the tuning cells we wrap scaling in a `Pipeline` so cross-validation folds are scaled correctly and automatically.)

---

### Cell 9 — Train the Four Baseline Models

```python
models = {
    "LogisticRegression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "KNN":                KNeighborsClassifier(n_neighbors=5),
    "DecisionTree":       DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    "SVM_RBF":            SVC(kernel="rbf", gamma="scale",
                             probability=True, random_state=RANDOM_STATE),
}

# Tree is scale-invariant, but using scaled data uniformly keeps the loop simple
# and does not hurt tree performance.
fitted = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    fitted[name] = model
    print(f"{name:20s} trained.")
```

**What & why:** We train all four classification workhorses on identical scaled data so the comparison is apples-to-apples. `max_iter=5000` prevents LogReg convergence warnings; `probability=True` lets the SVM produce scores for ROC-AUC; a shallow tree (`max_depth=4`) is a sane baseline that resists overfitting. Storing fitted models in a dict makes the evaluation loop trivial.

---

### Cell 10 — Evaluate All Models (Full Metric Suite)

```python
def evaluate(model, X_te, y_te):
    """Return a dict of the standard classification metrics."""
    y_pred = model.predict(X_te)
    # Prefer probabilities for AUC; fall back to decision_function.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_te)[:, 1]
    else:
        y_score = model.decision_function(X_te)
    return {
        "accuracy":  accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall":    recall_score(y_te, y_pred),
        "f1":        f1_score(y_te, y_pred),
        "roc_auc":   roc_auc_score(y_te, y_score),
    }

rows = {name: evaluate(m, X_test_scaled, y_test) for name, m in fitted.items()}
results_df = pd.DataFrame(rows).T.round(4).sort_values("f1", ascending=False)
print(results_df)
```

**What & why:** One helper computes the five metrics that matter for classification — **accuracy, precision, recall, F1, and ROC-AUC** — for every model. We collect them into a sortable DataFrame so the leaderboard is instantly readable. Reporting the full suite (not just accuracy) is the whole point on imbalanced medical data: a model can be accurate yet miss too many positives (low recall).

---

### Cell 11 — Confusion Matrices for Each Model

```python
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, model) in zip(axes, fitted.items()):
    cm = confusion_matrix(y_test, model.predict(X_test_scaled))
    ConfusionMatrixDisplay(cm, display_labels=["neg(0)", "pos(1)"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.show()

# Text report for the current best model (top of the leaderboard).
best_name = results_df.index[0]
print(f"=== classification_report: {best_name} ===")
print(classification_report(y_test, fitted[best_name].predict(X_test_scaled),
                            target_names=["neg(0)", "pos(1)"]))
```

**What & why:** The confusion matrix exposes **where** each model errs — false positives (top-right) vs false negatives (bottom-left). In a disease-prediction context, false negatives (missed cases) are usually the costlier error, so we inspect them directly rather than trusting a single scalar. `classification_report` prints per-class precision/recall/F1 for the leader.

---

### Cell 12 — Hyperparameter Tuning with GridSearchCV (Pipelines)

```python
# Each search wraps scaling + estimator in a Pipeline so every CV fold is
# scaled using ONLY that fold's training data (no leakage).
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search_spaces = {
    "LogisticRegression": (
        Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]),
        {"clf__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "KNN": (
        Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier())]),
        {"clf__n_neighbors": [3, 5, 7, 11, 15], "clf__weights": ["uniform", "distance"]},
    ),
    "SVM_RBF": (
        Pipeline([("sc", StandardScaler()), ("clf", SVC(probability=True))]),
        {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]},
    ),
}

best_estimators = {}
for name, (pipe, grid) in search_spaces.items():
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)                      # raw X: pipeline scales internally
    best_estimators[name] = gs.best_estimator_
    print(f"{name:20s} best_f1={gs.best_score_:.4f}  params={gs.best_params_}")
```

**What & why:** `GridSearchCV` exhaustively cross-validates every hyperparameter combination and keeps the best by F1. Crucially, we pass **raw** `X_train` (not pre-scaled) and let the `Pipeline` scale inside each fold — the correct, leakage-free way to tune scale-sensitive models. `StratifiedKFold` keeps class ratios stable across folds; `n_jobs=-1` parallelizes over all cores.

---

### Cell 13 — Cross-Validation of the Tuned Models (Stability Check)

```python
cv_summary = {}
for name, est in best_estimators.items():
    scores = cross_val_score(est, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    cv_summary[name] = {"cv_f1_mean": scores.mean(), "cv_f1_std": scores.std()}
    print(f"{name:20s} F1 = {scores.mean():.4f} +/- {scores.std():.4f}")

cv_df = pd.DataFrame(cv_summary).T.round(4).sort_values("cv_f1_mean", ascending=False)
cv_df
```

**What & why:** A single test score can be lucky. Re-running 5-fold CV on the **tuned** estimators reports the mean **and standard deviation** of F1 — the std tells us how stable/reliable each model is. A model with a slightly lower mean but much smaller std is often the safer production choice. This guards against selecting a model that merely got a favorable test split.

---

### Cell 14 — Final Evaluation on the Held-Out Test Set + Comparison Table

```python
# Refit each tuned pipeline on ALL training data, then judge once on the
# untouched test set.
final_rows = {}
for name, est in best_estimators.items():
    est.fit(X_train, y_train)
    final_rows[name] = evaluate(est, X_test, y_test)   # pipeline scales internally

# Add the shallow decision tree (tuned separately for completeness).
tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree_gs = GridSearchCV(tree, {"max_depth": [3, 4, 5, 7, None],
                              "min_samples_leaf": [1, 3, 5]},
                       cv=cv, scoring="f1", n_jobs=-1).fit(X_train, y_train)
best_estimators["DecisionTree"] = tree_gs.best_estimator_
final_rows["DecisionTree"] = evaluate(tree_gs.best_estimator_, X_test, y_test)

comparison = pd.DataFrame(final_rows).T.round(4).sort_values("roc_auc", ascending=False)
print("=== FINAL MODEL COMPARISON (held-out test set) ===")
print(comparison)
```

**What & why:** The moment of truth: each tuned model is refit on the full training set and scored **once** on the test set it has never seen. We assemble a single comparison table ranked by ROC-AUC (threshold-independent) so the winner is obvious. The decision tree is tuned here too, since trees don't need the scaling pipeline. This table is exactly what you'd present as your result.

---

### Cell 15 — ROC Curves for All Models (One Plot)

```python
plt.figure(figsize=(7, 6))
for name, est in best_estimators.items():
    if hasattr(est, "predict_proba"):
        y_score = est.predict_proba(X_test)[:, 1]
    else:
        y_score = est.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curves — Tuned Models")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
```

**What & why:** The ROC curve plots the true-positive rate against the false-positive rate across **all** decision thresholds, so it summarizes a model's ranking ability independent of any single cutoff. Overlaying every model on one axis makes the comparison visual: the curve closest to the top-left (largest area) is best; the dashed diagonal is a random baseline. AUC in the legend quantifies it.

---

### Cell 16 — Interpretation, Model Selection, and Next Steps

```python
winner = comparison.index[0]
print(f"Selected model: {winner}")
print(comparison.loc[winner])

# For a linear model, inspect coefficient magnitudes for interpretability.
if winner == "LogisticRegression":
    clf = best_estimators[winner].named_steps["clf"]
    coefs = pd.Series(clf.coef_[0], index=X_cols).sort_values(key=np.abs, ascending=False)
    print("\nTop 8 drivers (|coef|):")
    print(coefs.head(8).round(3))
```

**What & why (wrap-up):**

- **Model choice.** Pick the winner by the metric that matches the clinical cost. On this data LogReg and RBF-SVM typically tie near the top (AUC ~0.99). Prefer **Logistic Regression** when interpretability matters — its coefficients tell a clinician *which* features drive risk; prefer **SVM** if it gives a clear F1/recall edge and interpretability is secondary.
- **Metric priority.** For disease prediction, **recall** (catching true cases) usually outranks raw accuracy; if false positives are costly (expensive follow-up tests), balance with precision via F1. ROC-AUC ranks overall discrimination.
- **Guarding against overfitting.** The CV std (Cell 13) and the gap between CV F1 and test F1 tell you if a model is over-tuned. Small, stable gaps = trustworthy.
- **Threshold tuning.** The default `0.5` cutoff is rarely optimal on imbalanced data. Use the ROC/PR curves to pick a threshold that hits your target recall, then re-report precision at that operating point.
- **Next steps.** Add `class_weight="balanced"` if positives are rare, try ensemble models (RandomForest / GradientBoosting) as stronger baselines, and validate on an external cohort before trusting the model in practice.


---

# Part 3 — Model Optimization (Practical)

## Part A — Coding Questions

> **How to use this section:** Each question mirrors what interviewers actually ask in a live coding round or a Jupyter lab exam. Read the *Problem Statement*, try to solve it before peeking at the implementation, and pay special attention to every place I mention **data leakage** — it is the single most common reason strong-looking solutions get rejected. If you can articulate *why* fitting a scaler on the full dataset is wrong, you are already ahead of most candidates.

---

# Practical Question 1: Implement StandardScaler From Scratch

**Difficulty:** Easy
**Estimated Time:** 15 min
**Concepts Tested:** Feature scaling math, fit/transform API contract, train-only fitting, avoiding leakage

## Problem Statement
Implement a `StandardScaler` class from scratch that mirrors the scikit-learn API: a `fit` method that learns the per-feature mean and standard deviation, a `transform` method that standardizes data using the *learned* statistics, and a convenience `fit_transform`. The critical rule: statistics must be learned on the **training set only** and then reused on the test set. Standardization maps each feature to zero mean and unit variance:

```
z = (x - mean) / std
```

## Example Input
```python
import numpy as np
X_train = np.array([[1.0, 100.0],
                    [2.0, 200.0],
                    [3.0, 300.0]])
X_test  = np.array([[4.0, 400.0]])
```

## Example Output
```python
# Train means = [2.0, 200.0], stds = [0.8165, 81.65]
scaled_train ≈ [[-1.2247, -1.2247],
                [ 0.0000,  0.0000],
                [ 1.2247,  1.2247]]
# Test uses TRAIN stats, NOT its own:
scaled_test  ≈ [[3.6742, 3.6742]]   # extrapolates beyond train range — expected
```

## Approach
1. In `fit`, compute the mean and standard deviation **column-wise** (`axis=0`) and store them as instance attributes.
2. Guard against divide-by-zero: if a feature has zero variance (a constant column), replace its std with `1.0` so the output is simply `x - mean = 0`.
3. In `transform`, apply the stored statistics — never recompute them.
4. `fit_transform` is just `fit` followed by `transform`.
5. Use the **population** standard deviation (`ddof=0`) to match scikit-learn's `StandardScaler`.

## Python Implementation
```python
import numpy as np


class StandardScalerScratch:
    """A from-scratch re-implementation of sklearn's StandardScaler.

    Learns per-feature mean/std on fit() and reuses them on transform().
    This separation is what prevents test-set statistics from leaking
    into preprocessing.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None  # std deviation per feature

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)                 # per-column mean, O(n*d)
        std = X.std(axis=0, ddof=0)                 # population std (ddof=0)
        # A constant column has std 0 -> would divide by zero.
        # sklearn replaces 0 with 1 so those features map to 0.
        std[std == 0.0] = 1.0
        self.scale_ = std
        return self  # return self so calls can be chained, like sklearn

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_        # broadcasting, O(n*d)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# --- Verify against sklearn ---------------------------------------------
if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler

    X_train = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    X_test = np.array([[4.0, 400.0]])

    mine = StandardScalerScratch().fit(X_train)
    ref = StandardScaler().fit(X_train)

    assert np.allclose(mine.transform(X_train), ref.transform(X_train))
    assert np.allclose(mine.transform(X_test), ref.transform(X_test))
    print("Scratch scaler matches sklearn. Test scaled:", mine.transform(X_test))
```

**Complexity:** `fit` and `transform` are both `O(n·d)` time (one pass over the matrix) and `O(d)` extra space to store the statistics.

## Alternative Solution
Use `np.nanmean` / `np.nanstd` if the data may contain `NaN`s, or subclass `sklearn.base.BaseEstimator, TransformerMixin` to get `fit_transform` and grid-search compatibility for free:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self
    def transform(self, X):
        return (X - self.mean_) / self.scale_
```
Inheriting `TransformerMixin` means it drops straight into a `Pipeline`.

## Interview Variations
- Implement `inverse_transform` (`x = z * scale_ + mean_`).
- Implement `MinMaxScaler` (`(x - min) / (max - min)`) or `RobustScaler` (`(x - median) / IQR`) from scratch.
- Extend to handle sparse matrices without densifying (only `MaxAbsScaler`-style scaling is safe on sparse data because centering destroys sparsity).

## Common Follow-up Questions
- **"Why fit on train only?"** Because the scaler's mean/std are model parameters learned from data. If you compute them over train+test, information about the test distribution bleeds into training — leakage — and your cross-validation scores become optimistically biased.
- **"Which scaler for outliers?"** `RobustScaler` — it uses median and IQR, which are far less sensitive to extreme values than mean/std.
- **"Do tree models need scaling?"** No. Decision trees, random forests, and gradient boosting split on thresholds and are invariant to monotonic feature rescaling. Scaling matters for distance/gradient-based models: KNN, SVM, logistic/linear regression with regularization, PCA, and neural nets.

---

# Practical Question 2: Compare Standard vs MinMax vs Robust Scaling

**Difficulty:** Easy
**Estimated Time:** 12 min
**Concepts Tested:** Choosing a scaler, effect of outliers, reading distributions

## Problem Statement
Given a feature with a heavy outlier, apply `StandardScaler`, `MinMaxScaler`, and `RobustScaler`. Report the transformed statistics and explain which scaler you would choose and why.

## Example Input
```python
import numpy as np
x = np.array([10, 12, 11, 13, 12, 11, 500]).reshape(-1, 1)  # 500 is an outlier
```

## Example Output
```
StandardScaler -> outlier pulls mean up; most points squashed near a small negative z.
MinMaxScaler   -> outlier becomes 1.0, all normal points crammed into ~[0, 0.006].
RobustScaler   -> normal points spread around 0; outlier large but the rest stay usable.
```

## Approach
Fit each scaler, transform, and compare how the "normal" cluster is represented. The takeaway: mean/min/max are outlier-sensitive; median/IQR are not.

## Python Implementation
```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

x = np.array([10, 12, 11, 13, 12, 11, 500], dtype=float).reshape(-1, 1)

for name, scaler in [
    ("Standard", StandardScaler()),
    ("MinMax", MinMaxScaler()),
    ("Robust", RobustScaler()),
]:
    z = scaler.fit_transform(x).ravel()
    normal = z[:-1]                     # exclude the outlier
    print(f"{name:8s}: normal range [{normal.min():+.3f}, {normal.max():+.3f}]"
          f"  outlier={z[-1]:+.2f}")
```

## Alternative Solution
For features that must land in a bounded, distribution-agnostic space, `QuantileTransformer(output_distribution="normal")` or `PowerTransformer` (Yeo-Johnson) can be stronger than any linear scaler — they reshape the distribution, not just its location/scale.

## Interview Variations
- "Your model is KNN and one feature has outliers — which scaler?" → `RobustScaler`, because KNN distances are dominated by unscaled/outlier-heavy features.
- "You need features strictly in [0, 1] for a neural net image pipeline." → `MinMaxScaler` (or divide by 255), assuming outliers are already controlled.

## Common Follow-up Questions
- **"MinMax vs Standard for gradient descent?"** Both help convergence. Standard is usually preferred when features are roughly Gaussian; MinMax when you need a bounded range or the data isn't Gaussian.
- **"Does RobustScaler bound the output?"** No — outliers can still map to large values; it just prevents them from compressing the rest.

---

# Practical Question 3: Build a Leakage-Free Pipeline (Scaling + Model + CV)

**Difficulty:** Medium
**Estimated Time:** 20 min
**Concepts Tested:** `Pipeline`, cross-validation, preventing preprocessing leakage, StratifiedKFold

## Problem Statement
A junior teammate scaled the entire dataset once, then ran `cross_val_score`. Explain why this leaks, then write the **correct** version where scaling is fit *inside* each CV fold using a scikit-learn `Pipeline`. Quantify the difference in reported scores if you can.

## Example Input
```python
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
```

## Example Output
```
Leaky CV accuracy (scale-then-split):    ~0.976
Correct CV accuracy (pipeline per fold): ~0.974   # honest estimate
```
(The leaky number is usually slightly inflated; on strongly-scaled or small datasets the gap grows.)

## Approach
1. **The trap:** `StandardScaler().fit_transform(X)` computes the mean/std over *all* rows, including the rows that will land in the validation fold. Every fold's validation data has already influenced the scaling — leakage.
2. **The fix:** wrap the scaler and the estimator in a `Pipeline`. When `cross_val_score` splits the data, it calls `pipeline.fit` on the *train* portion only, so the scaler learns statistics from that fold's training rows and merely `transform`s the validation rows.
3. Use `StratifiedKFold` for classification so each fold preserves the class ratio.

## Python Implementation
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- WRONG: fit the scaler on ALL data, then cross-validate --------------
X_leaky = StandardScaler().fit_transform(X)          # sees the whole dataset
leaky = cross_val_score(
    LogisticRegression(max_iter=5000), X_leaky, y, cv=cv, scoring="accuracy"
)

# ---- RIGHT: scaler is refit inside every fold via a Pipeline -------------
pipe = Pipeline([
    ("scaler", StandardScaler()),                    # fit on fold-train only
    ("clf", LogisticRegression(max_iter=5000)),
])
correct = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

print(f"Leaky   CV accuracy: {leaky.mean():.4f} +/- {leaky.std():.4f}")
print(f"Correct CV accuracy: {correct.mean():.4f} +/- {correct.std():.4f}")
```

**Why the pipeline is correct:** `cross_val_score` clones the pipeline for each fold and calls `.fit(X_train_fold, y_train_fold)`. Inside `.fit`, the scaler step runs `fit_transform` on the *training* rows and the classifier trains on those. On the validation rows it runs only `.transform` (via the pipeline's `.predict`). The validation fold never touches the scaler's `fit`.

## Alternative Solution
`make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))` is a terser constructor that auto-names steps. For heterogeneous columns (numeric + categorical), use `ColumnTransformer` inside the pipeline so each column group gets the right preprocessing — still leakage-free because the whole `ColumnTransformer` is fit per fold.

## Interview Variations
- Add a `SelectKBest` feature-selection step — stress that feature selection must *also* be inside the pipeline, or you leak by selecting features using the validation labels.
- Replace `cross_val_score` with `cross_validate` to return train scores, fit times, and multiple metrics at once.

## Common Follow-up Questions
- **"Name three steps that must go inside the CV loop."** Scaling/normalization, imputation, feature selection, resampling (SMOTE), and target encoding — anything that *learns from data*.
- **"Is a single train/test split immune?"** No. You must `fit` preprocessing on train and only `transform` test — same rule, applied once instead of per fold.
- **"Where is it acceptable to preprocess the whole dataset?"** Only deterministic, row-independent operations that don't learn parameters — e.g., `log1p` on a column, or dropping a column by name. Even then, keeping it in the pipeline is safer.

---

# Practical Question 4: Apply SMOTE Correctly Inside Cross-Validation

**Difficulty:** Hard
**Estimated Time:** 25 min
**Concepts Tested:** Imbalanced data, SMOTE, imblearn Pipeline, resampling leakage, correct metrics

## Problem Statement
On an imbalanced dataset, a candidate ran `SMOTE().fit_resample(X, y)` and *then* did cross-validation. Explain why this is a leakage bug, then implement the correct approach using `imblearn.pipeline.Pipeline` so SMOTE is applied to the **training portion of each fold only**. Evaluate with metrics that survive imbalance (precision, recall, F1, ROC-AUC / PR-AUC), not accuracy.

## Example Input
```python
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=3000, n_features=20, weights=[0.95, 0.05],
                           n_informative=6, random_state=42)   # 5% positives
```

## Example Output
```
Leaky   (resample-then-CV) F1:  ~0.71   # optimistic — synthetic points leak
Correct (SMOTE-in-pipeline) F1: ~0.63   # honest
```

## Approach
1. **The leak:** SMOTE synthesizes new minority points by interpolating between *neighbors*. If you resample before splitting, a synthetic point can be built from a real point that later lands in the validation fold — the validation fold's information is baked into the training data. Worse, duplicated/near-duplicated points can appear on both sides of the split.
2. **The fix:** use `imblearn`'s `Pipeline`, which is SMOTE-aware. Resampling steps run **only during `fit`** (on training data) and are automatically skipped during `predict`/`transform` on validation data — exactly what you want.
3. **Order matters:** scale (or not) → SMOTE → classifier. SMOTE uses distances, so if you scale, scale *before* SMOTE, all inside the pipeline.
4. Score with F1 / recall / PR-AUC; accuracy is meaningless when 95% of rows are one class.

## Python Implementation
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # NOTE: imblearn's Pipeline

X, y = make_classification(n_samples=3000, n_features=20, weights=[0.95, 0.05],
                           n_informative=6, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- WRONG: SMOTE the entire dataset, then cross-validate ---------------
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)   # leaks across folds
leaky = cross_val_score(
    RandomForestClassifier(random_state=42),
    X_res, y_res, cv=cv, scoring="f1",
)

# ---- RIGHT: SMOTE lives inside the pipeline, refit per fold --------------
pipe = ImbPipeline([
    ("scaler", StandardScaler()),          # scale before distance-based SMOTE
    ("smote", SMOTE(random_state=42)),     # runs on fit() only, per fold
    ("clf", RandomForestClassifier(random_state=42)),
])
correct = cross_val_score(pipe, X, y, cv=cv, scoring="f1")

print(f"Leaky   (resample-then-CV) F1: {leaky.mean():.3f}")
print(f"Correct (SMOTE-in-pipeline) F1: {correct.mean():.3f}")
```

**Key API detail:** you MUST import `Pipeline` from `imblearn.pipeline`, not `sklearn.pipeline`. The sklearn one has no notion of a `fit_resample` step and will error or behave wrongly. The imblearn pipeline calls `fit_resample` on samplers during `fit` and bypasses them at predict time.

## Alternative Solution
Often you don't need SMOTE at all. Many estimators accept `class_weight="balanced"`, which reweights the loss so minority errors count more — no synthetic data, no resampling leakage risk, and it's cheaper:

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(class_weight="balanced", random_state=42)
```
For gradient boosting (XGBoost/LightGBM) use `scale_pos_weight = n_negative / n_positive`. Undersampling (`RandomUnderSampler`) or hybrids (`SMOTETomek`, `SMOTEENN`) are other options; benchmark them — no method universally wins.

## Interview Variations
- "SMOTE for categorical features?" → use `SMOTENC` (handles nominal columns) or `SMOTEN` (all-categorical); plain SMOTE interpolates and would create invalid category values.
- "Combine SMOTE with `RandomizedSearchCV`?" → put the imblearn pipeline as the estimator; tune `smote__k_neighbors` and the classifier params together. Still leakage-free because search uses CV internally.

## Common Follow-up Questions
- **"Should you SMOTE the test set?"** Never. The test set must reflect the real, imbalanced distribution — otherwise your metrics don't represent production.
- **"Why is accuracy a trap here?"** A model predicting the majority class for everything scores 95% accuracy while catching zero positives. Use recall/precision/F1/PR-AUC.
- **"PR-AUC vs ROC-AUC under heavy imbalance?"** PR-AUC (average precision) is more informative because ROC-AUC can look deceptively high when true negatives dominate.

---

# Practical Question 5: GridSearchCV vs RandomizedSearchCV (with Timing)

**Difficulty:** Medium
**Estimated Time:** 20 min
**Concepts Tested:** Hyperparameter tuning, search-space design, compute budget trade-offs, `best_params_`

## Problem Statement
Tune a `RandomForestClassifier` on the same data with both `GridSearchCV` (exhaustive) and `RandomizedSearchCV` (sampled). Time both, compare best scores and the number of fits, and explain when you'd pick each.

## Example Input
```python
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
```

## Example Output
```
GridSearchCV     : 108 fits, best F1 ~0.965, time ~9.4s
RandomizedSearchCV: 20 fits,  best F1 ~0.963, time ~1.9s   # ~5x faster, ~equal score
```

## Approach
1. Define a discrete grid for grid search. `n_fits = (#combinations) × n_splits`.
2. For randomized search, use distributions (`scipy.stats`) and cap the budget with `n_iter`.
3. Time each with `time.perf_counter()`. Compare `best_score_`, `best_params_`, and fit count.
4. Message: grid search cost grows *combinatorially* with the grid; randomized search lets you fix a budget and often finds a near-optimal point far cheaper, especially when only a few hyperparameters truly matter.

## Python Implementation
```python
import time
import numpy as np
from scipy.stats import randint
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold)

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
rf = RandomForestClassifier(random_state=42)

# ---- Exhaustive grid search ---------------------------------------------
grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"],
}  # 3*3*2*2 = 36 combos * 3 folds = 108 fits
gs = GridSearchCV(rf, grid, cv=cv, scoring="f1", n_jobs=-1)
t0 = time.perf_counter()
gs.fit(X, y)
gs_time = time.perf_counter() - t0

# ---- Randomized search over distributions -------------------------------
dist = {
    "n_estimators": randint(100, 600),
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": randint(2, 11),
    "max_features": ["sqrt", "log2"],
}
rs = RandomizedSearchCV(rf, dist, n_iter=20, cv=cv, scoring="f1",
                        random_state=42, n_jobs=-1)  # 20 combos * 3 = 60 fits
t0 = time.perf_counter()
rs.fit(X, y)
rs_time = time.perf_counter() - t0

print(f"Grid   : best F1={gs.best_score_:.4f}, time={gs_time:.2f}s, "
      f"params={gs.best_params_}")
print(f"Random : best F1={rs.best_score_:.4f}, time={rs_time:.2f}s, "
      f"params={rs.best_params_}")
```

**Complexity intuition:** grid search is `O(∏ |param_i| × folds)` — adding one hyperparameter with `k` values multiplies the cost by `k`. Randomized search is `O(n_iter × folds)`, fixed regardless of dimensionality.

## Alternative Solution
For expensive models, use `HalvingGridSearchCV` / `HalvingRandomSearchCV` (successive halving — cheap configs on little data, promote survivors) or Bayesian optimization (`optuna`, `skopt`) which model the score surface and sample smarter than random.

## Interview Variations
- "Tune a full pipeline." → use double-underscore keys: `{"clf__n_estimators": ..., "scaler__with_mean": ...}`.
- "Avoid overfitting the validation set during tuning." → use nested CV: an inner CV for tuning, an outer CV for the honest performance estimate.

## Common Follow-up Questions
- **"When does grid search win?"** Small, low-dimensional grids where you want a guaranteed sweep of every combination.
- **"Why does random search work so well?"** Bergstra & Bengio (2012): usually only a few hyperparameters matter; random sampling explores those important axes with more distinct values than a coarse grid does for the same budget.
- **"Is `best_score_` your final performance?"** No — it's the CV score used to *pick* params and is mildly optimistic. Report performance on a held-out test set or via nested CV.

---

# Practical Question 6: Permutation Importance vs Impurity Importance

**Difficulty:** Medium
**Estimated Time:** 18 min
**Concepts Tested:** Model interpretation, permutation importance, why built-in importances mislead, computing on the test set

## Problem Statement
Train a random forest and compare its built-in (`feature_importances_`, mean impurity decrease) importances with `permutation_importance`. Explain why permutation importance is often more trustworthy and on which dataset split it should be computed.

## Example Input
```python
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
```

## Example Output
```
Top features by impurity:     ['worst area', 'worst concave points', ...]
Top features by permutation:  ['worst concave points', 'worst area', ...]
# Ranks are similar here, but impurity inflates high-cardinality features.
```

## Approach
1. Train/test split, fit the model on train.
2. Read `model.feature_importances_` (mean decrease in impurity — MDI).
3. Compute `permutation_importance` on the **test set**: shuffle one feature's column, measure how much the score drops. Big drop = important feature.
4. Explain the bias: MDI is computed on training data and inflates features with many split points (high-cardinality/continuous). Permutation importance measures the effect on *held-out* predictive performance, so it reflects generalization.

## Python Implementation
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_tr, y_tr)

# Built-in impurity-based importance (computed on the training splits)
mdi = rf.feature_importances_

# Permutation importance on the TEST set: model-agnostic, generalization-aware
perm = permutation_importance(
    rf, X_te, y_te, n_repeats=20, random_state=42, scoring="accuracy", n_jobs=-1)

names = np.array(data.feature_names)
print("Top 5 by impurity (MDI):")
for i in mdi.argsort()[::-1][:5]:
    print(f"  {names[i]:24s} {mdi[i]:.4f}")

print("\nTop 5 by permutation importance (test set):")
for i in perm.importances_mean.argsort()[::-1][:5]:
    print(f"  {names[i]:24s} {perm.importances_mean[i]:.4f}"
          f" +/- {perm.importances_std[i]:.4f}")
```

**Complexity:** permutation importance costs roughly `O(n_features × n_repeats)` model evaluations — it re-scores the model once per shuffled feature per repeat. It's model-agnostic: works on any fitted estimator with a `.predict`.

## Alternative Solution
For correlated features, permutation importance can *understate* importance (the model recovers the signal from a correlated twin). Mitigate by clustering correlated features (hierarchical clustering on Spearman correlations) and permuting whole clusters, or use `drop-column importance` (retrain without each feature — more accurate, far more expensive).

## Interview Variations
- "Why not just trust `feature_importances_`?" → MDI is biased toward high-cardinality features and is computed on train data, so it can reward overfitting.
- "Compute permutation importance for a pipeline." → pass the fitted pipeline as the estimator; it permutes raw input columns and re-runs the whole pipeline.

## Common Follow-up Questions
- **"Train or test set for permutation importance?"** Test (or validation). On train it measures how much the model *relied* on a feature to fit; on test it measures generalizable importance.
- **"A feature has near-zero permutation importance — drop it?"** Maybe — but check for correlation with another feature first; the pair might share the signal.

---

# Practical Question 7: Explain a Prediction with SHAP

**Difficulty:** Hard
**Estimated Time:** 20 min
**Concepts Tested:** Local vs global interpretation, SHAP values, additive explanations, TreeExplainer

## Problem Statement
Train a tree model and use SHAP to (a) explain a single prediction locally and (b) produce a global feature-importance ranking. Explain what a SHAP value means and how it differs from permutation importance.

## Example Input
```python
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
```

## Example Output
```
Base value (mean model output): ~0.63
Sample #0 prediction explained: base + sum(shap_values) = model output
Global |SHAP| ranking:  worst concave points > worst area > mean concave points > ...
```

## Approach
1. Train a tree model (SHAP's `TreeExplainer` is exact and fast for trees).
2. Compute SHAP values on a sample of the data.
3. **Local:** for one row, SHAP decomposes the prediction as `base_value + Σ shap_i = f(x)`. Each `shap_i` is that feature's signed contribution (in log-odds or probability space).
4. **Global:** average `|shap_i|` across rows to rank features — a theoretically grounded importance.
5. Contrast with permutation importance: SHAP attributes *per prediction* (local + additive, grounded in Shapley values from game theory); permutation importance is a single global score per feature.

## Python Implementation
```python
# pip install shap
import numpy as np
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_tr, y_tr)

# TreeExplainer is exact for tree ensembles and much faster than KernelExplainer
explainer = shap.TreeExplainer(model)
sv = explainer(X_te)                    # SHAP Explanation object

# For binary classifiers shap returns values per class; take the positive class
# (newer shap: sv[..., 1]; the values are additive around the base value)
shap_pos = sv[..., 1] if sv.values.ndim == 3 else sv

# --- Local explanation for one test row ----------------------------------
row = 0
contribs = shap_pos.values[row]
base = shap_pos.base_values[row]
print(f"Base value: {base:.4f}")
print(f"Sum of SHAP contributions: {contribs.sum():+.4f}")
print(f"Reconstructed output: {base + contribs.sum():.4f}")
order = np.argsort(np.abs(contribs))[::-1][:5]
for i in order:
    print(f"  {X_te.columns[i]:24s} contributes {contribs[i]:+.4f}")

# --- Global importance: mean absolute SHAP across all test rows ----------
global_imp = np.abs(shap_pos.values).mean(axis=0)
print("\nTop 5 global features by mean |SHAP|:")
for i in np.argsort(global_imp)[::-1][:5]:
    print(f"  {X_te.columns[i]:24s} {global_imp[i]:.4f}")

# Visualizations (in a notebook):
#   shap.plots.waterfall(shap_pos[row])     # local, one prediction
#   shap.plots.beeswarm(shap_pos)           # global, distribution of effects
#   shap.plots.bar(shap_pos)                # global, mean |SHAP| bar chart
```

**Note on cost:** `TreeExplainer` is polynomial and exact for trees. For arbitrary models use `shap.KernelExplainer` (model-agnostic but slow — sample your background/data) or `shap.PermutationExplainer`.

## Alternative Solution
`LIME` fits a local linear surrogate around one prediction — cheaper conceptually and model-agnostic, but its explanations are less stable and not additive/consistent the way SHAP's are:

```python
# pip install lime
from lime.lime_tabular import LimeTabularExplainer
lime_exp = LimeTabularExplainer(
    X_tr.values, feature_names=list(X_tr.columns),
    class_names=["malignant", "benign"], mode="classification")
exp = lime_exp.explain_instance(X_te.values[0], model.predict_proba, num_features=5)
print(exp.as_list())
```

## Interview Variations
- "Explain a deep learning model." → `shap.DeepExplainer` / `GradientExplainer`, or KernelSHAP with a background sample.
- "Regulator asks why one customer was denied." → local SHAP waterfall for that customer's row is the standard answer.

## Common Follow-up Questions
- **"What exactly is a SHAP value?"** The Shapley value from cooperative game theory: a feature's average marginal contribution to the prediction over all possible feature orderings. It's the unique attribution satisfying local accuracy (additivity), missingness, and consistency.
- **"SHAP vs permutation importance?"** SHAP is local + additive (per-prediction, then aggregate); permutation importance is a single global number per feature. SHAP shows *direction* (pushes prediction up/down); permutation only shows *magnitude* of impact on a metric.
- **"SHAP and correlated features?"** SHAP splits credit among correlated features, which can dilute apparent importance; interpret with the correlation structure in mind.

---

## Part B — Notebook Workflows

> These are meant to be pasted into Jupyter cell-by-cell. Run them top to bottom. I have kept each workflow self-contained (its own imports) so you can drop either one into a fresh notebook.

---

### Workflow 1 — Correct vs Leaky Preprocessing

**Goal:** *feel* data leakage numerically, then eliminate it with a `Pipeline` + `StratifiedKFold`. This is the demonstration that separates candidates who *say* "don't leak" from those who can *show* it.

---

### Cell 1 — Imports and setup
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```
We import both a scaler *and* a feature selector on purpose — leakage most often sneaks in through feature selection, not just scaling. Setting a global seed keeps the demo reproducible.

---

### Cell 2 — Load data and inspect the class balance
```python
data = load_breast_cancer()
X, y = data.data, data.target
print("Shape:", X.shape)
print("Class balance:", np.bincount(y), "(0=malignant, 1=benign)")
print("Feature scales differ wildly:")
print(pd.DataFrame(X, columns=data.feature_names).iloc[:, :3].describe().loc[["mean", "std"]])
```
The three features shown already have means/stds spanning orders of magnitude — exactly the situation where scaling matters and where a leaky scaler can quietly distort results.

---

### Cell 3 — The LEAKY way: scale and select on the full dataset first
```python
# ANTI-PATTERN. Do not do this in real work.
X_scaled_all = StandardScaler().fit_transform(X)                 # sees every row
X_selected_all = SelectKBest(f_classif, k=10).fit_transform(X_scaled_all, y)  # sees every label

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
leaky_scores = cross_val_score(
    LogisticRegression(max_iter=5000), X_selected_all, y, cv=cv, scoring="accuracy")
print(f"LEAKY CV accuracy: {leaky_scores.mean():.4f} +/- {leaky_scores.std():.4f}")
```
Both the scaler and `SelectKBest` were fit on the **entire** dataset — including every row and label that will later serve as validation data. `SelectKBest` even peeks at `y` for all rows. The CV score below is therefore contaminated.

---

### Cell 4 — The CORRECT way: wrap every learned step in a Pipeline
```python
pipe = Pipeline([
    ("scaler", StandardScaler()),                 # refit each fold, train rows only
    ("select", SelectKBest(f_classif, k=10)),     # feature selection also per fold
    ("clf", LogisticRegression(max_iter=5000)),
])
correct_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print(f"CORRECT CV accuracy: {correct_scores.mean():.4f} +/- {correct_scores.std():.4f}")
```
Now `cross_val_score` clones the pipeline per fold and calls `.fit` on the training rows only. The scaler learns its mean/std and `SelectKBest` chooses its top-10 features using **fold-train data alone**; the validation rows are merely transformed. This is a leakage-free, honest estimate.

---

### Cell 5 — Put the two estimates side by side
```python
print(f"Leaky   : {leaky_scores.mean():.4f}")
print(f"Correct : {correct_scores.mean():.4f}")
print(f"Difference (leaky - correct): {leaky_scores.mean() - correct_scores.mean():+.4f}")
```
On this clean, easy dataset the gap is small — but it is almost always in the *optimistic* direction, and it widens on smaller datasets, higher-dimensional feature selection, or target-based encoders. In a real project a small inflated CV number can mean a model that underperforms in production.

---

### Cell 6 — Amplify the leak so it is unmistakable
```python
# Add 5000 pure-noise features. Feature selection on the full data will "find"
# noise columns that happen to correlate with y across ALL rows -> big leak.
rng = np.random.RandomState(RANDOM_STATE)
X_noise = np.hstack([X, rng.randn(X.shape[0], 5000)])

# Leaky: select k best using every label
X_leak = SelectKBest(f_classif, k=20).fit_transform(StandardScaler().fit_transform(X_noise), y)
leaky_noise = cross_val_score(LogisticRegression(max_iter=5000), X_leak, y, cv=cv).mean()

# Correct: selection inside the pipeline, per fold
pipe_noise = Pipeline([("scaler", StandardScaler()),
                       ("select", SelectKBest(f_classif, k=20)),
                       ("clf", LogisticRegression(max_iter=5000))])
correct_noise = cross_val_score(pipe_noise, X_noise, y, cv=cv).mean()

print(f"With 5000 noise features -> LEAKY: {leaky_noise:.4f}  CORRECT: {correct_noise:.4f}")
```
This is the classic demonstration. Selecting features using all labels lets noise columns that spuriously correlate with `y` sneak in, inflating the leaky score well above the honest one. Same code structure, dramatically different (and dishonest) result.

---

### Cell 7 — The same rule for a single train/test split
```python
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

scaler = StandardScaler().fit(X_tr)     # FIT ON TRAIN ONLY
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)  # transform both

clf = LogisticRegression(max_iter=5000).fit(X_tr_s, y_tr)
print(f"Held-out test accuracy: {accuracy_score(y_te, clf.predict(X_te_s)):.4f}")
```
Cross-validation is not special: the same discipline applies to a plain split. Fit preprocessing on train, transform test. If you had written `StandardScaler().fit_transform(X)` before splitting, you would have leaked test statistics into training here too.

---

### Cell 8 — Takeaways
```python
print("""
LEAKAGE CHECKLIST
1. Anything that LEARNS from data (scaler, imputer, feature selector,
   target encoder, resampler, PCA) must be fit INSIDE the CV loop.
2. Use sklearn Pipeline so cross_val_score/GridSearchCV refit per fold.
3. Fit on TRAIN, transform TEST -- never fit_transform the whole dataset.
4. Use StratifiedKFold for classification to preserve class ratios.
5. If your CV score looks 'too good', suspect leakage first.
""")
```
Memorize this checklist. In interviews, spotting the leak in someone else's snippet — and naming the fix (`Pipeline` + fit-on-train) — is a high-value signal.

---

### Workflow 2 — Handling an Imbalanced Dataset End-to-End

**Goal:** take a heavily imbalanced dataset from the "accuracy trap" all the way to a tuned, interpretable model, using the right metrics at every step. This is a complete mini-project you can narrate in an interview.

---

### Cell 1 — Imports
```python
import numpy as np
from scipy.stats import randint
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support,
                             average_precision_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
```
Note the two pipeline worlds: we use `imblearn`'s `Pipeline` (aliased `ImbPipeline`) because only it knows how to run SMOTE on fit-time training data and skip it at predict time.

---

### Cell 2 — Create an imbalanced dataset and split it
```python
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=6, n_redundant=4,
    weights=[0.94, 0.06], flip_y=0.01, class_sep=0.9, random_state=RANDOM_STATE)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

print("Train class counts:", np.bincount(y_tr))
print("Test  class counts:", np.bincount(y_te))
print(f"Positive rate: {y.mean():.1%}")
```
About 6% positives. We `stratify` on `y` so both splits keep that ratio — never let a random split hand you a test set with a different positive rate than production.

---

### Cell 3 — The accuracy trap: a baseline that predicts nothing useful
```python
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
acc = dummy.score(X_te, y_te)
pred = dummy.predict(X_te)
prec, rec, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
print(f"'Always majority' accuracy: {acc:.3f}")
print(f"...but precision={prec:.3f}, recall={rec:.3f}, F1={f1:.3f}")
```
A model that always predicts the majority class scores ~94% accuracy while catching **zero** positives (recall = 0). This is the accuracy trap: on imbalanced data, high accuracy can mean a useless model. From here on we track precision/recall/F1/PR-AUC instead.

---

### Cell 4 — Honest baseline: plain logistic regression, proper metrics
```python
base = ImbPipeline([("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=5000))]).fit(X_tr, y_tr)
proba = base.predict_proba(X_te)[:, 1]
pred = base.predict(X_te)
print(classification_report(y_te, pred, digits=3))
print(f"ROC-AUC: {roc_auc_score(y_te, proba):.3f}   PR-AUC: {average_precision_score(y_te, proba):.3f}")
```
This is our real baseline. Look at the minority-class (label 1) recall — the fraction of true positives we catch. That recall, plus PR-AUC (average precision), is what the next techniques must improve.

---

### Cell 5 — Technique 1: class_weight="balanced" (no resampling)
```python
cw = ImbPipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]
                 ).fit(X_tr, y_tr)
proba_cw = cw.predict_proba(X_te)[:, 1]
pred_cw = cw.predict(X_te)
p, r, f, _ = precision_recall_fscore_support(y_te, pred_cw, average="binary")
print(f"class_weight  -> precision={p:.3f} recall={r:.3f} F1={f:.3f} "
      f"PR-AUC={average_precision_score(y_te, proba_cw):.3f}")
```
`class_weight="balanced"` reweights the loss so minority mistakes cost more (inversely proportional to class frequency). Typically recall jumps and precision drops — the model now *wants* to catch positives. No synthetic data, no leakage risk, essentially free.

---

### Cell 6 — Technique 2: SMOTE, correctly, inside a pipeline
```python
smote_pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=RANDOM_STATE)),   # fit-time only, train rows only
    ("clf", LogisticRegression(max_iter=5000)),
]).fit(X_tr, y_tr)

proba_sm = smote_pipe.predict_proba(X_te)[:, 1]
pred_sm = smote_pipe.predict(X_te)
p, r, f, _ = precision_recall_fscore_support(y_te, pred_sm, average="binary")
print(f"SMOTE         -> precision={p:.3f} recall={r:.3f} F1={f:.3f} "
      f"PR-AUC={average_precision_score(y_te, proba_sm):.3f}")
print("Test set positive count (unchanged, as it must be):", int(y_te.sum()))
```
SMOTE synthesizes minority examples by interpolating between neighbors — but only on the **training** portion, because it lives inside the imblearn pipeline and runs during `fit` only. The **test set stays imbalanced** (we never resample test), so these metrics reflect reality.

---

### Cell 7 — Cross-validate the SMOTE pipeline (prove no fold leakage)
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
from sklearn.model_selection import cross_val_score
cv_f1 = cross_val_score(smote_pipe, X_tr, y_tr, cv=cv, scoring="f1")
cv_ap = cross_val_score(smote_pipe, X_tr, y_tr, cv=cv, scoring="average_precision")
print(f"CV F1:      {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")
print(f"CV PR-AUC:  {cv_ap.mean():.3f} +/- {cv_ap.std():.3f}")
```
Because SMOTE is a pipeline step, each of the 5 folds resamples only its own training rows and scores on untouched validation rows. Had we called `SMOTE().fit_resample(X_tr, y_tr)` *before* cross-validating, synthetic points would leak across folds and inflate these numbers.

---

### Cell 8 — Tune the whole imbalanced pipeline with RandomizedSearchCV
```python
search_pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
])
param_dist = {
    "smote__k_neighbors": randint(3, 8),                 # tune the sampler too
    "clf__n_estimators": randint(200, 600),
    "clf__max_depth": [None, 6, 12, 20],
    "clf__min_samples_leaf": randint(1, 8),
    "clf__max_features": ["sqrt", "log2"],
}
search = RandomizedSearchCV(
    search_pipe, param_dist, n_iter=25, cv=cv,
    scoring="average_precision",       # optimize PR-AUC, the imbalance-aware metric
    random_state=RANDOM_STATE, n_jobs=-1, refit=True)
search.fit(X_tr, y_tr)
print("Best PR-AUC (CV):", round(search.best_score_, 3))
print("Best params:", search.best_params_)
```
We tune SMOTE and the classifier **jointly**, scoring on `average_precision` (PR-AUC) because that is the metric that matters under imbalance. `RandomizedSearchCV` samples 25 configs instead of exhaustively gridding — cheaper and, empirically, nearly as good. Every fold re-runs SMOTE internally, so tuning stays leakage-free.

---

### Cell 9 — Evaluate the tuned model on the held-out test set
```python
best = search.best_estimator_
proba_best = best.predict_proba(X_te)[:, 1]
pred_best = best.predict(X_te)
print(classification_report(y_te, pred_best, digits=3))
print("Confusion matrix [ [TN FP] [FN TP] ]:")
print(confusion_matrix(y_te, pred_best))
print(f"Test PR-AUC: {average_precision_score(y_te, proba_best):.3f}   "
      f"ROC-AUC: {roc_auc_score(y_te, proba_best):.3f}")
```
The confusion matrix tells the operational story: false negatives (missed positives, bottom-left) vs false positives (false alarms, top-right). Which is worse is a *business* decision — for fraud/disease you usually prioritize recall (minimize FN); adjust the decision threshold on the predicted probabilities accordingly rather than always using 0.5.

---

### Cell 10 — Interpret the tuned model with permutation importance
```python
# Permutation importance on the TEST set: which features drive PR-AUC?
perm = permutation_importance(
    best, X_te, y_te, n_repeats=20, random_state=RANDOM_STATE,
    scoring="average_precision", n_jobs=-1)

order = perm.importances_mean.argsort()[::-1][:8]
print("Top features by permutation importance (drop in PR-AUC when shuffled):")
for i in order:
    print(f"  feature_{i:<2d}  {perm.importances_mean[i]:+.4f} "
          f"+/- {perm.importances_std[i]:.4f}")
```
Permutation importance is model-agnostic and computed on the **test set**, so it reflects which features actually help the tuned pipeline generalize. We score on `average_precision` to stay consistent with how we tuned. Features whose shuffling barely changes PR-AUC are candidates for removal.

---

### Cell 11 — Summary of the end-to-end story
```python
print("""
IMBALANCED WORKFLOW RECAP
1. Never trust accuracy -- a majority-class predictor 'wins' it (recall=0).
2. Track precision / recall / F1 / PR-AUC; keep the TEST set imbalanced.
3. class_weight='balanced' is the cheap first move (no resampling leakage).
4. SMOTE must live INSIDE an imblearn Pipeline -> resample train folds only.
5. Tune sampler + model jointly with RandomizedSearchCV, scoring on PR-AUC.
6. Pick the decision threshold from the business cost of FN vs FP.
7. Interpret with permutation importance (or SHAP) on the test set.
""")
```
If you can walk an interviewer through these seven steps with the code above, you have demonstrated the full lifecycle: correct metrics, leakage-free resampling, principled tuning, and interpretation. That is exactly what a practical ML round is checking for.


---

# Part 4 — Ensemble Learning (Practical)

**Scope:** Bagging, Boosting, Stacking, Voting, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, AdaBoost.

**How to use this guide:** Part A drills the *isolated* skills an interviewer probes ("implement bagging from scratch", "wire up a stacking model"). Part B strings them into one realistic end-to-end notebook you can run top-to-bottom. Read the mentor notes between code blocks — in a lab exam the *why* is what earns marks when your first model doesn't converge.

**Environment assumed:**

```bash
# Core (almost always preinstalled in exam images)
pip install numpy pandas scikit-learn matplotlib seaborn
# Boosting libraries — may need install; code below degrades gracefully if missing
pip install xgboost lightgbm catboost imbalanced-learn
```

---

# PART A — CODING QUESTIONS

These are ordered roughly by difficulty. Each is a self-contained problem an interviewer can hand you on a shared screen. Type the imports every time in a real exam — a NameError on `np` costs you momentum.

---

# Practical Question 1: Implement a Bagging Ensemble from Scratch

**Difficulty:** Easy–Medium
**Estimated Time:** 20 min
**Concepts Tested:** Bootstrap sampling, variance reduction, majority voting, NumPy indexing, the *definition* of bagging (not just calling `BaggingClassifier`).

## Problem Statement
Without using `sklearn.ensemble.BaggingClassifier`, implement a bagging classifier. Given a base estimator, train `n_estimators` copies, each on a bootstrap sample (sample-with-replacement of size `n`) of the training data. Predict by majority vote across the ensemble. Use NumPy for the bootstrap indices and the vote aggregation.

## Example Input
```python
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, n_informative=8,
                           n_classes=2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
```

## Example Output
```
Single deep tree accuracy : 0.876
Bagging (50 trees)        : 0.916
```
(Exact numbers vary; the point is bagging > single high-variance tree.)

## Approach
1. Bagging reduces *variance* by averaging many high-variance, low-bias learners trained on perturbed data.
2. The perturbation is the **bootstrap sample**: draw `n` indices uniformly *with replacement* from `range(n)`. About 63.2% of unique rows appear in each sample; the ~36.8% left out are the *out-of-bag* (OOB) rows.
3. Clone the base estimator each round (never refit the same object — you'd overwrite it).
4. Aggregate: for classification take the mode (majority vote) per row; for regression you'd average.

## Python Implementation
```python
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class BaggingFromScratch:
    """Minimal bagging classifier: bootstrap + majority vote.

    Parameters
    ----------
    base_estimator : a fitted-capable sklearn-style classifier (has fit/predict)
    n_estimators   : number of bootstrap models
    random_state   : reproducibility seed
    """

    def __init__(self, base_estimator=None, n_estimators=50, random_state=None):
        self.base_estimator = base_estimator or DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []          # trained clones
        self.oob_indices_ = []         # rows NOT used by each estimator

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        self.estimators_, self.oob_indices_ = [], []

        for _ in range(self.n_estimators):
            # Bootstrap: n draws WITH replacement -> O(n) per tree
            idx = rng.integers(0, n, size=n)
            oob = np.setdiff1d(np.arange(n), idx, assume_unique=False)
            model = clone(self.base_estimator)     # fresh unfitted copy
            model.fit(X[idx], y[idx])
            self.estimators_.append(model)
            self.oob_indices_.append(oob)
        return self

    def predict(self, X):
        X = np.asarray(X)
        # Stack per-estimator predictions -> shape (n_estimators, n_samples)
        preds = np.stack([est.predict(X) for est in self.estimators_], axis=0)
        # Majority vote column-wise. bincount is fast for small integer classes.
        maj = np.apply_along_axis(
            lambda col: np.bincount(col, minlength=self.classes_.max() + 1).argmax(),
            axis=0, arr=preds)
        return maj

    def oob_score(self, X, y):
        """Estimate generalization using only rows each tree never saw."""
        X, y = np.asarray(X), np.asarray(y)
        n = X.shape[0]
        # votes[i, c] = number of trees (for which i was OOB) that predicted class c
        votes = np.zeros((n, len(self.classes_)), dtype=int)
        for est, oob in zip(self.estimators_, self.oob_indices_):
            if len(oob) == 0:
                continue
            p = est.predict(X[oob])
            for row, cls in zip(oob, p):
                votes[row, np.searchsorted(self.classes_, cls)] += 1
        covered = votes.sum(axis=1) > 0          # rows that were OOB at least once
        oob_pred = self.classes_[votes[covered].argmax(axis=1)]
        return accuracy_score(y[covered], oob_pred)


# --- Demo ---
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=8,
                               random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    single = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
    print(f"Single deep tree accuracy : {accuracy_score(y_te, single.predict(X_te)):.3f}")

    bag = BaggingFromScratch(DecisionTreeClassifier(random_state=42),
                             n_estimators=50, random_state=42).fit(X_tr, y_tr)
    print(f"Bagging (50 trees)        : {accuracy_score(y_te, bag.predict(X_te)):.3f}")
    print(f"OOB score estimate        : {bag.oob_score(X_tr, y_tr):.3f}")
```

**Complexity:** fitting is `O(n_estimators * cost_of_base_fit)`. For trees that's `O(T * n * m * log n)` with `m` features. Prediction is `O(T * n)`. Bootstrap sampling is `O(n)` per estimator.

## Alternative Solution
The idiomatic version is a one-liner with sklearn — worth knowing so you can contrast:
```python
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42),
                        n_estimators=50, oob_score=True, bootstrap=True,
                        n_jobs=-1, random_state=42).fit(X_tr, y_tr)
print(bag.oob_score_)   # built-in OOB estimate
```
Note: in sklearn ≥1.2 the parameter is `estimator=` (was `base_estimator=` before — a very common exam gotcha).

## Interview Variations
- "Make it a regressor" → replace majority vote with `np.mean` of predictions, drop `bincount`.
- "Add soft voting" → have base estimators expose `predict_proba`, average the probabilities, then `argmax`.
- "Parallelize the fit" → wrap the loop body in `joblib.Parallel`/`delayed`.
- "Sample features too, not just rows" → that turns bagging into a Random Forest-style random subspace method.

## Common Follow-up Questions
- *Why ~63.2%?* The probability a given row is *not* picked in one draw is `(1 - 1/n)`; over `n` draws that's `(1 - 1/n)^n → e^-1 ≈ 0.368`, so ~63.2% are in-bag.
- *Does bagging reduce bias?* No — mainly variance. Averaging unbiased-ish high-variance models keeps bias roughly constant while shrinking variance.
- *Why does bagging help trees so much?* Deep trees are low-bias/high-variance; they're the ideal candidate for variance reduction.
- *What's OOB good for?* A free validation estimate — no separate hold-out needed.

---

# Practical Question 2: Random Forest — Feature Importance & OOB Score

**Difficulty:** Easy–Medium
**Estimated Time:** 20 min
**Concepts Tested:** `RandomForestClassifier`, `oob_score_`, impurity-based vs permutation importance, plotting, the pitfall of biased importances.

## Problem Statement
Train a Random Forest on a classification dataset. (a) Report the OOB accuracy. (b) Extract the impurity-based feature importances and plot them sorted. (c) Also compute permutation importance and explain why the two can disagree.

## Example Input
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
```

## Example Output
```
OOB score: 0.958
Top feature (impurity): worst area
Top feature (permutation): worst concave points
```

## Approach
1. Set `oob_score=True` and `bootstrap=True` so the forest self-validates on out-of-bag rows.
2. `rf.feature_importances_` gives **mean decrease in impurity (MDI)** — fast but biased toward high-cardinality / continuous features.
3. `permutation_importance` shuffles one column at a time on held-out data and measures the score drop — slower but model-agnostic and less biased. Prefer it for reporting.
4. Plot both as horizontal bar charts, sorted descending.

## Python Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)

rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",   # decorrelates trees -> the "random" in random forest
    oob_score=True,        # free validation via out-of-bag rows
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
).fit(X_tr, y_tr)

print(f"OOB score : {rf.oob_score_:.3f}")
print(f"Test acc  : {rf.score(X_te, y_te):.3f}")

# (a) Impurity-based importance (MDI) with std across trees
mdi = rf.feature_importances_
mdi_std = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
order = np.argsort(mdi)[::-1]

# (b) Permutation importance on the TEST set (unbiased, model-agnostic)
perm = permutation_importance(rf, X_te, y_te, n_repeats=20,
                              random_state=42, n_jobs=-1)
perm_order = np.argsort(perm.importances_mean)[::-1]

# --- Plot side by side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
feat = np.array(X.columns)

axes[0].barh(feat[order][:10][::-1], mdi[order][:10][::-1],
             xerr=mdi_std[order][:10][::-1], color="#4C78A8")
axes[0].set_title("Impurity-based importance (MDI)")

axes[1].barh(feat[perm_order][:10][::-1], perm.importances_mean[perm_order][:10][::-1],
             xerr=perm.importances_std[perm_order][:10][::-1], color="#F58518")
axes[1].set_title("Permutation importance (test set)")

plt.tight_layout()
plt.savefig("rf_importances.png", dpi=120)  # or plt.show()
print("Top (MDI):", feat[order][0])
print("Top (perm):", feat[perm_order][0])
```

## Alternative Solution
For correlated features, MDI *and* single-feature permutation both mislead (importance gets split/shared). Use **hierarchical clustering on the correlation matrix**, keep one feature per cluster, then re-measure. Or use SHAP values (`shap.TreeExplainer(rf)`) for consistent, additive attributions.

## Interview Variations
- "Why is OOB close to CV score?" → OOB is essentially a leave-some-out estimate baked into training.
- "Tune the forest" → key knobs: `n_estimators` (more never hurts accuracy, only time), `max_depth`, `max_features`, `min_samples_leaf`.
- "Regression version" → `RandomForestRegressor`, importances work identically.

## Common Follow-up Questions
- *Why is MDI biased?* It favors features with many split points (continuous / high-cardinality categoricals), even random ones.
- *Does more trees overfit?* No — accuracy plateaus; only compute grows. Overfitting in RF comes from too-deep trees, not too many of them.
- *`max_features="sqrt"` — why?* Limiting candidate features per split decorrelates trees, which is what makes averaging effective.

---

# Practical Question 3: VotingClassifier — Hard vs Soft, and StackingClassifier

**Difficulty:** Medium
**Estimated Time:** 25 min
**Concepts Tested:** Heterogeneous ensembling, hard vs soft voting, weighted voting, stacking with a meta-learner, `passthrough`, avoiding meta-learner leakage.

## Problem Statement
Given three diverse base learners (logistic regression, SVM, random forest): (a) build a hard-voting and a soft-voting ensemble and compare; (b) build a stacking ensemble whose meta-learner combines the base predictions. Explain when soft beats hard and why stacking can beat voting.

## Example Input
```python
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=0)
```

## Example Output
```
LogReg     : 0.842
SVM        : 0.861
RandomForest: 0.888
Hard vote  : 0.879
Soft vote  : 0.892
Stacking   : 0.901
```

## Approach
1. **Hard voting** = majority of predicted *labels*. **Soft voting** = argmax of *averaged probabilities* — usually better when base models are well-calibrated, because it uses confidence, not just the winner.
2. Soft voting requires every estimator to support `predict_proba` (SVM needs `probability=True`, which is slower).
3. **Weighted voting**: pass `weights=[...]` to upweight stronger models.
4. **Stacking**: base models' out-of-fold predictions become features for a meta-learner (`final_estimator`). sklearn uses internal cross-validation (`cv=`) to generate those features *without leakage* — a base model never predicts on rows it trained on.
5. `passthrough=True` also feeds the original features to the meta-learner.

## Python Implementation
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=0)

# Base learners. Scale-sensitive models get their own scaler pipeline.
clf_lr  = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
clf_svm = make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))
clf_rf  = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)

estimators = [("lr", clf_lr), ("svm", clf_svm), ("rf", clf_rf)]

# (a) Voting
hard = VotingClassifier(estimators, voting="hard").fit(X_tr, y_tr)
soft = VotingClassifier(estimators, voting="soft",
                        weights=[1, 1, 2]).fit(X_tr, y_tr)  # trust RF more

# (b) Stacking — meta-learner sees base out-of-fold probabilities
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method="predict_proba",  # feed probabilities to the meta-learner
    cv=5,                          # internal CV prevents leakage
    passthrough=False,             # set True to also pass raw features
    n_jobs=-1,
).fit(X_tr, y_tr)

for name, model in [("LogReg", clf_lr), ("SVM", clf_svm), ("RandomForest", clf_rf),
                    ("Hard vote", hard), ("Soft vote", soft), ("Stacking", stack)]:
    # refit-safe scoring on the held-out test set
    m = model if hasattr(model, "classes_") else model.fit(X_tr, y_tr)
    print(f"{name:12s}: {m.score(X_te, y_te):.3f}")
```

## Alternative Solution
Roll your own stacking to show you understand it: use `cross_val_predict(base, X_tr, y_tr, method="predict_proba", cv=5)` for each base model, `hstack` those out-of-fold probabilities into a meta-feature matrix, fit a meta-model on it, and at inference time refit each base on all of `X_tr` and stack their probabilities. That is exactly what `StackingClassifier` automates.

## Interview Variations
- "Meta-learner choice?" → keep it simple (logistic regression) to avoid overfitting the base predictions; complex meta-learners often hurt.
- "Voting with no probabilities?" → you're forced into hard voting.
- "Multi-layer stacking" → stack of stacks; rarely worth it, diminishing returns and huge leakage risk.

## Common Follow-up Questions
- *When does soft lose to hard?* When base probabilities are badly calibrated (e.g., a raw SVM decision score), soft voting can be misled; calibrate first (`CalibratedClassifierCV`).
- *Why does stacking beat voting?* Voting uses fixed/manual weights; stacking *learns* how to combine models, including non-linear interactions.
- *Biggest stacking risk?* Leakage — using in-fold base predictions inflates the meta-features. Always generate them out-of-fold.

---

# Practical Question 4: AdaBoost & Gradient Boosting — the Boosting Mechanism

**Difficulty:** Medium
**Estimated Time:** 25 min
**Concepts Tested:** Sequential ensembling, reweighting (AdaBoost) vs residual-fitting (Gradient Boosting), weak learners, staged predictions, bias reduction.

## Problem Statement
(a) Train an `AdaBoostClassifier` and a `GradientBoostingClassifier` on the same data. (b) Plot test accuracy as a function of the number of boosting stages using `staged_predict`, to visualize how boosting reduces error over rounds and where it starts to overfit.

## Example Input
```python
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=3000, n_features=20, n_informative=6,
                           flip_y=0.1, random_state=1)
```

## Example Output
A curve where test accuracy climbs steeply for the first ~50 stages, plateaus, and (for GBM with high learning rate) may slowly *decline* — the signature of boosting overfitting.

## Approach
1. **Boosting is sequential**: each new weak learner focuses on the mistakes of the ensemble so far. This reduces **bias** (unlike bagging, which reduces variance).
2. **AdaBoost** reweights *samples* — misclassified rows get larger weights next round; final prediction is a weighted vote of stumps.
3. **Gradient Boosting** fits each new tree to the *negative gradient of the loss* (for squared error, that's the residual). It's gradient descent in function space.
4. `staged_predict` yields the ensemble prediction after each stage — perfect for a learning curve without retraining.

## Python Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=3000, n_features=20, n_informative=6,
                           flip_y=0.1, random_state=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=1)

# AdaBoost with decision stumps (the classic weak learner)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # stump
    n_estimators=300, learning_rate=0.5, random_state=1,
).fit(X_tr, y_tr)

# Gradient Boosting with shallow trees
gbm = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=3,
    subsample=0.8,          # stochastic GB -> a touch of variance reduction
    random_state=1,
).fit(X_tr, y_tr)

def staged_curve(model, X, y):
    """Accuracy after each boosting stage using staged_predict (no retrain)."""
    return [accuracy_score(y, pred) for pred in model.staged_predict(X)]

ada_curve = staged_curve(ada, X_te, y_te)
gbm_curve = staged_curve(gbm, X_te, y_te)

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(ada_curve) + 1), ada_curve, label="AdaBoost (stumps)")
plt.plot(range(1, len(gbm_curve) + 1), gbm_curve, label="Gradient Boosting (depth 3)")
plt.xlabel("Number of boosting stages")
plt.ylabel("Test accuracy")
plt.title("Boosting: error reduction & overfitting over stages")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("boosting_curve.png", dpi=120)

print(f"AdaBoost final : {ada_curve[-1]:.3f}")
print(f"GBM final      : {gbm_curve[-1]:.3f}")
print(f"GBM best stage : {np.argmax(gbm_curve)+1} (acc {max(gbm_curve):.3f})")
```

## Alternative Solution
`HistGradientBoostingClassifier` is sklearn's modern, histogram-based GBM — much faster on large data, supports early stopping natively (`early_stopping=True`) and handles NaNs. It's the sklearn analogue of LightGBM and usually the right default over the classic `GradientBoostingClassifier`.

## Interview Variations
- "Make AdaBoost overfit" → increase `learning_rate` and `n_estimators`, use deeper base trees.
- "Which is more noise-robust?" → Gradient Boosting; AdaBoost aggressively upweights outliers/mislabeled points and can chase noise.
- "Early stopping" → monitor a validation score and stop when it stops improving (`n_iter_no_change`, `validation_fraction`).

## Common Follow-up Questions
- *Bagging vs boosting in one line?* Bagging = parallel, independent, variance-reduction; Boosting = sequential, dependent, bias-reduction.
- *Why shallow trees in GBM?* Each tree is a weak learner correcting a small piece; deep trees overfit residuals fast.
- *What loss does GBM optimize for classification?* Log loss (deviance); it fits trees to the pseudo-residuals (negative gradients) of that loss.

---

# Practical Question 5: XGBoost vs LightGBM vs CatBoost with Early Stopping

**Difficulty:** Hard
**Estimated Time:** 30 min
**Concepts Tested:** The three production boosting libraries, early stopping with a validation set, categorical handling, API differences, fair benchmarking (accuracy + wall-clock).

## Problem Statement
Train XGBoost, LightGBM, and CatBoost on the same split with **early stopping** on a validation set. Report ROC-AUC and training time for each. Handle the case where a library isn't installed. Note the API differences (each library's early-stopping mechanism differs by version).

## Example Input
```python
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=20000, n_features=30, n_informative=12,
                           weights=[0.7, 0.3], random_state=7)
```

## Example Output
```
model      auc     fit_time(s)  best_iter
xgboost    0.951   1.8          142
lightgbm   0.953   0.7          210
catboost   0.952   2.4          380
```
(Illustrative — numbers depend on hardware.)

## Approach
1. Carve out a **validation set** distinct from test; early stopping watches it and halts when the metric stops improving for `early_stopping_rounds` iterations, keeping the best iteration.
2. Set a large `n_estimators` and let early stopping choose the real count — the standard idiom.
3. API differences to memorize:
   - **XGBoost** (≥1.6): pass `early_stopping_rounds` in the constructor (or a callback), then `fit(..., eval_set=[(X_val, y_val)])`.
   - **LightGBM** (≥4.0): early stopping via `callbacks=[lgb.early_stopping(...)]` in `fit`.
   - **CatBoost**: `early_stopping_rounds` in constructor, `eval_set=(X_val, y_val)` in `fit`; native categorical support via `cat_features`.
4. Time each fit with `time.perf_counter`. Wrap imports in try/except so a missing library skips rather than crashes.

## Python Implementation
```python
import time
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

X, y = make_classification(n_samples=20000, n_features=30, n_informative=12,
                           weights=[0.7, 0.3], random_state=7)
# Three-way split: train / validation (early stopping) / test (final report)
X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2,
                                            stratify=y, random_state=7)
X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2,
                                            stratify=y_tmp, random_state=7)

results = []

# ---------- XGBoost ----------
try:
    from xgboost import XGBClassifier
    t0 = time.perf_counter()
    xgb = XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", early_stopping_rounds=50,
        tree_method="hist", n_jobs=-1, random_state=7,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    dt = time.perf_counter() - t0
    auc = roc_auc_score(y_te, xgb.predict_proba(X_te)[:, 1])
    results.append(("xgboost", auc, dt, xgb.best_iteration))
except ImportError:
    print("xgboost not installed — skipping (pip install xgboost)")

# ---------- LightGBM ----------
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    t0 = time.perf_counter()
    lgbm = LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=7,
    )
    lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc",
             callbacks=[lgb.early_stopping(50, verbose=False)])
    dt = time.perf_counter() - t0
    auc = roc_auc_score(y_te, lgbm.predict_proba(X_te)[:, 1])
    results.append(("lightgbm", auc, dt, lgbm.best_iteration_))
except ImportError:
    print("lightgbm not installed — skipping (pip install lightgbm)")

# ---------- CatBoost ----------
try:
    from catboost import CatBoostClassifier
    t0 = time.perf_counter()
    cat = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6,
        eval_metric="AUC", early_stopping_rounds=50,
        random_seed=7, verbose=False,
    )
    cat.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    dt = time.perf_counter() - t0
    auc = roc_auc_score(y_te, cat.predict_proba(X_te)[:, 1])
    results.append(("catboost", auc, dt, cat.get_best_iteration()))
except ImportError:
    print("catboost not installed — skipping (pip install catboost)")

print(f"\n{'model':10s} {'auc':>6s} {'fit(s)':>8s} {'best_iter':>10s}")
for name, auc, dt, it in results:
    print(f"{name:10s} {auc:6.3f} {dt:8.2f} {str(it):>10s}")
```

## Alternative Solution
For a fully controlled benchmark, use each library's **native training API** (`xgb.train` on `DMatrix`, `lgb.train` on `Dataset`, `catboost.train` on `Pool`) rather than the sklearn wrappers. The native APIs expose more knobs (custom objectives, richer callbacks) and are what you'll see in Kaggle-grade code. Also add `StratifiedKFold` CV around each to get variance bars, not a single-split point estimate.

## Interview Variations
- "Categorical features?" → CatBoost: pass `cat_features=[...]`, no manual encoding. LightGBM: `categorical_feature=[...]`. XGBoost: needs `enable_categorical=True` with pandas `category` dtype, or one-hot/target encode.
- "GPU?" → `tree_method="gpu_hist"` (XGB), `device="gpu"` (LGBM), `task_type="GPU"` (CatBoost).
- "Class imbalance?" → `scale_pos_weight` (XGB/LGBM) or `class_weights`/`auto_class_weights` (CatBoost).

## Common Follow-up Questions
- *XGBoost vs LightGBM growth?* XGBoost grows level-wise (depth-balanced); LightGBM grows **leaf-wise** (splits the highest-loss leaf) — faster and often more accurate, but more prone to overfitting on small data (control with `num_leaves`, `min_child_samples`).
- *Why is CatBoost good on categoricals?* Ordered target statistics + ordered boosting reduce target leakage; strong defaults.
- *What exactly does early stopping return?* The model at `best_iteration` (best validation metric), not the last iteration — prevents overfitting and saves compute.

---

# Practical Question 6: learning_rate vs n_estimators Tradeoff

**Difficulty:** Medium–Hard
**Estimated Time:** 25 min
**Concepts Tested:** Shrinkage, the inverse relationship between learning rate and number of trees, validation-curve reasoning, regularization intuition.

## Problem Statement
Empirically demonstrate the learning-rate / n_estimators tradeoff in gradient boosting: a smaller learning rate needs more estimators to reach the same fit but generalizes better (to a point). Sweep a grid of learning rates and plot validation AUC vs number of trees for each.

## Example Input
```python
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=8000, n_features=25, n_informative=10,
                           random_state=3)
```

## Example Output
Curves where `lr=0.3` peaks early then overfits, `lr=0.1` peaks later at a higher/steadier plateau, and `lr=0.01` climbs slowly and needs many more trees to catch up.

## Approach
1. **Shrinkage**: each tree's contribution is scaled by `learning_rate`. Small `lr` = each tree matters less = you need more trees, but the ensemble is smoother and generalizes better.
2. Rule of thumb: `learning_rate * n_estimators ≈ constant` for a comparable fit. Halve the LR → roughly double the trees.
3. Use `staged_predict` to trace validation AUC across all stages for each learning rate in one fit — cheap and informative.
4. The practical recipe: pick a *small* LR (0.01–0.1), set a large `n_estimators`, and let **early stopping** choose the count.

## Python Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=8000, n_features=25, n_informative=10,
                           random_state=3)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3,
                                            stratify=y, random_state=3)

learning_rates = [0.3, 0.1, 0.03, 0.01]
N_TREES = 500

plt.figure(figsize=(10, 6))
for lr in learning_rates:
    gbm = GradientBoostingClassifier(
        n_estimators=N_TREES, learning_rate=lr, max_depth=3,
        subsample=0.9, random_state=3,
    ).fit(X_tr, y_tr)

    # staged_predict_proba -> validation AUC after each stage
    aucs = [roc_auc_score(y_val, p[:, 1])
            for p in gbm.staged_predict_proba(X_val)]
    best = int(np.argmax(aucs))
    plt.plot(range(1, N_TREES + 1), aucs,
             label=f"lr={lr}  (best n={best+1}, auc={aucs[best]:.3f})")

plt.xlabel("Number of trees (n_estimators)")
plt.ylabel("Validation ROC-AUC")
plt.title("Shrinkage tradeoff: learning_rate vs n_estimators")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("lr_vs_ntrees.png", dpi=120)
```

## Alternative Solution
Do it as a proper 2D grid with `GridSearchCV` over `{learning_rate, n_estimators}` and cross-validated AUC, then visualize the score surface as a heatmap. That's more rigorous (accounts for split variance) but far more expensive than the single-fit `staged_predict` trick above.

## Interview Variations
- "Add regularization" → sweep `max_depth`, `subsample`, `min_samples_leaf` and show they interact with LR.
- "Do it for XGBoost" → same idea; use `eval_set` + `evals_result_` to pull the per-iteration validation metric.
- "What if training is too slow at lr=0.01?" → use a warm start / early stopping so you don't train all 500 trees when 200 suffice.

## Common Follow-up Questions
- *Why does small LR generalize better?* It's a form of regularization — small steps avoid overfitting any single residual pattern; the ensemble averages many gentle corrections.
- *Can LR be too small?* Yes — underfitting within your tree budget, and impractically slow training.
- *Interaction with early stopping?* Small LR + large `n_estimators` + early stopping is the canonical production setup; early stopping makes the exact `n_estimators` a non-issue.

---
---

# PART B — END-TO-END NOTEBOOK: Customer Churn Prediction with Ensembles

**Goal:** Predict which customers churn (binary classification) and compare ensemble methods end to end. This is the exact shape of a lab-exam notebook: load → explore → clean → model → tune → evaluate → interpret.

**Dataset choice:** We use `make_classification` shaped to *look like* a telco churn table (churn is the minority class ~27%), so the notebook is 100% runnable with zero downloads. A commented `pd.read_csv` fallback shows how you'd swap in the real Kaggle Telco Churn CSV. Pick one and stay consistent — we default to the synthetic generator.

Run the cells top to bottom.

---

### Cell 1 — Imports & environment setup

```python
# Core stack (preinstalled in most exam images)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, RocCurveDisplay)

# Optional boosting libs — pip install if missing. Flags let later cells skip gracefully.
# pip install xgboost lightgbm catboost imbalanced-learn
try:
    from xgboost import XGBClassifier; HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier; HAS_LGB = True
except ImportError:
    HAS_LGB = False
try:
    from catboost import CatBoostClassifier; HAS_CAT = True
except ImportError:
    HAS_CAT = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")
print(f"xgboost={HAS_XGB}  lightgbm={HAS_LGB}  catboost={HAS_CAT}")
```

**Explanation.** We import everything up front (exam-safe: no mid-notebook `ImportError` surprises) and probe for the three boosting libraries, storing availability flags so every downstream cell degrades gracefully instead of crashing. Setting a single `RANDOM_STATE` everywhere makes results reproducible — graders reward determinism.

---

### Cell 2 — Load the dataset (synthetic telco-style churn)

```python
from sklearn.datasets import make_classification

# --- Option A (default, runnable anywhere): synthetic churn-shaped data ---
X_arr, y_arr = make_classification(
    n_samples=7043,          # same row count as the classic Telco Churn dataset
    n_features=18, n_informative=10, n_redundant=4,
    weights=[0.735, 0.265],  # ~26.5% churn -> realistic imbalance
    flip_y=0.03, class_sep=0.9, random_state=RANDOM_STATE,
)
num_cols = [f"num_{i}" for i in range(14)]
cat_cols = ["contract", "internet", "payment", "gender"]
df = pd.DataFrame(X_arr[:, :14], columns=num_cols)
# Synthesize a few categorical columns from the remaining signal
df["contract"] = pd.cut(X_arr[:, 14], bins=3, labels=["month", "one_year", "two_year"])
df["internet"] = pd.cut(X_arr[:, 15], bins=3, labels=["dsl", "fiber", "none"])
df["payment"]  = pd.cut(X_arr[:, 16], bins=2, labels=["auto", "manual"])
df["gender"]   = np.where(X_arr[:, 17] > 0, "F", "M")
df["Churn"] = y_arr

# --- Option B (real data) — uncomment to use the Kaggle Telco Churn CSV instead ---
# df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# df = df.drop(columns=["customerID"])
# df["Churn"] = (df["Churn"] == "Yes").astype(int)

print(df.shape)
df.head()
```

**Explanation.** We build a DataFrame that mixes 14 numeric columns with 4 categorical ones and a binary `Churn` target, mirroring a real telco table (contract type, internet service, payment method, gender). The commented Option B shows the two classic gotchas of the real CSV: `TotalCharges` ships as text with blank strings (coerce to numeric), and `Churn` is `"Yes"/"No"` (map to 1/0). Keeping both paths documents that you know the real dataset.

---

### Cell 3 — Exploratory Data Analysis (EDA)

```python
print("Target distribution:")
print(df["Churn"].value_counts(normalize=True).round(3))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
# 1) Class balance
df["Churn"].value_counts().plot.bar(ax=axes[0], color=["#4C78A8", "#F58518"])
axes[0].set_title("Churn class balance"); axes[0].set_xticklabels(["stay", "churn"], rotation=0)
# 2) A numeric feature by churn
sns.kdeplot(data=df, x="num_0", hue="Churn", fill=True, ax=axes[1])
axes[1].set_title("num_0 distribution by churn")
# 3) Churn rate by contract type
(df.groupby("contract")["Churn"].mean()
   .plot.bar(ax=axes[2], color="#54A24B"))
axes[2].set_title("Churn rate by contract"); axes[2].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.show()

print("\nMissing values:\n", df.isna().sum().sum(), "total")
print("\nNumeric summary:\n", df[num_cols].describe().T[["mean", "std", "min", "max"]].head())
```

**Explanation.** EDA answers three questions before any modeling: *How imbalanced is the target?* (~26% churn — enough to matter), *Do features separate the classes?* (KDE overlap tells us discriminative power), and *Which segments churn most?* (grouped churn rate by contract is the kind of business insight graders love). We also confirm no missing values and eyeball numeric ranges to decide on scaling.

---

### Cell 4 — Train/test split (stratified, early)

```python
X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

print(f"Train: {X_train.shape}, churn rate {y_train.mean():.3f}")
print(f"Test : {X_test.shape}, churn rate {y_test.mean():.3f}")
```

**Explanation.** Split *before* fitting any transformer — otherwise scaling/encoding statistics leak test information into training. `stratify=y` preserves the ~26% churn rate in both splits, which is essential for a reliable minority-class estimate. We hold the test set out and never touch it until final evaluation.

---

### Cell 5 — Preprocessing pipeline (encode + scale)

```python
numeric_features = num_cols
categorical_features = cat_cols

preprocess = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
])

# Fit on train only, then inspect the expanded feature space
X_train_pp = preprocess.fit_transform(X_train)
X_test_pp  = preprocess.transform(X_test)
feature_names = preprocess.get_feature_names_out()
print(f"Feature space after encoding: {X_train_pp.shape[1]} columns")
```

**Explanation.** A `ColumnTransformer` applies the right transform to the right columns: `StandardScaler` for numeric (helps scale-sensitive models and RandomizedSearch stability; tree ensembles are scale-invariant but it doesn't hurt), and `OneHotEncoder(handle_unknown="ignore")` for categoricals so unseen categories at inference don't blow up. `drop="first"` avoids the dummy-variable trap for linear meta-learners. Wrapping this in a transformer (not manual `pd.get_dummies`) is what prevents leakage and keeps train/test columns aligned.

---

### Cell 6 — Handle class imbalance

```python
# Strategy A: class weights (built into most estimators) — cheap, no resampling
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos      # for XGBoost/LightGBM
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

# Strategy B: SMOTE oversampling (optional; only if imbalanced-learn is installed)
USE_SMOTE = False
try:
    from imblearn.over_sampling import SMOTE
    if USE_SMOTE:
        X_train_pp, y_train_bal = SMOTE(random_state=RANDOM_STATE).fit_resample(
            X_train_pp, y_train)
        print("Applied SMOTE:", np.bincount(y_train_bal))
    else:
        y_train_bal = y_train
except ImportError:
    y_train_bal = y_train
    print("imbalanced-learn not installed; relying on class weights instead.")
```

**Explanation.** Two standard levers. **Class weighting** (`class_weight="balanced"` for sklearn, `scale_pos_weight` for XGB/LGBM) tells the loss to penalize minority errors more — no data duplication, no leakage risk. **SMOTE** synthesizes new minority examples; if you use it, fit it on the *training fold only* (never before the split, never on test). We default to class weights (simpler, safer) and leave SMOTE as an opt-in flag. Because churn is only mildly imbalanced (~1:2.8), weighting is usually enough.

---

### Cell 7 — Baseline: Random Forest

```python
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced",
    oob_score=True, n_jobs=-1, random_state=RANDOM_STATE,
).fit(X_train_pp, y_train)

print(f"RF OOB score: {rf.oob_score_:.3f}")
rf_pred  = rf.predict(X_test_pp)
rf_proba = rf.predict_proba(X_test_pp)[:, 1]
print(f"RF test accuracy : {accuracy_score(y_test, rf_pred):.3f}")
print(f"RF test ROC-AUC  : {roc_auc_score(y_test, rf_proba):.3f}")
```

**Explanation.** Random Forest is our sturdy baseline: little tuning, `class_weight="balanced"` for the imbalance, and a free `oob_score_` sanity check that should track the test score. If a fancy boosting model can't beat this, something is wrong with the fancy model's setup — always establish the baseline first.

---

### Cell 8 — Gradient Boosting (sklearn)

```python
gbm = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=3,
    subsample=0.8, random_state=RANDOM_STATE,
).fit(X_train_pp, y_train)

gbm_proba = gbm.predict_proba(X_test_pp)[:, 1]
print(f"GBM test accuracy: {accuracy_score(y_test, gbm.predict(X_test_pp)):.3f}")
print(f"GBM test ROC-AUC : {roc_auc_score(y_test, gbm_proba):.3f}")
```

**Explanation.** Sklearn's Gradient Boosting brings the sequential, bias-reducing counterpart to the forest's variance reduction. We use a small learning rate (0.05) with `subsample=0.8` (stochastic gradient boosting) for a bit of regularization. Note there's no `class_weight` here — GBM lacks it, so on strongly imbalanced data you'd lean on SMOTE or `sample_weight`; here the mild imbalance is tolerable.

---

### Cell 9 — XGBoost, LightGBM, CatBoost (with early stopping)

```python
# Validation slice carved from training data for early stopping
Xtr2, Xval, ytr2, yval = train_test_split(
    X_train_pp, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE)

boost_models = {}

if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        eval_metric="auc", early_stopping_rounds=50,
        tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)
    xgb.fit(Xtr2, ytr2, eval_set=[(Xval, yval)], verbose=False)
    boost_models["XGBoost"] = xgb
    print(f"XGB best_iteration = {xgb.best_iteration}")

if HAS_LGB:
    import lightgbm as lgb
    lgbm = LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE)
    lgbm.fit(Xtr2, ytr2, eval_set=[(Xval, yval)], eval_metric="auc",
             callbacks=[lgb.early_stopping(50, verbose=False)])
    boost_models["LightGBM"] = lgbm
    print(f"LGB best_iteration = {lgbm.best_iteration_}")

if HAS_CAT:
    cat = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=5, eval_metric="AUC",
        early_stopping_rounds=50, auto_class_weights="Balanced",
        random_seed=RANDOM_STATE, verbose=False)
    cat.fit(Xtr2, ytr2, eval_set=(Xval, yval))
    boost_models["CatBoost"] = cat
    print(f"CatBoost best_iteration = {cat.get_best_iteration()}")

print("Trained boosting models:", list(boost_models.keys()))
```

**Explanation.** All three gradient-boosting heavyweights, each with a large `n_estimators` reined in by **early stopping** on a dedicated validation slice — the model keeps its best iteration, not the last. Note the per-library API dialects: XGBoost takes `early_stopping_rounds` in the constructor, LightGBM uses a `callbacks` list, CatBoost uses `eval_set` as a tuple. Each also gets imbalance handling in its native way (`scale_pos_weight` / `class_weight` / `auto_class_weights`). The `HAS_*` flags mean this cell runs even if some libraries are absent.

---

### Cell 10 — Unified evaluation (accuracy / precision / recall / F1 / ROC-AUC)

```python
def evaluate(name, model, X_te, y_te):
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy":  accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall":    recall_score(y_te, pred),
        "f1":        f1_score(y_te, pred),
        "roc_auc":   roc_auc_score(y_te, proba),
    }

all_models = {"RandomForest": rf, "GradientBoosting": gbm, **boost_models}
scores = pd.DataFrame([evaluate(n, m, X_test_pp, y_test) for n, m in all_models.items()])
scores = scores.set_index("model").round(3).sort_values("roc_auc", ascending=False)
scores
```

**Explanation.** One function, every model, the same test set — apples to apples. For churn we care most about **recall** (catching customers who will actually leave) and **ROC-AUC** (threshold-independent ranking quality); raw accuracy is misleading under imbalance because predicting "nobody churns" already scores ~74%. Sorting by ROC-AUC surfaces the best ranker.

---

### Cell 11 — Confusion matrix & ROC curves

```python
best_name = scores.index[0]
best_model = all_models[best_name]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Confusion matrix for the best model
cm = confusion_matrix(y_test, best_model.predict(X_test_pp))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["stay", "churn"], yticklabels=["stay", "churn"])
axes[0].set_title(f"Confusion matrix — {best_name}")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

# ROC curves for all models
for name, m in all_models.items():
    RocCurveDisplay.from_predictions(
        y_test, m.predict_proba(X_test_pp)[:, 1], name=name, ax=axes[1])
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_title("ROC curves")
plt.tight_layout(); plt.show()

print(classification_report(y_test, best_model.predict(X_test_pp),
                            target_names=["stay", "churn"]))
```

**Explanation.** The confusion matrix exposes the *cost structure*: false negatives (missed churners) are the expensive quadrant for a retention team, false positives just waste a discount coupon. Overlaid ROC curves compare ranking quality across models at every threshold. The `classification_report` gives per-class precision/recall so you can articulate the tradeoff, not just a single number.

---

### Cell 12 — Hyperparameter tuning with RandomizedSearchCV

```python
tune_estimator = RandomForestClassifier(
    class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

param_dist = {
    "n_estimators":      [200, 300, 500, 800],
    "max_depth":         [None, 6, 10, 16, 24],
    "max_features":      ["sqrt", "log2", 0.5],
    "min_samples_leaf":  [1, 2, 4, 8],
    "min_samples_split": [2, 5, 10],
}

search = RandomizedSearchCV(
    tune_estimator, param_distributions=param_dist,
    n_iter=25, scoring="roc_auc",
    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
).fit(X_train_pp, y_train)

print("Best CV ROC-AUC:", round(search.best_score_, 3))
print("Best params:", search.best_params_)
rf_tuned = search.best_estimator_
print("Tuned RF test ROC-AUC:",
      round(roc_auc_score(y_test, rf_tuned.predict_proba(X_test_pp)[:, 1]), 3))
```

**Explanation.** `RandomizedSearchCV` samples 25 random configurations rather than exhaustively gridding — far cheaper and, in high dimensions, usually finds a near-optimal point faster than `GridSearchCV`. We optimize `roc_auc` (the right metric for imbalanced ranking) with stratified 5-fold CV so each fold keeps the churn ratio. The same pattern tunes any boosting model — just swap the estimator and the `param_dist` (e.g. `learning_rate`, `num_leaves`, `subsample`).

---

### Cell 13 — Cross-validation of the shortlist

```python
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
cv_targets = {"RandomForest(tuned)": rf_tuned, "GradientBoosting": gbm}
if HAS_LGB: cv_targets["LightGBM"] = LGBMClassifier(
    n_estimators=300, learning_rate=0.05, class_weight="balanced",
    n_jobs=-1, random_state=RANDOM_STATE)

for name, model in cv_targets.items():
    aucs = cross_val_score(model, X_train_pp, y_train, cv=cv,
                           scoring="roc_auc", n_jobs=-1)
    print(f"{name:22s} ROC-AUC {aucs.mean():.3f} +/- {aucs.std():.3f}")
```

**Explanation.** A single train/test split has variance; cross-validation reports mean ± std so we can tell whether model A *reliably* beats model B or just got a lucky split. If two models overlap within one standard deviation, prefer the simpler/faster one. Note we CV on the *training* data only — the test set stays sacred for the final unbiased estimate.

---

### Cell 14 — Feature importance comparison

```python
importances = {}

# Random Forest (tuned) impurity importance
importances["RandomForest"] = pd.Series(rf_tuned.feature_importances_, index=feature_names)
# Gradient Boosting
importances["GradientBoosting"] = pd.Series(gbm.feature_importances_, index=feature_names)
# LightGBM if available
if HAS_LGB and "LightGBM" in boost_models:
    importances["LightGBM"] = pd.Series(
        boost_models["LightGBM"].feature_importances_, index=feature_names)

imp_df = pd.DataFrame(importances).fillna(0)
# Normalize each column so models are comparable, then rank by mean importance
imp_df = imp_df.div(imp_df.sum(axis=0), axis=1)
imp_df["mean"] = imp_df.mean(axis=1)
top = imp_df.sort_values("mean", ascending=False).head(12).drop(columns="mean")

top[::-1].plot.barh(figsize=(10, 7))
plt.title("Top-12 feature importances across models (normalized)")
plt.xlabel("Relative importance"); plt.tight_layout(); plt.show()
```

**Explanation.** Different ensembles agree on the *strong* signals and disagree on the noise — plotting them together tells you which features are robustly predictive versus artifacts of one model's inductive bias. We normalize each column so a model that reports raw split counts (LightGBM) is comparable to one reporting fractional impurity decrease (RF/GBM). For a production report you'd escalate to permutation importance or SHAP, since impurity importance is biased toward high-cardinality features (the caveat from Part A, Q2).

---

### Cell 15 — Model comparison table (accuracy / speed / importance)

```python
rows = []
for name, model in all_models.items():
    t0 = time.perf_counter()
    model.predict_proba(X_test_pp)             # inference latency proxy
    infer_ms = (time.perf_counter() - t0) * 1000
    top_feat = (pd.Series(getattr(model, "feature_importances_", np.zeros(len(feature_names))),
                          index=feature_names).idxmax()
                if hasattr(model, "feature_importances_") else "n/a")
    s = evaluate(name, model, X_test_pp, y_test)
    rows.append({"model": name, "accuracy": s["accuracy"], "recall": s["recall"],
                 "f1": s["f1"], "roc_auc": s["roc_auc"],
                 "infer_ms": round(infer_ms, 1), "top_feature": top_feat})

comparison = pd.DataFrame(rows).set_index("model").round(3).sort_values("roc_auc", ascending=False)
comparison
```

**Explanation.** The deliverable a hiring manager actually reads: one table ranking every model by ROC-AUC alongside recall/F1, inference latency, and its most important feature. This is where you make the *engineering* argument — LightGBM might win AUC by 0.002 while being 5x faster to score, which matters at scale; or Random Forest might be "good enough" and easier to explain to stakeholders. Accuracy alone never tells this story.

---

### Cell 16 — Interpretation & deployment considerations

```python
print(f"""
CHURN MODEL — SUMMARY
=====================
Best model (by ROC-AUC): {comparison.index[0]}
  ROC-AUC : {comparison.iloc[0]['roc_auc']}
  Recall  : {comparison.iloc[0]['recall']}  (share of true churners caught)
  Top driver: {comparison.iloc[0]['top_feature']}

Recommendation: deploy '{comparison.index[0]}' behind a churn-risk score.
Tune the decision threshold to the business cost ratio of a missed churner
(lost lifetime value) vs a wasted retention offer, rather than defaulting to 0.5.
""")
```

**Explanation & deployment checklist.**
- **Threshold tuning:** the 0.5 cutoff is arbitrary. If retaining a customer is worth far more than a coupon, lower the threshold to raise recall (catch more churners) and accept more false positives. Pick it from a precision-recall curve tied to the cost matrix.
- **Calibration:** if you act on the *probability* (e.g., "offer bigger discount above 80% risk"), calibrate with `CalibratedClassifierCV` — raw tree-ensemble probabilities are often miscalibrated.
- **Data / concept drift:** churn drivers shift with pricing and competitors. Monitor feature distributions and AUC over time; schedule retraining.
- **Reproducibility & serving:** persist the *entire* pipeline (`joblib.dump(Pipeline([('prep', preprocess), ('model', best_model)]), 'churn.joblib')`) so preprocessing and model travel together — the #1 source of train/serve skew is re-implementing preprocessing at inference.
- **Explainability:** for regulated or stakeholder-facing use, ship SHAP explanations per prediction ("this customer scored high because month-to-month contract + high monthly charge").
- **Leakage audit:** double-check no post-churn field (e.g., a cancellation flag) sneaked into features — it inflates offline metrics and collapses in production.

---

## Quick-reference cheat sheet

| Concept | One-liner to remember |
|---|---|
| Bagging | Parallel, independent learners on bootstrap samples → **reduces variance** |
| Boosting | Sequential learners fixing prior errors → **reduces bias** |
| Random Forest | Bagging + random feature subsets (`max_features`) to decorrelate trees |
| OOB score | Free validation from the ~37% rows each tree didn't train on |
| AdaBoost | Reweights *samples*; weighted vote of stumps |
| Gradient Boosting | Fits new tree to negative gradient (residuals) of the loss |
| XGBoost | Level-wise growth, regularized, `scale_pos_weight` for imbalance |
| LightGBM | **Leaf-wise** growth, histogram-based, fastest on big data (`num_leaves`) |
| CatBoost | Native categoricals, ordered boosting, great defaults |
| Voting | Hard = majority label; Soft = argmax of averaged probabilities |
| Stacking | Meta-learner on out-of-fold base predictions (`cv=` prevents leakage) |
| LR vs n_estimators | Smaller `learning_rate` ⇒ more trees; use early stopping to pick the count |
| Metric for churn | Optimize ROC-AUC / recall, not accuracy, under class imbalance |


---

# Part 5 — Model Evaluation (Practical)

Part A drills the "code it from scratch" muscle that lab exams and interviews love: metrics, confusion matrices, ROC/PR curves, threshold tuning, and cross-validation with nothing but NumPy. Part B is a full end-to-end playbook you can paste cell-by-cell into a notebook. Read the prose, not just the code — the metric you *choose* is graded more heavily than the metric you *compute*.

---

# PART A — CODING QUESTIONS

The golden rule for every question below: **the metric is a business decision, not a math default.** Before you write a single line, ask "what does a false positive cost here versus a false negative?" That framing is what separates a junior who memorized `sklearn.metrics` from an engineer.

---

# Practical Question 1: Confusion Matrix from Scratch + Derived Metrics

**Difficulty:** Easy
**Estimated Time:** 15 min
**Concepts Tested:** Confusion matrix construction, TP/FP/FN/TN bookkeeping, precision, recall, specificity, F1, vectorized NumPy

## Problem Statement
Given two arrays `y_true` and `y_pred` of binary labels (0/1), build a 2x2 confusion matrix **without** using `sklearn`. Then derive accuracy, precision, recall (sensitivity), specificity, and F1-score from the four cells. Convention: positive class = 1.

The confusion matrix layout you should standardize on:

```
                 Predicted 0     Predicted 1
Actual 0            TN               FP
Actual 1            FN               TP
```

## Example Input
```python
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
```

## Example Output
```
Confusion Matrix:
[[4 1]
 [1 4]]
Accuracy:    0.800
Precision:   0.800
Recall:      0.800
Specificity: 0.800
F1-score:    0.800
```

## Approach
1. Coerce inputs to NumPy arrays so we can use boolean masks.
2. TP = count where (true==1 & pred==1); FP = (true==0 & pred==1); FN = (true==1 & pred==0); TN = (true==0 & pred==0). Each is a single vectorized `np.sum` over a boolean AND — O(n).
3. Assemble the 2x2 matrix in `[[TN, FP], [FN, TP]]` order to match sklearn's `confusion_matrix`.
4. Derive metrics, guarding every denominator against divide-by-zero (an all-negative-prediction model has precision 0/0).

## Python Implementation
```python
import numpy as np


def confusion_matrix_scratch(y_true, y_pred):
    """Return a 2x2 confusion matrix [[TN, FP], [FN, TP]] for binary labels.

    Positive class is assumed to be 1. Runs in O(n) time, O(1) extra space.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    return np.array([[tn, fp], [fn, tp]]), (tn, fp, fn, tp)


def _safe_div(numerator, denominator):
    """Divide, returning 0.0 when the denominator is 0 (undefined metric)."""
    return numerator / denominator if denominator else 0.0


def classification_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, specificity, and F1 from scratch."""
    cm, (tn, fp, fn, tp) = confusion_matrix_scratch(y_true, y_pred)

    accuracy    = _safe_div(tp + tn, tp + tn + fp + fn)
    precision   = _safe_div(tp, tp + fp)          # of predicted positives, how many correct
    recall      = _safe_div(tp, tp + fn)          # of actual positives, how many caught (sensitivity)
    specificity = _safe_div(tn, tn + fp)          # of actual negatives, how many caught (TNR)
    f1          = _safe_div(2 * precision * recall, precision + recall)

    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


if __name__ == "__main__":
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

    m = classification_metrics(y_true, y_pred)
    print("Confusion Matrix:")
    print(m["confusion_matrix"])
    for key in ("accuracy", "precision", "recall", "specificity", "f1"):
        print(f"{key.capitalize():12s} {m[key]:.3f}")
```

**Complexity:** O(n) time, O(1) auxiliary space (the matrix is fixed 2x2).

## Alternative Solution
For an arbitrary number of classes, index into a KxK matrix with a single pass. This generalizes gracefully and is the pattern sklearn uses internally:

```python
def confusion_matrix_multiclass(y_true, y_pred, num_classes=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = num_classes or int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((k, k), dtype=int)
    # np.add.at handles repeated indices correctly (unlike cm[rows, cols] += 1)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm
```

## Interview Variations
- "Now do it for 3+ classes and compute **macro** vs **micro** vs **weighted** F1." Macro = unweighted mean of per-class F1; micro = pool all TP/FP/FN globally (equals accuracy in single-label multiclass); weighted = mean weighted by class support.
- "The labels are strings like `'spam'`/`'ham'`." Map to indices via `np.unique(..., return_inverse=True)`.
- "Return per-class precision/recall in one pass." Compute row sums (support) and column sums (predicted counts); diagonal is TP per class.

## Common Follow-up Questions
- **Why is accuracy misleading on imbalanced data?** A 99%-negative dataset gets 99% accuracy from a model that always predicts negative — recall on the positive class is 0. Report precision/recall/F1 or balanced accuracy instead.
- **Precision vs recall trade-off?** Precision protects against false alarms (spam filter eating real mail); recall protects against misses (cancer screening). You rarely maximize both.
- **What is F1 actually?** The harmonic mean of precision and recall — it punishes imbalance between the two, so you can't game it by maxing one.

---

# Practical Question 2: Regression Metrics from Scratch (R², Adjusted R², RMSE)

**Difficulty:** Easy-Medium
**Estimated Time:** 15 min
**Concepts Tested:** MAE, MSE, RMSE, R² (coefficient of determination), Adjusted R², degrees-of-freedom correction

## Problem Statement
Implement MAE, MSE, RMSE, R², and Adjusted R² using only NumPy. `y_true` and `y_pred` are continuous. Adjusted R² also needs `n` (samples) and `p` (number of predictors/features).

## Example Input
```python
y_true = [3.0, -0.5, 2.0, 7.0, 4.2]
y_pred = [2.5,  0.0, 2.1, 7.8, 3.9]
n_features = 2
```

## Example Output
```
MAE:          0.420
MSE:          0.230
RMSE:         0.480
R2:           0.958
Adjusted R2:  0.917
```

## Approach
- **MAE** = mean(|error|). Robust to outliers, same units as target, interpretable.
- **MSE** = mean(error²). Penalizes large errors quadratically; units are squared.
- **RMSE** = sqrt(MSE). Back to target units; still outlier-sensitive.
- **R²** = 1 − SS_res / SS_tot, where SS_res = Σ(y−ŷ)² and SS_tot = Σ(y−ȳ)². It's the fraction of variance explained. A model predicting the mean scores 0; a perfect model scores 1; a model worse than the mean goes negative.
- **Adjusted R²** = 1 − (1−R²)·(n−1)/(n−p−1). Penalizes adding predictors that don't help — critical for honest model comparison, because plain R² never decreases when you add features.

## Python Implementation
```python
import numpy as np


def regression_metrics(y_true, y_pred, n_features=None):
    """MAE, MSE, RMSE, R2, and (optionally) Adjusted R2 from scratch."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    n = y_true.size

    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    if n_features is not None:
        p = n_features
        # Guard the denominator: undefined when n - p - 1 <= 0
        if n - p - 1 > 0:
            adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
        else:
            adj = float("nan")
        metrics["Adjusted R2"] = adj

    return metrics


if __name__ == "__main__":
    y_true = [3.0, -0.5, 2.0, 7.0, 4.2]
    y_pred = [2.5,  0.0, 2.1, 7.8, 3.9]
    for k, v in regression_metrics(y_true, y_pred, n_features=2).items():
        print(f"{k:12s} {v:.3f}")
```

## Alternative Solution
Verify against sklearn to catch off-by-one mistakes in the adjusted formula:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)                       # or root_mean_squared_error in sklearn >= 1.4
r2   = r2_score(y_true, y_pred)
```

Note: older sklearn used `mean_squared_error(..., squared=False)` for RMSE; that arg is deprecated in favor of `root_mean_squared_error`. In an exam, `np.sqrt(mse)` always works and sidesteps version drift.

## Interview Variations
- "Add MAPE (mean absolute percentage error)." `np.mean(np.abs(error / y_true))` — but guard against `y_true == 0`.
- "Why can R² be negative?" When the model fits worse than the horizontal line ȳ, SS_res > SS_tot.
- "When prefer MAE over RMSE?" When outliers are noise you don't want to chase; MAE treats all errors linearly.

## Common Follow-up Questions
- **MSE vs RMSE — why bother with both?** MSE is the differentiable loss you optimize; RMSE is the human-readable report in target units.
- **Adjusted R² intuition?** It asks: "did the new feature earn its keep?" If R² barely moves while p grows, adjusted R² falls.
- **Is high R² always good?** No — it can indicate overfitting or leakage. Always pair with a held-out set and residual plots.

---

# Practical Question 3: ROC Curve and AUC from Scratch

**Difficulty:** Medium
**Estimated Time:** 25 min
**Concepts Tested:** Threshold sweeping, TPR/FPR, ROC geometry, trapezoidal AUC, ranking interpretation of AUC

## Problem Statement
Given `y_true` (binary) and `y_scores` (continuous model confidence for the positive class), compute the ROC curve by sweeping every distinct threshold, and compute AUC via the trapezoidal rule — no sklearn. Then plot it with matplotlib.

ROC plots **TPR (recall)** on the y-axis against **FPR (1−specificity)** on the x-axis. Each threshold is one point on the curve.

## Example Input
```python
y_true   = [0, 0, 1, 1, 0, 1, 0, 1]
y_scores = [0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.5, 0.7]
```

## Example Output
```
AUC = 0.875
(plus a plotted ROC curve rising toward the top-left corner)
```

## Approach
1. Sort samples by descending score. Candidate thresholds are the unique scores (plus +inf so we start with everything classified negative).
2. As you lower the threshold, points flip from predicted-negative to predicted-positive one group at a time. Track cumulative TP and FP.
3. TPR = TP / P (total positives); FPR = FP / N (total negatives).
4. AUC = area under the (FPR, TPR) curve via `np.trapz`. Interpretation: **AUC = probability that a randomly chosen positive outranks a randomly chosen negative.** 0.5 = random, 1.0 = perfect.

## Python Implementation
```python
import numpy as np
import matplotlib.pyplot as plt


def roc_curve_scratch(y_true, y_scores):
    """Compute ROC points (fpr, tpr, thresholds) by sweeping thresholds.

    Returns arrays sorted by increasing FPR so they can be plotted directly.
    Complexity: O(n log n) for the sort, O(n) for the sweep.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    # Sort by score descending: high-confidence positives get crossed first.
    order = np.argsort(-y_scores)
    y_true = y_true[order]
    y_scores = y_scores[order]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    # Cumulative counts as the threshold sweeps from +inf down through each score.
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    # Keep only the last index of each distinct score (a threshold sits *between* scores).
    distinct = np.where(np.diff(y_scores) != 0)[0]
    idx = np.r_[distinct, y_scores.size - 1]

    tpr = np.r_[0, tps[idx] / P]          # prepend the (0,0) origin
    fpr = np.r_[0, fps[idx] / N]
    thresholds = np.r_[np.inf, y_scores[idx]]

    return fpr, tpr, thresholds


def auc_trapezoid(fpr, tpr):
    """Area under the ROC curve via the trapezoidal rule."""
    return np.trapz(tpr, fpr)


def plot_roc(y_true, y_scores, label="Model"):
    fpr, tpr, _ = roc_curve_scratch(y_true, y_scores)
    auc = auc_trapezoid(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, marker="o", label=f"{label} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return auc


if __name__ == "__main__":
    y_true   = [0, 0, 1, 1, 0, 1, 0, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.5, 0.7]
    print(f"AUC = {plot_roc(y_true, y_scores):.3f}")
```

## Alternative Solution
The "count concordant pairs" view of AUC is O(n log n) and needs no thresholds — great for a quick sanity check and a slick interview answer:

```python
def auc_rank(y_true, y_scores):
    """AUC = P(score(pos) > score(neg)) via the Mann-Whitney U statistic."""
    y_true = np.asarray(y_true)
    order = np.argsort(y_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_scores) + 1)  # average-tie handling omitted for brevity
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    sum_ranks_pos = np.sum(ranks[y_true == 1])
    return (sum_ranks_pos - P * (P + 1) / 2) / (P * N)
```

## Interview Variations
- "Plot multiple models on one axis." Loop and overlay — the model whose curve hugs the top-left wins (see Part B).
- "AUC is 0.9 but the model is useless in production — how?" AUC is threshold-independent and ignores calibration and class imbalance; on a 1:1000 dataset a great AUC can still mean terrible precision.
- "Handle ties in scores." Thresholds must sit between distinct scores; average tied ranks in the rank formula.

## Common Follow-up Questions
- **ROC vs PR curve — when to use which?** ROC can look optimistic under heavy class imbalance because FPR has a huge negative denominator. Prefer PR curves when positives are rare.
- **What does the diagonal mean?** Random guessing. Any point below it means the model is worse than a coin flip (invert predictions).
- **Why top-left is best?** High TPR, low FPR — catch positives without false alarms.

---

# Practical Question 4: Precision-Recall Curve from Scratch

**Difficulty:** Medium
**Estimated Time:** 20 min
**Concepts Tested:** Precision/recall as a function of threshold, PR curve, average precision, imbalanced evaluation, baseline reasoning

## Problem Statement
Given `y_true` and `y_scores`, sweep thresholds to trace precision vs recall, plot the PR curve, and compute Average Precision (AP) as the area under it. Contrast the baseline with ROC's baseline.

## Example Input
```python
y_true   = [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
y_scores = [0.2, 0.3, 0.7, 0.9, 0.4, 0.55, 0.35, 0.8, 0.1, 0.45]
```

## Example Output
```
Average Precision (AP) = 0.86
Baseline precision (prevalence) = 0.40
(a PR curve that starts high-precision and decays as recall approaches 1)
```

## Approach
1. Sort by descending score. Sweep the threshold downward; at each step accumulate TP and FP.
2. Precision = TP / (TP + FP); Recall = TP / P.
3. The PR baseline is the **positive class prevalence** P/(P+N), not 0.5 — a random classifier's precision equals prevalence at all recalls.
4. AP = Σ (R_k − R_{k−1}) · P_k — the step-wise area, the way sklearn's `average_precision_score` defines it.

## Python Implementation
```python
import numpy as np
import matplotlib.pyplot as plt


def pr_curve_scratch(y_true, y_scores):
    """Precision-recall points by sweeping thresholds. O(n log n)."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    order = np.argsort(-y_scores)
    y_true = y_true[order]
    y_scores = y_scores[order]

    P = np.sum(y_true == 1)
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    distinct = np.where(np.diff(y_scores) != 0)[0]
    idx = np.r_[distinct, y_scores.size - 1]

    precision = tps[idx] / (tps[idx] + fps[idx])
    recall = tps[idx] / P

    # sklearn convention: prepend precision=1, recall=0 as the curve's start.
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return precision, recall


def average_precision(precision, recall):
    """AP = sum over thresholds of (recall_delta * precision)."""
    return np.sum(np.diff(recall) * precision[1:])


def plot_pr(y_true, y_scores, label="Model"):
    precision, recall = pr_curve_scratch(y_true, y_scores)
    ap = average_precision(precision, recall)
    prevalence = np.mean(np.asarray(y_true) == 1)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, marker="o", label=f"{label} (AP = {ap:.3f})")
    plt.axhline(prevalence, ls="--", color="k",
                label=f"Baseline (prevalence = {prevalence:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()
    return ap


if __name__ == "__main__":
    y_true   = [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
    y_scores = [0.2, 0.3, 0.7, 0.9, 0.4, 0.55, 0.35, 0.8, 0.1, 0.45]
    print(f"AP = {plot_pr(y_true, y_scores):.3f}")
```

## Alternative Solution
Validate with sklearn's `precision_recall_curve` and `average_precision_score`:

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)
```

## Interview Variations
- "Why does the PR baseline move but the ROC baseline stay at 0.5?" Because precision depends on prevalence; TPR/FPR don't.
- "Your fraud model has AUC 0.97 but AP 0.30 — reconcile." Rare positives: ROC's giant negative pool masks the false positives that crush precision. Trust AP here.
- "Optimize the F1 point on this curve." F1 = 2PR/(P+R) at each threshold; pick the argmax (see Q5).

## Common Follow-up Questions
- **When is PR strictly better than ROC?** Highly imbalanced problems where the positive class is what you care about (fraud, disease, defect detection).
- **Is the PR curve monotonic?** No — precision can wiggle up and down as recall increases, which is why AP integrates the actual (possibly jagged) curve.

---

# Practical Question 5: Threshold Tuning (Maximize F1 / Hit a Recall Target)

**Difficulty:** Medium-Hard
**Estimated Time:** 25 min
**Concepts Tested:** The 0.5 threshold is a myth, operating-point selection, F1 optimization, recall-constrained selection, precision-recall trade-off in production

## Problem Statement
A classifier outputs probabilities. The default 0.5 cutoff is rarely optimal. Write functions that (a) find the threshold maximizing F1, and (b) find the **lowest** threshold that still meets a minimum recall target (e.g., "we must catch 90% of fraud") while maximizing precision subject to that constraint.

## Example Input
```python
y_true   = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
y_scores = [0.2, 0.8, 0.6, 0.55, 0.9, 0.4, 0.3, 0.45, 0.7, 0.25]
recall_target = 0.80
```

## Example Output
```
Best-F1 threshold: 0.60  -> F1=0.833, P=0.833, R=0.833
Recall>=0.80 threshold: 0.40 -> R=0.833, P=0.714
```

## Approach
1. Candidate thresholds = sorted unique scores (each produces a distinct confusion matrix).
2. For each threshold, predict `score >= t`, compute precision/recall/F1.
3. **Max-F1:** argmax over F1. **Recall target:** filter to thresholds where recall >= target, then among those pick the one with the highest precision (equivalently the highest threshold that still satisfies the constraint).
4. Return the chosen threshold and its metrics so the caller can justify the operating point.

## Python Implementation
```python
import numpy as np


def _prf_at_threshold(y_true, y_scores, t):
    """Precision, recall, F1 when predicting positive iff score >= t."""
    y_pred = (y_scores >= t).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def best_f1_threshold(y_true, y_scores):
    """Scan all candidate thresholds and return the one maximizing F1."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)
    candidates = np.unique(y_scores)

    best = {"threshold": 0.5, "f1": -1, "precision": 0, "recall": 0}
    for t in candidates:
        p, r, f1 = _prf_at_threshold(y_true, y_scores, t)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": f1, "precision": p, "recall": r}
    return best


def threshold_for_recall(y_true, y_scores, recall_target):
    """Highest-precision threshold whose recall still meets the target."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)
    candidates = np.unique(y_scores)

    feasible = []
    for t in candidates:
        p, r, f1 = _prf_at_threshold(y_true, y_scores, t)
        if r >= recall_target:
            feasible.append((t, p, r, f1))
    if not feasible:
        return None  # target unreachable at any threshold
    # Among feasible thresholds, maximize precision (ties -> higher threshold).
    t, p, r, f1 = max(feasible, key=lambda x: (x[1], x[0]))
    return {"threshold": float(t), "precision": p, "recall": r, "f1": f1}


if __name__ == "__main__":
    y_true   = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_scores = np.array([0.2, 0.8, 0.6, 0.55, 0.9, 0.4, 0.3, 0.45, 0.7, 0.25])

    bf = best_f1_threshold(y_true, y_scores)
    print(f"Best-F1 threshold: {bf['threshold']:.2f}  -> "
          f"F1={bf['f1']:.3f}, P={bf['precision']:.3f}, R={bf['recall']:.3f}")

    rt = threshold_for_recall(y_true, y_scores, recall_target=0.80)
    print(f"Recall>=0.80 threshold: {rt['threshold']:.2f} -> "
          f"R={rt['recall']:.3f}, P={rt['precision']:.3f}")
```

## Alternative Solution
Vectorize with sklearn's curve, then pick the operating point analytically:

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
f1 = 2 * precision * recall / (precision + recall + 1e-12)
best_idx = np.argmax(f1[:-1])      # last point has no threshold
best_threshold = thresholds[best_idx]
```

## Interview Variations
- "Optimize expected cost, not F1." Assign dollar costs to FP and FN, then minimize `cost_fp*FP + cost_fn*FN` over thresholds. This is the most production-realistic version.
- "Pick the threshold on a validation set, not the test set." Emphasize that tuning the threshold on test data leaks and inflates results.
- "Hit a precision floor instead of a recall floor." Symmetric: filter on precision >= target, maximize recall.

## Common Follow-up Questions
- **Why not always use 0.5?** 0.5 is only meaningful if the model is calibrated and classes are balanced and costs are symmetric — three assumptions that rarely all hold.
- **Does threshold tuning change AUC?** No — AUC/AP are threshold-independent. Tuning only picks your single operating point on the fixed curve.
- **Where should thresholds be chosen?** On a held-out validation split, then frozen before touching test/production.

---

# Practical Question 6: Manual k-Fold Cross-Validation Loop

**Difficulty:** Medium-Hard
**Estimated Time:** 30 min
**Concepts Tested:** k-fold splitting, stratification, mean±std reporting, avoiding leakage, model comparison, reproducibility

## Problem Statement
Implement k-fold cross-validation **from scratch** (no `cross_val_score`): shuffle indices, partition into k folds, and for each fold train on k−1 folds and evaluate on the held-out fold. Return per-fold scores plus mean and standard deviation. Support stratified folds so class balance is preserved in each fold.

## Example Input
```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
X, y = load_breast_cancer(return_X_y=True)
model_factory = lambda: LogisticRegression(max_iter=5000)
k = 5
```

## Example Output
```
Fold scores: [0.965, 0.956, 0.974, 0.939, 0.982]
CV accuracy: 0.963 +/- 0.014
```

## Approach
1. Build fold assignments. For **plain** k-fold: shuffle all indices and split into k near-equal chunks. For **stratified** k-fold: do the shuffle-and-split *within each class*, then merge, so every fold mirrors the overall class ratio.
2. Loop k times: the i-th fold is the validation set, the rest is training.
3. **Clone/rebuild a fresh model each fold** (via a factory) so state never leaks across folds.
4. Fit on train, score on validation, collect. Report mean ± std — the std tells you how *stable* the model is, which is as important as the mean.

## Python Implementation
```python
import numpy as np
from sklearn.base import clone


def make_folds(y, k=5, stratified=True, seed=42):
    """Return a list of validation-index arrays, one per fold."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = y.size

    if not stratified:
        idx = rng.permutation(n)
        return [f for f in np.array_split(idx, k)]

    # Stratified: split each class's indices into k parts, then zip parts together.
    fold_indices = [[] for _ in range(k)]
    for cls in np.unique(y):
        cls_idx = rng.permutation(np.where(y == cls)[0])
        for i, part in enumerate(np.array_split(cls_idx, k)):
            fold_indices[i].extend(part.tolist())
    return [np.array(f) for f in fold_indices]


def cross_validate_manual(model, X, y, k=5, stratified=True, seed=42, scorer=None):
    """Manual k-fold CV. `model` is any unfitted estimator; it is cloned per fold.

    scorer(model, X_val, y_val) -> float; defaults to accuracy.
    Returns (fold_scores, mean, std).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if scorer is None:
        scorer = lambda m, Xv, yv: np.mean(m.predict(Xv) == yv)

    folds = make_folds(y, k=k, stratified=stratified, seed=seed)
    all_idx = np.arange(y.size)

    scores = []
    for val_idx in folds:
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        fold_model = clone(model)                 # fresh, unfitted copy
        fold_model.fit(X[train_idx], y[train_idx])
        scores.append(scorer(fold_model, X[val_idx], y[val_idx]))

    scores = np.array(scores)
    return scores, scores.mean(), scores.std()


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression

    X, y = load_breast_cancer(return_X_y=True)
    model = LogisticRegression(max_iter=5000)
    fold_scores, mean, std = cross_validate_manual(model, X, y, k=5)

    print("Fold scores:", np.round(fold_scores, 3))
    print(f"CV accuracy: {mean:.3f} +/- {std:.3f}")
```

**Note on leakage:** if your pipeline includes scaling or feature selection, those must be `.fit()` *inside* each fold on the training portion only. The clean way is to pass a `sklearn.pipeline.Pipeline` as `model` so `clone` + `fit` handles it automatically.

## Alternative Solution
Once you've proven you can do it by hand, the production tool is:

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"{scores.mean():.3f} +/- {scores.std():.3f}")
```

## Interview Variations
- "Implement Leave-One-Out CV." k = n; each sample is its own fold. Expensive but low bias.
- "Time-series CV." No shuffling — train on past, validate on future (`TimeSeriesSplit`), because random folds leak the future into the past.
- "Nested CV." Outer loop estimates generalization; inner loop tunes hyperparameters — the correct way to avoid optimistic bias from tuning.

## Common Follow-up Questions
- **Why stratify?** With imbalanced classes, a random fold might contain zero positives, making its score meaningless. Stratification preserves prevalence.
- **What does high std across folds tell you?** The model is unstable / sensitive to the exact training data — often a sign of high variance or too little data.
- **Why clone the model each fold?** So a fold never benefits from another fold's fitted parameters — otherwise every score after the first is contaminated.

---

# PART B — NOTEBOOK WORKFLOW: Model Evaluation & Comparison Playbook

This is the sequence I run for *every* classification project. The narrative arc: load and split honestly, train several models, look at their confusion matrices and reports, compare them on threshold-independent curves (ROC, PR), confirm stability with cross-validation, *then* choose — and only then tune the operating threshold. Paste each cell in order into a Jupyter notebook. Everything is runnable end-to-end.

**The mentor's north star:** never choose a model by accuracy alone. Choose by the metric that maps to the business cost, confirm it's stable across folds, and document *why*.

---

### Cell 1 — Imports and Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.rcParams["figure.figsize"] = (7, 5)
```

We import everything up front so the notebook reads top-to-bottom without surprises. `make_pipeline` + `StandardScaler` matters because SVM and Logistic Regression are scale-sensitive — and putting the scaler *inside* a pipeline is what keeps cross-validation leak-free. Setting a single `RANDOM_STATE` everywhere makes results reproducible, which graders and reviewers expect.

---

### Cell 2 — Load a Classification Dataset

```python
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")   # 0 = malignant, 1 = benign

print("Shape:", X.shape)
print("Class balance:\n", y.value_counts(normalize=True).round(3))
X.head()
```

The Breast Cancer Wisconsin dataset: 569 samples, 30 numeric features, binary target. It's mildly imbalanced (~63% benign). **Crucially, decide which class is "positive" and what an error costs.** Here, missing a malignant tumor (a false negative on the malignant class) is far worse than a false alarm — so recall on malignancy is the metric that should drive decisions. Note sklearn labels malignant as 0; keep that straight when reading precision/recall per class.

---

### Cell 3 — Train-Test Split (Stratified)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

print("Train:", X_train.shape, " Test:", X_test.shape)
print("Train balance:", y_train.value_counts(normalize=True).round(3).to_dict())
print("Test balance :", y_test.value_counts(normalize=True).round(3).to_dict())
```

`stratify=y` guarantees both splits keep the same class ratio — non-negotiable for imbalanced data. The test set is a vault: we do not look at it, tune on it, or pick thresholds on it until the very end. All model selection happens via cross-validation on the training set. This discipline is the difference between an honest generalization estimate and a self-deceiving one.

---

### Cell 4 — Define and Train 3-4 Models

```python
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE
    ),
    "SVM (RBF)": make_pipeline(
        StandardScaler(), SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"Trained: {name}")
```

Three qualitatively different learners: a linear baseline (Logistic Regression), a bagged tree ensemble (Random Forest), and a kernel method (SVM). Logistic Regression and SVM sit inside scaling pipelines; Random Forest is scale-invariant so it doesn't need one. `SVC(probability=True)` enables `predict_proba` (via internal calibration) — required for ROC/PR curves. Comparing diverse model families guards against the trap of assuming your first choice is best.

---

### Cell 5 — Generate Predictions AND Predicted Probabilities

```python
preds = {}
probas = {}
for name, model in models.items():
    preds[name] = model.predict(X_test)
    # Probability of the positive class (label 1 = benign) for curves/thresholds.
    probas[name] = model.predict_proba(X_test)[:, 1]

print("Example predicted probabilities (SVM):", np.round(probas["SVM (RBF)"][:5], 3))
```

Two outputs per model: hard **labels** (for confusion matrices and the classification report) and soft **probabilities** (for ROC/PR curves and threshold tuning). `predict_proba(...)[:, 1]` grabs the positive-class column. Anytime you plan to draw a curve or tune a cutoff, you need probabilities — hard labels have already collapsed the information at the default 0.5 threshold and thrown away the ranking.

---

### Cell 6 — Confusion Matrices + Classification Reports

```python
fig, axes = plt.subplots(1, len(models), figsize=(15, 4))
for ax, (name, y_pred) in zip(axes, preds.items()):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["malignant", "benign"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(name)
plt.tight_layout()
plt.show()

for name, y_pred in preds.items():
    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred,
                                target_names=["malignant", "benign"], digits=3))
```

The confusion matrix is the ground truth of classifier behavior — every other classification metric is derived from its four cells. Read the `classification_report` per class: for this problem, zero in on **recall for the malignant class** (how many cancers did we catch?). A model with 97% accuracy but 88% malignant recall is quietly letting cancers through. This is where "the right metric" beats "the headline metric."

---

### Cell 7 — ROC Curves for All Models on One Axis

```python
plt.figure(figsize=(7, 7))
for name, y_score in probas.items():
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curves — Model Comparison")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

Overlaying ROC curves makes ranking visual: the curve nearest the top-left corner (highest TPR for lowest FPR) is strongest, and AUC summarizes each in one number. AUC is **threshold-independent** — it measures how well the model *ranks* positives above negatives across all cutoffs. Because this dataset is only mildly imbalanced, ROC is a fair comparison here; on a heavily skewed problem I'd lean on the PR curve in the next cell.

---

### Cell 8 — Precision-Recall Curves for All Models

```python
prevalence = np.mean(y_test == 1)

plt.figure(figsize=(7, 7))
for name, y_score in probas.items():
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    plt.plot(recall, precision, lw=2, label=f"{name} (AP = {ap:.3f})")

plt.axhline(prevalence, ls="--", color="k",
            label=f"Baseline (prevalence = {prevalence:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves — Model Comparison")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

PR curves focus on the positive class and ignore the true negatives, so they expose weaknesses that ROC can hide under imbalance. The dashed baseline sits at the positive-class prevalence — anything above it beats random. Average Precision (AP) is the area under the PR curve, the PR-analog of AUC. When positives are rare and precious (fraud, disease), AP is the number I report to stakeholders, not AUC.

---

### Cell 9 — Cross-Validate All Models and Tabulate Mean ± Std

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rows = []
for name, model in models.items():
    # Re-run CV on the *training* set only; use ROC-AUC as a robust ranking metric.
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    rows.append({
        "Model": name,
        "Mean AUC": scores.mean(),
        "Std AUC": scores.std(),
        "Fold scores": np.round(scores, 3),
    })

cv_table = pd.DataFrame(rows).sort_values("Mean AUC", ascending=False)
print(cv_table.to_string(index=False))
```

A single train-test split is one noisy sample of performance; cross-validation averages over five, and the **std** quantifies stability. Note we cross-validate on the *training* set only — the test set stays sealed. Reporting `mean ± std` is the professional standard: a model with 0.985 ± 0.004 is preferable to one at 0.990 ± 0.030 because you can trust it fold to fold. Scale-sensitive models are inside pipelines, so scaling is re-fit per fold with no leakage.

---

### Cell 10 — Pick the Best Model with Justification

```python
best_name = cv_table.iloc[0]["Model"]
best_model = models[best_name]

print(f"Selected model: {best_name}")
print(f"CV Mean AUC: {cv_table.iloc[0]['Mean AUC']:.4f} "
      f"+/- {cv_table.iloc[0]['Std AUC']:.4f}")

# Confirm on the held-out test set (first and only peek for final reporting).
test_auc = roc_auc_score(y_test, probas[best_name])
print(f"Held-out test AUC: {test_auc:.4f}")
```

Model selection is a *decision*, and decisions need a rationale. I pick on cross-validated AUC (highest mean, acceptably low std), then confirm — not select — on the sealed test set. If two models tie on AUC, I break the tie with the metric that matches the business cost (here, malignant recall) and with simplicity/latency (Logistic Regression deploys and explains more easily than an SVM). Write the justification down; "it had the best number" is not a justification, "best stable AUC and highest malignant recall at our operating point" is.

---

### Cell 11 — Threshold Selection Demonstration

```python
y_score = probas[best_name]
precision, recall, thresholds = precision_recall_curve(y_test, y_score)

# F1 per threshold (drop the last precision/recall point, which has no threshold).
f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Default threshold 0.50 -> F1 = "
      f"{f1_score(y_test, (y_score >= 0.5).astype(int)):.3f}")
print(f"Tuned threshold {best_threshold:.3f} -> F1 = {f1_scores[best_idx]:.3f}, "
      f"P = {precision[best_idx]:.3f}, R = {recall[best_idx]:.3f}")

plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.plot(thresholds, f1_scores, label="F1")
plt.axvline(best_threshold, color="k", ls="--", label=f"Best F1 @ {best_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title(f"Threshold Tuning — {best_name}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

The 0.5 cutoff is a default, not a decision. This cell sweeps thresholds and shows precision, recall, and F1 as functions of the cutoff — you can literally see the trade-off. For a cancer screen we might instead pick the lowest threshold that keeps malignant recall above, say, 0.98, accepting more false positives to miss fewer cancers. In production you'd select this threshold on a validation split, not the test set; it's shown on test here purely for the demonstration.

---

### Cell 12 — Regression Metrics Mini-Section

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load a regression dataset and split.
Xr, yr = fetch_california_housing(return_X_y=True)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    Xr, yr, test_size=0.25, random_state=RANDOM_STATE
)

reg = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
reg.fit(Xr_tr, yr_tr)
yr_pred = reg.predict(Xr_te)

mae  = mean_absolute_error(yr_te, yr_pred)
mse  = mean_squared_error(yr_te, yr_pred)
rmse = np.sqrt(mse)
r2   = r2_score(yr_te, yr_pred)

# Adjusted R2 penalizes free parameters: 1 - (1-R2)(n-1)/(n-p-1)
n, p = Xr_te.shape
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE         : {mae:.3f}")
print(f"MSE         : {mse:.3f}")
print(f"RMSE        : {rmse:.3f}")
print(f"R2          : {r2:.3f}")
print(f"Adjusted R2 : {adj_r2:.3f}")
```

Classification metrics don't transfer to regression, so here's the parallel toolkit on California Housing. **MAE** is the average error in target units (interpretable, outlier-robust). **RMSE** punishes big misses harder (outlier-sensitive) and is what you report when large errors are especially costly. **R²** is the fraction of variance explained; **Adjusted R²** discounts it for the number of predictors so you can't inflate the score by dumping in junk features. Report MAE *and* RMSE together — if RMSE >> MAE, a few large errors are dominating.

---

### Cell 13 — Final Interpretation and Recommendation

```python
print("SUMMARY")
print("-" * 60)
print(cv_table[["Model", "Mean AUC", "Std AUC"]].to_string(index=False))
print("-" * 60)
print(f"Recommended model : {best_name}")
print(f"Operating threshold: {best_threshold:.3f} (tuned for F1)")
print(f"Held-out test AUC : {test_auc:.4f}")
print("\nReasoning: selected on stable cross-validated AUC; threshold shifted")
print("off 0.5 to balance precision/recall for the malignant class, where a")
print("false negative (missed cancer) is the costly error.")
```

Close every evaluation with a plain-language recommendation a non-ML stakeholder can act on: which model, which threshold, and *why* — grounded in the business cost of each error type. The evaluation pipeline you just walked — honest split, diverse models, confusion matrices, ROC/PR comparison, cross-validated stability, deliberate model choice, and threshold tuning — is exactly what a lab exam or interviewer wants to see you reason through. Memorize the *order* and the *justifications*, not just the function names, and you'll handle any variation they throw at you.

---

## Quick Reference: Which Metric When

| Situation | Metric to optimize | Why |
|---|---|---|
| Balanced classes, symmetric costs | Accuracy / AUC | Simple, fair when classes and costs are even |
| Imbalanced, positives rare & precious | Precision-Recall / AP, F1 | ROC/accuracy hide false positives under imbalance |
| False negatives are dangerous (disease, fraud) | Recall (with a floor) | Missing a positive is the costly error |
| False positives are expensive (spam, alerts) | Precision | A false alarm wastes resources / trust |
| Need one balanced number | F1 | Harmonic mean punishes P/R imbalance |
| Ranking quality, threshold-agnostic | ROC-AUC / AP | Judges the ordering, not a single cutoff |
| Regression, outliers matter | RMSE | Quadratic penalty on large errors |
| Regression, robust average error | MAE | Linear, interpretable, outlier-tolerant |
| Regression, variance explained | R² / Adjusted R² | Adjusted version penalizes extra predictors |

The single most important habit: **state the business cost of each error type first, choose the metric that reflects it, then let that metric drive model and threshold selection.**
