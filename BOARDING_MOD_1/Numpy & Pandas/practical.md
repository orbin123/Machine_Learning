# NumPy & Pandas — Practical & Coding Assessment Guide

> Companion to `theory.md`. This document predicts the coding questions an interviewer or lab examiner is likely to ask on **NumPy, Pandas, Matplotlib, Seaborn, Summary Statistics, Distribution Analysis, Correlation/Covariance, and EDA**, with complete, production-quality solutions, complexity analysis, alternatives, variations, and follow-ups.
>
> **How to practice:** cover the solution, attempt the problem on paper or in a notebook, then compare. Focus on being able to *explain* your approach and its complexity — assessors probe reasoning.

## Table of Contents

- [Part A — NumPy Coding Questions](#part-a--numpy-coding-questions)
- [Part B — Pandas Coding Questions](#part-b--pandas-coding-questions)
- [Part C — Visualization Tasks (Matplotlib & Seaborn)](#part-c--visualization-tasks-matplotlib--seaborn)
- [Part D — Statistics: Summary, Distribution, Correlation, Covariance](#part-d--statistics-summary-distribution-correlation-covariance)
- [Part E — Notebook Workflows (End-to-End)](#part-e--notebook-workflows-end-to-end)
- [Part F — Rapid-Fire Coding Snippets & Gotchas](#part-f--rapid-fire-coding-snippets--gotchas)

---

# Part A — NumPy Coding Questions

## Practical Question 1 — Normalize / Standardize an Array

**Difficulty:** Easy
**Estimated Time:** 5 min
**Concepts Tested:** Vectorization, broadcasting, axis-aware reductions

**Problem Statement**
Given a 2-D array of shape `(n_samples, n_features)`, standardize each **column** (feature) to zero mean and unit variance (z-score scaling), without any Python loops.

**Example Input**
```
X = [[1, 100],
     [2, 200],
     [3, 300]]
```
**Example Output**
```
[[-1.2247, -1.2247],
 [ 0.,      0.    ],
 [ 1.2247,  1.2247]]
```

**Approach**
1. Compute the per-column mean with `axis=0` (collapses rows → one value per feature).
2. Compute the per-column standard deviation the same way.
3. Subtract the mean and divide by the std; broadcasting aligns the `(n_features,)` stats against the `(n_samples, n_features)` matrix.
4. Guard against divide-by-zero for constant columns.

## Python Implementation
```python
import numpy as np

def standardize(X: np.ndarray) -> np.ndarray:
    """Z-score standardize each column of X to mean 0, std 1."""
    X = np.asarray(X, dtype=float)          # ensure float, avoid int truncation
    mean = X.mean(axis=0)                    # shape (n_features,)
    std = X.std(axis=0)                       # population std (ddof=0)
    std_safe = np.where(std == 0, 1.0, std)   # avoid /0 on constant columns
    return (X - mean) / std_safe              # broadcasting does the work

X = np.array([[1, 100], [2, 200], [3, 300]])
print(standardize(X))
```
**Key lines explained:**
- `X.mean(axis=0)` / `X.std(axis=0)` — naming `axis=0` reduces *down the rows*, giving one statistic per column.
- `np.where(std == 0, 1.0, std)` — replaces zero std (constant column) with 1 so we divide by 1 instead of 0, leaving those columns at 0 after centering.
- `(X - mean) / std_safe` — pure broadcasting; no loop, one fused vectorized pass.

**Complexity:** Time **O(n·m)** (each element visited a constant number of times); Space **O(n·m)** for the output (plus O(m) for the stats).

## Alternative Solution
Use scikit-learn's `StandardScaler` in an ML pipeline (it stores the fitted mean/std to apply to test data, preventing leakage):
```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

## Interview Variations
- Standardize **rows** instead of columns (`axis=1`, reshape stats to `(n,1)`).
- Min-max scale to `[0,1]`: `(X - X.min(0)) / (X.max(0) - X.min(0))`.
- Robust scale using median and IQR instead of mean/std.

## Common Follow-up Questions
- *Why `axis=0` and not `axis=1`?* — Features are columns; we want per-feature stats.
- *Why guard divide-by-zero?* — A constant column has std 0; dividing yields NaN/inf.
- *Population vs sample std here?* — For scaling it rarely matters; be explicit with `ddof`.

---

## Practical Question 2 — Replace Values Conditionally Without a Loop

**Difficulty:** Easy
**Estimated Time:** 5 min
**Concepts Tested:** Boolean masking, `np.where`, vectorization

**Problem Statement**
Given a 1-D array, replace every negative value with 0 and cap every value above 100 at 100, in a single vectorized expression.

**Example Input** `[-5, 20, 150, 0, 99, -1]`
**Example Output** `[0, 20, 100, 0, 99, 0]`

**Approach**
Use `np.clip` for the natural bounds, or nested `np.where`. `clip` directly expresses "bound between low and high."

## Python Implementation
```python
import numpy as np

a = np.array([-5, 20, 150, 0, 99, -1])

# Cleanest: clip to [0, 100]
result = np.clip(a, 0, 100)             # [0 20 100 0 99 0]

# Equivalent with np.where (shows the general conditional pattern)
result2 = np.where(a < 0, 0, np.where(a > 100, 100, a))
```
**Key lines:** `np.clip(a, lo, hi)` is a single C loop that floors at `lo` and ceils at `hi`. The `np.where` version demonstrates element-wise conditional selection when the logic isn't a simple bound.

**Complexity:** Time **O(n)**, Space **O(n)**.

## Alternative Solution
In-place boolean-mask assignment (avoids a new array):
```python
a[a < 0] = 0
a[a > 100] = 100
```

## Interview Variations
- Replace values *equal to* a sentinel (e.g. −999) with NaN: `np.where(a == -999, np.nan, a)` (needs float dtype).
- Keep values in a range, zero out the rest.

## Common Follow-up Questions
- *`np.where` vs Python `if/else` loop?* — `where` is vectorized C; the loop is slow.
- *Why does replacing with `np.nan` require float?* — NaN only exists in floating-point dtypes.

---

## Practical Question 3 — Broadcasting: Pairwise Euclidean Distance Matrix

**Difficulty:** Medium
**Estimated Time:** 12 min
**Concepts Tested:** Broadcasting, `newaxis`, reductions, avoiding loops

**Problem Statement**
Given `A` of shape `(m, d)` and `B` of shape `(n, d)` (each row a point in d-dimensional space), compute the `(m, n)` matrix of Euclidean distances between every point in A and every point in B — no Python loops.

**Example Input**
```
A = [[0, 0], [1, 1]]      # 2 points
B = [[0, 0], [0, 1], [1, 0]]  # 3 points
```
**Example Output** (shape 2×3)
```
[[0.,    1.,    1.   ],
 [1.414, 1.,    1.   ]]
```

**Approach**
1. Insert a new axis so `A` becomes `(m, 1, d)` and `B` becomes `(1, n, d)`.
2. Their difference broadcasts to `(m, n, d)` — every A point minus every B point.
3. Square, sum over the last axis (`d`), and square-root → `(m, n)` distances.

## Python Implementation
```python
import numpy as np

def pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distance between every row of A and every row of B."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # (m,1,d) - (1,n,d) -> (m,n,d) via broadcasting
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))     # reduce over d

A = np.array([[0, 0], [1, 1]])
B = np.array([[0, 0], [0, 1], [1, 0]])
print(pairwise_distances(A, B))
```
**Key lines:**
- `A[:, np.newaxis, :]` reshapes `(m,d)`→`(m,1,d)`; `B[np.newaxis,:,:]` → `(1,n,d)`.
- The subtraction broadcasts the size-1 axes to `(m,n,d)` — the full grid of coordinate differences, computed without loops (stride-0 trick, no copies of A/B).
- `.sum(axis=2)` collapses the coordinate axis; `np.sqrt` gives distances.

**Complexity:** Time **O(m·n·d)** (unavoidable — that many distances); Space **O(m·n·d)** for the intermediate `diff`. For large m,n this temporary is the bottleneck.

## Alternative Solution
The `(a-b)² = a² + b² − 2ab` identity avoids the big 3-D temporary:
```python
def pairwise_distances_fast(A, B):
    A2 = (A**2).sum(1)[:, None]       # (m,1)
    B2 = (B**2).sum(1)[None, :]       # (1,n)
    cross = A @ B.T                    # (m,n)
    return np.sqrt(np.maximum(A2 + B2 - 2*cross, 0))
```
Uses BLAS matrix multiply, O(m·n·d) time but only O(m·n) extra memory — the standard trick. `np.maximum(...,0)` guards tiny negatives from float error. `scipy.spatial.distance.cdist(A,B)` is the library one-liner.

## Interview Variations
- Manhattan (L1) distance: `np.abs(diff).sum(axis=2)`.
- Distance of every point to a *single* reference point (simpler broadcasting).
- Find each A point's nearest B point: `dist.argmin(axis=1)`.

## Common Follow-up Questions
- *What's the memory cost for m=n=10,000, d=100?* — The 3-D temp is 10⁴·10⁴·100·8 bytes ≈ 80 GB → must use the a²+b²−2ab trick.
- *Why does broadcasting avoid copying A and B?* — Size-1 axes use stride 0; only the output is materialized.

---

## Practical Question 4 — Reshape & Aggregate (Reduce Along an Axis)

**Difficulty:** Medium
**Estimated Time:** 8 min
**Concepts Tested:** Reshaping, axis reductions, `-1` inference

**Problem Statement**
You have a flat array of 24 monthly readings (2 years × 12 months). Compute the average reading per month across the two years, and the total per year.

**Example Input** `np.arange(1, 25)` (values 1..24)
**Example Output**
- Per-month average across years: shape `(12,)`
- Per-year total: shape `(2,)`

**Approach**
Reshape the flat 24-vector into `(2, 12)` (year × month), then reduce along the appropriate axis: `axis=0` averages across years (per month), `axis=1` sums across months (per year).

## Python Implementation
```python
import numpy as np

data = np.arange(1, 25)                 # 24 readings
by_year_month = data.reshape(2, -1)     # (2, 12); -1 infers 12

monthly_avg = by_year_month.mean(axis=0)  # (12,) average of the two years
yearly_total = by_year_month.sum(axis=1)  # (2,)  total per year

print(monthly_avg)     # month-by-month mean
print(yearly_total)    # [78, 222]
```
**Key lines:**
- `reshape(2, -1)` — `-1` infers the second dimension as 24/2 = 12; returns a view (no copy).
- `mean(axis=0)` collapses the *year* axis → one mean per month.
- `sum(axis=1)` collapses the *month* axis → one total per year.

**Complexity:** Time **O(n)**; Space **O(1)** extra (reshape is a view; reductions output small arrays).

## Alternative Solution
If data were in a DataFrame with a datetime index, `df.resample('M').mean()` or `groupby(df.index.month)` would express this at a higher level.

## Interview Variations
- 3-D reshape: `(years, months, days)` and reduce over multiple axes.
- Use `keepdims=True` to subtract the monthly mean back from each year (broadcasting).

## Common Follow-up Questions
- *Why is reshape usually free?* — It only changes shape/strides over the same buffer (a view).
- *What if the total size doesn't divide evenly?* — `reshape` raises; you can't invent/drop elements.

---

## Practical Question 5 — Views vs Copies Trap

**Difficulty:** Medium
**Estimated Time:** 6 min
**Concepts Tested:** Views vs copies, slicing semantics

**Problem Statement**
Explain and fix the bug: a function meant to return a modified copy accidentally mutates the caller's array.
```python
def zero_first_row(mat):
    m = mat[0:1]      # intended "copy" of first row
    m[:] = 0
    return mat
```

**Approach**
`mat[0:1]` is a **slice**, which returns a *view* sharing memory with `mat`. Writing `m[:] = 0` therefore zeros the original's first row. Fix by taking an explicit copy.

## Python Implementation
```python
import numpy as np

def zero_first_row(mat: np.ndarray) -> np.ndarray:
    out = mat.copy()      # independent buffer
    out[0, :] = 0
    return out

a = np.ones((3, 3))
b = zero_first_row(a)
print(a[0].sum())   # 3.0 -> original untouched
print(b[0].sum())   # 0.0 -> only the copy changed
```
**Key line:** `mat.copy()` allocates a fresh buffer so subsequent writes don't touch the caller's data. Basic slicing (`mat[0:1]`) is a view; boolean/fancy indexing returns a copy — knowing which is essential.

**Complexity:** Time/Space **O(n·m)** for the copy.

## Interview Variations
- Show that `arr[arr>0] = 0` (boolean mask) mutates in place but `arr[arr>0]` (read) is a copy.
- Detect a view: `b.base is a`.

## Common Follow-up Questions
- *Which indexing returns views vs copies?* — Basic slicing → view; fancy/boolean → copy.
- *Why does NumPy default to views?* — Zero-copy performance; copies are opt-in.

---

# Part B — Pandas Coding Questions

## Practical Question 6 — Filter, Select, and Assign Safely

**Difficulty:** Easy
**Estimated Time:** 6 min
**Concepts Tested:** Boolean filtering, `.loc`, avoiding `SettingWithCopyWarning`

**Problem Statement**
Given an employees DataFrame, (a) select the names of employees in `Eng` earning over 80k, and (b) give everyone older than 30 a `'Senior'` level — without triggering a copy warning.

**Example Input**
```
name   age  dept  salary
Ana    23   HR    50
Ben    35   Eng   90
Cara   29   Eng   85
Dan    41   HR    60
```
**Example Output (a)** `['Ben', 'Cara']`

**Approach**
Use a parenthesized boolean mask with `&` for the filter, and a single `.loc[mask, col]` assignment to update in place safely.

## Python Implementation
```python
import pandas as pd

df = pd.DataFrame({
    'name':   ['Ana', 'Ben', 'Cara', 'Dan'],
    'age':    [23, 35, 29, 41],
    'dept':   ['HR', 'Eng', 'Eng', 'HR'],
    'salary': [50, 90, 85, 60],
})

# (a) select with a boolean mask
mask = (df['dept'] == 'Eng') & (df['salary'] > 80)
names = df.loc[mask, 'name'].tolist()        # ['Ben', 'Cara']

# (b) safe conditional assignment (single combined indexer)
df.loc[df['age'] > 30, 'level'] = 'Senior'
df['level'] = df['level'].fillna('Junior')
```
**Key lines:**
- `(cond1) & (cond2)` — element-wise AND; each condition parenthesized because `&` binds tighter than `>`/`==`.
- `df.loc[mask, 'name']` — selects rows and a column in one operation → a Series; `.tolist()` for a Python list.
- `df.loc[df['age']>30, 'level'] = 'Senior'` — one indexer writes directly into `df`, so no `SettingWithCopyWarning` (a chained `df[df.age>30]['level']=...` would silently fail).

**Complexity:** Time **O(n)** to scan/assign; Space **O(n)** for the mask.

## Alternative Solution
`df.query('dept == "Eng" and salary > 80')['name']` for the filter; `np.where(df.age>30,'Senior','Junior')` for the assignment.

## Interview Variations
- Filter with `isin(['Eng','Sales'])` or `between(25, 40)`.
- Multi-condition with `|` (OR) and `~` (NOT).

## Common Follow-up Questions
- *Why `&` not `and`?* — `and` needs a single truth value; a Series is ambiguous.
- *Why the parentheses?* — Operator precedence: `&`/`|` bind tighter than comparisons.
- *Why `.loc` for assignment?* — Avoids chained-indexing writes to a throwaway copy.

---

## Practical Question 7 — GroupBy Aggregation (Split-Apply-Combine)

**Difficulty:** Medium
**Estimated Time:** 10 min
**Concepts Tested:** `groupby`, `agg`, named aggregations, `reset_index`

**Problem Statement**
From a sales DataFrame (`region`, `product`, `revenue`, `units`), produce per-region: total revenue, average revenue per order, order count, and total units — as a tidy DataFrame with clean column names.

**Example Output**
```
region  total_rev  avg_rev  n_orders  total_units
East    ...        ...      ...       ...
West    ...        ...      ...       ...
```

**Approach**
Group by `region` and use **named aggregations** (`NamedAgg`) so output columns are explicit and readable; `reset_index` turns the group key back into a column.

## Python Implementation
```python
import pandas as pd

sales = pd.DataFrame({
    'region':  ['East','East','West','West','East'],
    'product': ['A','B','A','C','C'],
    'revenue': [100, 150, 200, 80, 120],
    'units':   [1, 2, 2, 1, 1],
})

summary = (
    sales.groupby('region')
         .agg(
             total_rev  =('revenue', 'sum'),
             avg_rev    =('revenue', 'mean'),
             n_orders   =('revenue', 'size'),
             total_units=('units',   'sum'),
         )
         .reset_index()               # region back to a column
         .sort_values('total_rev', ascending=False)
)
print(summary)
```
**Key lines:**
- `groupby('region')` — split rows into per-region groups.
- `.agg(name=('col','func'))` — named aggregation: each output column is `(source_column, function)`, giving clean, explicit names in one pass.
- `size` counts rows per group (includes NaN); `count` would exclude NaN.
- `reset_index()` converts the `region` index back into a regular column for a tidy result.

**Complexity:** Time **O(n)** for built-in aggregations (Cython); Space **O(g)** for g groups.

## Alternative Solution
`pivot_table` for a cross-tab of region × product:
```python
sales.pivot_table(index='region', columns='product',
                  values='revenue', aggfunc='sum', fill_value=0)
```

## Interview Variations
- Group by **multiple keys**: `groupby(['region','product'])`.
- `transform('mean')` to add a per-group average back to each row.
- `filter(lambda g: g.revenue.sum() > 250)` to keep only large-revenue regions.

## Common Follow-up Questions
- *`agg` vs `transform`?* — `agg` reduces to one row per group; `transform` returns input-shaped, aligned to rows.
- *Why is `mean` fast but `apply(lambda)` slow?* — Built-ins run in Cython; `apply` runs Python per group.
- *`size` vs `count`?* — `size` counts all rows; `count` excludes NaN.

---

## Practical Question 8 — Handle Missing Values Strategically

**Difficulty:** Medium
**Estimated Time:** 10 min
**Concepts Tested:** `isna`, imputation strategy, median vs mean, indicator features

**Problem Statement**
A DataFrame has numeric `age` (5% missing, skewed), numeric `income` (missing, has outliers), and categorical `city` (missing). Impute each appropriately and add a missingness indicator for `income`. Explain each choice.

**Approach**
- `age`: median (robust to skew).
- `income`: median (robust to outliers) + a `income_missing` flag (missingness may be informative).
- `city`: mode (most frequent category).

## Python Implementation
```python
import numpy as np, pandas as pd

df = pd.DataFrame({
    'age':    [25, np.nan, 40, 33, np.nan, 29],
    'income': [50_000, 60_000, np.nan, 1_000_000, 55_000, np.nan],
    'city':   ['NYC', 'LA', None, 'NYC', None, 'SF'],
})

# 1) Inspect missingness first
print(df.isna().mean())        # fraction missing per column

# 2) Flag income missingness BEFORE imputing (it may carry signal)
df['income_missing'] = df['income'].isna().astype(int)

# 3) Impute
df['age']    = df['age'].fillna(df['age'].median())      # robust to skew
df['income'] = df['income'].fillna(df['income'].median())# robust to outliers
df['city']   = df['city'].fillna(df['city'].mode()[0])   # most frequent

assert df.isna().sum().sum() == 0
```
**Key lines:**
- `df.isna().mean()` — fraction missing per column, the basis for the strategy.
- `df['income'].isna().astype(int)` — creates a 0/1 indicator *before* filling, preserving the "was missing" signal.
- `fillna(median)` for numeric (robust), `fillna(mode()[0])` for categorical (`mode()` returns a Series; `[0]` takes the top value).

**Complexity:** Time **O(n)** per column; Space **O(n)** for the new columns.

## Alternative Solution
`sklearn.impute.SimpleImputer(strategy='median')` inside a `ColumnTransformer` — crucial in ML because it *fits on train* and *applies to test*, preventing data leakage. For richer imputation, `KNNImputer` or `IterativeImputer`.

## Interview Variations
- Forward-fill (`ffill`) or interpolate for a **time series**.
- Drop rows/columns with `dropna(thresh=...)` when missingness is extreme.
- Group-wise imputation: fill `income` with the median *per city*.

## Common Follow-up Questions
- *Why median not mean here?* — Skewed/outlier data; median is robust.
- *Why impute after the train/test split?* — Computing the median on all data leaks test info.
- *Why an indicator column?* — Missingness itself can be predictive (informative missingness).

---

## Practical Question 9 — Merge / Join Two DataFrames

**Difficulty:** Medium
**Estimated Time:** 10 min
**Concepts Tested:** `merge`, join types, key alignment, handling unmatched rows

**Problem Statement**
Given `orders` (`order_id`, `customer_id`, `amount`) and `customers` (`customer_id`, `name`, `city`), produce a table of every order with the customer's name and city. Then produce a list of customers with **no** orders.

**Approach**
- Order+customer table: inner or left join on `customer_id`.
- Customers with no orders: left join customers→orders and keep rows where the order side is null (or use an anti-join via `indicator`).

## Python Implementation
```python
import pandas as pd

orders = pd.DataFrame({
    'order_id':    [1, 2, 3, 4],
    'customer_id': [10, 10, 20, 99],   # 99 has no customer record
    'amount':      [100, 50, 200, 75],
})
customers = pd.DataFrame({
    'customer_id': [10, 20, 30],       # 30 has no orders
    'name':        ['Ana', 'Ben', 'Cara'],
    'city':        ['NYC', 'LA', 'SF'],
})

# Every order enriched with customer info (LEFT keeps all orders)
enriched = orders.merge(customers, on='customer_id', how='left')

# Customers with NO orders (anti-join)
merged = customers.merge(orders, on='customer_id', how='left', indicator=True)
no_orders = merged.loc[merged['_merge'] == 'left_only', ['customer_id', 'name']]
```
**Key lines:**
- `merge(..., on='customer_id', how='left')` — matches rows by key; `how='left'` keeps every order even if the customer is missing (name/city become NaN for id 99).
- `indicator=True` adds a `_merge` column (`both`/`left_only`/`right_only`); filtering `left_only` yields the anti-join — customers with no matching orders.

**Complexity:** Time **O(n + m)** average (hash join); Space **O(n + m)**.

## Alternative Solution
`pd.merge(a, b, how='inner')` for only matched rows; `~customers.customer_id.isin(orders.customer_id)` as a simpler anti-join mask.

## Interview Variations
- `how='inner'` vs `'left'` vs `'right'` vs `'outer'` — what each keeps.
- Join on differently-named keys: `left_on`/`right_on`.
- Concatenate (`pd.concat`) stacked DataFrames vs merge (key-based).

## Common Follow-up Questions
- *Difference between merge and concat?* — merge = key-based SQL join; concat = stack along an axis.
- *What causes row duplication after a merge?* — Many-to-many keys multiply rows.
- *How do you detect unmatched keys?* — `indicator=True` / `isin`.

---

## Practical Question 10 — Top-N per Group

**Difficulty:** Hard
**Estimated Time:** 12 min
**Concepts Tested:** `groupby`, sorting, ranking, `nlargest` per group

**Problem Statement**
For each department, return the top 2 highest-paid employees.

**Approach**
Group by department and apply `nlargest` on salary within each group; or use `sort_values` + `groupby().head`. Ranking with `groupby().rank` is a third route.

## Python Implementation
```python
import pandas as pd

df = pd.DataFrame({
    'name':   ['A','B','C','D','E','F'],
    'dept':   ['Eng','Eng','Eng','HR','HR','HR'],
    'salary': [90, 85, 95, 60, 70, 65],
})

# Method 1: sort then take head per group (clear + fast)
top2 = (df.sort_values('salary', ascending=False)
          .groupby('dept')
          .head(2)
          .sort_values(['dept', 'salary'], ascending=[True, False]))

# Method 2: group-wise nlargest via apply
top2_alt = (df.groupby('dept', group_keys=False)
              .apply(lambda g: g.nlargest(2, 'salary')))
```
**Key lines:**
- **Method 1** sorts the whole frame by salary descending once, then `groupby('dept').head(2)` keeps the first 2 rows of each group — which, because it's pre-sorted, are the top 2. Efficient and readable.
- **Method 2** applies `nlargest(2,'salary')` inside each group; `group_keys=False` avoids adding an extra index level.

**Complexity:** Method 1 **O(n log n)** (one global sort). Method 2 is O(n log k) per group but pays Python `apply` overhead — Method 1 is usually faster.

## Alternative Solution
Ranking:
```python
df['rk'] = df.groupby('dept')['salary'].rank(method='first', ascending=False)
top2 = df[df['rk'] <= 2]
```

## Interview Variations
- Top-N by a *composite* key (dept + gender).
- The single highest per group (`idxmax` / `nlargest(1)`).
- Bottom-N with `nsmallest`.

## Common Follow-up Questions
- *Why is sort-then-head efficient?* — One sort amortizes across all groups.
- *How to break ties deterministically?* — `rank(method='first')` or add a tiebreaker column.
- *Difference between `rank` methods?* — `first`, `dense`, `min`, `average` handle ties differently.

---

## Practical Question 11 — Time-Series Resampling & Rolling

**Difficulty:** Hard
**Estimated Time:** 12 min
**Concepts Tested:** Datetime index, `resample`, `rolling`, `ffill`

**Problem Statement**
Given daily stock prices with some missing days, compute the monthly average price and a 7-day rolling mean (smoothing).

**Approach**
Parse dates, set a `DatetimeIndex`, `resample('M').mean()` for monthly aggregation, and `rolling(7).mean()` for the moving average. Forward-fill gaps if needed.

## Python Implementation
```python
import pandas as pd, numpy as np

dates = pd.date_range('2024-01-01', periods=90, freq='D')
prices = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 90)),
                   index=dates, name='price')

# Monthly average (downsampling)
monthly_avg = prices.resample('ME').mean()      # 'ME' = month-end

# 7-day rolling mean (smoothing), require full window
rolling_7 = prices.rolling(window=7, min_periods=7).mean()

# Fill occasional gaps by carrying last value forward
filled = prices.asfreq('D').ffill()
```
**Key lines:**
- `resample('ME')` groups the datetime index into month-end buckets; `.mean()` aggregates each — the time-series analogue of `groupby`.
- `rolling(window=7).mean()` computes the mean of each 7-day trailing window; `min_periods=7` yields NaN until a full window exists.
- `asfreq('D').ffill()` regularizes to daily frequency and forward-fills missing days.

**Complexity:** `resample`/`rolling` are **O(n)** (rolling uses an efficient windowed algorithm); Space **O(n)**.

## Alternative Solution
`prices.ewm(span=7).mean()` for an exponentially-weighted moving average (weights recent points more). `groupby(prices.index.to_period('M')).mean()` as a resample alternative.

## Interview Variations
- Rolling std / rolling max instead of mean.
- Upsample (daily→hourly) with interpolation.
- Compute month-over-month percentage change: `monthly_avg.pct_change()`.

## Common Follow-up Questions
- *`resample` vs `groupby`?* — `resample` is groupby specialized for time buckets on a datetime index.
- *`rolling` vs `ewm`?* — rolling weights the window equally; ewm decays weights exponentially.
- *Why `min_periods`?* — Controls whether partial leading windows produce a value or NaN.

---

## Practical Question 12 — Apply vs Vectorize (Performance)

**Difficulty:** Medium
**Estimated Time:** 8 min
**Concepts Tested:** Vectorization, avoiding `apply`/`iterrows`, `np.select`

**Problem Statement**
Add a `category` column: `'low'` if `score < 50`, `'mid'` if `50 ≤ score < 80`, else `'high'`. Do it without row-wise `apply`.

## Python Implementation
```python
import numpy as np, pandas as pd

df = pd.DataFrame({'score': [30, 55, 80, 95, 49]})

# Vectorized multi-condition mapping
conditions = [df['score'] < 50,
              df['score'] < 80]          # evaluated in order
choices    = ['low', 'mid']
df['category'] = np.select(conditions, choices, default='high')
```
**Key lines:**
- `np.select(conditions, choices, default)` — evaluates each boolean condition in order and picks the corresponding choice for the first True, else the default; fully vectorized (C speed), unlike `df.apply(func, axis=1)` which runs Python per row.
- Order matters: the first matching condition wins, so `< 50` is checked before `< 80`.

**Complexity:** Time **O(n)** vectorized; `apply(axis=1)` would also be O(n) but with large Python-interpreter constant factors (often 50–100× slower).

## Alternative Solution
`pd.cut(df['score'], bins=[-np.inf,50,80,np.inf], labels=['low','mid','high'])` — purpose-built for binning a continuous variable into ordered categories.

## Interview Variations
- Bin ages into groups with `pd.cut` / `pd.qcut` (quantile bins).
- Map a categorical column via a dict with `.map`.

## Common Follow-up Questions
- *Why avoid `apply(axis=1)`?* — It's a Python loop over rows; vectorized ops are far faster.
- *`np.select` vs `np.where`?* — `where` handles one condition; `select` handles many.
- *`cut` vs `qcut`?* — `cut` uses fixed value edges; `qcut` uses equal-frequency quantiles.

---

# Part C — Visualization Tasks (Matplotlib & Seaborn)

## Practical Question 13 — Build a Labeled Multi-Series Line Plot

**Difficulty:** Easy
**Estimated Time:** 8 min
**Concepts Tested:** Matplotlib OO interface, titles/labels/legend/grid

**Problem Statement**
Plot two time series on one figure with a title, axis labels, a legend, gridlines, and a set figure size; save to PNG at 300 dpi.

## Python Implementation
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y1, y2 = np.sin(x), np.cos(x)

fig, ax = plt.subplots(figsize=(9, 4))          # explicit Figure + Axes
ax.plot(x, y1, label='sin(x)', color='steelblue', linewidth=2)
ax.plot(x, y2, label='cos(x)', color='darkorange', linestyle='--')
ax.set_title('Sine vs Cosine')
ax.set_xlabel('x'); ax.set_ylabel('value')
ax.legend(loc='upper right')                     # uses the label= values
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('trig.png', dpi=300, bbox_inches='tight')
plt.show()
```
**Key lines:**
- `fig, ax = plt.subplots(figsize=...)` — the recommended object-oriented entry point; `ax` is the plot you configure.
- `label=` on each `plot` feeds `ax.legend()`.
- `set_title`/`set_xlabel`/`set_ylabel` add context; `grid(alpha=0.3)` adds subtle gridlines; `tight_layout` prevents clipping; `savefig(dpi=300, bbox_inches='tight')` exports crisply.

## Interview Variations
- 2×2 subplot grid (`plt.subplots(2, 2)`), addressing `axes[i, j]`.
- Add annotations (`ax.annotate`) or a secondary y-axis (`ax.twinx()`).

## Common Follow-up Questions
- *Figure vs Axes?* — Figure is the canvas; Axes is a single plot within it.
- *pyplot vs OO?* — OO is explicit about which Axes you modify; preferred for multi-panel.

---

## Practical Question 14 — Distribution & Category Plots (Histogram + Bar)

**Difficulty:** Easy
**Estimated Time:** 8 min
**Concepts Tested:** Histogram vs bar plot, bins, `value_counts`

**Problem Statement**
Given a numeric `age` column and a categorical `dept` column: draw a histogram of ages (choosing sensible bins) and a bar chart of counts per department.

## Python Implementation
```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Histogram: distribution of a CONTINUOUS variable
ax1.hist(df['age'], bins=20, color='steelblue', edgecolor='white')
ax1.set_title('Age Distribution'); ax1.set_xlabel('age'); ax1.set_ylabel('count')

# Bar: counts across CATEGORIES
counts = df['dept'].value_counts()
ax2.bar(counts.index, counts.values, color='seagreen')
ax2.set_title('Employees per Department'); ax2.set_ylabel('count')

fig.tight_layout(); plt.show()
```
**Key lines:**
- `ax1.hist(..., bins=20)` — bins a continuous variable; bar heights are counts per range. Bin count controls granularity (too few hides structure, too many is noisy).
- `df['dept'].value_counts()` returns category frequencies; `ax2.bar(index, values)` compares categories with gapped bars.

## Interview Variations
- Overlaid/normalized histograms per group (`density=True`).
- Horizontal bar (`barh`) for long category names.

## Common Follow-up Questions
- *Histogram vs bar plot?* — Histogram = distribution of one continuous variable (adjacent bars); bar = comparison across categories (gapped bars).
- *How to choose bins?* — Rules like Sturges/Freedman–Diaconis, or try a few and inspect.

---

## Practical Question 15 — Seaborn: Box Plot, Correlation Heatmap, Pair Plot

**Difficulty:** Medium
**Estimated Time:** 10 min
**Concepts Tested:** Seaborn statistical plots, tidy data, correlation visualization

**Problem Statement**
For a dataset with several numeric features and one categorical group: (a) compare a numeric feature's distribution across groups with a box plot, (b) visualize the correlation matrix as an annotated heatmap, (c) show all pairwise relationships with a pair plot colored by group.

## Python Implementation
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='whitegrid')

# (a) Box plot: distribution of 'salary' per 'dept', shows spread + outliers
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x='dept', y='salary')
plt.title('Salary by Department'); plt.show()

# (b) Correlation heatmap of numeric features
corr = df.corr(numeric_only=True)
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=.5)
plt.title('Feature Correlations'); plt.show()

# (c) Pair plot: all pairwise scatters + diagonal distributions, colored by group
sns.pairplot(df, hue='dept', corner=True)
plt.show()
```
**Key lines:**
- `sns.boxplot(x='dept', y='salary')` — one box per department showing median, IQR, whiskers, and outlier dots; ideal for comparing group distributions.
- `sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)` — colors each correlation; `center=0` makes positive/negative visually distinct; `annot` prints values. Reveals strong/redundant feature pairs.
- `sns.pairplot(df, hue='dept')` — grid of every pairwise scatter with per-feature distributions on the diagonal, colored by group; a fast multivariate overview (slow on big data).

## Interview Variations
- `violinplot` instead of box to reveal distribution shape/modality.
- Mask the upper triangle of the heatmap (`np.triu`) to reduce redundancy.
- `sns.regplot`/`lmplot` to add a fitted regression line.

## Common Follow-up Questions
- *What does a box plot's whisker/dot mean?* — Whiskers ≈ 1.5×IQR; dots are outliers.
- *Box vs violin?* — Violin adds the KDE shape (reveals modality) the box hides.
- *Why does Seaborn want tidy data?* — It maps columns to `x`/`y`/`hue`; long format enables that.

---

# Part D — Statistics: Summary, Distribution, Correlation, Covariance

## Practical Question 16 — Compute Summary Statistics From Scratch

**Difficulty:** Medium
**Estimated Time:** 12 min
**Concepts Tested:** Mean/median/mode, variance/std, understanding formulas

**Problem Statement**
Without using `np.mean/np.std` (implement the math), compute the mean, variance (sample), standard deviation, and median of a 1-D array. Then verify against NumPy.

## Python Implementation
```python
import numpy as np

def summary_stats(x):
    x = np.asarray(x, dtype=float)
    n = x.size

    mean = x.sum() / n                              # arithmetic mean
    var  = ((x - mean) ** 2).sum() / (n - 1)        # sample variance (ddof=1)
    std  = var ** 0.5                                # std = sqrt(variance)

    s = np.sort(x)
    mid = n // 2
    median = s[mid] if n % 2 else (s[mid-1] + s[mid]) / 2  # middle / avg of two

    return {'mean': mean, 'variance': var, 'std': std, 'median': median}

data = [2, 4, 4, 4, 5, 5, 7, 9]
print(summary_stats(data))
# Verify
print(np.mean(data), np.var(data, ddof=1), np.std(data, ddof=1), np.median(data))
```
**Key lines:**
- `((x - mean)**2).sum()/(n-1)` — the sample variance: mean of squared deviations, dividing by n−1 (Bessel's correction) for an unbiased estimate.
- Median: sort, then pick the middle element (odd n) or average the two middle elements (even n).

**Complexity:** Time **O(n log n)** (dominated by the sort for median); the moments alone are O(n). Space O(n).

## Alternative Solution
Vectorized NumPy: `x.mean()`, `x.var(ddof=1)`, `x.std(ddof=1)`, `np.median(x)`; `scipy.stats.mode(x)` for the mode.

## Interview Variations
- Add skewness (3rd standardized moment) and kurtosis (4th).
- Weighted mean: `np.average(x, weights=w)`.
- Rolling/streaming mean & variance (Welford's algorithm) for large data.

## Common Follow-up Questions
- *Why divide by n−1 not n?* — Unbiased sample variance (Bessel's correction).
- *Why sqrt for std?* — Return to the data's original units.
- *Mean vs median for skewed data?* — Median is robust; mean is pulled by the tail.

---

## Practical Question 17 — Skewness, Outliers (IQR), and Transformation

**Difficulty:** Hard
**Estimated Time:** 14 min
**Concepts Tested:** Skewness, IQR outlier rule, log transform, before/after comparison

**Problem Statement**
For a right-skewed feature: (a) measure its skewness, (b) detect outliers via the IQR rule, (c) apply a log transform and show that skewness decreases.

## Python Implementation
```python
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
income = pd.Series(rng.lognormal(mean=10, sigma=1.0, size=1000))  # right-skewed

# (a) Skewness
print('skew before:', income.skew())            # strongly positive

# (b) IQR outlier detection
Q1, Q3 = income.quantile([0.25, 0.75])
IQR = Q3 - Q1
low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers = income[(income < low) | (income > high)]
print('num outliers:', outliers.size, 'upper bound:', round(high))

# (c) Log transform reduces right skew
income_log = np.log1p(income)                    # log(1+x), safe at 0
print('skew after :', income_log.skew())         # much closer to 0
```
**Key lines:**
- `income.skew()` — sample skewness; large positive value confirms a long right tail (mean > median).
- IQR rule: bounds at `Q1 − 1.5·IQR` and `Q3 + 1.5·IQR`; points outside are flagged — robust because it's quartile-based (the box-plot rule).
- `np.log1p(income)` — compresses the long right tail, pulling skewness toward 0 and making the feature more symmetric/normal-like for modeling.

**Complexity:** Time **O(n log n)** (quantiles/sort); Space O(n).

## Alternative Solution
Z-score outliers (`abs((x-mean)/std) > 3`) — but less robust than IQR on skewed data. `scipy.stats.boxcox` / `PowerTransformer` for a principled normalizing transform (requires positive data).

## Interview Variations
- Winsorize (cap) outliers to the IQR bounds instead of removing.
- Compare mean vs median before/after transform.
- Detect *multivariate* outliers (Mahalanobis distance).

## Common Follow-up Questions
- *Why IQR over z-score for skewed data?* — Quartiles are robust; mean/std are inflated by the tail (masking).
- *Why `log1p` not `log`?* — Handles zeros safely (`log(0)` is −inf).
- *When should you NOT remove outliers?* — When they're the signal (fraud, rare events).

---

## Practical Question 18 — Correlation & Covariance Analysis

**Difficulty:** Medium
**Estimated Time:** 12 min
**Concepts Tested:** Pearson vs Spearman, covariance matrix, interpreting relationships

**Problem Statement**
For a numeric DataFrame: (a) compute Pearson and Spearman correlation matrices, (b) compute the covariance matrix and explain its diagonal, (c) identify the most correlated feature pair, (d) demonstrate why Pearson can miss a nonlinear relationship.

## Python Implementation
```python
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, 500)
df = pd.DataFrame({
    'x':       x,
    'lin':     2*x + rng.normal(0, 0.5, 500),   # linear with x
    'quad':    x**2 + rng.normal(0, 0.5, 500),  # nonlinear (U-shaped) with x
    'noise':   rng.normal(0, 1, 500),
})

pearson  = df.corr(method='pearson')            # linear
spearman = df.corr(method='spearman')           # monotonic (rank-based)
cov      = df.cov()                              # diagonal = variances

# (c) Most correlated pair (exclude self-correlations)
c = pearson.abs().where(~np.eye(len(pearson), dtype=bool))
pair = c.stack().idxmax()
print('most correlated pair:', pair, 'r =', round(pearson.loc[pair], 3))

# (d) Pearson misses the U-shape: x vs quad
print('Pearson  x~quad:', round(df['x'].corr(df['quad']), 3))   # ~0
print('Spearman x~quad:', round(df['x'].corr(df['quad'], method='spearman'), 3))
```
**Key lines:**
- `df.corr(method='pearson'/'spearman')` — Pearson captures linear association; Spearman correlates ranks, capturing monotonic/nonlinear-but-monotonic relationships and resisting outliers.
- `df.cov()` — the covariance matrix; its **diagonal entries are each feature's variance** (a variable's covariance with itself), off-diagonals are pairwise covariances.
- The U-shaped `quad ~ x` gives Pearson ≈ 0 (the up and down halves cancel) even though they're strongly related — proving Pearson only sees *linear* structure. (Spearman also stays low here because the relationship is non-monotonic — a scatter plot is the real tell.)

**Complexity:** Correlation/covariance matrices are **O(k²·n)** for k features, n rows; Space O(k²).

## Alternative Solution
`sns.heatmap(pearson, annot=True, cmap='coolwarm', center=0)` to visualize; `scipy.stats.pearsonr`/`spearmanr` also return p-values.

## Interview Variations
- Convert covariance to correlation manually (`cov / outer(std, std)`).
- Detect multicollinearity: flag `|r| > 0.9` feature pairs.
- Correlate every feature with a target and rank by absolute correlation.

## Common Follow-up Questions
- *Pearson vs Spearman?* — Linear (values) vs monotonic (ranks, robust).
- *Why is the covariance diagonal the variance?* — cov(X,X) = var(X).
- *Correlation ≠ causation — example?* — Ice-cream sales & drownings (confounder: heat).
- *Why prefer correlation over covariance to compare pairs?* — Correlation is standardized/unitless.

---

# Part E — Notebook Workflows (End-to-End)

These are structured as **Jupyter notebook cells**. In a lab exam you'll often be handed a CSV and asked to "explore and clean it" — this is the template. Each cell has a clear purpose and explanation.

## Workflow 1 — Complete EDA on a Dataset

> Scenario: you're given `data.csv` (a typical tabular dataset with numeric + categorical columns and some missing values). Produce a full EDA.

**Cell 1 — Import libraries**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', 50)
# Purpose: standard analysis stack + readable display + consistent plot theme.
```

**Cell 2 — Load the dataset**
```python
df = pd.read_csv('data.csv')
# Best practice: be explicit when you know the schema, e.g.
# df = pd.read_csv('data.csv', parse_dates=['date'], dtype={'zip': str})
df.shape
# Purpose: ingest data; print (rows, columns) to know its size immediately.
```

**Cell 3 — First look / structure**
```python
display(df.head())          # sample rows
df.info()                    # dtypes + non-null counts + memory
df.dtypes                    # confirm each column's type is sensible
# Purpose: understand columns, types, and obvious issues (wrong dtype, object columns).
```

**Cell 4 — Data quality: missing values & duplicates**
```python
missing = df.isna().mean().sort_values(ascending=False)
print(missing[missing > 0])            # fraction missing per column
print('duplicate rows:', df.duplicated().sum())
# Purpose: quantify missingness (drives imputation strategy) and detect duplicates.
```

**Cell 5 — Univariate: numeric summary & distributions**
```python
num_cols = df.select_dtypes(include='number').columns
display(df[num_cols].describe().T)     # count/mean/std/min/quartiles/max

df[num_cols].hist(figsize=(12, 8), bins=30)
plt.tight_layout(); plt.show()
# Purpose: see center, spread, and distribution shape (skew/outliers) per numeric feature.
```

**Cell 6 — Univariate: categorical frequencies**
```python
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for c in cat_cols:
    print(f'\n{c} — {df[c].nunique()} unique')
    print(df[c].value_counts(dropna=False).head(10))
# Purpose: understand category cardinality, dominant values, and class imbalance.
```

**Cell 7 — Data cleaning: impute & fix types**
```python
# Numeric -> median (robust to skew/outliers)
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
# Categorical -> mode
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

df = df.drop_duplicates()
assert df.isna().sum().sum() == 0
# Purpose: produce a complete, clean frame; median/mode are safe defaults.
```

**Cell 8 — Outlier detection (IQR)**
```python
def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    return ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()

outlier_counts = {c: iqr_outliers(df[c]) for c in num_cols}
print(pd.Series(outlier_counts).sort_values(ascending=False))

plt.figure(figsize=(10, 4))
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=45); plt.title('Numeric feature spread & outliers'); plt.show()
# Purpose: quantify and visualize outliers before deciding to cap/keep/transform them.
```

**Cell 9 — Bivariate: correlation analysis**
```python
corr = df[num_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix'); plt.show()
# Purpose: find strongly related (and redundant/collinear) feature pairs at a glance.
```

**Cell 10 — Bivariate: category vs numeric**
```python
# Example: how a numeric target varies across a key category
key_cat = cat_cols[0]; key_num = num_cols[0]
sns.boxplot(data=df, x=key_cat, y=key_num)
plt.xticks(rotation=45); plt.title(f'{key_num} by {key_cat}'); plt.show()

df.groupby(key_cat)[key_num].agg(['mean', 'median', 'count'])
# Purpose: reveal group differences (a precursor to feature importance / segmentation).
```

**Cell 11 — Interpretation & insights (markdown)**
```markdown
### Key Findings
- Missingness concentrated in `<col>` (X%); imputed with median/mode.
- `<featureA>` is right-skewed (skew=…); consider a log transform for modeling.
- `<featureA>` and `<featureB>` are highly correlated (r=…) → possible redundancy.
- `<num>` outliers in `<col>` flagged by IQR; likely <errors|genuine> because …
- `<target>` differs markedly across `<category>` (means: …), suggesting predictive value.

### Next Steps
- Feature engineering: … ; Feature selection: drop one of the collinear pair.
- Validate hypotheses on held-out data (EDA findings are hypotheses, not conclusions).
```

**Why each step matters:** the workflow moves *structure → quality → univariate → cleaning → outliers → bivariate → insight*, mirroring exactly how a senior analyst de-risks a dataset before modeling. In an exam, narrating this order earns marks even if a specific plot is imperfect.

---

## Workflow 2 — Data Cleaning + Feature Engineering (Assignment-Style)

> Scenario: turn a raw messy DataFrame into a clean, model-ready feature table.

**Cell 1 — Setup & load**
```python
import numpy as np, pandas as pd
df = pd.read_csv('raw.csv')
raw_shape = df.shape
# Purpose: load and remember the starting shape to track how much we drop.
```

**Cell 2 — Standardize column names & types**
```python
df.columns = (df.columns.str.strip().str.lower().str.replace(' ', '_'))
# Fix obvious type issues
if 'date' in df:  df['date'] = pd.to_datetime(df['date'], errors='coerce')
if 'price' in df: df['price'] = pd.to_numeric(df['price'], errors='coerce')
# Purpose: consistent names + correct dtypes prevent downstream bugs.
```

**Cell 3 — Handle missing values (with indicators)**
```python
for c in df.select_dtypes('number'):
    if df[c].isna().any():
        df[f'{c}_missing'] = df[c].isna().astype(int)   # informative-missingness flag
        df[c] = df[c].fillna(df[c].median())
for c in df.select_dtypes('object'):
    df[c] = df[c].fillna('Unknown')
# Purpose: keep the "was missing" signal, then impute robustly.
```

**Cell 4 — Treat outliers (winsorize/cap)**
```python
def cap_iqr(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    return s.clip(q1 - k*iqr, q3 + k*iqr)

for c in df.select_dtypes('number'):
    df[c] = cap_iqr(df[c])
# Purpose: limit extreme values' leverage without deleting rows (keeps sample size).
```

**Cell 5 — Feature engineering**
```python
# Datetime parts
if 'date' in df:
    df['year']      = df['date'].dt.year
    df['month']     = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend']= (df['dayofweek'] >= 5).astype(int)

# Ratios / interactions (domain-driven)
if {'revenue', 'units'} <= set(df):
    df['price_per_unit'] = df['revenue'] / df['units'].replace(0, np.nan)

# Binning a continuous variable into ordered categories
if 'age' in df:
    df['age_group'] = pd.cut(df['age'], bins=[0,18,35,60,120],
                             labels=['minor','young','adult','senior'])
# Purpose: create informative features that expose structure to models.
```

**Cell 6 — Encode categoricals**
```python
# Low-cardinality -> one-hot; high-cardinality would need other encodings
df = pd.get_dummies(df, columns=[c for c in df.select_dtypes('object')
                                 if df[c].nunique() <= 10], drop_first=True)
# Purpose: convert categories to numeric columns models can consume.
```

**Cell 7 — Scale numeric features**
```python
num = df.select_dtypes('number').columns
df[num] = (df[num] - df[num].mean()) / df[num].std()
# Purpose: put features on a comparable scale (needed by distance/gradient-based models).
# NOTE: in a real pipeline, fit the scaler on TRAIN only to avoid leakage.
```

**Cell 8 — Final validation**
```python
print('shape:', raw_shape, '->', df.shape)
assert df.isna().sum().sum() == 0
df.head()
# Purpose: confirm the table is complete, numeric, and ready; document row/col changes.
```

**Why this ordering:** clean names/types → missingness → outliers → engineer → encode → scale → validate. Encoding and scaling come *last* because they assume the data is already clean and complete; doing them earlier corrupts imputation and outlier logic.

---

# Part F — Rapid-Fire Coding Snippets & Gotchas

Interviewers love quick "what does this output / fix this" questions. Master these.

**1. Select column(s):**
```python
df['age']        # Series (1-D)
df[['age']]      # DataFrame (2-D)
df[['age','pay']]# DataFrame, multiple columns
```

**2. Filter (parentheses + `&`/`|`):**
```python
df[(df.age > 30) & (df.dept == 'Eng')]     # AND
df[df.dept.isin(['Eng','HR'])]             # membership
df.query('age > 30 and dept == "Eng"')     # SQL-like
```

**3. Safe conditional assignment (no chained indexing):**
```python
df.loc[df.age > 30, 'level'] = 'Senior'     # correct
# df[df.age>30]['level'] = 'Senior'          # WRONG: writes to a copy
```

**4. Missing values — detect with `.isna`, NOT `== NaN`:**
```python
df[df.col.isna()]           # correct
# df[df.col == np.nan]       # WRONG: NaN != NaN, always empty
df.col.fillna(df.col.median())
```

**5. GroupBy summary:**
```python
df.groupby('dept')['salary'].agg(['mean','count'])
df.groupby('dept')['salary'].transform('mean')   # same shape as input
```

**6. Value counts / uniques:**
```python
df.dept.value_counts()          # frequencies
df.dept.value_counts(normalize=True)  # proportions
df.dept.nunique()               # number of distinct values
```

**7. Sort & top-N:**
```python
df.sort_values('salary', ascending=False)
df.nlargest(3, 'salary')        # efficient top-3
```

**8. Apply a dict mapping (vectorized recode):**
```python
df['dept_code'] = df['dept'].map({'HR':0,'Eng':1,'Sales':2})
```

**9. NumPy axis intuition (the named axis disappears):**
```python
X.sum(axis=0)   # collapse rows -> per-column result
X.sum(axis=1)   # collapse cols -> per-row result
```

**10. Broadcasting to standardize columns:**
```python
(X - X.mean(axis=0)) / X.std(axis=0)
```

**11. View vs copy:**
```python
b = a[1:3]          # VIEW — mutating b changes a
b = a[1:3].copy()   # independent COPY
b = a[a > 0]        # boolean index -> COPY
```

**12. dtype trap — leading zeros / mixed:**
```python
pd.read_csv('f.csv', dtype={'zip': str})   # keep "00123" as text
```

**13. Reset index after groupby:**
```python
df.groupby('dept', as_index=False)['salary'].mean()
```

**14. Reshape with inferred dimension:**
```python
a.reshape(-1, 1)     # column vector
a.reshape(1, -1)     # row vector
a.ravel()            # flatten (view if possible)
```

**15. Common "why is it slow" answer:** you're using `df.apply(..., axis=1)`, `df.iterrows()`, or a Python loop — **vectorize** (column operations, `np.where`, `np.select`, `map`, `groupby`) instead.

---

## Final Practical Checklist

- [ ] Standardize/normalize an array with broadcasting (`axis=0`).
- [ ] Conditional replace with `np.where`/`np.clip`/`np.select` (no loops).
- [ ] Broadcasting distance matrix (and the a²+b²−2ab memory trick).
- [ ] View vs copy — predict and fix mutation bugs.
- [ ] Filter with parenthesized `&`/`|`; assign with a single `.loc`.
- [ ] GroupBy with named aggregations; `agg` vs `transform`.
- [ ] Impute missing values (median/mode + indicator; leakage awareness).
- [ ] Merge/join with correct `how`; detect unmatched keys.
- [ ] Top-N per group (sort+head, nlargest, or rank).
- [ ] Resample & rolling on a datetime index.
- [ ] Matplotlib labeled figure (title/labels/legend/grid); histogram vs bar.
- [ ] Seaborn box/violin/heatmap/pairplot; tidy data.
- [ ] Summary stats from scratch (variance n−1, median logic).
- [ ] Skewness + IQR outliers + log transform.
- [ ] Pearson vs Spearman; covariance matrix diagonal = variance.
- [ ] Full EDA notebook workflow, in order.
- [ ] Cleaning + feature-engineering notebook workflow, in order.

*Reminder: in coding rounds, always state your **time and space complexity** and call out the **view/copy** and **vectorization** implications — that's what separates a senior answer from a working one.*
