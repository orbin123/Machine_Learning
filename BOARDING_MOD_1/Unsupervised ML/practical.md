# Unsupervised Machine Learning — Practical & Coding Assessment Guide

> **How to use this document**
> This is the hands-on companion to `theory.md`. It predicts the coding and notebook questions you'll face in practical labs, coding rounds, and viva demos, and gives **production-quality, fully-explained** solutions.
> Every solution states **time & space complexity**, offers **alternatives**, lists **interview variations**, and ends with **follow-up questions**.
> Machine-learning topics are structured as **Jupyter notebook workflows** (one concept per cell) exactly as you'd present them in a lab exam.

## Environment setup (run this first in every notebook)

```python
# Standard imports used across all workflows
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, silhouette_samples)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

np.random.seed(42)          # reproducibility — ALWAYS set a seed in assessments
sns.set_style("whitegrid")
```

## Table of Contents

1. [Distance Metrics from Scratch](#1-distance-metrics-from-scratch)
2. [K-Means from Scratch + scikit-learn](#2-k-means-from-scratch--scikit-learn)
3. [Choosing K — Elbow & Silhouette Workflow](#3-choosing-k--elbow--silhouette-workflow)
4. [Hierarchical Clustering & Dendrograms](#4-hierarchical-clustering--dendrograms)
5. [DBSCAN & Outlier Detection](#5-dbscan--outlier-detection)
6. [Clustering Evaluation & Algorithm Comparison](#6-clustering-evaluation--algorithm-comparison)
7. [PCA — Full Notebook Workflow](#7-pca--full-notebook-workflow)
8. [t-SNE Visualization](#8-t-sne-visualization)
9. [SVD & Recommendation / LSA](#9-svd--recommendation--lsa)
10. [Feature Scaling Comparison](#10-feature-scaling-comparison)
11. [End-to-End Project: Customer Segmentation](#11-end-to-end-project-customer-segmentation)
12. [End-to-End Project: Image Compression](#12-end-to-end-project-image-compression)
13. [Coding Questions Bank (Easy / Medium / Hard)](#13-coding-questions-bank)

---

# 1. Distance Metrics from Scratch

## Practical Question 1

**Difficulty:** Easy–Medium
**Estimated Time:** 15–20 min
**Concepts Tested:** distance metrics, vectorization with NumPy, cosine vs Euclidean intuition

**Problem Statement**
Implement Euclidean, Manhattan, Minkowski, Cosine similarity, and Hamming distance **from scratch** (no `scipy.spatial.distance`). Then verify against SciPy.

**Example Input**
```
a = [1, 2, 3]
b = [4, 0, 3]
```
**Example Output**
```
Euclidean ≈ 3.606
Manhattan  = 5
Cosine sim ≈ 0.837
```

**Approach**
Each metric is a direct translation of its formula. Use NumPy arrays so operations are vectorized (fast, clean). Cosine needs a guard against zero-norm vectors.

## Python Implementation

```python
import numpy as np

def euclidean(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.sqrt(np.sum((a - b) ** 2))        # L2 norm of the difference

def manhattan(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.sum(np.abs(a - b))                 # sum of absolute differences

def minkowski(a, b, p=3):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.sum(np.abs(a - b) ** p) ** (1 / p) # p=1→Manhattan, p=2→Euclidean

def cosine_similarity(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:                               # guard: undefined for zero vectors
        return 0.0
    return np.dot(a, b) / denom                  # cosθ = (a·b)/(‖a‖‖b‖)

def cosine_distance(a, b):
    return 1 - cosine_similarity(a, b)

def hamming(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, "sequences must be equal length"
    return np.sum(a != b)                         # count of differing positions

# --- verify against SciPy ---
from scipy.spatial import distance
a, b = [1, 2, 3], [4, 0, 3]
print(euclidean(a, b), distance.euclidean(a, b))
print(cosine_similarity(a, b), 1 - distance.cosine(a, b))
```

**Line notes**
- `np.asarray(..., float)` makes the functions accept lists or arrays and avoids integer overflow.
- `np.linalg.norm` computes the L2 norm; the zero-norm guard prevents division-by-zero (a common bug interviewers watch for).
- `np.sum(a != b)` — boolean array summed = count of mismatches (Hamming).

**Complexity:** each is **O(d)** time (d = dimensions), **O(1)** extra space (O(d) if you count the difference array).

## Alternative Solution
Use library functions directly: `scipy.spatial.distance.euclidean/cityblock/minkowski/cosine/hamming`, or `sklearn.metrics.pairwise` for **matrix-wise** pairwise distances (vectorized over all pairs, far faster than Python loops).

```python
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity as cos
X = np.array([[1,2,3],[4,0,3],[0,0,1]])
euclidean_distances(X)      # full n×n distance matrix in one call
```

## Interview Variations
- Compute the **pairwise distance matrix** for a dataset (n×n) — watch for O(n²) memory.
- Implement **weighted** Euclidean distance (per-feature weights).
- Given two documents as word-count dicts, compute **cosine similarity** (sparse).
- Implement **k-nearest-neighbors** search using your distance function.

## Common Follow-up Questions
- *"Why guard the cosine denominator?"* — zero-vector → division by zero / NaN.
- *"Which metric for TF-IDF text and why?"* — cosine (ignores document length/magnitude).
- *"How does this scale to a million points?"* — O(n²) pairwise is infeasible; use vectorized libs, KD-trees, or approximate NN (FAISS).
- *"Why does Euclidean fail in high dimensions?"* — distance concentration.

---

# 2. K-Means from Scratch + scikit-learn

## Practical Question 2

**Difficulty:** Medium
**Estimated Time:** 25–35 min
**Concepts Tested:** K-Means internals, the EM loop, convergence, vectorization, comparing with sklearn

**Problem Statement**
Implement K-Means clustering from scratch (assignment + update loop), then reproduce the result with `sklearn.cluster.KMeans`.

**Example Input**
A 2-D blob dataset of 300 points in 4 natural groups.

**Example Output**
4 cluster labels per point and 4 final centroids; a scatter plot colored by cluster with centroids marked.

**Approach (step by step)**
1. Initialize K centroids (pick K random distinct points).
2. **Assignment:** compute each point's distance to every centroid, assign to the nearest.
3. **Update:** recompute each centroid as the mean of its assigned points.
4. Repeat until centroids stop moving (or `max_iter`).
5. Track inertia (WCSS) to confirm it decreases.

## Python Implementation

```python
import numpy as np

class KMeansScratch:
    def __init__(self, k=3, max_iter=100, tol=1e-4, random_state=42):
        self.k, self.max_iter, self.tol = k, max_iter, tol
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        # --- Step 1: initialize centroids from random data points ---
        idx = rng.choice(len(X), self.k, replace=False)
        self.centroids = X[idx].copy()

        for it in range(self.max_iter):
            # --- Step 2: ASSIGNMENT ---
            # distances: (n_points, k) via broadcasting; ||x - c||
            dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)          # nearest centroid per point

            # --- Step 3: UPDATE ---
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j)
                else self.centroids[j]                 # empty-cluster guard: keep old
                for j in range(self.k)
            ])

            # --- Step 4: convergence check ---
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:                        # centroids barely moved
                break

        self.labels_ = labels
        self.inertia_ = self._inertia(X, labels)
        return self

    def _inertia(self, X, labels):
        # WCSS: sum of squared distances to assigned centroid
        return sum(np.sum((X[labels == j] - self.centroids[j]) ** 2)
                   for j in range(self.k))

    def predict(self, X):
        dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

# --- Demo ---
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

model = KMeansScratch(k=4).fit(X)
print("Scratch inertia:", round(model.inertia_, 2))
```

**Line notes**
- `X[:, None, :] - self.centroids[None, :, :]` — broadcasting produces an `(n, k, d)` difference tensor; `np.linalg.norm(..., axis=2)` collapses the feature axis to an `(n, k)` distance matrix. **Fully vectorized — no Python loops over points.**
- The **empty-cluster guard** (`else self.centroids[j]`) prevents `mean of empty slice → NaN`, a classic crash.
- Convergence via **centroid shift < tol** (equivalent to labels stabilizing).

**Complexity:** **O(n · k · d · i)** time (n points, k clusters, d dims, i iterations) — effectively linear in n. **O(n·k + k·d)** space for the distance matrix and centroids.

## scikit-learn version (what you'd actually use)

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels = km.fit_predict(X)
print("sklearn inertia:", round(km.inertia_, 2))
print("centroids:\n", km.cluster_centers_)

# visualize
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=25)
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
            c='red', marker='X', s=200, label='centroids')
plt.legend(); plt.title("K-Means (k=4)"); plt.show()
```
- `init='k-means++'` — smart seeding (default). `n_init=10` — run 10 times, keep lowest inertia (mitigates local minima).

## Alternative Solution
- **MiniBatchKMeans** for large data — updates centroids on random mini-batches, far faster with minor quality loss.
- **GaussianMixture** for soft/elliptical clusters.

## Interview Variations
- Add **K-Means++ initialization** to the scratch version.
- Return the **iteration count** and plot inertia vs iteration (show monotonic decrease).
- Implement **K-Medians** (use median + Manhattan) for outlier robustness.
- Handle **empty clusters** by reinitializing to the farthest point.

## Common Follow-up Questions
- *"Why do you set `n_init`?"* — K-Means finds local minima; multiple restarts pick the best.
- *"Why vectorize instead of looping?"* — NumPy broadcasting is orders of magnitude faster than Python loops.
- *"What if a cluster becomes empty?"* — reinitialize its centroid (farthest point / random).
- *"Prove your loop terminates."* — inertia is non-increasing and bounded below by 0.

---

# 3. Choosing K — Elbow & Silhouette Workflow

## Practical Question 3

**Difficulty:** Medium
**Estimated Time:** 20–25 min
**Concepts Tested:** model selection, elbow method, silhouette analysis, reading diagnostic plots

**Problem Statement**
Given an unlabeled dataset, determine the optimal number of clusters K using both the **Elbow Method** (inertia) and **Silhouette Score**, then justify your choice.

This is a classic **notebook workflow** — present it cell by cell.

### Cell 1 — Imports & data
```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)
```
*We use blobs with 4 true centers so we can confirm the methods recover K=4.*

### Cell 2 — Scale the features (mandatory)
```python
X_scaled = StandardScaler().fit_transform(X)
```
*K-Means is distance-based, so we standardize first — otherwise large-range features dominate.*

### Cell 3 — Compute inertia & silhouette across K
```python
K_range = range(2, 11)
inertias, silhouettes = [], []

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)                       # WCSS for elbow
    silhouettes.append(silhouette_score(X_scaled, labels))  # separation quality
```
*We loop K from 2 to 10 (silhouette is undefined for K=1). Each K stores its inertia and average silhouette.*

### Cell 4 — Plot the Elbow
```python
plt.plot(list(K_range), inertias, 'o-')
plt.xlabel('K'); plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method'); plt.show()
```
*Look for the "elbow" — the K where inertia's decrease sharply flattens. Expect a bend at K=4.*

### Cell 5 — Plot Silhouette vs K
```python
plt.plot(list(K_range), silhouettes, 'o-', color='green')
plt.xlabel('K'); plt.ylabel('Average Silhouette Score')
plt.title('Silhouette Analysis'); plt.show()
best_k = list(K_range)[int(np.argmax(silhouettes))]
print("Best K by silhouette:", best_k)
```
*The K with the **highest** average silhouette is the most defensible, objective choice.*

### Cell 6 — Silhouette plot for the chosen K (advanced, impresses examiners)
```python
k = best_k
km = KMeans(n_clusters=k, n_init=10, random_state=42)
labels = km.fit_predict(X_scaled)
sample_sil = silhouette_samples(X_scaled, labels)

y_lower = 0
for i in range(k):
    vals = np.sort(sample_sil[labels == i])
    y_upper = y_lower + len(vals)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, vals)
    y_lower = y_upper
plt.axvline(silhouette_score(X_scaled, labels), color='red', linestyle='--')
plt.xlabel('Silhouette coefficient'); plt.ylabel('Cluster')
plt.title(f'Silhouette plot (k={k})'); plt.show()
```
*Each cluster's "knife" should be wide and extend past the red average line. Clusters dipping below 0 signal misassignment.*

### Cell 7 — Interpretation
```
Elbow suggests K≈4 (bend where marginal inertia gain drops).
Silhouette peaks at K=4 (highest average separation).
Both agree → choose K=4, consistent with the 4 true blobs.
```

## Complexity
- Elbow loop: fitting K-Means for each K → **O(|K_range| · n·k·d·i)**.
- Silhouette: **O(n²)** per K (pairwise distances) — the expensive part; subsample for large n.

## Alternative Solution
- **Gap statistic** — compares inertia to that of random uniform data (more principled than the elbow).
- **`KElbowVisualizer`** from the **Yellowbrick** library automates the elbow + timing.
- **Davies-Bouldin / Calinski-Harabasz** as additional votes.

## Interview Variations
- Do it for **DBSCAN** (no K — instead sweep `eps` via the k-distance plot).
- Automate: return the K that **maximizes silhouette** programmatically.
- Compare elbow vs silhouette when they **disagree** and justify.

## Common Follow-up Questions
- *"Why is silhouette more reliable than the elbow?"* — it's a quantitative score, not a subjective bend.
- *"Why can't you use inertia alone?"* — it always decreases with K.
- *"Silhouette is O(n²) — what for 10M points?"* — subsample, or use Calinski-Harabasz (cheap).

---

# 4. Hierarchical Clustering & Dendrograms

## Practical Question 4

**Difficulty:** Medium
**Estimated Time:** 20–30 min
**Concepts Tested:** agglomerative clustering, linkage methods, reading/cutting dendrograms

**Problem Statement**
Perform agglomerative hierarchical clustering, plot a dendrogram, choose the number of clusters by cutting it, and compare linkage methods.

### Cell 1 — Imports & data
```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

X, _ = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=42)
X = StandardScaler().fit_transform(X)      # scale first
```

### Cell 2 — Build the linkage matrix & plot dendrogram
```python
Z = linkage(X, method='ward')              # Ward = variance-minimizing (default choice)

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=20) # show last 20 merges for readability
plt.title('Ward Dendrogram')
plt.xlabel('Sample index / cluster size'); plt.ylabel('Merge distance')
plt.axhline(y=10, color='red', linestyle='--')  # candidate cut
plt.show()
```
*`linkage` returns the merge matrix `Z`: each row = [cluster_a, cluster_b, distance, sample_count]. The horizontal red line is a candidate cut; it crosses 3 vertical lines → 3 clusters.*

### Cell 3 — Choosing the cut (largest gap heuristic)
```python
# distances at which merges happened (last column-ish)
merge_dists = Z[:, 2]
gaps = np.diff(merge_dists)
print("Biggest jump between merges at index:", np.argmax(gaps))
# Cut BELOW the biggest vertical gap → the natural number of clusters.
```
*The tallest vertical gap in the dendrogram = where merging becomes "expensive" = the natural split point.*

### Cell 4 — Extract flat clusters
```python
# Option A: cut the scipy tree at a distance threshold
labels_scipy = fcluster(Z, t=10, criterion='distance')

# Option B: sklearn with a fixed number of clusters
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=25)
plt.title('Agglomerative (Ward, k=3)'); plt.show()
```

### Cell 5 — Compare linkage methods
```python
from sklearn.metrics import silhouette_score
for method in ['single', 'complete', 'average', 'ward']:
    lab = AgglomerativeClustering(n_clusters=3, linkage=method).fit_predict(X)
    print(f"{method:>9}: silhouette = {silhouette_score(X, lab):.3f}")
```
*Expect Ward/complete/average to score well on blobs; single linkage often underperforms due to chaining.*

## Complexity
- `linkage` (agglomerative): **O(n²) memory**, **O(n² log n)** time (Ward). **Does not scale** beyond ~10k–50k points.

## Alternative Solution
- For large data, cluster a **sample** or first reduce with K-Means, then run hierarchical on the centroids.
- Use `scipy` `linkage` for the dendrogram + `AgglomerativeClustering` for labels (sklearn ≥1.2 can also return distances for its own dendrogram).

## Interview Variations
- Plot dendrograms for **all four linkages** side by side and explain the shape differences.
- Use a **non-Euclidean metric** (e.g. cosine) with `average` linkage (Ward requires Euclidean!).
- Cut the tree at **different heights** and show how cluster count changes.

## Common Follow-up Questions
- *"How do you pick the number of clusters from a dendrogram?"* — cut at the largest vertical gap.
- *"Why does single linkage chain?"* — it merges on the *closest pair*, so bridges fuse clusters.
- *"Why can't Ward use cosine distance?"* — Ward minimizes variance, an L2 (Euclidean) concept.
- *"Why is this infeasible for 1M rows?"* — O(n²) distance matrix.

---

# 5. DBSCAN & Outlier Detection

## Practical Question 5

**Difficulty:** Medium–Hard
**Estimated Time:** 25–35 min
**Concepts Tested:** density-based clustering, k-distance elbow for `eps`, noise/outlier identification, non-convex shapes

**Problem Statement**
Cluster a dataset with non-spherical shapes and noise using DBSCAN. Tune `eps` with the k-distance graph, identify outliers, and compare against K-Means on the same data.

### Cell 1 — Data: two moons + noise (K-Means killer)
```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
X = StandardScaler().fit_transform(X)      # DBSCAN uses a global eps → scale!
```

### Cell 2 — Choose `eps` via the k-distance plot
```python
min_samples = 5
nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
dists, _ = nbrs.kneighbors(X)
k_dist = np.sort(dists[:, -1])             # distance to the k-th nearest neighbor

plt.plot(k_dist)
plt.ylabel(f'{min_samples}-th NN distance'); plt.xlabel('points sorted')
plt.title('k-distance graph — pick eps at the knee'); plt.show()
# The sharp "knee" (elbow) of this curve is a good eps.
```
*Below the knee = intra-cluster distances; above = noise. Read the y-value at the knee as `eps` (≈0.2 here).*

### Cell 3 — Fit DBSCAN
```python
db = DBSCAN(eps=0.2, min_samples=5).fit(X)
labels = db.labels_                        # -1 == noise/outlier

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"clusters: {n_clusters}, noise points: {n_noise}")
```

### Cell 4 — Visualize clusters + outliers
```python
mask = labels == -1
plt.scatter(X[~mask,0], X[~mask,1], c=labels[~mask], cmap='viridis', s=25)
plt.scatter(X[mask,0], X[mask,1], c='red', marker='x', s=60, label='noise')
plt.legend(); plt.title('DBSCAN — two moons + outliers'); plt.show()
```

### Cell 5 — K-Means on the same data (show the failure)
```python
km = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=km, cmap='coolwarm', s=25)
plt.title('K-Means slices the moons — wrong!'); plt.show()
```
*K-Means draws a straight boundary and cuts each moon; DBSCAN follows the crescents and isolates noise. This side-by-side is the canonical demonstration.*

### Cell 6 — Interpretation
```
DBSCAN: 2 crescent clusters + N noise points (the injected outliers).
K-Means: fails — assumes convex blobs, splits the moons linearly.
Takeaway: choose density-based methods for non-convex + noisy data.
```

## Complexity
- DBSCAN with a spatial index: **O(n log n)** average, **O(n²)** worst case. Memory O(n).

## Alternative Solution
- **HDBSCAN** — handles **varying density** and auto-selects `eps`-equivalents; often superior.
- **Isolation Forest / LOF** for pure anomaly detection (below).

## Bonus — dedicated anomaly detectors
```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

iso = IsolationForest(contamination=0.05, random_state=42).fit_predict(X)  # -1 = anomaly
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit_predict(X) # -1 = anomaly
print("IsoForest anomalies:", (iso == -1).sum(), " LOF anomalies:", (lof == -1).sum())
```

## Interview Variations
- Tune **both** `eps` and `min_samples` with a small grid and report cluster/noise counts.
- Apply DBSCAN to **geospatial** (lat/long) data using the **haversine** metric.
- Replace DBSCAN noise detection with **Isolation Forest** and compare flagged points.

## Common Follow-up Questions
- *"How did you choose `eps`?"* — k-distance graph knee.
- *"What does label −1 mean?"* — noise/outlier, not a cluster.
- *"Why does DBSCAN beat K-Means here?"* — density follows arbitrary shapes; handles noise.
- *"When does DBSCAN fail?"* — varying-density clusters, high dimensions → use HDBSCAN / reduce dims.

---

# 6. Clustering Evaluation & Algorithm Comparison

## Practical Question 6

**Difficulty:** Medium
**Estimated Time:** 20–25 min
**Concepts Tested:** internal validity metrics, comparing algorithms fairly on one dataset

**Problem Statement**
Run K-Means, Agglomerative, and DBSCAN on the same dataset and compare them using Silhouette, Davies-Bouldin, and Calinski-Harabasz. Recommend the best and justify.

### Cell 1 — Setup
```python
import numpy as np, pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)
X = StandardScaler().fit_transform(X)
```

### Cell 2 — Fit all three
```python
results = {}
results['KMeans']        = KMeans(4, n_init=10, random_state=42).fit_predict(X)
results['Agglomerative'] = AgglomerativeClustering(4).fit_predict(X)
results['DBSCAN']        = DBSCAN(eps=0.3, min_samples=5).fit_predict(X)
```

### Cell 3 — Score them (helper handles DBSCAN's noise label)
```python
def score(X, labels):
    # metrics need >=2 clusters and can't score all-noise; ignore noise for a fair number
    mask = labels != -1
    n_lab = len(set(labels[mask]))
    if n_lab < 2:
        return (np.nan, np.nan, np.nan)
    return (silhouette_score(X[mask], labels[mask]),          # higher better
            davies_bouldin_score(X[mask], labels[mask]),      # lower  better
            calinski_harabasz_score(X[mask], labels[mask]))   # higher better

table = pd.DataFrame(
    {name: score(X, lab) for name, lab in results.items()},
    index=['Silhouette↑', 'DaviesBouldin↓', 'CalinskiHarabasz↑']
).T
print(table.round(3))
```

### Cell 4 — Interpretation
```
Read each column with its arrow:
  - Highest Silhouette  = best-separated.
  - Lowest Davies-Bouldin = most compact & separated.
  - Highest Calinski-Harabasz = best variance ratio.
On clean blobs, K-Means & Agglomerative usually win all three;
DBSCAN may lag because blobs are convex (its strength is non-convex+noise).
Recommendation depends on data: blobs → K-Means; moons/noise → DBSCAN.
```

## Complexity
- Silhouette: **O(n²)**; Davies-Bouldin & Calinski-Harabasz: **O(n·k·d)** (cheap). Prefer the cheap ones at scale.

## Alternative Solution
- If you have *some* labels: **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** (external metrics).
```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
```

## Interview Variations
- Add **GaussianMixture** to the comparison.
- Loop over multiple synthetic datasets (blobs, moons, circles, anisotropic) and show **no algorithm wins everywhere** (the "no free lunch" demo).
- Automate: pick the algorithm with the best composite rank.

## Common Follow-up Questions
- *"Why did DBSCAN score worst on blobs?"* — silhouette favors convex clusters; DBSCAN's edge is non-convex + noise.
- *"Which metric would you trust for DBSCAN?"* — density-aware (DBCV) or visual; silhouette can mislead.
- *"Metrics disagree — what now?"* — weigh silhouette + business interpretability; inspect visually.

---

# 7. PCA — Full Notebook Workflow

## Practical Question 7

**Difficulty:** Medium–Hard
**Estimated Time:** 30–40 min
**Concepts Tested:** PCA end-to-end, explained variance, scree plot, choosing components, PCA from scratch, PCA-before-clustering

**Problem Statement**
Apply PCA to a high-dimensional dataset (Iris/Wine/Digits). Choose the number of components via explained variance, visualize in 2-D, interpret loadings, and (bonus) implement PCA from scratch to prove you understand it.

### Cell 1 — Imports & load data
```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_wine()
X, y, names = data.data, data.target, data.feature_names
print(X.shape)                    # (178, 13) — 13 features
```

### Cell 2 — Standardize (mandatory for PCA)
```python
X_scaled = StandardScaler().fit_transform(X)
```
*PCA maximizes variance, which is scale-dependent — standardize so no feature dominates by unit alone.*

### Cell 3 — Fit PCA (all components) & inspect explained variance
```python
pca = PCA().fit(X_scaled)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
for i,(e,c) in enumerate(zip(evr, cum), 1):
    print(f"PC{i}: {e:.3f}  cumulative {c:.3f}")
```

### Cell 4 — Scree plot + cumulative variance
```python
fig, ax = plt.subplots(1, 2, figsize=(12,4))
ax[0].plot(range(1,len(evr)+1), evr, 'o-'); ax[0].set_title('Scree plot')
ax[0].set_xlabel('component'); ax[0].set_ylabel('explained variance')
ax[1].plot(range(1,len(cum)+1), cum, 'o-'); ax[1].axhline(0.95, color='r', ls='--')
ax[1].set_title('Cumulative variance'); ax[1].set_xlabel('n components')
plt.show()

n95 = np.argmax(cum >= 0.95) + 1
print("Components for 95% variance:", n95)
```
*Keep the smallest number of components crossing the 95% line — that's your compression target.*

### Cell 5 — Transform to 2-D & visualize
```python
X_2d = PCA(n_components=2).fit_transform(X_scaled)
plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap='viridis', s=30)
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Wine in 2 PCs')
plt.colorbar(label='class'); plt.show()
```
*Even in 2 components the three wine classes separate visibly — PCA preserved the discriminative structure.*

### Cell 6 — Interpret loadings (which features drive each PC)
```python
pca2 = PCA(n_components=2).fit(X_scaled)
loadings = pd.DataFrame(pca2.components_.T, index=names, columns=['PC1','PC2'])
print(loadings.sort_values('PC1', key=abs, ascending=False).head())
```
*Large-magnitude loadings tell you what each component "means" — name PC1 by its top-loading features.*

### Cell 7 — Bonus: PCA from scratch (proves understanding)
```python
def pca_scratch(X, k):
    Xc = X - X.mean(axis=0)                 # 1. center
    cov = np.cov(Xc, rowvar=False)          # 2. covariance matrix (d×d)
    eigvals, eigvecs = np.linalg.eigh(cov)  # 3. eig (eigh: symmetric matrix)
    order = np.argsort(eigvals)[::-1]       # 4. sort descending
    W = eigvecs[:, order[:k]]               # 5. top-k eigenvectors
    return Xc @ W, eigvals[order]           # 6. project

X_manual, eigvals = pca_scratch(X_scaled, 2)
# sign may flip vs sklearn (eigenvectors are unique up to sign) — that's expected
print("scratch explained-var ratio:", (eigvals/eigvals.sum())[:2].round(3))
```

**Line notes**
- `np.linalg.eigh` (not `eig`) — the covariance matrix is symmetric, `eigh` is faster and returns real, sorted-ish eigenvalues.
- **Sign flips** between scratch and sklearn are normal — eigenvectors are defined up to sign.
- sklearn uses **SVD** internally (more stable than covariance eig), but results match.

**Complexity:** covariance **O(n·d²)** + eig **O(d³)**. For `d ≫ n` or sparse data, SVD/TruncatedSVD is better.

## Alternative Solution
- **`TruncatedSVD`** for sparse data (no centering).
- **`IncrementalPCA`** for data too large for memory (batch-wise).
- **Kernel PCA** for non-linear structure.

## PCA before clustering (very common combined question)
```python
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
pipe = Pipeline([('scale', StandardScaler()),
                 ('pca', PCA(n_components=0.95)),   # keep 95% variance
                 ('km', KMeans(3, n_init=10, random_state=42))])
labels = pipe.fit_predict(X)
```
*The Pipeline prevents data leakage by fitting scaler+PCA only on training folds during CV.*

## Interview Variations
- Reconstruct the data from k components and report **reconstruction error** vs k.
- Use **`PCA(n_components=0.95)`** (float = variance target) instead of an integer.
- Show PCA **fails** on a non-linear dataset; switch to Kernel PCA.
- Compare model accuracy **with vs without** PCA (speed vs accuracy trade-off).

## Common Follow-up Questions
- *"Why `eigh` not `eig`?"* — covariance is symmetric.
- *"Why do signs differ from sklearn?"* — eigenvectors are unique up to sign.
- *"How many components did you keep and why?"* — smallest count reaching 95% cumulative variance.
- *"PCA reduced accuracy — why?"* — it's unsupervised; discriminative direction may be low-variance (use LDA).
- *"Why fit PCA on train only?"* — avoid leaking test info into the transform.

---

# 8. t-SNE Visualization

## Practical Question 8

**Difficulty:** Medium
**Estimated Time:** 20–25 min
**Concepts Tested:** t-SNE for visualization, perplexity, PCA-before-t-SNE, PCA vs t-SNE comparison

**Problem Statement**
Visualize the high-dimensional Digits dataset (64-D) in 2-D using t-SNE, tune perplexity, and compare the result to PCA.

### Cell 1 — Load high-D data
```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

digits = load_digits()
X, y = digits.data, digits.target          # (1797, 64), 10 classes
X = StandardScaler().fit_transform(X)
```

### Cell 2 — PCA first (denoise + speed up t-SNE)
```python
X_pca50 = PCA(n_components=30).fit_transform(X)   # 64 → 30 dims
```
*Standard recipe: reduce to ~30 dims with PCA before t-SNE — faster and less noisy.*

### Cell 3 — Run t-SNE
```python
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto',
            n_iter=1000, init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_pca50)
```
*`init='pca'` gives a stable, reproducible start; `random_state` fixes the (otherwise non-deterministic) layout.*

### Cell 4 — Plot t-SNE vs PCA side by side
```python
X_pca2 = PCA(n_components=2).fit_transform(X)
fig, ax = plt.subplots(1, 2, figsize=(14,6))
ax[0].scatter(X_pca2[:,0], X_pca2[:,1], c=y, cmap='tab10', s=10)
ax[0].set_title('PCA (2D) — overlapping')
ax[1].scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='tab10', s=10)
ax[1].set_title('t-SNE (2D) — clean clusters')
plt.show()
```
*t-SNE separates the 10 digit clusters far more clearly than PCA — it captures non-linear local structure.*

### Cell 5 — Effect of perplexity
```python
for p in [5, 30, 50]:
    emb = TSNE(2, perplexity=p, init='pca', random_state=42).fit_transform(X_pca50)
    plt.figure(figsize=(4,4))
    plt.scatter(emb[:,0], emb[:,1], c=y, cmap='tab10', s=8)
    plt.title(f'perplexity={p}'); plt.show()
```
*Low perplexity → fragmented; high → merged. Try several; there's no single "correct" value.*

### Cell 6 — Interpretation & cautions
```
t-SNE reveals 10 well-separated digit clusters (local structure).
CAUTION: cluster SIZES and DISTANCES between clusters are NOT meaningful.
Never feed t-SNE coordinates into a classifier — visualization only.
```

## Complexity
- t-SNE: ~**O(n²)** (Barnes-Hut → O(n log n) for 2-D). Slow; PCA-first mitigates.

## Alternative Solution
- **UMAP** — faster, preserves more global structure, supports `.transform()` on new data. Often preferred now.
```python
# import umap ; UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X)
```

## Interview Variations
- Compare **t-SNE vs UMAP vs PCA** runtime and visual quality.
- Show a **misleading** t-SNE (bad perplexity) and explain the artifact.
- Fix `random_state` and show reproducibility; remove it and show variation.

## Common Follow-up Questions
- *"Can you cluster on t-SNE output?"* — No; it's viz-only, distances aren't reliable.
- *"Why PCA before t-SNE?"* — denoise + speed.
- *"Why non-deterministic?"* — random init + non-convex KL optimization.
- *"What does perplexity control?"* — effective neighborhood size (local vs global balance).

---

# 9. SVD & Recommendation / LSA

## Practical Question 9

**Difficulty:** Hard
**Estimated Time:** 30–40 min
**Concepts Tested:** SVD/matrix factorization, TruncatedSVD, latent factors, recommendation, LSA on text

**Problem Statement (Part A — Recommendation)**
Given a small user-item rating matrix with missing values, use SVD-based matrix factorization to predict missing ratings and recommend items.

### Cell 1 — Build the rating matrix
```python
import numpy as np, pandas as pd
# rows = users, cols = movies; 0 = unrated
R = np.array([
    [5, 4, 0, 1, 0],
    [4, 0, 0, 1, 2],
    [1, 1, 0, 5, 0],
    [0, 0, 5, 4, 5],
    [0, 1, 4, 0, 4],
], dtype=float)
movies = ['Action1','Action2','Rom1','Rom2','Rom3']
```

### Cell 2 — Mean-center, factor with SVD
```python
# Fill missing with each user's mean so 0 doesn't mean "hated it"
user_mean = np.true_divide(R.sum(1), (R != 0).sum(1))
R_filled = np.where(R == 0, user_mean[:, None], R)
R_demeaned = R_filled - user_mean[:, None]

U, sigma, Vt = np.linalg.svd(R_demeaned, full_matrices=False)
k = 2                                     # keep top-2 latent factors
U_k, S_k, Vt_k = U[:, :k], np.diag(sigma[:k]), Vt[:k, :]
```
*We keep only the top-k singular values — the dominant latent "taste" factors (e.g. action-lover vs romance-lover).*

### Cell 3 — Reconstruct & predict
```python
pred = (U_k @ S_k @ Vt_k) + user_mean[:, None]   # add the mean back
pred_df = pd.DataFrame(pred.round(2), columns=movies)
print(pred_df)

# Recommend: highest predicted rating among items the user hasn't rated
def recommend(user_id):
    unseen = np.where(R[user_id] == 0)[0]
    best = unseen[np.argmax(pred[user_id, unseen])]
    return movies[best]
print("Recommend for user 0:", recommend(0))
```
*The dot product of a user's and item's latent vectors predicts the missing rating; we recommend the top unseen item.*

### Cell 4 — Explained variance of the factors
```python
print("Explained variance ratio:", (sigma**2 / (sigma**2).sum()).round(3))
```

**Part B — LSA on text (TruncatedSVD)**

### Cell 5 — TF-IDF + TruncatedSVD for latent topics
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

docs = ["cats and dogs are pets", "the stock market fell today",
        "my dog loves the cat", "investors sold stocks and bonds"]
tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf.fit_transform(docs)         # sparse matrix

lsa = TruncatedSVD(n_components=2, random_state=42)
X_lsa = lsa.fit_transform(X_tfidf)          # docs → 2 latent topics
print("Topic space:\n", X_lsa.round(2))
print("Explained variance:", lsa.explained_variance_ratio_.round(3))
```
*TruncatedSVD works **directly on the sparse TF-IDF matrix** (no centering → sparsity preserved). Docs 0&2 (pets) and 1&3 (finance) cluster in the 2-topic space.*

**Line notes**
- Filling missing ratings with the **user mean** (not 0) avoids telling the model "unrated = strongly disliked."
- `np.linalg.svd(..., full_matrices=False)` returns the economy SVD (smaller, faster).
- **TruncatedSVD, not PCA**, for text — PCA would densify the sparse matrix via centering.

**Complexity:** full SVD **O(min(mn², m²n))**; TruncatedSVD (randomized) ~**O(mnk)** — scales to large sparse matrices.

## Alternative Solution
- **`scipy.sparse.linalg.svds`** for large sparse matrices.
- Dedicated libraries: **`surprise`** (SVD/SVD++ for recommenders), **`implicit`** (ALS).
- Deep learning: neural collaborative filtering / autoencoders.

## Interview Variations
- Predict a **specific** (user, movie) rating and explain the latent-factor reasoning.
- Vary `k` and show reconstruction error vs number of factors.
- Use LSA output to compute **document similarity** (cosine on topic vectors).

## Common Follow-up Questions
- *"Why fill missing with the mean, not 0?"* — 0 would be interpreted as a low rating.
- *"Why TruncatedSVD over PCA for text?"* — no centering → keeps sparsity.
- *"Is this the 'real' SVD used at Netflix?"* — no; production uses regularized matrix factorization trained on observed entries only (SGD/ALS).
- *"How do singular values relate to importance?"* — larger σ = more variance/energy in that latent factor.

---

# 10. Feature Scaling Comparison

## Practical Question 10

**Difficulty:** Easy–Medium
**Estimated Time:** 15 min
**Concepts Tested:** why scaling matters, StandardScaler vs MinMax vs Robust, effect on clustering

**Problem Statement**
Demonstrate that unscaled K-Means gives wrong clusters, then compare the three scalers and their effect on silhouette.

### Cell 1 — Data with wildly different feature ranges
```python
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score

rng = np.random.default_rng(42)
age    = rng.integers(18, 70, 300)                    # range ~50
income = rng.integers(10_000, 2_000_000, 300)         # range ~2,000,000
X = np.column_stack([age, income]).astype(float)
```

### Cell 2 — Cluster WITHOUT scaling (broken)
```python
lab_raw = KMeans(3, n_init=10, random_state=42).fit_predict(X)
print("Unscaled silhouette:", round(silhouette_score(X, lab_raw), 3))
# Clusters split almost entirely on income; age is ignored.
```

### Cell 3 — Compare the three scalers
```python
scalers = {'Standard': StandardScaler(),
           'MinMax':   MinMaxScaler(),
           'Robust':   RobustScaler()}

for name, sc in scalers.items():
    Xs = sc.fit_transform(X)
    lab = KMeans(3, n_init=10, random_state=42).fit_predict(Xs)
    print(f"{name:>9} silhouette: {silhouette_score(Xs, lab):.3f}")
```

### Cell 4 — Interpretation
```
Unscaled: income (range 2M) dominates → age contributes ~nothing.
StandardScaler: both features contribute → clusters reflect age & income.
MinMaxScaler: bounds to [0,1]; fine here but fragile if outliers exist.
RobustScaler: best when extreme incomes (outliers) are present.
Rule: ALWAYS scale before distance-based clustering.
```

## Complexity
- Scaling is **O(n·d)**; negligible vs the clustering itself.

## Alternative Solution
- **`Normalizer`** (L2) — scales each *row* to unit norm; use when direction matters (cosine-style), not per-feature scaling.
- Log-transform highly skewed features before scaling.

## Interview Variations
- Inject outliers and show MinMax collapsing while Robust survives.
- Show the same effect on **PCA** (unscaled PCA loads entirely on the big-range feature).
- Demonstrate leakage: scaling before train/test split vs inside a Pipeline.

## Common Follow-up Questions
- *"Which scaler by default?"* — StandardScaler.
- *"Outliers present — which?"* — RobustScaler.
- *"Why is unscaled K-Means wrong here?"* — Euclidean distance dominated by income's huge range.
- *"How do you prevent scaling leakage?"* — fit on train only / use a Pipeline.

---

# 11. End-to-End Project: Customer Segmentation

## Practical Question 11

**Difficulty:** Hard (the flagship notebook question)
**Estimated Time:** 45–60 min
**Concepts Tested:** the *entire* unsupervised pipeline — load → EDA → scale → choose K → cluster → reduce → interpret

**Problem Statement**
You're given a mall/customer dataset (age, annual income, spending score). Segment customers into actionable groups and describe each segment for the marketing team.

This is the **complete workflow** examiners most often assign. Present it as ordered cells.

### Cell 1 — Import libraries
```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
np.random.seed(42)
```

### Cell 2 — Load dataset
```python
# Typical columns: CustomerID, Gender, Age, Annual_Income, Spending_Score
df = pd.read_csv('Mall_Customers.csv')      # or build a synthetic one
df.head()
```

### Cell 3 — Explore (EDA)
```python
print(df.describe())
print(df.isnull().sum())                    # check missing values
sns.pairplot(df[['Age','Annual_Income','Spending_Score']])
plt.show()
```
*EDA reveals feature ranges, missing values, and visible groupings — guides preprocessing.*

### Cell 4 — Data cleaning & feature selection
```python
df = df.dropna()
X = df[['Age', 'Annual_Income', 'Spending_Score']].copy()
# (encode Gender only if you include it: df['Gender'].map({'Male':0,'Female':1}))
```

### Cell 5 — Feature scaling
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # required before K-Means
```

### Cell 6 — Choose K (elbow + silhouette)
```python
inertias, sils = [], []
Ks = range(2, 11)
for k in Ks:
    km = KMeans(k, n_init=10, random_state=42).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, km.labels_))

fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].plot(Ks, inertias, 'o-'); ax[0].set_title('Elbow')
ax[1].plot(Ks, sils, 'o-'); ax[1].set_title('Silhouette')
plt.show()
best_k = list(Ks)[int(np.argmax(sils))]
print("Chosen K:", best_k)
```

### Cell 7 — Fit final model
```python
km = KMeans(best_k, n_init=10, random_state=42)
df['Cluster'] = km.fit_predict(X_scaled)
```

### Cell 8 — Visualize clusters (PCA to 2-D)
```python
X_2d = PCA(2).fit_transform(X_scaled)
plt.scatter(X_2d[:,0], X_2d[:,1], c=df['Cluster'], cmap='viridis', s=30)
plt.title('Customer segments (PCA projection)'); plt.show()
```

### Cell 9 — Interpret each cluster (the business payoff)
```python
# profile clusters in ORIGINAL units (not scaled)
profile = df.groupby('Cluster')[['Age','Annual_Income','Spending_Score']].mean().round(1)
print(profile)
```
*Read the profile table and name each segment, e.g.:*
```
Cluster 0: high income, high spending  → "VIP / Target with premium offers"
Cluster 1: high income, low spending   → "Careful wealthy → win-back campaigns"
Cluster 2: low income, high spending   → "Young enthusiasts → loyalty programs"
Cluster 3: average income & spending   → "Standard → general marketing"
```

### Cell 10 — Deliverable / next steps
```
- Export segment labels back to CRM for targeting.
- Monitor: re-cluster quarterly; watch for drift.
- Validate segments with the marketing team (domain check).
```

## Complexity
- Dominated by K-Means fits across the K sweep and the O(n²) silhouette: fine for a few thousand customers.

## Alternative Solution
- **RFM analysis** (Recency, Frequency, Monetary) as engineered features before clustering.
- **Hierarchical clustering** for a dendrogram-based segment story on small data.
- **GMM** for soft membership ("60% VIP, 40% standard").

## Interview Variations
- Add **Gender** and engineered features (e.g. income-per-age).
- Segment with **DBSCAN** and compare (does it find a "noise" outlier segment?).
- Build the whole thing as a **scikit-learn Pipeline** to avoid leakage.

## Common Follow-up Questions
- *"How did you pick K?"* — elbow + silhouette agreement + business sense.
- *"Why interpret in original units?"* — scaled centroids are meaningless to stakeholders (inverse-transform).
- *"How do you validate segments?"* — stability across runs + domain expert sign-off.
- *"How do you deploy/monitor this?"* — save scaler+model, re-score new customers, re-cluster periodically for drift.

---

# 12. End-to-End Project: Image Compression

## Practical Question 12

**Difficulty:** Medium–Hard
**Estimated Time:** 25–30 min
**Concepts Tested:** K-Means as vector quantization, applying clustering to pixels, compression ratio

**Problem Statement**
Compress an image by reducing it to **K colors** using K-Means (color quantization). Show the compressed image and compute the compression ratio.

### Cell 1 — Load image as pixels
```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

img = load_sample_image('china.jpg')        # (h, w, 3) uint8
img = np.array(img, dtype=np.float64) / 255  # normalize to [0,1]
h, w, c = img.shape
pixels = img.reshape(-1, 3)                  # (h*w, 3): each row = an RGB pixel
print("pixels:", pixels.shape)
```
*We treat every pixel as a 3-D point (R,G,B). Clustering the pixels finds K representative colors.*

### Cell 2 — Cluster colors with K-Means
```python
K = 16                                       # compress to 16 colors
km = KMeans(n_clusters=K, n_init=3, random_state=42).fit(pixels)
palette = km.cluster_centers_                # the 16 representative colors
labels = km.labels_                          # which color each pixel maps to
```
*Each centroid is one color in the new palette; each pixel is replaced by its nearest centroid.*

### Cell 3 — Rebuild the compressed image
```python
compressed = palette[labels].reshape(h, w, 3)

fig, ax = plt.subplots(1, 2, figsize=(12,5))
ax[0].imshow(img); ax[0].set_title('Original (16.7M colors)')
ax[1].imshow(compressed); ax[1].set_title(f'Compressed ({K} colors)')
for a in ax: a.axis('off')
plt.show()
```

### Cell 4 — Compression ratio
```python
orig_bits = h * w * 24                        # 24-bit color per pixel
# compressed: each pixel stores a K-index (log2 K bits) + palette (K*24 bits)
comp_bits = h * w * np.ceil(np.log2(K)) + K * 24
print(f"Compression ratio ≈ {orig_bits / comp_bits:.1f}x")
```
*With K=16, each pixel needs only 4 bits (index) instead of 24 → ~6x smaller, plus a tiny palette.*

### Cell 5 — Interpretation
```
K-Means performs VECTOR QUANTIZATION: 16.7M possible colors → K prototypes.
Larger K → better quality, less compression. Smaller K → more compression, more banding.
This is Advanced K-Means in action (color quantization application).
```

## Complexity
- K-Means over `h*w` pixels: **O(h·w·K·i)**. Use `MiniBatchKMeans` for large images.

## Alternative Solution
- **MiniBatchKMeans** — much faster on megapixel images with negligible quality loss.
- **Median-cut / octree** quantization (classic non-ML algorithms).

## Interview Variations
- Sweep K ∈ {2,4,8,16,32,64} and plot **quality vs compression**.
- Use **MiniBatchKMeans** and compare runtime.
- Reconstruct and compute **MSE / PSNR** between original and compressed.

## Common Follow-up Questions
- *"Why is K-Means suited to this?"* — centroids act as a color codebook (vector quantization).
- *"Do you need to scale RGB?"* — no; all channels already share the same 0–255 range.
- *"K=2 looks terrible — why?"* — only 2 colors can't represent the image → severe banding.
- *"How to speed up on a 4K image?"* — MiniBatchKMeans / subsample pixels to fit centroids.

---

# 13. Coding Questions Bank

A predicted question bank organized by difficulty. Each notes **why an interviewer asks it** — the underlying skill being probed.

## Easy

**E1. Compute the Euclidean distance between two points without libraries.**
*Why asked:* tests basic loop/formula fluency and whether you know the L2 definition.
```python
def euclidean(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
```

**E2. Given cluster labels, count how many points are in each cluster.**
*Why asked:* basic data manipulation; also checks you know DBSCAN uses −1 for noise.
```python
from collections import Counter
counts = Counter(labels)             # {-1: noise_count, 0: ..., 1: ...}
```

**E3. Standardize a feature matrix (z-score) with NumPy.**
*Why asked:* confirms you know scaling and can vectorize it.
```python
def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)
```

**E4. Fit K-Means with k=3 on given data and print the centroids.**
*Why asked:* baseline sklearn API familiarity (`fit_predict`, `cluster_centers_`, `n_init`).

**E5. Explain in code the difference between `fit`, `transform`, and `fit_transform`.**
*Why asked:* a leakage-awareness check — `fit` on train, `transform` on test.

## Medium

**M1. Implement the elbow method and return the suggested K programmatically.**
*Why asked:* model-selection reasoning + turning a visual heuristic into code (e.g. via the "knee" of inertia, or max curvature).
```python
def elbow_k(X, k_max=10):
    inertias = [KMeans(k, n_init=10, random_state=42).fit(X).inertia_
                for k in range(1, k_max+1)]
    # second difference approximates curvature; largest = elbow
    diffs = np.diff(inertias, 2)
    return int(np.argmax(diffs)) + 2
```

**M2. Given a fitted PCA, return the number of components needed for 95% variance.**
*Why asked:* explained-variance understanding.
```python
def n_components_for(pca, thresh=0.95):
    return int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= thresh) + 1)
```

**M3. Write a function that flags outliers as DBSCAN noise points.**
*Why asked:* connects clustering to anomaly detection.
```python
def dbscan_outliers(X, eps, min_samples):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return np.where(labels == -1)[0]     # indices of outliers
```

**M4. Implement cosine similarity between all pairs of rows (n×n matrix).**
*Why asked:* vectorization + high-dim/text similarity.
```python
def cosine_matrix(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return Xn @ Xn.T
```

**M5. Build a leakage-free Pipeline: scale → PCA(0.95) → KMeans.**
*Why asked:* production hygiene; the `Pipeline` prevents fitting on test data.

**M6. Given a dendrogram linkage matrix, cut it to produce exactly 4 clusters.**
*Why asked:* hierarchical clustering fluency (`fcluster(Z, 4, criterion='maxclust')`).

## Hard

**H1. Implement K-Means++ initialization from scratch.**
*Why asked:* deep understanding of why smart seeding matters and probability-proportional sampling.
```python
def kmeans_pp_init(X, k, rng):
    centroids = [X[rng.integers(len(X))]]           # first centroid random
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c)**2, axis=1) for c in centroids], axis=0)
        probs = d2 / d2.sum()                       # ∝ squared distance
        centroids.append(X[rng.choice(len(X), p=probs)])
    return np.array(centroids)
```

**H2. Implement PCA from scratch and reconstruct the data; report reconstruction error.**
*Why asked:* the linear-algebra core (covariance → eig → project → inverse-project).
```python
def pca_reconstruct(X, k):
    Xc = X - X.mean(0)
    cov = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    W = vecs[:, np.argsort(vals)[::-1][:k]]
    Z = Xc @ W                        # project
    X_rec = Z @ W.T + X.mean(0)       # reconstruct
    return X_rec, np.mean((X - X_rec)**2)   # MSE reconstruction error
```

**H3. Build a mini recommender: predict a user's missing rating via SVD latent factors.**
*Why asked:* ties SVD to a real application; tests mean-centering and dot-product prediction.

**H4. Given two clusterings, compute the Adjusted Rand Index from scratch (or explain it).**
*Why asked:* external evaluation understanding and combinatorics.

**H5. Implement a silhouette score for a single sample from scratch.**
*Why asked:* proves you truly understand cohesion `a` vs separation `b`.
```python
def silhouette_one(X, labels, i):
    same = labels == labels[i]
    a = np.mean([np.linalg.norm(X[i]-X[j]) for j in np.where(same)[0] if j != i])
    b = min(np.mean(np.linalg.norm(X[i]-X[labels==c], axis=1))
            for c in set(labels) if c != labels[i])
    return (b - a) / max(a, b)
```

**H6. Cluster a large sparse text corpus: TF-IDF → TruncatedSVD (LSA) → KMeans; print top terms per cluster.**
*Why asked:* the full NLP unsupervised stack; why TruncatedSVD (not PCA) on sparse data.

---

## Final practical checklist (say/do these in any lab exam)

- [ ] **Set a random seed** at the top (`np.random.seed(42)`, `random_state=42`).
- [ ] **Scale features** before K-Means / DBSCAN / PCA.
- [ ] Use **`n_init=10`** and **`init='k-means++'`** for K-Means.
- [ ] Justify **K** with elbow **and** silhouette.
- [ ] Choose **PCA components** by cumulative explained variance (≥95%).
- [ ] Tune DBSCAN **`eps`** with the k-distance plot; remember **−1 = noise**.
- [ ] Use **TruncatedSVD** (not PCA) for sparse text; **fill missing ratings with mean** (not 0).
- [ ] Never feed **t-SNE** output into a model — visualization only.
- [ ] **Interpret clusters in original units** (inverse-transform the scaler).
- [ ] Wrap preprocessing + model in a **Pipeline** to avoid data leakage.
- [ ] Always state **time & space complexity** when asked.

---

*End of practical guide. Pair with `theory.md` for concepts, definitions, and interview Q&A.*

