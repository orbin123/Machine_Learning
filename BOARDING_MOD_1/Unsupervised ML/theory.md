# Unsupervised Machine Learning — Complete Theory & Interview Guide

> **How to use this document**
> This is a first-principles study guide for technical interviews, viva, written assessments, and lab exams. Every topic follows the same structure: *What → Why → How → Internal Working → Advantages → Limitations → Applications → Interview Questions with Model Answers → Common Mistakes → Related Concepts.*
> Read it slowly. Do not memorize definitions — understand the *reasoning*, because interviewers probe the "why", not the "what".

---

## Table of Contents

1. [Introduction to Unsupervised Learning](#1-introduction-to-unsupervised-learning)
2. [Distance & Similarity Measures](#2-distance--similarity-measures)
3. [K-Means Clustering](#3-k-means-clustering)
4. [Advanced K-Means](#4-advanced-k-means)
5. [Hierarchical Clustering](#5-hierarchical-clustering)
6. [DBSCAN](#6-dbscan)
7. [Comparing Clustering Algorithms & Evaluation Metrics](#7-comparing-clustering-algorithms--evaluation-metrics)
8. [Curse of Dimensionality & Dimensionality Reduction](#8-curse-of-dimensionality--dimensionality-reduction)
9. [Principal Component Analysis (PCA)](#9-principal-component-analysis-pca)
10. [Advanced PCA](#10-advanced-pca)
11. [t-SNE](#11-t-sne)
12. [Singular Value Decomposition (SVD)](#12-singular-value-decomposition-svd)
13. [Feature Scaling Before Clustering](#13-feature-scaling-before-clustering)
14. [Cluster Interpretation](#14-cluster-interpretation)
15. [Outlier / Anomaly Detection](#15-outlier--anomaly-detection)
16. [Master Interview Checklist](#16-master-interview-checklist)

---

# 1. Introduction to Unsupervised Learning

## What is it?

Unsupervised learning is a family of machine-learning techniques where the algorithm is given **only input data (`X`) with no labels (`y`)** and is asked to discover structure, patterns, or organization in that data on its own.

Think of it like this: in **supervised learning** you hand a student flashcards where each question already has the correct answer on the back — the student learns the mapping from question → answer. In **unsupervised learning** you hand the student a shoebox full of unlabeled photographs and say *"organize these into groups that make sense."* Nobody told the student what the groups should be. The student must invent the grouping criteria by noticing which photos *resemble* each other.

Formally: given a dataset `X = {x₁, x₂, …, xₙ}` where each `xᵢ` is a feature vector and **there is no target variable**, the goal is to model the underlying structure or distribution of the data — for example by grouping similar points (clustering) or by finding a lower-dimensional representation (dimensionality reduction).

## Why is it needed?

The single most important real-world motivation: **labeled data is expensive, and unlabeled data is everywhere.**

- Labeling requires humans. To train a supervised fraud detector you need thousands of transactions each hand-marked "fraud" or "legit" by an analyst. That costs money and time.
- The world generates oceans of *unlabeled* data every second — clickstreams, sensor logs, images, text, transactions. Roughly 80–90% of enterprise data has no labels.
- Sometimes **you don't even know what the labels should be.** A marketing team may not know how many customer segments exist or what defines them. Unsupervised learning *discovers* the segments instead of assuming them.

So unsupervised learning solves three broad problems:
1. **Discovery** — "What natural groups exist in my customers?" (clustering)
2. **Compression / simplification** — "Can I represent this 1000-feature dataset with 20 features and lose almost nothing?" (dimensionality reduction)
3. **Anomaly detection** — "Which points don't fit any pattern?" (outlier detection)

## How does it work?

At a high level every unsupervised method rests on one idea: **a notion of similarity or structure.**

```
        Unsupervised Learning
        ├── Clustering               → group similar points together
        │     (K-Means, Hierarchical, DBSCAN)
        ├── Dimensionality Reduction → compress features, keep information
        │     (PCA, t-SNE, SVD)
        └── Association Rule Learning → find "items bought together" rules
              (Apriori, FP-Growth)   → e.g. Market Basket Analysis
```

- **Clustering** partitions data into groups such that points in the same group are *similar* (small distance) and points in different groups are *dissimilar* (large distance).
- **Dimensionality reduction** finds a new, smaller set of axes that still captures most of the variation in the data.
- **Association rule learning** finds co-occurrence patterns like *"customers who buy bread and butter also buy milk."*

## Internal Working

There is no single algorithm, but all of them share this behind-the-scenes loop:

1. Define a **similarity/distance function** (Euclidean, cosine, etc.).
2. Define an **objective** that encodes what "good structure" means (e.g. minimize within-cluster variance, or maximize variance captured along new axes).
3. Use an **optimization procedure** (iterative refinement, eigen-decomposition, gradient descent) to search for the structure that best satisfies the objective.

Because there are no labels, there is **no ground-truth error to minimize**. Instead the algorithm optimizes an *internal* criterion. This is exactly why evaluation is hard — we cannot compute accuracy, so we rely on internal metrics (silhouette, inertia) or downstream business value.

## Supervised vs Unsupervised — the core contrast

| Aspect | Supervised Learning | Unsupervised Learning |
|---|---|---|
| Data | Labeled `(X, y)` | Unlabeled `X` only |
| Goal | Predict `y` for new `X` | Discover structure in `X` |
| Feedback signal | Ground-truth labels | Internal criterion (no ground truth) |
| Typical tasks | Classification, Regression | Clustering, Dim. reduction, Anomaly detection |
| Evaluation | Accuracy, F1, RMSE (objective) | Silhouette, Davies-Bouldin, human judgment (subjective) |
| Example | Spam vs not-spam | "How many kinds of email users do I have?" |

There is also **semi-supervised learning** (a few labels + lots of unlabeled data) and **self-supervised learning** (labels generated automatically from the data itself, e.g. predicting a masked word) — worth naming in interviews as the middle ground.

## Advantages

- Works on the vast majority of data that has **no labels**.
- **Discovers unknown patterns** you didn't think to look for.
- Excellent for **exploratory data analysis** — a first look at what's in a dataset.
- Great preprocessing step: reduce dimensions or create cluster-based features before supervised learning.
- Naturally suited to **anomaly detection**, since anomalies are simply points that fit no cluster.

## Limitations

- **Evaluation is hard and subjective** — no accuracy score, results need human interpretation.
- Results can be **ambiguous** — different algorithms (or different `K`) give different valid-looking answers.
- **Sensitive to preprocessing** — scaling, distance metric, and hyperparameters strongly change output.
- Harder to know when you're "done" or "correct."
- Can find patterns that are **statistically real but business-meaningless.**

## Real-world Applications

- **Customer Segmentation** — retailers/banks group users to target marketing (K-Means, hierarchical).
- **Recommendation Systems** — Netflix/Amazon use SVD/matrix factorization on user-item matrices.
- **Fraud & Anomaly Detection** — banks flag transactions that fit no normal cluster (DBSCAN, Isolation Forest).
- **Market Basket Analysis** — supermarkets find products bought together (association rules).
- **Image Compression** — reduce colors via K-Means color quantization.
- **Topic Modeling** — group documents by theme (LSA/SVD, LDA).
- **Gene Expression Analysis** — cluster genes/patients with similar expression profiles.
- **Data Visualization** — project 1000-D embeddings to 2-D with t-SNE/PCA.

## Interview Questions

**Beginner**
1. What is unsupervised learning and how does it differ from supervised learning?
2. Give three real-world applications of unsupervised learning.
3. Name the two main categories of unsupervised learning.

**Intermediate**
4. Why is labeled data a bottleneck, and how does unsupervised learning help?
5. Why is evaluating an unsupervised model harder than a supervised one?
6. What is the difference between clustering and dimensionality reduction?

**Advanced / "Why" / Scenario**
7. You have 5 million unlabeled customer records. Walk me through how you'd extract value.
8. Why can two clustering algorithms give completely different but equally "valid" results on the same data?
9. Where does semi-supervised learning fit between supervised and unsupervised?

**Comparison**
10. Compare supervised vs unsupervised vs reinforcement learning in one sentence each.

## Model Answers

**Q1 — What is unsupervised learning and how does it differ from supervised learning?**
Unsupervised learning trains on input data that has **no labels**; the algorithm's job is to find inherent structure — groups, low-dimensional representations, or anomalies — rather than to predict a known target. The key difference is the *feedback signal*: supervised learning has ground-truth labels and optimizes a measurable error (accuracy, RMSE), whereas unsupervised learning has no ground truth and instead optimizes an internal criterion such as within-cluster variance. As a consequence, supervised results can be scored objectively, while unsupervised results must be judged with internal metrics or human/business interpretation. Example: supervised = "is this email spam?" (labels exist); unsupervised = "what natural groups of email behavior exist?" (no labels).

**Q4 — Why is labeled data a bottleneck, and how does unsupervised learning help?**
Labels require human effort — someone must manually annotate each example, which is slow, costly, and sometimes needs domain experts (e.g. a radiologist labeling tumors). Meanwhile organizations collect enormous volumes of *unlabeled* data automatically (logs, transactions, images). Unsupervised learning turns that unlabeled data into value directly: it can segment customers, reduce dimensionality, or flag anomalies without a single label. It's also used to *bootstrap* supervised learning — e.g. cluster the data, label a few representatives per cluster, and propagate labels — dramatically cutting labeling cost.

**Q5 — Why is evaluating an unsupervised model harder?**
Because there is no ground truth to compare against. In supervised learning you can hold out a labeled test set and compute accuracy. In unsupervised learning there is no "correct" clustering to check against, so you fall back on: (a) **internal metrics** that measure geometric quality of clusters (silhouette, Davies-Bouldin, Calinski-Harabasz), which only tell you if clusters are *compact and separated*, not if they're *meaningful*; (b) **external metrics** if you happen to have some labels (ARI, NMI); and (c) **human/business validation** — does a domain expert agree the segments are actionable? This subjectivity is the fundamental difficulty.

**Q8 — Why can two algorithms give different but equally valid results?**
Because each algorithm optimizes a *different definition of "structure."* K-Means assumes clusters are spherical blobs of similar size and minimizes within-cluster variance; DBSCAN assumes clusters are dense regions separated by sparse ones. On the same data, K-Means might split one long snake-shaped cluster into three balls, while DBSCAN keeps it as one. Neither is "wrong" — they encode different assumptions. The "right" answer depends on what structure matches your domain, which is why you must choose the algorithm based on expected cluster shape, density, and business meaning, then validate.

## Common Mistakes

- Assuming unsupervised learning is "easier" because there are no labels — evaluation and interpretation make it *harder*.
- Forgetting to **scale features** — distance-based methods are dominated by large-magnitude features.
- Treating cluster IDs as meaningful categories without validating them against domain knowledge.
- Expecting the algorithm to tell you the "true" number of groups — that's usually a human decision.
- Confusing dimensionality reduction (change the features) with feature selection (drop some features).

## Related Concepts

- [Distance & Similarity Measures](#2-distance--similarity-measures) — the foundation of all clustering.
- [Feature Scaling](#13-feature-scaling-before-clustering) — mandatory preprocessing.
- Semi-supervised & self-supervised learning — the hybrids.
- Association Rule Learning (Apriori) — the third pillar for market-basket problems.

---

# 2. Distance & Similarity Measures

## What is it?

A **distance metric** is a function `d(a, b)` that returns a number describing *how far apart* two data points are. A **similarity measure** does the opposite — it returns a large value when two points are *alike*. They are two sides of the same coin (often `similarity = 1 − normalized_distance` or `distance = 1 − similarity`).

Almost every clustering and many dimensionality-reduction methods rest entirely on this one idea: *to group "similar" points, you first need a precise, numerical definition of "similar."* Change the distance function and you change what "similar" means — and therefore change the clusters.

## Why is it needed?

Clustering algorithms don't understand your data semantically; they only see numbers. When K-Means asks *"which centroid is this point closest to?"* or DBSCAN asks *"how many points lie within `eps` of this one?"*, the word **"closest"** and **"within"** are defined by a distance metric. Pick the wrong metric and even a perfect algorithm produces nonsense.

Different data types need different metrics:
- Physical measurements (height, weight) → straight-line **Euclidean** distance.
- City-block movement / robust to outliers → **Manhattan**.
- Direction matters more than magnitude (text, ratings) → **Cosine**.
- Binary strings / categorical codes → **Hamming**.

## How does it work?

Let `a = (a₁, …, aₙ)` and `b = (b₁, …, bₙ)` be two points in n-dimensional space.

### Euclidean Distance (L2)
The ordinary straight-line distance — "as the crow flies."

```
d(a,b) = √( Σ (aᵢ − bᵢ)² )
```
Example in 2-D: distance between (0,0) and (3,4) = √(9+16) = **5**.

### Manhattan Distance (L1 / City-Block / Taxicab)
Distance if you can only move along axes, like a taxi on a grid.

```
d(a,b) = Σ |aᵢ − bᵢ|
```
Example: (0,0) to (3,4) = |3| + |4| = **7**.

### Minkowski Distance (the generalization)
A parameterized family that contains both above.

```
d(a,b) = ( Σ |aᵢ − bᵢ|ᵖ )^(1/p)
```
- `p = 1` → Manhattan
- `p = 2` → Euclidean
- `p → ∞` → Chebyshev (max coordinate difference)

So Minkowski is the "parent"; Euclidean and Manhattan are special cases.

### Cosine Similarity
Measures the **angle** between two vectors, ignoring their magnitude.

```
cos(θ) = (a · b) / (‖a‖ · ‖b‖)
```
Range: −1 (opposite) → 0 (orthogonal/unrelated) → 1 (same direction).

### Cosine Distance
```
cosine_distance = 1 − cosine_similarity
```
Note: this is **not a true metric** (it violates the triangle inequality), but it's extremely useful in practice.

### Hamming Distance
Counts the number of positions at which two equal-length sequences differ.

```
Hamming("10110", "10011") = 2   (positions 3 and 4 differ)
```
Used for binary vectors, categorical one-hot codes, DNA sequences, error-correcting codes.

### Visual intuition (Euclidean vs Manhattan)
```
      B
      |\
    4 | \  Euclidean = 5 (diagonal, dashed)
      |  \
      |___\
        3   A
   Manhattan = 3 + 4 = 7 (along the walls, solid)
```

## Internal Working

- A valid **distance metric** must satisfy four axioms:
  1. Non-negativity: `d(a,b) ≥ 0`
  2. Identity: `d(a,b) = 0 ⇔ a = b`
  3. Symmetry: `d(a,b) = d(b,a)`
  4. Triangle inequality: `d(a,c) ≤ d(a,b) + d(b,c)`
  Euclidean, Manhattan, Minkowski (p≥1), and Hamming satisfy all four. Cosine *distance* violates the triangle inequality, so it's a "semi-metric."
- **Why cosine ignores magnitude:** dividing by the norms `‖a‖‖b‖` normalizes both vectors to unit length before comparing, so only *direction* remains. This is why two documents of very different lengths but the same topic score as highly similar.
- **The high-dimensional catch:** as dimensions grow, Euclidean distances between all pairs of points become nearly equal (distance concentration). Cosine and Manhattan often behave better in high dimensions — a direct consequence of the [Curse of Dimensionality](#8-curse-of-dimensionality--dimensionality-reduction).

## When to Use Each (the decision table interviewers love)

| Metric | Best for | Sensitive to | Notes |
|---|---|---|---|
| **Euclidean** | Low-dim numeric data, continuous features | Scale, outliers | Default for K-Means |
| **Manhattan** | Grid-like data, more robust to outliers | Scale | Less affected by extreme values than L2 |
| **Minkowski** | Tunable general case | Choice of `p` | Parent of L1/L2 |
| **Cosine** | Text/TF-IDF, embeddings, high-dim sparse | Direction only (ignores magnitude) | Great when magnitude is noise |
| **Hamming** | Binary/categorical/DNA | Position-wise mismatches | Requires equal length |

## Advantages

- Give you a **precise, tunable definition of similarity** for any data type.
- Cheap to compute (mostly vector arithmetic).
- Choosing the right one can dramatically improve clustering quality *without changing the algorithm*.

## Limitations

- **Scale-dependent** (except cosine on normalized data) — a feature in the thousands dominates one in the 0–1 range. → must scale first.
- Euclidean degrades badly in **high dimensions**.
- Cosine ignores magnitude, which is sometimes exactly the information you need.
- Hamming needs equal-length, comparable-position sequences.

## Real-world Applications

- **Cosine similarity** → search engines, document/text similarity, recommendation of "similar items," semantic search over embeddings.
- **Euclidean** → K-Means customer segmentation, image pixel clustering.
- **Manhattan** → routing/logistics on grids, robust clustering with outliers.
- **Hamming** → spell-checkers, DNA sequence comparison, deduplication, error detection.

## Interview Questions

**Beginner**
1. What is the difference between Euclidean and Manhattan distance?
2. What does cosine similarity measure that Euclidean distance does not?

**Intermediate**
3. How is Minkowski distance related to Euclidean and Manhattan?
4. Why is cosine similarity preferred for text data?
5. Is cosine distance a true distance metric? Why or why not?

**Advanced / Scenario / "Why"**
6. Why does Euclidean distance become unreliable in high-dimensional space?
7. You're clustering TF-IDF document vectors. Which metric do you choose and why?
8. Two users rated 3 movies: User A = [5,5,5], User B = [1,1,1]. Compare them using Euclidean vs cosine and explain the difference in interpretation.

**Comparison**
9. When would Manhattan beat Euclidean?

## Model Answers

**Q2 — What does cosine similarity measure that Euclidean does not?**
Cosine measures the **angle** (direction) between two vectors, completely ignoring their magnitude, whereas Euclidean measures straight-line distance, which is dominated by magnitude. This matters when the *pattern* matters more than the *scale*. Classic example: two documents about the same topic, one short and one long. Their word-count vectors point in nearly the same direction (same topic) but have very different lengths (different sizes). Euclidean says they're far apart (because of length); cosine says they're very similar (because of direction). For topic similarity, cosine is correct.

**Q5 — Is cosine distance a true metric?**
No. Cosine distance = 1 − cosine similarity satisfies non-negativity, identity, and symmetry, but it **violates the triangle inequality**, so strictly it's a *semi-metric*, not a true metric. In practice this rarely causes problems, but it means algorithms that rely on the triangle inequality for correctness or speed-ups (like some metric-tree nearest-neighbor structures) can misbehave. If you need a proper metric on directional data, use *angular distance* = arccos(cosine)/π.

**Q6 — Why does Euclidean fail in high dimensions?**
Because of **distance concentration**: as the number of dimensions increases, the ratio between the nearest and farthest neighbor distances approaches 1 — everything becomes roughly equidistant. Mathematically, the contrast `(max_dist − min_dist)/min_dist → 0`. When all points look equally far apart, "nearest neighbor" loses meaning, so Euclidean-based clustering and KNN degrade. This is a face of the curse of dimensionality. Remedies: reduce dimensions first (PCA), or switch to cosine/Manhattan which concentrate less severely.

**Q8 — Scenario: A=[5,5,5], B=[1,1,1].**
Euclidean distance = √((4²)+(4²)+(4²)) = √48 ≈ **6.93** — they look far apart. Cosine similarity = (5+5+5)/(√75·√3) = 15/15 = **1.0** — identical direction. Interpretation: if these are movie ratings, Euclidean says the users are very different (A rates high, B rates low — a *severity* difference). Cosine says they have **identical taste** (both like all three movies equally *relative to each other*). Which is "right" depends on the question: for "do they have similar preferences?" cosine is better; for "do they rate with similar intensity?" Euclidean is better.

## Common Mistakes

- Using Euclidean on **unscaled** features so one big-magnitude feature dominates.
- Using Euclidean on **high-dimensional sparse text** where cosine is far more appropriate.
- Forgetting that cosine **ignores magnitude** — sometimes magnitude is signal, not noise.
- Assuming cosine distance obeys the triangle inequality in algorithms that require it.
- Applying Hamming to sequences of different lengths.

## Related Concepts

- [K-Means](#3-k-means-clustering) and [DBSCAN](#6-dbscan) — both are parameterized by a distance metric.
- [Feature Scaling](#13-feature-scaling-before-clustering) — makes distances meaningful.
- [Curse of Dimensionality](#8-curse-of-dimensionality--dimensionality-reduction) — why metrics break in high-D.

---

# 3. K-Means Clustering

## What is it?

K-Means is a **centroid-based, partitional** clustering algorithm. You tell it how many clusters you want (`K`), and it partitions the data into exactly `K` groups such that each point belongs to the cluster whose **centroid (mean point) is nearest**. It's the "hello world" of clustering — simple, fast, and everywhere.

The name says it all: **K** clusters, each represented by the **mean** of its members.

## Why is it needed?

You often need to split data into a *fixed, known number* of groups quickly and at scale: 5 customer tiers, 16 colors for image compression, 3 product categories. K-Means is the go-to because it is:
- **Fast** — near-linear in the number of points, so it scales to millions of rows.
- **Simple** to implement and explain to stakeholders.
- **Interpretable** — each cluster has a centroid you can read as its "prototype" (e.g. "the average customer in cluster 2 spends ₹5000/month and visits weekly").

## How does it work?

K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**, also called **inertia**:

```
WCSS = Σ (over clusters k)  Σ (over points x in cluster k)  ‖x − μₖ‖²
```
where `μₖ` is the centroid of cluster `k`. Smaller WCSS = tighter, more compact clusters.

### Algorithm steps (Lloyd's algorithm)
```
1. Choose K (number of clusters).
2. Initialize K centroids (randomly, or with K-Means++).
3. ASSIGNMENT step:
      For each point, assign it to the nearest centroid (usually Euclidean).
4. UPDATE step:
      Recompute each centroid as the mean of the points assigned to it.
5. Repeat steps 3–4 until centroids stop moving (convergence)
   or max_iter is reached.
```

### Worked mini-example (1-D, K=2)
```
Data: 1, 2, 3, 10, 11, 12
Init centroids: c1=1, c2=2
Assign: {1}→c1 ; {2,3,10,11,12}→c2
Update: c1=1, c2=(2+3+10+11+12)/5=7.6
Assign: {1,2,3}→c1 ; {10,11,12}→c2
Update: c1=2, c2=11
Assign: unchanged → CONVERGED
Final clusters: {1,2,3} and {10,11,12}
```

## Internal Working

- It's an instance of **Expectation-Maximization (EM)**:
  - **E-step** = assignment (given centroids, assign points).
  - **M-step** = update (given assignments, recompute centroids as means).
- Each iteration **provably lowers or keeps WCSS** the same, so it **always converges** — but only to a **local minimum**, not necessarily the global best. That's why the result depends on initialization.
- **Why the mean?** For each fixed assignment, the point that minimizes the sum of squared Euclidean distances to a group is exactly that group's **mean**. So the "update to mean" step is the mathematically optimal M-step *for the squared-Euclidean objective.* This is also why K-Means is tied to Euclidean distance — the mean minimizes L2, not L1.
- **Convergence criterion:** centroids move less than a tolerance `tol`, or assignments stop changing, or `max_iter` reached.
- **Complexity:** `O(n · K · d · i)` where n=points, K=clusters, d=dimensions, i=iterations. Effectively linear in n → scalable.

## Choosing the Right Number of Clusters (K)

K-Means can't tell you `K` — you must supply it. Common techniques:

### Elbow Method
Plot WCSS (inertia) vs `K`. As `K` increases, WCSS always decreases (more clusters = tighter fit). Look for the **"elbow"** — the point where the curve bends and adding more clusters gives diminishing returns.
```
WCSS
 |*
 | *
 |  *
 |   *___  ← elbow at K=3 (marginal gain flattens)
 |       *___
 |___________*____ K
   1  2  3  4  5
```

### Silhouette Score
For each point: `s = (b − a) / max(a, b)` where `a` = mean distance to points in its **own** cluster, `b` = mean distance to points in the **nearest other** cluster. Ranges −1 to +1. Average silhouette closest to **+1** ⇒ best-separated clusters. Choose the `K` with the highest average silhouette. More rigorous than the elbow.

### Inertia
Inertia *is* the WCSS — the raw number the elbow plots. Lower is tighter, but it always drops with more K, so never use it alone.

## Initialization

- **Random initialization** — pick K random points as centroids. Fast but risky: a bad start → poor local minimum, and results vary run to run.
- **K-Means++** (the modern default in scikit-learn) — pick the first centroid randomly, then pick each next centroid with probability proportional to its squared distance from the nearest already-chosen centroid. This **spreads initial centroids apart**, giving faster convergence and much better, more stable results.
- **`n_init`** — scikit-learn runs the whole algorithm several times with different seeds and keeps the best (lowest inertia) run, mitigating the local-minimum problem.

## Advantages

- **Fast and scalable** — near-linear time; handles large datasets.
- **Simple** to understand, implement, and explain.
- **Interpretable centroids** — each cluster has a readable prototype.
- Works well when clusters are **roughly spherical, similar-sized, and well-separated.**

## Limitations

- **You must choose K** in advance.
- **Assumes spherical, equal-size, equal-density clusters** — fails on elongated, nested, or irregular shapes.
- **Sensitive to initialization** → local minima (mitigated by K-Means++ and `n_init`).
- **Sensitive to feature scale** → must standardize first.
- **Sensitive to outliers** — because centroids are means, one extreme point drags a centroid.
- Only works cleanly with **Euclidean-style** distance (the mean minimizes L2).
- Can produce **empty clusters** if a centroid attracts no points.

## Real-world Applications

- **Customer segmentation** — group shoppers by spend/frequency/recency (RFM).
- **Image compression / color quantization** — reduce millions of colors to K representative colors.
- **Document/topic grouping** — cluster articles by TF-IDF vectors.
- **Anomaly detection** — points far from every centroid are candidates for outliers.
- **Vector quantization** — in signal processing and as a preprocessing step.

## Interview Questions

**Beginner**
1. Explain the K-Means algorithm step by step.
2. What does the "K" and the "Means" in K-Means refer to?
3. What is inertia / WCSS?

**Intermediate**
4. How do you choose the number of clusters K?
5. What is K-Means++ and why is it better than random initialization?
6. Why must you scale features before running K-Means?

**Advanced / "Why" / Scenario**
7. Why does K-Means always converge, and why only to a local minimum?
8. Why does K-Means use the *mean* to update centroids rather than the median?
9. Your K-Means clusters look terrible on a dataset shaped like two crescents. Why, and what would you use instead?
10. K-Means gives different results every time you run it. Why, and how do you make it stable?

**Comparison**
11. Elbow Method vs Silhouette Score — which is more reliable and why?

## Model Answers

**Q1 — Explain K-Means step by step.**
(1) Choose K. (2) Initialize K centroids, ideally with K-Means++. (3) *Assignment*: assign every point to its nearest centroid using Euclidean distance. (4) *Update*: move each centroid to the mean of the points assigned to it. (5) Repeat assignment and update until centroids stop moving (convergence) or a max-iteration cap is hit. The algorithm is minimizing the within-cluster sum of squared distances (WCSS/inertia) — each iteration lowers it — so it converges, though only to a local optimum, which is why we run it several times (`n_init`) and keep the best.

**Q5 — What is K-Means++ and why better?**
Random initialization can place two centroids right next to each other or all in one region, leading to slow convergence and poor local minima. K-Means++ seeds smartly: pick the first centroid at random, then pick each subsequent centroid with probability proportional to its **squared distance from the nearest existing centroid**. This biases the initial centroids to be *spread out* across the data, which empirically gives faster convergence and lower final inertia. It's the default in scikit-learn for exactly this reason.

**Q7 — Why converge, only locally?**
K-Means minimizes WCSS. The assignment step can only reduce (or hold) WCSS because each point moves to a *closer* centroid; the update step can only reduce (or hold) WCSS because the mean is the WCSS-minimizing center for a fixed assignment. So WCSS is **monotonically non-increasing** and bounded below by 0 — it must converge. But it's a **greedy, non-convex** optimization: depending on where centroids start, it settles into whichever "valley" is nearest, which may not be the global minimum. Hence sensitivity to initialization and the use of K-Means++ / multiple restarts.

**Q8 — Why the mean, not the median?**
Because the objective is the **sum of squared Euclidean distances**, and the point minimizing squared distance to a set is its arithmetic mean (take the derivative of Σ(x−μ)², set to 0 → μ = average). If instead you minimized the sum of *absolute* distances (L1), the optimal center would be the **median**, giving the "K-Medians" algorithm, which is more robust to outliers. So the choice of mean is dictated by the squared-error objective — and it's also why K-Means is sensitive to outliers.

**Q9 — Two crescents scenario.**
K-Means fails because it assumes **convex, spherical** clusters and draws linear (Voronoi) boundaries around centroids. Two interlocking crescents are non-convex and not linearly separable by centroid distance, so K-Means will slice each crescent in half or merge parts of both. The right tool is **DBSCAN** (density-based — follows arbitrary shapes) or **spectral clustering** (uses connectivity, not centroid distance). This is the canonical "K-Means shape limitation" question.

**Q11 — Elbow vs Silhouette.**
The Elbow Method plots WCSS vs K and looks for the bend; it's quick but **subjective** — the elbow is often ambiguous, and on real data the curve is smooth with no clear kink. The Silhouette Score measures how well-separated clusters are (cohesion vs separation) and gives a **single quantitative score per K**, so you can pick the K with the maximum average silhouette objectively. Silhouette is generally more reliable but more expensive (it needs pairwise distances). Best practice: use both — elbow to narrow the range, silhouette to confirm.

## Common Mistakes

- **Not scaling features** — the classic K-Means bug; a salary feature (₹) swamps an age feature (years).
- Using the **elbow method blindly** when the curve has no clear elbow.
- Forgetting K-Means finds **local** minima → not setting `n_init` / not using K-Means++.
- Applying K-Means to **non-spherical** or **very different-sized** clusters and trusting the output.
- Interpreting cluster labels (0,1,2) as if the numbers mean order or magnitude — they're arbitrary IDs.
- Leaving **outliers** in — they distort centroids.

## Related Concepts

- [Elbow / Silhouette / evaluation metrics](#7-comparing-clustering-algorithms--evaluation-metrics)
- [Feature Scaling](#13-feature-scaling-before-clustering) — mandatory before K-Means.
- [DBSCAN](#6-dbscan) and [Hierarchical Clustering](#5-hierarchical-clustering) — alternatives for non-spherical data.
- K-Medoids / K-Medians — outlier-robust cousins.

---

# 4. Advanced K-Means

## What is it?

This section covers the practical engineering knowledge that separates someone who *ran* K-Means from someone who *understands* it: choice of distance metric, mandatory feature scaling, empty-cluster handling, and the concrete limitations you must design around. These are the exact topics interviewers use to tell juniors from seniors.

## Different distance metrics

Standard K-Means is welded to **squared Euclidean distance** because the centroid-update step (take the mean) is only optimal for L2. If you genuinely need another metric:
- **K-Medoids (PAM)** — represents each cluster by an actual data point (the *medoid*) instead of a mean, so it works with **any** distance metric (Manhattan, cosine, Gower for mixed types) and is far more **robust to outliers**. Cost: much slower — `O(n²)` per iteration.
- **Spherical K-Means** — normalizes vectors and uses cosine similarity; the standard choice for **text/TF-IDF** clustering.
- Swapping in Manhattan naively (K-Medians) changes the update to the **median**, gaining outlier robustness.

Key interview point: *"you can't just plug cosine into vanilla K-Means, because the mean no longer minimizes the objective — you must change the algorithm (K-Medoids / spherical K-Means)."*

## Feature scaling before clustering

K-Means measures distance, and distance is **not scale-invariant**. Consider customers with `age ∈ [18, 70]` and `income ∈ [10,000, 2,000,000]`. The income differences are thousands of times larger, so Euclidean distance is essentially *income only* — age is ignored. Standardizing (z-score) puts every feature on a comparable scale so each contributes fairly.

Rule: **always scale before K-Means** (StandardScaler is the usual choice; see [Feature Scaling](#13-feature-scaling-before-clustering)). The only exception is when all features are already in the same, meaningful units (e.g. pixel RGB values 0–255).

## Empty cluster handling

Sometimes a centroid ends up with **zero points** assigned (common with bad init or too-large K). WCSS is undefined for an empty cluster, so the algorithm must react. Standard strategies:
- **Reinitialize** the empty centroid to the point that is **farthest from its current centroid** (scikit-learn's approach) — this usefully "steals" a poorly-served point.
- Reinitialize to a **random** point.
- Assign it the point contributing most to the total WCSS.

scikit-learn handles this automatically, but interviewers ask *"what would you do if a cluster goes empty?"* — the answer above is what they want.

## K-Means limitations (the design constraints)

1. **Must pre-specify K.**
2. **Assumes spherical, isotropic, similarly-sized clusters** — fails on elongated/irregular/varying-density shapes.
3. **Sensitive to initialization** → local minima (mitigate: K-Means++, `n_init`).
4. **Sensitive to outliers** — means get dragged.
5. **Scale-sensitive** — must standardize.
6. **Hard assignment** — every point belongs 100% to one cluster; no notion of "60% cluster A, 40% cluster B." For soft assignment use **Gaussian Mixture Models (GMM)**.
7. **Struggles in high dimensions** — Euclidean distance concentrates; reduce dimensions (PCA) first.

## Internal Working (extra detail)

- **Mini-Batch K-Means** — for huge datasets, update centroids using small random batches instead of the full data each iteration. Massively faster with a small quality trade-off; the standard scaling trick for millions of rows.
- **Voronoi tessellation** — K-Means partitions the feature space into **Voronoi cells** (the region closest to each centroid). Boundaries are straight lines/hyperplanes, which is exactly why it can only carve out **convex** regions.

## Advantages

- With scaling + K-Means++ + Mini-Batch, K-Means is **the** default for large-scale, well-behaved clustering.
- Centroids double as a **compression codebook** (image/vector quantization).
- Cluster-distance features can be fed into supervised models.

## Limitations

(As above.) The headline: *K-Means is fast and simple but makes strong geometric assumptions; know when they're violated.*

## Real-world Applications

- **Image compression / color quantization** — cluster pixels into K colors, replace each pixel with its centroid color. 24-bit → K-color image, huge size reduction.
- **Customer segmentation** at scale with Mini-Batch K-Means.
- **Feature engineering** — add "distance to each centroid" as new features.
- **Document clustering** with spherical K-Means.

## Interview Questions

**Beginner**
1. Why must features be scaled before K-Means?
2. What happens if a cluster becomes empty?

**Intermediate**
3. Why can't you simply use cosine distance inside standard K-Means?
4. How does K-Means perform image compression?
5. What is Mini-Batch K-Means and when do you use it?

**Advanced / Scenario / "Why"**
6. K-Means keeps getting dragged by a few extreme customers. What are two fixes?
7. Explain why K-Means produces only convex (Voronoi) cluster boundaries.
8. You have mixed numeric + categorical data. Can you use K-Means? What's better?
9. Compare K-Means vs Gaussian Mixture Models.

## Model Answers

**Q1 — Why scale before K-Means?**
Because K-Means clusters by Euclidean distance, and distance is dominated by whichever feature has the largest numeric range. If income ranges in the millions and age in the tens, the algorithm effectively clusters on income alone and ignores age. Standardizing every feature (mean 0, std 1) makes each contribute proportionally, so the clusters reflect *all* features, not just the biggest-magnitude one. Skipping this is the single most common K-Means mistake.

**Q3 — Why not just use cosine in standard K-Means?**
Because the "update centroid to the mean" step is mathematically the optimal move **only for squared Euclidean distance** — the mean minimizes L2 error. Under cosine, the arithmetic mean is no longer the point that minimizes total cosine distance, so mixing cosine assignment with mean update breaks the convergence guarantee and the objective. To cluster with cosine you must change the algorithm: **spherical K-Means** (normalize vectors, use cosine, and the mean-then-renormalize update is valid) or **K-Medoids** (uses actual data points as centers and accepts any metric).

**Q6 — Outliers dragging centroids, two fixes.**
(1) **Remove or cap outliers** in preprocessing (e.g. via IQR/z-score filtering or RobustScaler), since means are sensitive to extremes. (2) **Switch to an outlier-robust algorithm** — K-Medoids/K-Medians (uses medoids/medians instead of means) or DBSCAN (which explicitly labels outliers as noise instead of forcing them into a cluster). Both address the root cause: the mean is not robust.

**Q7 — Why convex boundaries?**
Each point is assigned to the nearest centroid, so the boundary between any two clusters is the set of points equidistant from two centroids — a straight hyperplane (perpendicular bisector). The intersection of such half-spaces around a centroid forms a **Voronoi cell**, which is always **convex**. Therefore K-Means can only produce convex, linearly-bounded clusters and cannot capture concave or nested shapes — the fundamental reason it fails on crescents/rings.

**Q9 — K-Means vs GMM.**
K-Means does **hard** assignment (each point in exactly one cluster) and assumes spherical, equal-variance clusters (it only tracks centroids). GMM does **soft/probabilistic** assignment (each point has a probability of belonging to each cluster) and models each cluster as a Gaussian with its own **mean and covariance**, so it can capture **elliptical, differently-shaped, differently-sized** clusters. K-Means is actually a special case of GMM with spherical, equal covariances and hard assignment. Trade-off: GMM is more flexible but slower, needs more data, and can overfit. Use GMM when clusters overlap or are elliptical and you want membership probabilities.

## Common Mistakes

- Not scaling (again — it's *the* mistake).
- Trying to force a non-Euclidean metric into vanilla K-Means.
- Ignoring outliers before clustering.
- Using full-batch K-Means on millions of rows instead of Mini-Batch.
- Applying K-Means to categorical data via one-hot without realizing Euclidean on 0/1 is dubious (use K-Modes / K-Prototypes).

## Related Concepts

- [Feature Scaling](#13-feature-scaling-before-clustering)
- [DBSCAN](#6-dbscan) — outlier-aware, shape-flexible alternative.
- Gaussian Mixture Models, K-Medoids, K-Modes/K-Prototypes.
- [Image compression project](#14-cluster-interpretation) — color quantization.

---

# 5. Hierarchical Clustering

## What is it?

Hierarchical clustering builds a **tree (hierarchy) of clusters** instead of a flat partition. Rather than committing to `K` groups up front, it produces a full nested structure — from "every point is its own cluster" all the way up to "everything is one big cluster" — and lets you *cut* the tree at whatever level gives the number of clusters you want. The tree is visualized as a **dendrogram**.

## Why is it needed?

- You **don't know K** in advance and want to *see* the cluster structure at every granularity.
- Your data has a genuine **nested/taxonomic structure** — species → genus → family (biology), or product → subcategory → category (retail).
- You want a **visual, explainable** clustering you can show a stakeholder (the dendrogram is very intuitive).
- Small-to-medium datasets where K-Means' assumptions don't hold.

## How does it work?

### Two directions
- **Agglomerative (bottom-up)** — start with each point as its own cluster; repeatedly **merge** the two closest clusters until one remains. *This is the common one* (scikit-learn's `AgglomerativeClustering`).
- **Divisive (top-down)** — start with all points in one cluster; repeatedly **split** the least-cohesive cluster until each point is alone. Conceptually the reverse; computationally expensive and rarely used.

### Agglomerative steps
```
1. Start: N points = N clusters.
2. Compute pairwise distances between all clusters.
3. Merge the two closest clusters (per the chosen LINKAGE).
4. Update the distance matrix.
5. Repeat 3–4 until one cluster remains.
6. Cut the resulting dendrogram at the desired height → K clusters.
```

## Linkage Methods (how "distance between clusters" is defined)

This is the crux. Given two clusters A and B:

| Linkage | Distance between clusters = | Behavior |
|---|---|---|
| **Single** | distance between the **two closest** points | Finds elongated/"chaining" clusters; sensitive to noise |
| **Complete** | distance between the **two farthest** points | Compact, roughly equal-diameter clusters; sensitive to outliers |
| **Average** | **average** of all pairwise distances | Balance between single & complete |
| **Ward** | merge that **minimizes the increase in total within-cluster variance** | Tends to produce compact, equal-sized, spherical clusters; usually the best default |

- **Single linkage** suffers from **chaining** — clusters can string together through bridge points.
- **Ward** is the most popular; it's like "K-Means-flavored" hierarchical clustering (variance-minimizing) and requires Euclidean distance.

## Dendrogram

A dendrogram is the tree diagram of merges. The **height** of each merge = the distance at which those clusters joined.
```
        ┌────────┴────────┐          ← high merge (very different)
     ┌──┴──┐          ┌───┴───┐
   ┌─┴─┐  ┌┴┐        ┌┴┐    ┌─┴─┐
   A   B  C D        E F    G   H
   |________|        |________|
     low merges (very similar points join early)
```
- **Reading it:** points that merge low (short vertical lines) are very similar; merges high up join dissimilar groups.
- **Choosing clusters:** draw a **horizontal cut** across the dendrogram. The number of vertical lines it crosses = number of clusters. A good cut goes through the **tallest vertical gap** (the biggest jump in merge distance), because that's where merging becomes "expensive" — i.e., you'd be forcing dissimilar groups together.

## Internal Working

- The naive algorithm maintains an `N×N` distance matrix and repeatedly finds the minimum → **`O(n³)`** time, **`O(n²)`** memory. Optimized methods (SLINK for single, Lance-Williams update formula) reach `O(n²)`. Either way it **does not scale** past ~10k–50k points.
- The **Lance-Williams formula** lets you update cluster distances after a merge without recomputing from scratch — it's how all linkages are computed efficiently under one framework.
- **Ward's method** specifically merges the pair that yields the smallest increase in the error sum of squares (ESS), which is why it mirrors K-Means' variance objective.

## Advantages

- **No need to pre-specify K** — decide after seeing the dendrogram.
- Produces a **full hierarchy** — insight at every granularity.
- **Dendrogram is highly interpretable/visual** — great for presentations.
- Can capture **nested** structure.
- **Deterministic** — no random initialization, so same input → same output (unlike K-Means).
- Works with **any distance metric** and any linkage (flexible).

## Limitations

- **Computationally expensive** — `O(n²)` memory and `O(n²–n³)` time; infeasible for large datasets.
- **Greedy and irreversible** — a merge, once made, is never undone, so early mistakes propagate.
- **Sensitive to noise and outliers** (especially single/complete linkage).
- Choice of **linkage and metric strongly changes** the result.
- No single objective being globally optimized (unlike K-Means' WCSS).

## Real-world Applications

- **Taxonomy / phylogenetics** — building evolutionary trees from gene sequences.
- **Market/customer segmentation** on modest datasets where the hierarchy is informative.
- **Document/topic hierarchies.**
- **Gene expression clustering** (the classic heatmap-with-dendrogram in bioinformatics).
- **Social network community detection** at small scale.

## Interview Questions

**Beginner**
1. What is a dendrogram and how do you read it?
2. What's the difference between agglomerative and divisive clustering?

**Intermediate**
3. Explain single, complete, average, and Ward linkage.
4. How do you decide the number of clusters from a dendrogram?
5. Why is hierarchical clustering deterministic while K-Means is not?

**Advanced / Scenario / "Why"**
6. Why does single linkage cause "chaining", and when is that bad?
7. Why is hierarchical clustering unsuitable for a 10-million-row dataset?
8. Ward linkage requires Euclidean distance — why?
9. When would you prefer hierarchical clustering over K-Means?

**Comparison**
10. K-Means vs Hierarchical vs DBSCAN — one strength and one weakness each.

## Model Answers

**Q1 — Dendrogram, how to read.**
A dendrogram is a tree diagram showing the sequence of merges in agglomerative clustering. Leaves are individual data points; each internal node is a merge, and the **height** of that node is the distance at which the two clusters joined. Points joined near the bottom are very similar; joins near the top combine dissimilar groups. To extract clusters, cut the tree horizontally — the number of vertical lines the cut crosses is the number of clusters. You choose the cut height by finding the **largest vertical gap** between successive merges, which indicates the most "natural" separation.

**Q3 — Linkage methods.**
Linkage defines the distance between two *clusters* (not points). **Single**: the distance between their two closest members — tends to form long, chained clusters and can follow non-spherical shapes but is noise-sensitive. **Complete**: the distance between their two farthest members — produces compact, similar-diameter clusters but is outlier-sensitive. **Average**: the mean of all cross-cluster pairwise distances — a compromise between single and complete. **Ward**: merges the pair that minimizes the increase in total within-cluster variance — yields compact, roughly equal-sized, spherical clusters and is the usual default (it needs Euclidean distance because variance is an L2 concept).

**Q6 — Single-linkage chaining.**
Single linkage defines inter-cluster distance as the *minimum* distance between any two members. So if there's a chain of closely spaced points bridging two otherwise-distinct blobs, single linkage will keep merging along that bridge — the two blobs "chain" into one elongated cluster through a thin path. This is great when clusters are genuinely elongated (like DBSCAN-friendly shapes) but bad when the bridge is just noise, because a few stray points can erroneously fuse two real clusters. Complete/Ward linkage avoid this by considering farthest points / variance.

**Q7 — Why not 10M rows?**
Because agglomerative clustering needs the pairwise distance matrix, which is `O(n²)` in memory — for 10M points that's 10¹⁴ entries, utterly infeasible — and the merge process is `O(n²)` to `O(n³)` in time. K-Means, by contrast, is roughly linear in n. For large data you use K-Means / Mini-Batch K-Means, or run hierarchical clustering only on a **sample** or on **pre-aggregated** cluster centroids.

**Q9 — Hierarchical over K-Means when?**
When (a) you don't know K and want to explore structure at multiple granularities; (b) the data is small/medium so `O(n²)` is affordable; (c) the domain is genuinely hierarchical (biology, taxonomies); (d) you want a deterministic, explainable, visual result (dendrogram) to show stakeholders; or (e) clusters aren't spherical and you can exploit single/average linkage. If the data is large or you already know K and clusters are blob-like, K-Means wins.

## Common Mistakes

- Running agglomerative clustering on a huge dataset and hitting memory limits.
- Using **single linkage** on noisy data and getting one giant chained cluster.
- Cutting the dendrogram at an arbitrary height instead of the largest gap.
- Forgetting Ward needs Euclidean distance (pairing it with cosine, etc.).
- Not scaling features (distances still need to be comparable).

## Related Concepts

- [Distance metrics](#2-distance--similarity-measures) & [linkage] — the two knobs.
- [DBSCAN](#6-dbscan) — density-based, also handles arbitrary shapes without K.
- [Cluster evaluation](#7-comparing-clustering-algorithms--evaluation-metrics) — silhouette works here too.
- [Feature Scaling](#13-feature-scaling-before-clustering).

---

# 6. DBSCAN

## What is it?

**DBSCAN** = *Density-Based Spatial Clustering of Applications with Noise*. Instead of grouping by distance-to-a-center (like K-Means), it groups by **density**: a cluster is a **dense region of points** separated from other dense regions by **sparse regions**. Points in sparse areas that belong to no dense region are labeled **noise (outliers)**.

Crucially, DBSCAN **does not require you to specify the number of clusters** — it discovers however many dense regions exist — and it can find **arbitrarily shaped** clusters (crescents, rings, spirals) that defeat K-Means.

## Why is it needed?

K-Means and hierarchical clustering have three pain points DBSCAN solves:
1. You must choose K → DBSCAN finds it automatically.
2. They assume convex/spherical clusters → DBSCAN follows any shape.
3. They force **every** point into a cluster, mangling outliers → DBSCAN explicitly labels outliers as noise.

So DBSCAN is the natural choice for **anomaly detection** and for data with **irregular cluster shapes and noise** — geospatial data, fraud, sensor readings.

## How does it work?

Two hyperparameters define "dense":
- **`eps` (ε)** — the radius of a neighborhood around a point.
- **`min_samples`** — the minimum number of points required within `eps` (including the point itself) for a region to be considered dense.

Every point is classified as one of three types:
- **Core point** — has **at least `min_samples`** points within its `eps` neighborhood. These are the interior of clusters.
- **Border point** — within `eps` of a core point but **doesn't itself** have enough neighbors. The edge of a cluster.
- **Noise point** — neither core nor border. An outlier.

```
   .  .  .            ● = core (dense neighborhood)
  . ●● ● .            ○ = border (near a core, but sparse itself)
 .  ●●●● ○      ✕     ✕ = noise (isolated)
  . ●● ● .
   .  .  .
```

### Algorithm
```
1. Pick an unvisited point p.
2. Find all points within eps of p (its neighborhood).
3. If |neighborhood| >= min_samples → p is a CORE point:
      start a new cluster, and expand it by recursively
      absorbing all density-reachable points (core neighbors
      and their neighbors...), pulling in border points too.
4. Else → mark p as NOISE (may later become a border point).
5. Repeat until all points are visited.
```

Two key relations:
- **Directly density-reachable:** q is within `eps` of core point p.
- **Density-connected:** p and q are linked through a chain of core points. A cluster = a maximal set of density-connected points.

## Internal Working

- With a spatial index (KD-tree / Ball-tree) neighborhood queries are fast, giving **`O(n log n)`** average time; worst case `O(n²)`.
- **Border-point ambiguity:** a border point reachable from two clusters is assigned to whichever core reaches it first → results can be *slightly* order-dependent for border points (but core-point cluster structure is stable).
- **Choosing `eps` — the k-distance graph:** compute each point's distance to its `k`-th nearest neighbor (k = `min_samples`), sort ascending, and plot. Look for the **"knee"/elbow** — the sharp rise — and set `eps` there. Below the knee = dense (intra-cluster) distances; above = sparse (noise) distances.
- **Choosing `min_samples`:** rule of thumb `min_samples ≥ dimensions + 1`, often `2 × dimensions`. Larger values → more points labeled noise, more robust to noise.

## Hyperparameters

| Param | Meaning | Effect if too small | Effect if too large |
|---|---|---|---|
| `eps` | Neighborhood radius | Many points become noise; clusters fragment | Clusters merge together; noise disappears |
| `min_samples` | Density threshold | Noise gets absorbed into clusters | More points labeled noise; small clusters vanish |

## Advantages

- **No need to specify K** — discovers the number of clusters.
- Finds **arbitrarily shaped** clusters (non-convex, nested outlines).
- **Robust to outliers** — explicitly labels them as noise (built-in anomaly detection).
- Only two intuitive hyperparameters.
- Deterministic core structure (unlike K-Means' random init).

## Limitations

- **Sensitive to `eps` and `min_samples`** — bad values give garbage; tuning needs the k-distance plot.
- **Struggles with clusters of varying density** — a single global `eps` can't be right for both a dense and a sparse cluster simultaneously. (Fix: **HDBSCAN**, which handles variable density.)
- **Degrades in high dimensions** — density and distance both become meaningless (curse of dimensionality); reduce dimensions first.
- **Border points can be ambiguous / order-dependent.**
- Not great when clusters are close and similar-density but you want them split — density can't separate them.

## Real-world Applications

- **Anomaly / fraud detection** — noise points *are* the anomalies.
- **Geospatial clustering** — GPS hotspots, crime maps, delivery clustering, identifying points of interest.
- **Image segmentation** and object detection in dense point clouds (LIDAR).
- **Network intrusion detection.**
- Any domain with **irregular shapes + noise.**

## Interview Questions

**Beginner**
1. What do core, border, and noise points mean in DBSCAN?
2. What are the two hyperparameters of DBSCAN?

**Intermediate**
3. How does DBSCAN decide how many clusters exist?
4. How do you choose a good `eps`?
5. Why is DBSCAN good at handling outliers?

**Advanced / Scenario / "Why"**
6. Why does DBSCAN fail on clusters with very different densities, and what fixes it?
7. Why does DBSCAN struggle in high-dimensional data?
8. You have two crescent-shaped clusters plus scattered noise. Compare K-Means and DBSCAN here.
9. Are DBSCAN results fully deterministic? Explain the border-point subtlety.

**Comparison**
10. K-Means vs DBSCAN — list four fundamental differences.

## Model Answers

**Q1 — Core, border, noise.**
A **core point** has at least `min_samples` points (including itself) within radius `eps` — it sits in a dense region and forms the backbone of a cluster. A **border point** lies within `eps` of a core point but doesn't have enough neighbors to be core itself — it's on the fringe of a cluster. A **noise point** is neither: it's in a sparse region reachable from no core point, so DBSCAN labels it an outlier (cluster label −1 in scikit-learn). Clusters are built by connecting core points that are within `eps` of each other and attaching their border points.

**Q4 — Choosing eps.**
Use the **k-distance graph**: for each point compute the distance to its k-th nearest neighbor (with k = `min_samples`), sort these distances in ascending order, and plot them. The curve stays low and flat for points inside clusters, then rises sharply at the "knee" where you transition to sparse/noise points. Set `eps` at that knee. This is the density analog of the elbow method. Also always **scale features first**, since `eps` is a single distance threshold applied across all dimensions.

**Q6 — Varying density failure + fix.**
DBSCAN uses one global `eps` and one `min_samples` to define density everywhere. If one cluster is very dense and another is sparse, no single `eps` works: a small `eps` correctly finds the dense cluster but shreds the sparse one into noise; a large `eps` captures the sparse cluster but merges the dense one with its neighbors. The fix is **HDBSCAN** (Hierarchical DBSCAN), which builds a hierarchy over a range of density levels and extracts clusters at *locally* appropriate densities, so it handles variable-density data without a single global `eps`.

**Q8 — Crescents + noise scenario.**
K-Means fails: it assumes spherical, convex clusters and would slice each crescent and misassign points, and it has no concept of noise — every scattered point gets forced into a cluster, distorting centroids. DBSCAN excels: the crescents are **dense connected regions**, so density-reachability traces each crescent's full arc regardless of shape, and the scattered points fall in sparse regions and are correctly labeled **noise**. This scenario is the textbook demonstration of DBSCAN's two big wins — arbitrary shapes and outlier handling.

**Q10 — K-Means vs DBSCAN.**
(1) **K**: K-Means needs it specified; DBSCAN discovers it. (2) **Shape**: K-Means only convex/spherical; DBSCAN any shape. (3) **Outliers**: K-Means forces all points into clusters; DBSCAN labels sparse points as noise. (4) **Basis**: K-Means groups by distance to centroids (partitional); DBSCAN groups by density connectivity. Add-on: K-Means scales near-linearly and handles high-D better; DBSCAN struggles with varying density and high dimensions but needs no K.

## Common Mistakes

- Not scaling features before DBSCAN (a global `eps` on unscaled data is meaningless).
- Guessing `eps` instead of using the k-distance elbow.
- Applying DBSCAN to **high-dimensional** data without reducing dimensions.
- Expecting it to separate clusters of very **different densities** (use HDBSCAN).
- Forgetting that label `−1` means noise, not a cluster.

## Related Concepts

- **HDBSCAN** — variable-density successor.
- [Outlier Detection](#15-outlier--anomaly-detection) — DBSCAN noise, Isolation Forest.
- [K-Means](#3-k-means-clustering) / [Hierarchical](#5-hierarchical-clustering) — the alternatives.
- [Curse of Dimensionality](#8-curse-of-dimensionality--dimensionality-reduction).

---

# 7. Comparing Clustering Algorithms & Evaluation Metrics

## What is it?

Two skills interviewers test together: (a) **choosing** the right clustering algorithm for a dataset, and (b) **measuring** how good a clustering is when you have no labels. This section is your decision framework plus the four metrics you must know cold.

## Why is it needed?

There is no universally best clustering algorithm — each encodes different assumptions. And because unsupervised learning has no ground truth, you need **internal validity metrics** to compare results objectively rather than eyeballing plots. Being able to say *"I chose DBSCAN because the clusters are non-convex with noise, and I validated with silhouette = 0.62 and Davies-Bouldin = 0.7"* is exactly what separates a strong candidate.

## Algorithm comparison

| Algorithm | Best for | Cluster shape | Needs K? | Handles noise? | Scalability | Key weakness |
|---|---|---|---|---|---|---|
| **K-Means** | Large, spherical, well-separated | Convex/spherical | Yes | No | Excellent (≈linear) | Assumes shape/size; scale & init sensitive |
| **Hierarchical** | Small data, nested structure, visualization | Flexible (linkage-dependent) | No (cut tree) | No | Poor (`O(n²)`) | Expensive; greedy merges |
| **DBSCAN** | Irregular shapes + outliers | Arbitrary | No | **Yes** | Good (`O(n log n)`) | Varying density; high-D; `eps` tuning |
| **GMM** | Overlapping, elliptical clusters, soft membership | Elliptical | Yes | Soft (low prob) | Moderate | Slower; can overfit |

**Decision heuristics**
- Know K, big data, blobs → **K-Means**.
- Small data, want a hierarchy/visual → **Hierarchical**.
- Weird shapes + outliers, don't know K → **DBSCAN**.
- Overlapping/elliptical, want probabilities → **GMM**.

## Evaluation Metrics (internal — no labels needed)

### 1. Silhouette Score
For each point: `s = (b − a) / max(a, b)`, where `a` = mean intra-cluster distance (cohesion), `b` = mean distance to the nearest *other* cluster (separation).
- Range **−1 → +1**. Near **+1** = well inside its cluster; **0** = on a boundary; **negative** = probably misassigned.
- Report the **average** over all points. **Higher is better.** Great for choosing K.

### 2. Davies-Bouldin Index (DBI)
Average over clusters of the "worst-case similarity" — each cluster's ratio of *within-cluster scatter* to *between-cluster separation*, taking the most similar other cluster.
- **Lower is better** (0 is ideal). Compact, well-separated clusters → low DBI.
- Cheap to compute; a good complement to silhouette.

### 3. Calinski-Harabasz Index (Variance Ratio Criterion)
Ratio of **between-cluster variance** to **within-cluster variance** (scaled by degrees of freedom).
- **Higher is better.** Rewards clusters that are dense internally and far apart.
- Fast; works well with convex clusters (biased toward K-Means-style solutions).

### 4. Inertia / WCSS
Sum of squared distances of points to their assigned centroid.
- **Lower is better**, but it **monotonically decreases** with more clusters, so **never use it alone** to pick K — only via the **elbow**.

### External metrics (only if you *have* some labels)
- **Adjusted Rand Index (ARI)** — agreement between predicted and true clusterings, corrected for chance (−1 to 1).
- **Normalized Mutual Information (NMI)** — shared information between clusterings (0 to 1).
Mention these when the interviewer says "what if you had labels to validate?"

## How does it work (choosing K with metrics)

```
For K in 2..10:
    fit clustering with K
    record: silhouette (max), davies-bouldin (min),
            calinski-harabasz (max), inertia (elbow)
Pick K where silhouette peaks AND DBI troughs AND CH peaks.
Cross-check with the elbow of inertia and with business sense.
```
When metrics disagree, prefer **silhouette** for separation quality and **business interpretability** as the tie-breaker.

## Advantages / Limitations of the metrics

- **Silhouette** — intuitive, bounded, good for K selection; but `O(n²)` (slow on big data), assumes convex clusters, can mislead on density-based clusters.
- **Davies-Bouldin** — fast, simple; but favors convex/spherical and is distance-scale dependent.
- **Calinski-Harabasz** — fast; but biased toward more clusters and convex shapes.
- **Inertia** — trivial to compute; but decreases with K, only usable via elbow, K-Means-specific.
- **All internal metrics** measure *geometry*, not *meaning* — high silhouette ≠ business-useful clusters.

## Real-world Applications

- Automated **K selection** in production pipelines (grid over K, pick best silhouette).
- **Model selection** — comparing K-Means vs DBSCAN vs GMM on the same data.
- **Monitoring** — tracking cluster quality drift over time.

## Interview Questions

**Beginner**
1. Name three internal clustering evaluation metrics and their "better" direction.
2. What does a silhouette score of −0.3 tell you?

**Intermediate**
3. Why can't you use inertia alone to choose K?
4. Difference between internal and external evaluation metrics?
5. How do you pick between K-Means, DBSCAN, and hierarchical clustering?

**Advanced / Scenario / "Why"**
6. Silhouette says K=4 is best but the business expects 3 segments. What do you do?
7. Why might silhouette score mislead you on DBSCAN's density-based clusters?
8. Your metrics disagree (silhouette likes K=3, Calinski-Harabasz likes K=6). How do you decide?

**Comparison**
9. Silhouette vs Davies-Bouldin vs Calinski-Harabasz — one line each.

## Model Answers

**Q2 — Silhouette −0.3 meaning.**
A negative silhouette means the point is, on average, **closer to a neighboring cluster than to its own** — it's likely **misclassified**. If many points score negative, the clustering (or the chosen K) is poor: clusters overlap or K is wrong. Silhouette near 0 means points sit on cluster boundaries (ambiguous), and near +1 means clean, well-separated assignment. So −0.3 is a red flag prompting you to try a different K, algorithm, or better feature scaling/reduction.

**Q3 — Why not inertia alone.**
Inertia (WCSS) is the sum of squared distances to centroids, and it **always decreases** as K increases — at K = n (one cluster per point) it hits 0. So "minimize inertia" would trivially pick the maximum K, which is useless. You can only use it via the **elbow method**: plot inertia vs K and find where the *rate* of decrease sharply flattens, indicating diminishing returns. Even then it's subjective, so pair it with silhouette or Davies-Bouldin.

**Q6 — Metric vs business (K=4 vs 3).**
Clustering serves the business, not the metric. I'd first check *how much better* K=4 is — if silhouette for 3 and 4 are close (say 0.61 vs 0.63), the gain is marginal and I'd favor the interpretable, actionable **3 segments** the business can act on. If K=4 is *dramatically* better, I'd investigate the extra cluster: maybe it's a genuine, valuable niche the business hasn't recognized, in which case I'd present it. The metric is a guide; the deliverable is *actionable, explainable* segments, so I balance statistical quality with business utility and validate with a domain expert.

**Q7 — Silhouette misleading DBSCAN.**
Silhouette assumes **convex, compact** clusters and rewards low intra-cluster distance. DBSCAN produces **arbitrarily shaped** clusters (crescents, rings) where points at opposite ends of the same cluster are far apart, inflating the intra-cluster distance `a` and pushing silhouette down — even though the clustering is *correct*. Also, DBSCAN's noise points don't fit the silhouette framework cleanly. So a low silhouette on DBSCAN doesn't necessarily mean bad clustering; for density-based methods, prefer **density-aware validation** (e.g. DBCV) or visual inspection.

## Common Mistakes

- Using inertia alone to choose K.
- Applying silhouette to non-convex (DBSCAN) clusters and trusting a low score.
- Optimizing a metric and ignoring whether clusters are **interpretable/actionable**.
- Comparing metrics computed on **unscaled** vs scaled data.
- Forgetting metrics measure geometry, not business value.

## Related Concepts

- [Elbow & Silhouette](#3-k-means-clustering) in K-Means.
- [Cluster Interpretation](#14-cluster-interpretation) — turning clusters into meaning.
- All three clustering algorithms above.

---

# 8. Curse of Dimensionality & Dimensionality Reduction

## What is it?

**Dimensionality reduction** is the process of transforming data from a high-dimensional space (many features) into a lower-dimensional one while **preserving as much meaningful information as possible.** The reason we bother is the **curse of dimensionality** — a collection of counterintuitive problems that arise as the number of features grows.

## The Curse of Dimensionality

As dimensions increase:
1. **Data becomes sparse.** The volume of the space grows exponentially, so a fixed number of points becomes vanishingly sparse. To keep the same density you'd need exponentially more data. (10 points cover a line well; 10 points in a 10-D cube cover almost nothing.)
2. **Distances concentrate.** The distance to the nearest and farthest neighbor become nearly equal — "everything is far from everything." This breaks distance-based methods (KNN, K-Means, DBSCAN).
3. **Overfitting risk rises.** More features → more parameters → models memorize noise.
4. **Computation explodes** and **visualization is impossible** beyond 3-D.

```
Points needed to keep constant density:
1-D: 10        2-D: 100        3-D: 1,000   ...   10-D: 10,000,000,000
                                         (exponential blow-up)
```

## Why is dimensionality reduction needed?

- **Beat the curse** — restore meaningful distances and reduce sparsity.
- **Visualization** — project 100-D data to 2-D/3-D to actually *see* structure.
- **Speed & memory** — fewer features = faster training, smaller models.
- **Noise reduction** — dropping low-variance directions often removes noise, improving generalization.
- **Remove multicollinearity** — decorrelate redundant features (PCA outputs are uncorrelated).
- **Storage/transmission** — compression.

## Feature Extraction vs Feature Selection (a must-know distinction)

| | Feature **Selection** | Feature **Extraction** |
|---|---|---|
| Idea | **Keep a subset** of original features, drop the rest | **Create new features** as combinations of originals |
| Interpretability | High (original features retained) | Lower (new axes are mixtures) |
| Examples | Filter (correlation), Wrapper (RFE), Embedded (Lasso) | **PCA, t-SNE, SVD**, autoencoders |
| Info from dropped features | Lost entirely | Partially retained (folded into new features) |

PCA/t-SNE/SVD are all **feature extraction** — they build new axes.

## How does it work (the two families)?

- **Linear methods (PCA, SVD)** — find new axes that are *linear combinations* of the originals, capturing maximum variance. Fast, interpretable-ish, invertible.
- **Non-linear / manifold methods (t-SNE, UMAP, autoencoders)** — unfold curved structure ("manifolds") that linear methods can't, mainly for visualization.

Underlying idea: real high-dimensional data usually lives on a **much lower-dimensional manifold** (e.g. images of faces vary along far fewer degrees of freedom than pixel count). Dimensionality reduction finds that manifold.

## Internal Working

- Linear methods rely on **linear algebra** — eigen-decomposition of the covariance matrix (PCA) or singular value decomposition of the data matrix (SVD). Both find orthogonal directions ordered by how much variance they explain.
- Manifold methods rely on **preserving neighborhoods** — t-SNE keeps nearby points nearby by modeling pairwise similarities as probabilities and minimizing a divergence.

## Advantages

- Faster training, less memory, less overfitting.
- Enables visualization and noise reduction.
- Removes correlated/redundant features.

## Limitations

- **Information loss** — you always discard *something*.
- **Reduced interpretability** — new features are mixtures of originals.
- Choosing the number of components is a judgment call.
- Non-linear methods can distort global structure or be hard to reproduce.

## Real-world Applications

- **Visualization** of embeddings (words, images, genes) in 2-D.
- **Preprocessing** before clustering/classification (PCA before K-Means).
- **Image/signal compression.**
- **Recommender systems** (SVD/matrix factorization).
- **Noise filtering** in sensor and financial data.

## Interview Questions

**Beginner**
1. What is the curse of dimensionality?
2. Difference between feature selection and feature extraction?

**Intermediate**
3. Give three reasons to reduce dimensionality.
4. Why do distance-based algorithms break in high dimensions?

**Advanced / Scenario / "Why"**
5. Why does dimensionality reduction often *improve* a downstream model's accuracy?
6. Your KNN classifier works great on 5 features but terribly on 500. Why, and what do you do?
7. What does it mean that "data lies on a low-dimensional manifold"?

## Model Answers

**Q1 — Curse of dimensionality.**
It's the set of problems that appear as feature count grows: (a) the space's volume grows exponentially, so data becomes extremely **sparse** and you'd need exponentially more samples to maintain density; (b) **distances concentrate** — nearest and farthest points become almost equidistant, so "similarity" loses meaning and distance-based methods (KNN, K-Means, DBSCAN) fail; (c) **overfitting** becomes easier with so many features; and (d) computation and visualization become intractable. The practical takeaway: more features is not always better, and reducing dimensions often helps.

**Q2 — Selection vs extraction.**
**Feature selection** keeps a *subset* of the original features and discards the rest (e.g. drop low-variance or highly correlated columns, or use Lasso/RFE) — the surviving features stay interpretable. **Feature extraction** *transforms* the data into new features that are combinations of the originals (e.g. PCA components = weighted sums of original features), typically capturing more information in fewer dimensions but at the cost of interpretability. PCA, t-SNE, and SVD are extraction methods; correlation filtering and RFE are selection methods.

**Q5 — Why DR improves accuracy.**
Two reasons. First, **noise reduction**: low-variance directions are often noise; dropping them removes distractions the model would otherwise fit, improving generalization. Second, it mitigates the **curse of dimensionality** — fewer, decorrelated features mean distances are meaningful again and the model has fewer parameters, reducing overfitting and variance. It also removes **multicollinearity** (PCA components are orthogonal), which stabilizes many models. The caveat: reduce too far and you lose signal, hurting accuracy — it's a bias/variance trade-off tuned via explained variance.

**Q6 — KNN 5 vs 500 features.**
KNN relies entirely on distances, and in 500 dimensions **distance concentration** makes all points roughly equidistant, so "nearest neighbors" are essentially random — accuracy collapses. Fixes: (1) **reduce dimensions** with PCA (keep ~95% variance) so distances regain meaning; (2) **feature selection** to drop irrelevant/redundant features; (3) switch to a metric less affected by high-D (cosine) or a model that doesn't depend on raw distances (tree-based). PCA-then-KNN is the classic remedy.

## Common Mistakes

- Confusing feature selection with extraction.
- Reducing dimensions **without scaling first** (PCA is variance-based → scale-sensitive).
- Over-reducing and destroying signal.
- Assuming more features always help.
- Using t-SNE output as *features* for a model (it's for visualization only).

## Related Concepts

- [PCA](#9-principal-component-analysis-pca), [t-SNE](#11-t-sne), [SVD](#12-singular-value-decomposition-svd).
- [Distance metrics](#2-distance--similarity-measures) & distance concentration.
- [Feature Scaling](#13-feature-scaling-before-clustering).

---

# 9. Principal Component Analysis (PCA)

## What is it?

PCA is the most widely used **linear dimensionality-reduction** technique. It finds a new set of axes — the **principal components** — that are (a) **orthogonal** (uncorrelated) and (b) ordered so that the **first captures the most variance** in the data, the second the most remaining variance, and so on. By keeping only the first few components you compress the data while retaining most of its "information" (variance).

Intuition: imagine a flat, tilted pancake of points floating in 3-D. Most of the spread is along the pancake's length and width; almost none is along its thickness. PCA rotates the axes to align with length (PC1), width (PC2), thickness (PC3), then lets you drop the near-flat thickness axis — going 3-D → 2-D with negligible loss.

## Why is it needed?

- **Compress** many correlated features into a few informative ones.
- **Visualize** high-D data in 2-D/3-D.
- **Remove multicollinearity** — components are uncorrelated by construction.
- **De-noise** — low-variance components are often noise.
- **Speed up** downstream models and reduce overfitting.

## How does it work? (step by step)

```
1. STANDARDIZE the data (mean 0, unit variance per feature).  ← critical
2. Compute the COVARIANCE MATRIX of the features (d×d).
3. Compute its EIGENVALUES and EIGENVECTORS.
4. Sort eigenvectors by descending eigenvalue.
      - Each eigenvector = a principal component (a direction).
      - Its eigenvalue = the amount of variance along that direction.
5. Pick the top k eigenvectors → projection matrix W (d×k).
6. PROJECT the data: X_reduced = X_standardized · W   (n×k).
```

- **PC1** is the direction of maximum variance. **PC2** is the direction of maximum *remaining* variance, orthogonal to PC1. Etc.
- The transformation is a **rotation** of the coordinate system to align with the directions of greatest spread.

## The linear algebra (interviewers love this)

- **Covariance matrix `Σ`** — a `d×d` symmetric matrix where entry `(i,j)` is the covariance between features i and j. It encodes how features vary together.
- **Eigenvectors of Σ** — the principal component *directions*. Because Σ is symmetric, its eigenvectors are **orthogonal**.
- **Eigenvalues of Σ** — the **variance captured** along each corresponding eigenvector. Bigger eigenvalue = more important component.
- The defining equation: `Σ v = λ v` — the eigenvector `v` is the direction Σ merely *stretches* (by factor λ) without rotating.
- Equivalent formulation via **SVD**: decompose the (centered) data matrix `X = U S Vᵀ`; the columns of `V` are the principal components and the singular values relate to eigenvalues by `λ = s²/(n−1)`. (scikit-learn's PCA actually uses SVD internally — more numerically stable than forming the covariance matrix.)

## Choosing the number of components

### Explained Variance Ratio
Each component explains `λᵢ / Σλ` of the total variance. E.g. PC1 might explain 62%, PC2 24%, PC3 8%…

### Cumulative Explained Variance
Add them up and keep enough components to reach a threshold — commonly **95%** (or 90%/99%): "keep the smallest k such that cumulative variance ≥ 0.95."

### Scree Plot
Plot eigenvalue (or explained variance) vs component index. Look for the **elbow** where the curve flattens — components after the elbow add little.
```
Var
 |*
 | *
 |  *___  ← elbow: keep first 3 components
 |      *___
 |__________*__ component
   1  2  3  4  5
```

## Internal Working (behind the scenes)

- **Why maximum variance = maximum information?** PCA equates "information" with "variance" — a direction along which points spread out lets you distinguish them; a direction where all points have the same value carries no discriminative information. Mathematically, maximizing projected variance is equivalent to **minimizing the reconstruction error** (squared distance between original points and their projection). So PC1 is simultaneously the max-variance direction *and* the best-fit line minimizing perpendicular distances.
- **Centering is mandatory** (subtract the mean) so variance is measured about the origin; **scaling** is needed so a large-unit feature doesn't dominate the covariance.
- Components are **uncorrelated** because eigenvectors of the symmetric covariance matrix are orthogonal.

## Advantages

- **Fast, deterministic, well-understood.**
- Removes correlation/redundancy; components are orthogonal.
- **Invertible** (approximately) — you can reconstruct data from components (used in compression/denoising).
- Excellent general-purpose preprocessing.
- Global structure preserved (unlike t-SNE).

## Limitations

- **Linear only** — can't capture curved/non-linear manifolds (t-SNE/UMAP/kernel PCA for that).
- **Components are hard to interpret** — each is a mixture of all original features.
- **Variance ≠ importance** — a low-variance direction might be the discriminative one for your task (PCA is unsupervised, ignores labels; LDA is the supervised alternative).
- **Sensitive to scaling and outliers** (variance is outlier-sensitive).
- Assumes the principal components (large variance) are what matter.

## Real-world Applications

- **Image compression / eigenfaces** (facial recognition).
- **Preprocessing** before clustering or classification.
- **Genomics** — reduce thousands of gene expressions to a few components.
- **Finance** — factor analysis of asset returns.
- **Visualization** of high-D datasets in 2-D.
- **Noise reduction** in signals/images.

## Interview Questions

**Beginner**
1. What is PCA and what does it do?
2. What is a principal component?
3. What is explained variance ratio?

**Intermediate**
4. Walk through the PCA algorithm step by step.
5. What role do eigenvalues and eigenvectors play in PCA?
6. Why must you standardize data before PCA?
7. How do you choose the number of components?

**Advanced / Scenario / "Why"**
8. Why does maximizing variance minimize reconstruction error?
9. Why are principal components orthogonal?
10. PCA hurt your classifier's accuracy. Give a plausible reason.
11. When would you use LDA instead of PCA?

**Comparison**
12. PCA vs t-SNE. PCA vs SVD.

## Model Answers

**Q4 — PCA step by step.**
(1) **Standardize** each feature to mean 0, unit variance. (2) Compute the `d×d` **covariance matrix** capturing how features co-vary. (3) Compute its **eigenvectors** (the principal-component directions) and **eigenvalues** (variance along each). (4) **Sort** eigenvectors by descending eigenvalue. (5) Choose the top **k** and stack them into a projection matrix `W`. (6) **Project** the standardized data onto `W` to get the k-dimensional representation. The first component captures maximum variance, each subsequent one captures the most remaining variance while being orthogonal to the previous. In practice scikit-learn does this via SVD of the centered data for numerical stability.

**Q5 — Role of eigenvalues/eigenvectors.**
The **eigenvectors** of the covariance matrix are the **directions** of the new axes (principal components), and because the covariance matrix is symmetric, they're mutually orthogonal (uncorrelated). Each **eigenvalue** is the **amount of variance** the data has along its eigenvector — so sorting by eigenvalue ranks the components by importance. Keeping the eigenvectors with the largest eigenvalues keeps the directions where the data varies most, which is where the information is. The eigenvalue divided by the sum of all eigenvalues gives that component's explained-variance ratio.

**Q6 — Why standardize before PCA.**
PCA finds directions of maximum **variance**, and variance depends on scale. If one feature is in the thousands (salary) and another in single digits (years of experience), the salary feature will have vastly larger variance and PC1 will point almost entirely along it — not because it's more informative, but because its units are bigger. Standardizing (z-score) puts every feature on equal footing so PCA reflects genuine correlation structure, not arbitrary units. (You'd skip scaling only when all features share the same meaningful unit.)

**Q8 — Max variance ⇔ min reconstruction error.**
For a projection onto a subspace, the total variance of the data splits into the variance *captured* in the subspace plus the variance *lost* perpendicular to it (Pythagoras on the deviations). Total variance is fixed, so **maximizing the captured (projected) variance is exactly the same as minimizing the perpendicular residual**, which is the reconstruction error. That's why PC1 is simultaneously the direction of greatest spread and the best-fitting line (minimizing sum of squared perpendicular distances) — two views of the same optimization.

**Q9 — Why orthogonal components.**
Because the principal components are the eigenvectors of the **covariance matrix, which is real and symmetric**, and the spectral theorem guarantees a symmetric matrix has an orthogonal set of eigenvectors. Geometrically, PCA deliberately picks each new component to be orthogonal to the previous ones so that they capture *non-overlapping* variance — otherwise components would be redundant. Orthogonality also means the resulting features are **uncorrelated**, which is one of PCA's main benefits (removes multicollinearity).

**Q10 — PCA hurt the classifier.**
Most likely because PCA is **unsupervised** — it keeps directions of maximum variance, which are not necessarily the directions that **separate the classes**. The discriminative signal might lie along a *low-variance* direction that PCA discarded. Other causes: you reduced too aggressively (dropped useful components), or you didn't standardize so PCA latched onto a high-variance but irrelevant feature. Fixes: keep more components (raise the explained-variance threshold), or use **LDA** (supervised, maximizes class separation) instead of/with PCA.

**Q11 — LDA instead of PCA.**
Use **LDA (Linear Discriminant Analysis)** when you *have labels* and the goal is **class separation**, not just variance capture. PCA ignores labels and finds max-variance directions; LDA finds directions that **maximize between-class variance while minimizing within-class variance**, which is often far better for a downstream classifier. LDA can project to at most `(#classes − 1)` dimensions, though. Rule: PCA for unsupervised compression/visualization; LDA for supervised dimensionality reduction before classification.

## Common Mistakes

- **Not standardizing** first → PC1 hijacked by the largest-unit feature.
- **Fitting PCA on the whole dataset including the test set** → data leakage; fit on train only, transform test.
- Assuming high-variance = most useful for the task (it's unsupervised).
- Over-reducing and losing signal.
- Trying to interpret components as if they were single original features.

## Related Concepts

- [SVD](#12-singular-value-decomposition-svd) — the algorithm PCA uses under the hood.
- [t-SNE](#11-t-sne) — non-linear counterpart for visualization.
- Eigenvalues/eigenvectors, covariance — the linear algebra core.
- LDA — the supervised alternative. Kernel PCA — the non-linear extension.
- [Feature Scaling](#13-feature-scaling-before-clustering).

---

# 10. Advanced PCA

## What is it?

The practical, deeper aspects of PCA that show up in senior interviews: **whitening**, **interpreting components via loadings**, PCA's **limitations in depth**, and using **PCA purely as a preprocessing step** for other models and for **2-D/3-D visualization**.

## Whitening

By default PCA rotates the data onto the principal axes but keeps the original variance magnitudes. **Whitening** additionally **scales each component to unit variance** (divides by the square root of its eigenvalue). The result: features that are both **uncorrelated** *and* have **equal (unit) variance** — i.e., an identity covariance matrix.

- **Why use it?** Some downstream algorithms (certain neural nets, algorithms assuming isotropic inputs, image models like eigenfaces) work better when inputs are "spherical." Whitening removes the dominance of high-variance components.
- **Trade-off:** it **amplifies noise** in low-variance components (dividing by a tiny eigenvalue blows up that direction), so use it judiciously. In scikit-learn: `PCA(whiten=True)`.

## Feature interpretation (loadings)

Even though components are mixtures, you *can* interpret them via **loadings** — the coefficients (`components_` in scikit-learn) that say how much each original feature contributes to each PC.

- A large positive/negative loading means that feature strongly drives that component.
- Example: if PC1 has high loadings on `income`, `spending`, `savings`, you might name it a **"wealth" axis**; if PC2 loads on `age` and `tenure`, call it a **"maturity" axis**.
- **Biplots** overlay data points and feature-loading arrows to visualize both together.

This is how you give business meaning to otherwise-abstract components — a common practical/viva task.

## PCA limitations (in depth)

1. **Linearity** — only captures linear correlations; fails on curved manifolds (Swiss roll). Use **Kernel PCA** (applies the kernel trick for non-linear PCA) or t-SNE/UMAP.
2. **Unsupervised** — ignores labels; max-variance ≠ max-discrimination (use LDA when labels matter).
3. **Interpretability** — components are combinations of all features.
4. **Outlier sensitivity** — variance is dominated by extremes; consider **RobustPCA** or outlier removal first.
5. **Scaling dependence** — must standardize.
6. **Assumes variance = importance** — not always true.
7. **Mean-centering assumption** — works on the covariance structure around the mean.

## PCA for preprocessing

A very common pipeline: **Scale → PCA → model.**
- Before **K-Means/DBSCAN**: PCA reduces dimensions so distances behave, speeds clustering, and removes correlated features. (Often "PCA to 2-D for visualization, but keep 95%-variance components for the actual clustering.")
- Before **classification/regression**: reduces overfitting and training time, removes multicollinearity (helpful for linear/logistic regression).
- **Critical rule:** `fit` PCA (and the scaler) on the **training set only**, then `transform` the test set — otherwise you leak test information. Use a scikit-learn `Pipeline` to enforce this.

## Visualization

- **2-D projection** — keep 2 components, scatter-plot; the go-to first look at cluster structure.
- **3-D projection** — keep 3 components for an interactive plot when 2-D loses too much.
- Compare with t-SNE: PCA preserves **global** structure and distances (fast, linear); t-SNE preserves **local** neighborhoods (slow, non-linear). A common workflow: **PCA to ~50-D first, then t-SNE to 2-D** — PCA denoises and speeds up t-SNE.

## Internal Working

- Whitening: after projecting onto eigenvectors, divide component `i` by `√λᵢ`. This equalizes variances but inflates directions with tiny `λ` (noise amplification).
- Loadings are literally the entries of the eigenvectors; `components_[i]` is the i-th eigenvector; `explained_variance_ratio_[i]` is `λᵢ/Σλ`.

## Advantages

- Whitening yields isotropic inputs for downstream models.
- Loadings/biplots restore interpretability.
- As preprocessing: faster, more robust, less overfit downstream models.

## Limitations

- Whitening amplifies noise.
- Interpretation still approximate.
- All base PCA limitations (linearity, unsupervised, scaling) persist.

## Real-world Applications

- **Eigenfaces** — whitened PCA components of face images for recognition.
- **PCA + t-SNE** pipelines for embedding visualization.
- **PCA before regression** in econometrics/finance (principal component regression).
- **Preprocessing** for clustering pipelines in production.

## Interview Questions

**Beginner**
1. What is whitening in PCA?
2. What are loadings?

**Intermediate**
3. How do you interpret a principal component?
4. Why fit PCA only on the training set?
5. What is a biplot?

**Advanced / Scenario / "Why"**
6. What are the risks of whitening?
7. Your data is a Swiss roll; PCA gives a useless 2-D map. Why, and what do you use?
8. Why is PCA-then-t-SNE a common pipeline?
9. How does PCA help a logistic regression model?

## Model Answers

**Q1 — Whitening.**
Whitening is PCA plus a rescaling step: after rotating data onto the principal components, each component is divided by the square root of its eigenvalue so that **all components have unit variance**. The output has an identity covariance matrix — uncorrelated *and* equal-variance ("spherical") features. It's useful when downstream algorithms assume isotropic inputs (some neural nets, eigenface recognition). The downside is **noise amplification**: dividing a low-variance (often noisy) component by a tiny eigenvalue blows that direction up, so whitening can hurt if you keep noisy components.

**Q3 — Interpreting a component.**
Look at its **loadings** — the weights that show how strongly each original feature contributes to that component. Features with large-magnitude loadings dominate the component; the sign shows direction of correlation. By reading which features co-load, you assign a semantic name: e.g. a component loading heavily on income, spending, and savings is a "financial capacity" axis. A **biplot** helps by drawing loading vectors alongside the projected points. This turns abstract components into business-meaningful dimensions, which is essential when presenting results.

**Q4 — Why fit PCA on train only.**
Because PCA learns its components (means, variances, eigenvectors) *from the data it's fit on*. If you fit on the full dataset including test rows, information about the test set leaks into the transformation, giving optimistically biased evaluation. The correct protocol: `fit` the scaler and PCA on the **training set**, store them, and only `transform` the validation/test sets. Wrapping scaler + PCA + model in a scikit-learn `Pipeline` (and doing CV over the pipeline) enforces this automatically.

**Q7 — Swiss roll.**
A Swiss roll is a 2-D sheet **rolled up non-linearly** in 3-D. PCA is linear — it can only rotate and project along straight axes — so it "flattens" the roll by projecting through the layers, collapsing points that are far apart along the sheet into the same 2-D spot. It cannot "unroll" curvature. You need **non-linear manifold methods**: **Kernel PCA**, **t-SNE**, **UMAP**, or **Isomap/LLE**, which respect the intrinsic (geodesic) structure and unroll the sheet correctly.

**Q8 — Why PCA-then-t-SNE.**
t-SNE is `O(n²)`-ish and slow, and it's sensitive to noise and to very high dimensions. Running **PCA first to ~30–50 components** (a) drastically cuts the dimensionality so t-SNE runs much faster and uses less memory, and (b) **removes noise** by discarding low-variance directions, which improves the quality and stability of the t-SNE embedding. PCA keeps ~global variance structure; t-SNE then refines local neighborhoods for the final 2-D visualization. It's the standard recipe for visualizing large high-D datasets.

## Common Mistakes

- Whitening and then keeping noisy low-variance components → amplified noise.
- Fitting PCA on train+test together (leakage).
- Expecting PCA to unroll non-linear manifolds.
- Naming components without checking loadings.
- Using PCA-reduced features but forgetting to apply the *same* fitted transform to new data.

## Related Concepts

- [PCA](#9-principal-component-analysis-pca) core, [t-SNE](#11-t-sne), [SVD](#12-singular-value-decomposition-svd).
- Kernel PCA, RobustPCA, LDA.
- [Feature Scaling](#13-feature-scaling-before-clustering); scikit-learn Pipelines.

---

# 11. t-SNE

## What is it?

**t-SNE** (*t-Distributed Stochastic Neighbor Embedding*) is a **non-linear dimensionality-reduction technique built specifically for visualization.** It takes high-dimensional data and produces a 2-D or 3-D map in which **points that were close in high-D stay close**, revealing clusters and local structure that linear methods like PCA miss.

Its defining property: it preserves **local structure** (neighborhoods) at the expense of **global structure** (distances between far-apart clusters are not meaningful).

## Why is it needed?

PCA is linear, so it can't unfold curved manifolds and often produces cluttered 2-D plots where distinct groups overlap. When your goal is purely to **see** whether high-D data (word embeddings, image features, single-cell genomics) forms clusters, t-SNE typically produces dramatically clearer, well-separated visual clusters than PCA. It's the standard tool for **exploratory visualization of embeddings.**

## How does it work?

Two-stage idea: model similarities in high-D as probabilities, then arrange points in 2-D so their probabilities match.

```
1. HIGH-D similarities: for each pair (i,j), compute a probability
   p(j|i) that i would pick j as a neighbor, using a Gaussian
   centered on i. Nearby points → high probability.
   (The Gaussian's width is set per-point by PERPLEXITY.)

2. LOW-D similarities: place points randomly in 2-D. Compute
   pairwise similarities q(i,j) using a STUDENT-t (Cauchy)
   distribution with heavy tails.

3. Minimize the KL DIVERGENCE between the high-D distribution P
   and the low-D distribution Q, using gradient descent — moving
   points in 2-D until neighborhoods match.
```

- **Why a t-distribution in low-D?** Its **heavy tails** give moderately distant points more room, counteracting the "crowding problem" (in 2-D there isn't enough space to place all neighbors) and producing well-separated clusters.

## Hyperparameters

| Param | Meaning | Guidance |
|---|---|---|
| **Perplexity** | Roughly the "effective number of neighbors" each point considers (balances local vs global) | Typical **5–50**; smaller → very local, larger → more global. Must be < n. The single most important knob. |
| **Learning rate** | Gradient-descent step size | Too low → stuck/clumped; too high → a diffuse "ball." Typical 10–1000 (or `auto`). |
| **n_iter** | Number of optimization iterations | ≥ 1000; too few → not converged. |

## Internal Working (behind the scenes)

- **Per-point Gaussian bandwidth:** each point gets its own Gaussian variance chosen so the neighborhood's entropy matches the target perplexity — this adapts to local density (dense regions get narrow Gaussians).
- **Symmetrized probabilities:** `p(i,j) = (p(j|i) + p(i|j)) / 2n`.
- **KL divergence is asymmetric:** it heavily penalizes putting *nearby* high-D points far apart in 2-D, but is lenient about far-apart points — which is exactly why **local** structure is preserved and **global** distances are not.
- **Non-convex optimization** with random initialization → **different runs give different maps** (non-deterministic unless you fix the seed).
- Complexity ~`O(n²)` (Barnes-Hut approximation brings it to `O(n log n)` for the common 2-D case).

## Advantages

- **Superb at revealing clusters** and local structure.
- Captures **non-linear** relationships PCA cannot.
- Produces visually striking, well-separated 2-D/3-D maps.

## Limitations

- **Visualization only** — the axes are meaningless; **do not** use t-SNE output as features for a model or for clustering distances.
- **Slow / memory-heavy** — poor for very large datasets (mitigate with PCA-first + Barnes-Hut).
- **Non-deterministic** — different runs/seeds give different layouts.
- **Global structure is unreliable** — distances *between* clusters and cluster *sizes* on the plot are **not** meaningful; don't over-interpret gaps.
- **Hyperparameter-sensitive** — perplexity dramatically changes the picture; misleading plots are easy to produce.
- No `transform` for new points (must re-run). Cannot naturally embed unseen data.

## t-SNE vs PCA (the classic comparison)

| | PCA | t-SNE |
|---|---|---|
| Type | Linear | Non-linear |
| Preserves | **Global** structure/variance | **Local** neighborhoods |
| Speed | Fast | Slow |
| Deterministic | Yes | No |
| Use for features? | Yes | **No (viz only)** |
| Interpretable axes? | Somewhat (loadings) | No |
| Handles curved manifolds | No | Yes |
| New data (`transform`) | Yes | No |

**UMAP** is a modern alternative: faster, preserves more global structure, and *can* transform new data — increasingly the default over t-SNE (worth mentioning).

## Real-world Applications

- **Visualizing word/sentence embeddings** (NLP).
- **Single-cell RNA-seq** clustering visualization (huge in bioinformatics).
- **Image feature/embedding** visualization (e.g. CNN penultimate layers, MNIST).
- **Exploratory analysis** to check whether classes/clusters are separable before modeling.

## Interview Questions

**Beginner**
1. What is t-SNE used for?
2. What is perplexity?

**Intermediate**
3. Why does t-SNE preserve local but not global structure?
4. Why is t-SNE non-deterministic?
5. Why shouldn't you use t-SNE output as model features?

**Advanced / Scenario / "Why"**
6. Why does t-SNE use a Student-t distribution in the low-D space?
7. Someone shows a t-SNE plot and argues cluster A is "twice as big" and "far from" cluster B. What's wrong with that?
8. How would you make t-SNE run on a large, high-dimensional dataset?
9. When would you choose PCA over t-SNE and vice versa?

## Model Answers

**Q2 — Perplexity.**
Perplexity controls how many neighbors each point effectively considers when t-SNE builds its high-D similarity distribution — loosely, the "effective neighborhood size." It sets the bandwidth of each point's Gaussian: low perplexity (~5) focuses on very local structure and can fragment clusters; high perplexity (~50) considers broader neighborhoods and captures more global layout but can merge clusters. Typical values are 5–50 and it must be less than the number of points. It's the single most influential hyperparameter, so you usually try several and compare.

**Q3 — Local not global.**
t-SNE minimizes the KL divergence between high-D and low-D neighbor distributions, and KL divergence is **asymmetric**: it imposes a large penalty when two points that are *close* in high-D end up *far* in 2-D, but only a small penalty for the reverse. So the optimization works hard to keep true neighbors together (local structure) while caring little about the placement of far-apart points. Combined with the crowding-relieving t-distribution, this means clusters are faithfully formed, but the **distances and gaps between clusters carry no reliable meaning** — you can't read global geometry off a t-SNE map.

**Q6 — Why Student-t in low-D.**
Because of the **crowding problem**: when you compress high-D neighborhoods into 2-D, there isn't enough area to place all the moderately-close neighbors at the right distance — they get crushed together. The Student-t (Cauchy) distribution has **heavy tails**, so a pair that's moderately far apart in 2-D still gets a reasonable similarity, allowing the optimizer to push clusters apart and use the low-D space efficiently. Using a Gaussian in low-D (as the original SNE did) caused clumping; switching to the t-distribution is exactly the "t" in t-SNE and is what yields clean, separated clusters.

**Q7 — Misreading a t-SNE plot.**
Both claims are unsound. t-SNE **does not preserve cluster sizes or inter-cluster distances** — it optimizes local neighborhoods, and the t-distribution plus per-point bandwidths mean a dense cluster can be drawn large or small and gaps between clusters are essentially arbitrary artifacts of the optimization and perplexity. So "A is twice as big" and "A is far from B" read global geometry that t-SNE deliberately discards. The only safe reading is *which points group together locally*. To compare sizes/distances, use PCA or actual metrics in the original space.

**Q8 — t-SNE on large high-D data.**
(1) **Reduce dimensions with PCA first** (e.g. to 30–50 components) to denoise and shrink the input. (2) Use the **Barnes-Hut** approximation (`method='barnes_hut'`, the scikit-learn default for 2-D) to cut complexity from `O(n²)` to `O(n log n)`. (3) **Subsample** if still too big. (4) Consider **UMAP** instead — it's much faster, scales better, preserves more global structure, and supports transforming new points. (5) Tune perplexity/learning rate and run enough iterations to converge.

## Common Mistakes

- Using t-SNE coordinates as **features** for clustering or a classifier.
- Interpreting **cluster sizes / distances / gaps** on the plot as meaningful.
- Running with default perplexity and trusting a single plot.
- Too few iterations (not converged) → misleading clumps.
- Forgetting to PCA-reduce/scale first on large high-D data.
- Expecting reproducible layouts without fixing `random_state`.

## Related Concepts

- [PCA](#9-principal-component-analysis-pca) — linear, global, fast; often run before t-SNE.
- **UMAP** — faster, transform-capable successor.
- [SVD](#12-singular-value-decomposition-svd); manifold learning (Isomap, LLE).

---

# 12. Singular Value Decomposition (SVD)

## What is it?

SVD is a fundamental **matrix factorization** from linear algebra that decomposes *any* `m×n` matrix `A` into three matrices:

```
A = U · Σ · Vᵀ
```
- **U** (`m×m`) — left singular vectors (orthonormal). Relate to the *rows* (e.g. users, documents).
- **Σ** (`m×n`) — a diagonal matrix of **singular values** `σ₁ ≥ σ₂ ≥ … ≥ 0`, sorted descending. Each measures the "strength" of a latent dimension.
- **Vᵀ** (`n×n`) — right singular vectors (orthonormal). Relate to the *columns* (e.g. items, terms).

By keeping only the top `k` singular values (**Truncated SVD**), you get the best rank-`k` approximation of `A` — this is the engine behind recommender systems, LSA, and PCA.

## Why is it needed?

- **Works on any matrix** — rectangular, non-square, and crucially **sparse** matrices (unlike PCA's covariance step), so it scales to huge sparse data like user-item ratings or term-document counts.
- **Latent factors** — it uncovers hidden structure: latent "tastes" in recommendations, latent "topics" in text.
- **Compression / low-rank approximation** — represent a big matrix with far fewer numbers while keeping the dominant structure (image compression, noise reduction).
- **The mathematical foundation of PCA** and many ML methods.

## How does it work?

- **Singular values** `σᵢ` tell you how much "energy"/variance each latent dimension carries; they're the square roots of the eigenvalues of `AᵀA` (and `AAᵀ`).
- **Truncated SVD**: keep the top `k` singular values and corresponding vectors → `A ≈ Uₖ Σₖ Vₖᵀ`. This is provably the **best rank-k approximation** in least-squares terms (Eckart–Young theorem).
- **For recommendations:** the user-item rating matrix `R` is factored so `R ≈ U Σ Vᵀ`; each user and item becomes a vector in a shared **latent factor space**, and a missing rating is predicted by the dot product of that user's and item's latent vectors.

```
        items →
       ┌───────────┐        ┌────┐ ┌───┐ ┌────────┐
users  │  ratings  │   ≈    │ U  │ │ Σ │ │  Vᵀ    │
  ↓    │  (sparse) │        │k   │ │k×k│ │ k      │
       └───────────┘        └────┘ └───┘ └────────┘
       m×n                  m×k    diag   k×n
              latent factors: user tastes × item traits
```

## SVD vs PCA (relationship)

- PCA on centered data `X` is equivalent to SVD of `X`: the **right singular vectors `V` are the principal components**, and singular values relate to eigenvalues by `λᵢ = σᵢ²/(n−1)`.
- scikit-learn's `PCA` uses SVD internally (numerically stabler than eigen-decomposing the covariance matrix).
- **Key difference:** PCA **centers** the data (subtracts the mean); **TruncatedSVD does not**, which is why TruncatedSVD works directly on **sparse** matrices (centering would destroy sparsity). That's why text pipelines use `TruncatedSVD` (called **LSA**), not PCA.

## Internal Working

- Computed via iterative numerical algorithms (e.g. randomized SVD for large/sparse matrices — `TruncatedSVD` uses this).
- `U` and `V` are orthonormal (rotations/reflections); `Σ` is a non-negative diagonal (scaling). So `A = U Σ Vᵀ` reads as **rotate → scale → rotate** — any linear map decomposes this way.
- Explained variance for TruncatedSVD: `σᵢ² / Σσ²` (analogous to PCA's explained variance ratio).

## Applications

- **Recommendation systems / collaborative filtering** — the famous Netflix-Prize approach (matrix factorization). Predict unseen user-item ratings via latent factors.
- **Latent Semantic Analysis (LSA)** — TruncatedSVD on a term-document TF-IDF matrix to find latent topics and enable semantic search.
- **Image compression** — keep top-k singular values of the pixel matrix to store the image with far fewer numbers.
- **Noise reduction** — small singular values often correspond to noise; dropping them denoises.
- **PCA** — the underlying computation.
- **Pseudo-inverse / solving least squares** — numerically stable via SVD.

## Advantages

- **Handles sparse and non-square** matrices; scales to huge data.
- Provides the **optimal low-rank approximation** (Eckart–Young).
- Uncovers **latent/hidden factors**.
- Numerically stable; foundational and well-understood.
- Efficient (randomized/truncated variants).

## Limitations

- **Latent factors are hard to interpret** (like PCA components).
- **Linear** method — can't capture non-linear structure.
- Basic SVD needs a **complete** matrix; for recommendation with missing entries you use specialized matrix-factorization (SGD/ALS on observed entries), not vanilla SVD — an important nuance ("SVD" in recommenders usually means *SVD-inspired* factorization).
- Sensitive to scaling.
- Full SVD is expensive for very large dense matrices (hence truncated/randomized versions).

## Real-world Applications

- **Netflix / Amazon / Spotify** recommendations (matrix factorization).
- **Search / NLP** — LSA for document similarity and topic discovery.
- **Image & video compression.**
- **Signal processing / PCA / data denoising.**

## Interview Questions

**Beginner**
1. What does SVD decompose a matrix into?
2. What are singular values?

**Intermediate**
3. How is SVD used in recommendation systems?
4. What is Truncated SVD and what is it good for?
5. What is Latent Semantic Analysis?

**Advanced / Scenario / "Why"**
6. How is SVD related to PCA? When would you pick TruncatedSVD over PCA?
7. Why can SVD handle sparse text data when PCA can't?
8. Explain how you'd compress an image with SVD.
9. Why is "SVD" in recommender systems not exactly the textbook SVD?

## Model Answers

**Q1 / Q2 — Decomposition & singular values.**
SVD factors any `m×n` matrix `A` into `U Σ Vᵀ`: `U` holds the left singular vectors (orthonormal, tied to rows), `Vᵀ` holds the right singular vectors (orthonormal, tied to columns), and `Σ` is diagonal with the **singular values** in descending order. A singular value `σᵢ` quantifies how much of the matrix's structure/energy lies along the i-th latent dimension — larger means more important. They're the square roots of the eigenvalues of `AᵀA`. Keeping the largest few gives the best low-rank approximation of `A`.

**Q3 — SVD in recommenders.**
You arrange the data as a user-item rating matrix `R` (rows = users, columns = items). Matrix factorization approximates `R ≈ U Σ Vᵀ`, mapping every user and every item into a shared **latent-factor space** — dimensions that capture hidden patterns like "likes action movies" or "prefers indie music." A missing rating for (user, item) is then predicted by the **dot product** of that user's latent vector and the item's latent vector. This generalizes from observed ratings to unseen ones, which is the core of collaborative filtering. The top-k factors compress millions of ratings into small dense vectors.

**Q6 — SVD vs PCA, when TruncatedSVD.**
PCA is essentially SVD applied to **mean-centered** data — the right singular vectors are the principal components, and singular values map to eigenvalues via `λ = σ²/(n−1)`. The practical difference: PCA centers the data, which **destroys sparsity** (a sparse matrix becomes dense after subtracting the mean), making it infeasible for huge sparse matrices. **TruncatedSVD does not center**, so it operates directly on sparse matrices. Therefore you pick TruncatedSVD for **sparse, high-dimensional data like TF-IDF text** (that's LSA), and PCA for dense numeric data where centering is fine.

**Q7 — SVD on sparse text.**
Text represented as TF-IDF is a huge, extremely **sparse** term-document matrix (mostly zeros). PCA requires computing the covariance matrix, which involves mean-centering; subtracting the column means turns all those zeros into non-zeros, exploding memory. SVD (specifically TruncatedSVD/randomized SVD) factors the matrix **without centering**, preserving sparsity and using efficient sparse linear algebra, so it scales to millions of documents and terms. Applying it to TF-IDF is exactly **LSA**, revealing latent topics.

**Q9 — Why recommender "SVD" isn't textbook SVD.**
Classical SVD is defined only for a **fully known** matrix, but rating matrices are mostly **missing** (users rate few items). You can't just fill missing entries with zeros — that would mean "rated 0." Instead, so-called "SVD" in recommenders (e.g. Simon Funk's method from the Netflix Prize) learns latent user/item vectors by **minimizing error only over the observed ratings**, using stochastic gradient descent or ALS, plus regularization and bias terms. It's *inspired by* SVD's latent-factor idea but is really regularized matrix factorization on observed entries — a crucial distinction.

## Common Mistakes

- Thinking vanilla SVD directly handles missing ratings (it needs a complete matrix).
- Using PCA (centered) on huge sparse text instead of TruncatedSVD.
- Not scaling before SVD when magnitudes vary.
- Over-interpreting latent factors as concrete concepts.
- Keeping too few/too many singular values (bad compression vs bad reconstruction).

## Related Concepts

- [PCA](#9-principal-component-analysis-pca) — SVD is its engine.
- LSA/LDA (topic modeling); collaborative filtering & matrix factorization.
- Eckart–Young theorem (best low-rank approximation).
- [Recommendation project](#15-outlier--anomaly-detection).

---

# 13. Feature Scaling Before Clustering

## What is it?

Feature scaling transforms features so they share a **comparable numeric range**. Because clustering and PCA are **distance-/variance-based**, features with larger numeric ranges dominate unless you scale. Scaling is not optional preprocessing — it's often the difference between meaningful and garbage clusters.

## Why is it needed?

Distance metrics sum contributions across features:
```
d = √( (age₁−age₂)² + (income₁−income₂)² )
```
If income is in the hundreds of thousands and age in the tens, the income term dwarfs age — the algorithm effectively clusters on income alone. Scaling equalizes each feature's influence. The same logic applies to PCA (variance is scale-dependent) and to `eps` in DBSCAN (a single radius across all dimensions).

## The three main techniques

### StandardScaler (Z-score standardization)
```
z = (x − mean) / std
```
- Result: mean 0, std 1. Range unbounded but centered.
- **Default choice** for K-Means, PCA, DBSCAN. Best when data is roughly Gaussian.
- Sensitive to outliers (mean and std are affected by extremes).

### MinMaxScaler (Normalization)
```
x' = (x − min) / (max − min)
```
- Result: squashed to **[0, 1]** (or a chosen range).
- Good when you need bounded features or the distribution isn't Gaussian; preserves the shape but is **very sensitive to outliers** (one extreme sets min/max).

### RobustScaler
```
x' = (x − median) / IQR        (IQR = Q3 − Q1)
```
- Uses **median and interquartile range** instead of mean/std, so it's **robust to outliers**.
- Best when the data has significant outliers you don't want to remove.

| Scaler | Centers on | Scales by | Outlier-robust? | Output range |
|---|---|---|---|---|
| StandardScaler | mean | std | No | unbounded, ~[−3,3] |
| MinMaxScaler | min | range | **No** (very sensitive) | [0, 1] |
| RobustScaler | median | IQR | **Yes** | unbounded |

## Internal Working

- Scalers are `fit` (learn the statistics: mean/std, min/max, median/IQR) then `transform`. **Fit on training data only**, then transform test/new data with the same stored statistics — otherwise leakage.
- For **cosine-based** clustering you often **L2-normalize** each *sample* (row) to unit length instead of per-feature scaling — different goal (direction vs magnitude).

## When you might NOT scale

- All features already share the same meaningful unit (e.g. RGB pixel values 0–255 for image clustering).
- Tree-based models (not distance-based) don't need it — but this section is about clustering, which does.

## Advantages

- Makes distances/variance **fair** across features → correct clusters.
- Speeds up and stabilizes convergence.
- Essential for PCA to reflect true correlation, not units.

## Limitations

- Wrong scaler choice can hurt (MinMax with outliers).
- Scaling can reduce interpretability (values no longer in original units).
- Doesn't fix non-numeric/categorical features (need encoding first).

## Real-world Applications

- Every real clustering/PCA pipeline scales first.
- RobustScaler in finance/sensor data with outliers.
- MinMax for image/neural-net inputs.

## Interview Questions

1. Why is scaling critical before K-Means and PCA? (Beginner)
2. StandardScaler vs MinMaxScaler vs RobustScaler — when each? (Intermediate)
3. Why fit the scaler on training data only? (Intermediate)
4. Your data has extreme outliers; which scaler and why? (Scenario)
5. When is scaling unnecessary? (Advanced)

## Model Answers

**Q1 — Why critical.**
K-Means and PCA both depend on magnitude: K-Means sums squared differences across features, and PCA finds max-variance directions. A feature with a large numeric range (income) contributes far more than a small-range feature (age), so without scaling the algorithm is effectively driven by one feature and ignores the rest. Scaling puts all features on comparable footing so each contributes fairly, producing clusters/components that reflect the true multidimensional structure rather than an artifact of units.

**Q2 — Which scaler when.**
**StandardScaler** (z-score) is the default — centers to mean 0, std 1 — ideal when features are roughly Gaussian and outliers are mild. **MinMaxScaler** squashes to [0,1], useful when you need bounded inputs (e.g. neural nets) or non-Gaussian data, but it's very sensitive to outliers because a single extreme defines the min/max. **RobustScaler** uses median and IQR, so it's the right choice when there are significant outliers you can't remove, since medians/quartiles resist extremes. Pick based on distribution shape and outlier presence.

**Q4 — Outliers scenario.**
Use **RobustScaler**. StandardScaler's mean and standard deviation are themselves distorted by outliers, so the "standardized" values get compressed and the outliers still dominate distances; MinMaxScaler is even worse because one extreme value sets the range and crushes everything else into a tiny band. RobustScaler centers on the **median** and scales by the **interquartile range**, both of which are insensitive to extreme values, so the bulk of the data is scaled sensibly while outliers don't hijack the transformation. (Alternatively, detect and handle outliers first, then StandardScale.)

## Common Mistakes

- Skipping scaling entirely (the #1 clustering bug).
- Using MinMax on outlier-heavy data.
- Fitting the scaler on the full dataset (leakage) instead of train-only.
- Scaling categorical/one-hot columns as if numeric without thought.
- Forgetting to apply the *same* fitted scaler to new/test data.

## Related Concepts

- [K-Means](#3-k-means-clustering), [PCA](#9-principal-component-analysis-pca), [DBSCAN](#6-dbscan) — all require it.
- [Distance metrics](#2-distance--similarity-measures).
- scikit-learn Pipelines (to prevent leakage).

---

# 14. Cluster Interpretation

## What is it?

Clustering produces group **labels** (0, 1, 2…), but those numbers are meaningless until you **interpret** each cluster — describe *what kind of thing* it contains and *why it's useful*. Interpretation is what turns a clustering into a business deliverable ("high-value at-risk customers"), and it's a guaranteed viva/interview topic because it's where analysis meets action.

## Why is it needed?

A model that says "customer #4821 is in cluster 2" is useless to a marketing team. They need "cluster 2 = young, high-frequency, low-basket shoppers — target with bundle offers." Interpretation bridges the algorithm's output and the decision it should drive. Without it, clustering is a math exercise.

## How does it work? (the interpretation workflow)

```
1. PROFILE each cluster: compute per-cluster summary statistics
   (mean/median of each feature) and compare to the overall average.
2. FIND DISTINGUISHING features: which features are unusually
   high/low in this cluster vs others? Those define the cluster.
3. For K-Means: read the CENTROIDS directly — each centroid is the
   "average member" (remember to inverse-transform scaling to get
   real units).
4. VISUALIZE: PCA/t-SNE 2-D scatter colored by cluster; box/violin
   plots of key features per cluster; heatmap of cluster-mean features.
5. NAME & narrate: assign a human label and a business meaning.
6. VALIDATE with a domain expert and check cluster SIZE (tiny
   clusters may be noise; huge ones may be under-segmented).
```

## Techniques in detail

- **Analyze cluster centers** — For K-Means, `cluster_centers_` gives the prototype per cluster. **Inverse-transform** the scaler to read them in original units ("centroid = ₹5,200 spend, 12 visits/month").
- **Identify cluster characteristics** — Build a table of `feature × cluster` means and highlight the largest deviations from the global mean; these are the cluster's "signature."
- **Snake plots / radar charts** — plot standardized cluster means across features to see each segment's profile at a glance.
- **Assign business meaning** — translate the signature into a persona: "Champions", "At-risk", "Budget-conscious", etc. (RFM segmentation is the classic template.)
- **Cross-tab with known attributes** — if you have any labels/metadata (region, plan type), cross-tabulate to sanity-check.

## Internal Working / practical notes

- Cluster labels are **arbitrary and unordered** — cluster "0" isn't "less than" cluster "1"; never treat them as numeric.
- Labels are **not stable across runs** (K-Means re-seeds) — align clusters by their centroids/profiles, not by ID.
- Always interpret in **original units** (inverse-transform), not scaled space, when communicating.

## Advantages

- Turns clusters into **actionable segments** and decisions.
- Builds trust — stakeholders see *why* the grouping makes sense.
- Surfaces new insights (unexpected niches).

## Limitations

- **Subjective** — different analysts may narrate clusters differently.
- Risk of **storytelling** — inventing meaning for statistically real but random groupings.
- Needs **domain knowledge** to be credible.
- Interpretation quality depends on feature quality.

## Real-world Applications

- **Customer segmentation** → targeted marketing, pricing, retention.
- **Patient stratification** in healthcare.
- **Product/user analytics** → feature prioritization.
- **Portfolio grouping** in finance.

## Interview Questions

1. After K-Means, how do you figure out what each cluster represents? (Intermediate)
2. Why can't you treat cluster labels 0,1,2 as ordered categories? (Beginner)
3. Your CEO asks "what are these 5 segments?" — walk me through your process. (Scenario)
4. How do you guard against inventing meaning that isn't really there? (Advanced/Why)
5. Why inverse-transform before reading centroids? (Intermediate)

## Model Answers

**Q1 — Interpreting K-Means clusters.**
I profile each cluster: compute the mean/median of every feature per cluster and compare against the overall average to find each cluster's *distinguishing* features. For K-Means I read the centroids directly (after inverse-transforming the scaler so they're in real units) — the centroid is the prototypical member. I then visualize: a PCA/t-SNE scatter colored by cluster to confirm separation, plus per-feature box plots or a heatmap of cluster means. From the signature (e.g. high spend + high frequency) I assign a business name ("VIP loyalists") and validate it with a domain expert and against cluster size.

**Q4 — Guarding against false meaning.**
Several checks: (1) validate cluster **stability** — re-run with different seeds/subsamples and confirm similar clusters emerge; (2) check **internal metrics** (silhouette) to ensure the clusters are genuinely separated, not arbitrary slices; (3) test whether the distinguishing features are **statistically significant** across clusters, not noise; (4) confirm the segments are **actionable and reproducible** on new data; and (5) get **domain-expert** sign-off. If clusters vanish or reshuffle across runs, or no feature meaningfully separates them, I treat the "story" as unreliable rather than forcing a narrative.

## Common Mistakes

- Treating cluster IDs as ordinal/meaningful numbers.
- Reading centroids in **scaled** units and reporting nonsense.
- Over-fitting a story to random clusters.
- Ignoring cluster **sizes** (tiny = possible noise).
- Not validating with a domain expert or across runs.

## Related Concepts

- [K-Means centroids](#3-k-means-clustering), [evaluation metrics](#7-comparing-clustering-algorithms--evaluation-metrics).
- [Feature Scaling](#13-feature-scaling-before-clustering) (inverse-transform).
- RFM analysis; [visualization](#11-t-sne).

---

# 15. Outlier / Anomaly Detection

## What is it?

Outlier (anomaly) detection is the task of identifying data points that **deviate significantly** from the majority — points that fit no normal pattern. In unsupervised learning it's closely tied to clustering: an outlier is essentially a point that belongs to **no dense cluster**. Unlike supervised anomaly detection (which needs labeled anomalies, usually scarce), unsupervised methods flag anomalies purely from the data's structure.

## Why is it needed?

- **Fraud detection** — a fraudulent transaction is a rare deviation from normal spending.
- **Fault/failure detection** — a sensor reading spike signals equipment failure.
- **Network intrusion** — an attack looks unlike normal traffic.
- **Data quality** — outliers may be errors to clean before modeling.
- **Medical** — abnormal test results.
Anomalies are usually **rare and unlabeled**, which is exactly why unsupervised methods shine — you can't collect enough labeled examples of every possible anomaly.

## Techniques

### 1. DBSCAN Noise Points
DBSCAN labels points in sparse regions as **noise (label −1)** — these *are* your outliers, for free, as a by-product of clustering. Great when normal data forms dense clusters and anomalies are isolated. No need to specify how many anomalies exist.

### 2. Distance / Density-based methods
- **Distance to k-th nearest neighbor** — points whose neighbors are far away are anomalies (kNN outlier score).
- **Distance to nearest centroid** (after K-Means) — points far from every centroid are candidates.
- **Local Outlier Factor (LOF)** — compares a point's local density to that of its neighbors; a point in a much sparser neighborhood than its neighbors is a **local** outlier (catches anomalies that global methods miss).

### 3. Isolation Forest (introduction)
A tree-based method built on a clever idea: **anomalies are easy to isolate.** It builds many random trees by repeatedly picking a random feature and a random split value. Anomalies, being few and different, get **separated in very few splits** (short path from root to leaf); normal points need many splits. The **average path length** across trees becomes the anomaly score — shorter = more anomalous.
- **Advantages:** fast, near-linear, scales to high dimensions, works well when anomalies are rare and "different." A go-to modern anomaly detector.
- **`contamination`** parameter sets the expected fraction of anomalies.

### Others worth naming
- **One-Class SVM** — learns a boundary around normal data; points outside are anomalies (good for high-D, but slower).
- **Autoencoders** — reconstruct normal data well; high **reconstruction error** flags anomalies (deep-learning approach).

## How does it work (general framing)?

```
1. Model "normal" — via density (DBSCAN/LOF), distance (kNN/centroid),
   isolation (iForest), a boundary (One-Class SVM), or reconstruction
   (autoencoder).
2. Score each point by how much it deviates from "normal".
3. Threshold the score (or use `contamination`) to flag anomalies.
```

## Internal Working

- **Isolation Forest:** path length `h(x)` averaged over trees; anomaly score `s = 2^(−E[h(x)]/c(n))` where `c(n)` normalizes for tree size. Score → 1 means anomaly, → 0.5 means normal.
- **LOF:** ratio of the average local reachability density of a point's neighbors to the point's own — >1 (much sparser than neighbors) → outlier.
- **DBSCAN:** noise = points not density-reachable from any core point.

## Advantages

- Works **without labeled anomalies** (unsupervised).
- DBSCAN/iForest need little tuning and scale reasonably.
- Isolation Forest and LOF handle high-D and local anomalies well.

## Limitations

- **No ground truth** → hard to validate; threshold/`contamination` is a guess.
- **Class imbalance** — anomalies are rare, so evaluation via accuracy is meaningless (use precision/recall on any labeled subset).
- DBSCAN/distance methods degrade in **high dimensions** and with **varying density**.
- Risk of flagging **rare-but-legitimate** points as anomalies.
- Defining "how anomalous is anomalous" is subjective.

## Real-world Applications

- **Banking/fintech** — credit-card fraud, money-laundering detection.
- **Cybersecurity** — intrusion/attack detection.
- **Manufacturing/IoT** — predictive maintenance from sensor anomalies.
- **Healthcare** — abnormal readings, disease outbreak detection.
- **Data cleaning** — remove erroneous records before modeling.

## Interview Questions

1. How does clustering help detect anomalies? (Beginner)
2. How does DBSCAN identify outliers? (Beginner)
3. Explain how Isolation Forest works and why it's efficient. (Intermediate/Why)
4. What is Local Outlier Factor and when does it beat global methods? (Advanced)
5. Anomalies are 0.1% of your data — why is accuracy a useless metric, and what do you use? (Scenario)
6. Compare DBSCAN vs Isolation Forest for anomaly detection. (Comparison)

## Model Answers

**Q2 — DBSCAN outliers.**
DBSCAN clusters by density: core points sit in dense regions, border points are on cluster edges, and any point that is **not density-reachable from a core point** — i.e., sits in a sparse region — is labeled **noise** with cluster ID −1. Those noise points are exactly the outliers. So DBSCAN gives anomaly detection as a free by-product of clustering, and unlike threshold methods you don't have to specify how many anomalies to expect — the density criteria (`eps`, `min_samples`) decide. It works best when normal data forms dense clusters and anomalies are genuinely isolated.

**Q3 — Isolation Forest.**
Isolation Forest builds many random binary trees; at each node it picks a random feature and a random split value, recursively partitioning the data. The key insight is that **anomalies are few and different, so they get isolated into their own leaf after very few splits**, giving a **short path length** from the root, whereas normal points are surrounded by similar points and require many splits. Averaging the path length over all trees yields an anomaly score — short average path = anomaly. It's efficient because it never computes distances or densities; it just does random partitioning, giving roughly **linear time** and good scaling to high dimensions.

**Q4 — LOF.**
Local Outlier Factor scores a point by comparing its **local density** to the local densities of its neighbors. If a point sits in a region far sparser than where its neighbors sit, its LOF is well above 1 and it's flagged as a **local** outlier. This beats global methods when the dataset has **regions of different density**: a point might not be a global outlier (it's not the farthest overall) yet be clearly anomalous *relative to its local neighborhood*. Global distance/centroid methods and a single-`eps` DBSCAN miss these; LOF catches them by being density-*relative*.

**Q5 — 0.1% anomalies, metrics.**
With 0.1% anomalies, a model that predicts "normal" for everything achieves **99.9% accuracy** while catching zero anomalies — accuracy is meaningless under extreme class imbalance. Instead use **precision** (of flagged points, how many are truly anomalous), **recall** (of true anomalies, how many we caught), their trade-off via **F1**, and the **precision-recall AUC** (more informative than ROC-AUC when positives are rare). In practice you tune the threshold/`contamination` to hit the business-required recall (catch enough fraud) at acceptable precision (not too many false alarms for investigators).

**Q6 — DBSCAN vs Isolation Forest.**
DBSCAN detects anomalies as low-density noise points and simultaneously clusters the normal data, but it struggles with **varying density**, needs `eps`/`min_samples` tuning, and degrades in high dimensions. Isolation Forest is purpose-built for anomaly detection: it's **distance-free**, near-linear, scales to **high dimensions**, handles large data, and directly outputs an anomaly score via isolation path length — but it doesn't cluster and assumes anomalies are "few and different." Rule of thumb: use DBSCAN when you also want clustering and data is low-D with uniform density; use Isolation Forest for scalable, high-dimensional, dedicated anomaly detection.

## Common Mistakes

- Using **accuracy** on highly imbalanced anomaly data.
- Applying distance-based detectors in high dimensions without reducing dimensions.
- Treating every outlier as an error (some are the *interesting* signal).
- Guessing `contamination` without validating on any labeled sample.
- Ignoring varying density (single-`eps` DBSCAN misses local anomalies → use LOF/HDBSCAN).

## Related Concepts

- [DBSCAN](#6-dbscan) noise points.
- Isolation Forest, LOF, One-Class SVM, Autoencoders.
- [Evaluation under imbalance](#7-comparing-clustering-algorithms--evaluation-metrics) (precision/recall).

---

# 16. Master Interview Checklist

Use this as a final rapid-revision sheet. If you can confidently explain each in 1–2 minutes, you're ready.

**Foundations**
- [ ] Supervised vs Unsupervised (feedback signal, evaluation difficulty).
- [ ] Types of unsupervised learning: clustering, dim. reduction, association rules.
- [ ] Distance metrics: Euclidean, Manhattan, Minkowski (p), Cosine, Hamming — and *when* to use each.
- [ ] Why cosine for text; why Euclidean breaks in high-D (distance concentration).

**Clustering**
- [ ] K-Means step by step; WCSS/inertia; EM view; converges to *local* min.
- [ ] K-Means++ vs random init; why the *mean* (L2 objective).
- [ ] Elbow vs Silhouette; why scale before clustering.
- [ ] K-Means limitations (spherical, K, outliers, init) and fixes.
- [ ] Hierarchical: agglomerative vs divisive; single/complete/average/**Ward** linkage; reading a dendrogram; `O(n²)` cost.
- [ ] DBSCAN: core/border/noise; `eps` & `min_samples`; k-distance plot; arbitrary shapes + noise; varying-density weakness (→ HDBSCAN).
- [ ] Choosing algorithm by data shape/size/noise.
- [ ] Metrics: Silhouette (↑), Davies-Bouldin (↓), Calinski-Harabasz (↑), Inertia (elbow only).

**Dimensionality Reduction**
- [ ] Curse of dimensionality; feature selection vs extraction.
- [ ] PCA: covariance → eigenvectors/eigenvalues → components; explained variance; scree plot; standardize first; variance = min reconstruction error; orthogonality.
- [ ] Advanced PCA: whitening, loadings/interpretation, LDA vs PCA, PCA-before-model (no leakage).
- [ ] t-SNE: local vs global, perplexity, Student-t/crowding, non-deterministic, viz-only.
- [ ] SVD: `A=UΣVᵀ`, singular values, TruncatedSVD/LSA, recommenders (latent factors), PCA relationship, sparse-data advantage.
- [ ] PCA vs t-SNE vs SVD comparison table.

**Practical wisdom**
- [ ] Feature scaling: StandardScaler / MinMax / Robust — when each; fit on train only.
- [ ] Cluster interpretation: profile centroids (inverse-transform!), name segments, validate.
- [ ] Outlier detection: DBSCAN noise, LOF, Isolation Forest; imbalance → precision/recall not accuracy.

**Golden rules to say out loud in interviews**
1. *"Always scale before distance-based clustering and before PCA."*
2. *"K-Means finds local minima — use K-Means++ and multiple restarts."*
3. *"Pick the algorithm from the data's shape, size, density, and noise — not by habit."*
4. *"t-SNE is for visualization only; never feed its output into a model."*
5. *"Unsupervised learning has no ground truth — validate with internal metrics *and* domain sense."*

---

*End of theory guide. Pair this with `practical.md` for coding implementations and notebook workflows.*

