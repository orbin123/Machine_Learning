# Unsupervised Machine Learning Revision Roadmap

## Focus Areas
- Clustering
- Dimensionality Reduction
- Similarity & Distance Metrics
- Cluster Evaluation
- Visualization
- Feature Extraction
- Scikit-learn Implementation

---

# Week 1: Foundations of Unsupervised Learning

## Introduction to Unsupervised Learning

### Topics
- What is Unsupervised Learning?
- How it differs from Supervised Learning
- Types of Unsupervised Learning
  - Clustering
  - Dimensionality Reduction
  - Association Rule Learning (Introduction)
- Real-world applications

### Applications
- Customer Segmentation
- Recommendation Systems
- Fraud Detection
- Market Basket Analysis
- Image Compression
- Topic Modeling
- Anomaly Detection
- Gene Expression Analysis

---

## Distance & Similarity Measures

Understanding distance metrics is essential for clustering algorithms.

### Distance Metrics
- Euclidean Distance
- Manhattan Distance
- Minkowski Distance
- Cosine Similarity
- Cosine Distance
- Hamming Distance

### When to Use Each
- Numerical data
- High-dimensional data
- Sparse datasets
- Text embeddings

---

# Week 2: Clustering Algorithms

## K-Means Clustering

### Theory
- What is K-Means?
- Centroid-based clustering
- Objective function (Within-Cluster Sum of Squares)

### Algorithm Steps
1. Choose **K**
2. Initialize centroids
3. Assign points to nearest centroid
4. Update centroids
5. Repeat until convergence

### Choosing the Right Number of Clusters
- Elbow Method
- Silhouette Score
- Inertia

### Initialization
- Random Initialization
- K-Means++

### Practical
- Implement K-Means using Scikit-learn
- Visualize clusters
- Interpret cluster centers

---

## Advanced K-Means

### Topics
- Different distance metrics
- Feature scaling before clustering
- Empty cluster handling
- K-Means limitations

### Applications
- Customer segmentation
- Image compression
- Color quantization

---

## Hierarchical Clustering

### Types
- Agglomerative Clustering
- Divisive Clustering

### Linkage Methods
- Single Linkage
- Complete Linkage
- Average Linkage
- Ward Linkage

### Dendrogram
- Reading dendrograms
- Choosing clusters from dendrograms

### Practical
- Implement Hierarchical Clustering
- Plot dendrograms
- Compare linkage methods

---

## DBSCAN

### Topics
- Density-based clustering
- Core Points
- Border Points
- Noise Points

### Hyperparameters
- `eps`
- `min_samples`

### Advantages
- Finds arbitrarily shaped clusters
- Handles noise well
- No need to specify number of clusters

### Limitations
- Sensitive to parameter selection
- Difficulty with varying densities

### Practical
- Implement DBSCAN
- Visualize clusters
- Identify outliers

---

## Comparing Clustering Algorithms

### Compare

| Algorithm | Best For | Advantages | Limitations |
|------------|----------|------------|-------------|
| K-Means | Spherical clusters | Fast | Requires K |
| Hierarchical | Small datasets | Easy visualization | Computationally expensive |
| DBSCAN | Irregular clusters | Detects noise | Parameter sensitive |

### Evaluation Metrics
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Inertia (K-Means)

---

# Week 3: Dimensionality Reduction

## Introduction

### Topics
- Curse of Dimensionality
- Why dimensionality reduction is important
- Feature extraction vs Feature selection

### Applications
- Data visualization
- Noise reduction
- Faster model training
- Improved generalization

---

## Principal Component Analysis (PCA)

### Theory
- Covariance Matrix
- Eigenvalues
- Eigenvectors
- Principal Components
- Explained Variance

### Choosing Components
- Explained Variance Ratio
- Scree Plot
- Cumulative Variance

### Practical
- Implement PCA
- Reduce dimensions
- Visualize transformed data

---

## Advanced PCA

### Topics
- Whitening
- Feature interpretation
- PCA limitations
- PCA for preprocessing

### Visualization
- 2D projection
- 3D projection

---

## t-Distributed Stochastic Neighbor Embedding (t-SNE)

### Theory
- Local structure preservation
- High-dimensional visualization

### Hyperparameters
- Perplexity
- Learning Rate
- Number of Iterations

### Advantages
- Excellent visualization
- Captures nonlinear relationships

### Limitations
- Slow
- Not suitable for feature extraction
- Non-deterministic

### Practical
- Implement t-SNE
- Visualize clusters
- Compare with PCA

---

## Singular Value Decomposition (SVD)

### Theory
- Matrix decomposition
- Singular values
- Singular vectors

### Applications
- Recommendation systems
- Collaborative filtering
- Latent Semantic Analysis (LSA)
- Noise reduction

### Practical
- Implement Truncated SVD
- Analyze explained variance

---

# Week 4: Advanced Concepts & Model Comparison

## Comparing Dimensionality Reduction Techniques

| Technique | Best For | Advantages | Limitations |
|------------|----------|------------|-------------|
| PCA | Linear data | Fast, interpretable | Linear assumption |
| t-SNE | Visualization | Preserves local structure | Computationally expensive |
| SVD | Sparse data | Efficient | Less interpretable |

---

## Visualization

Learn to visualize:

- PCA projections
- t-SNE embeddings
- Cluster distributions
- Dendrograms
- Silhouette plots

---

## Feature Scaling Before Clustering

Understand why clustering algorithms require scaling.

### Techniques
- StandardScaler
- MinMaxScaler
- RobustScaler

---

## Cluster Interpretation

Learn how to:

- Analyze cluster centers
- Identify cluster characteristics
- Assign business meaning to clusters

---

## Outlier Detection

Understand how clustering algorithms can identify anomalies.

### Techniques
- DBSCAN Noise Points
- Distance-based methods
- Isolation Forest (Introduction)

---

# Practical Sessions

## Clustering

Build and compare:
- K-Means
- Hierarchical Clustering
- DBSCAN

Evaluate using:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

---

## Dimensionality Reduction

Apply:
- PCA
- t-SNE
- Truncated SVD

Compare:
- Visualization quality
- Explained variance
- Runtime
- Interpretability

---

# Real-World Projects

## Customer Segmentation
- K-Means
- PCA visualization

---

## Image Compression
- K-Means

---

## Recommendation System
- SVD

---

## Market Segmentation
- Hierarchical Clustering

---

## Anomaly Detection
- DBSCAN

---

## High-Dimensional Data Visualization
- PCA
- t-SNE

---

# Topics to Revise

## Unsupervised Learning
- Clustering
- Dimensionality Reduction
- Similarity Measures

---

## Clustering Algorithms
- K-Means
- Hierarchical Clustering
- DBSCAN

---

## Distance Metrics
- Euclidean
- Manhattan
- Minkowski
- Cosine Similarity
- Hamming Distance

---

## Cluster Evaluation
- Elbow Method
- Silhouette Score
- Inertia
- Davies-Bouldin Index
- Calinski-Harabasz Score

---

## Dimensionality Reduction
- PCA
- t-SNE
- SVD

---

## Linear Algebra Concepts
- Covariance Matrix
- Eigenvalues
- Eigenvectors
- Principal Components
- Singular Values

---

## Visualization
- Scatter Plots
- Cluster Plots
- Dendrograms
- Scree Plots
- Silhouette Plots

---

# Interview Preparation Checklist

Be able to explain:

- Supervised vs Unsupervised Learning.
- K-Means algorithm step by step.
- Why feature scaling is important before clustering.
- Elbow Method vs Silhouette Score.
- K-Means vs Hierarchical Clustering vs DBSCAN.
- Core Points, Border Points, and Noise Points in DBSCAN.
- Agglomerative vs Divisive Clustering.
- Single, Complete, Average, and Ward Linkage.
- Curse of Dimensionality.
- PCA mathematics (covariance, eigenvalues, eigenvectors).
- Explained Variance Ratio.
- PCA vs t-SNE.
- PCA vs SVD.
- When to use t-SNE instead of PCA.
- How SVD is used in recommendation systems.
- Choosing the right clustering or dimensionality reduction algorithm based on the dataset.

---

# Practice Goals

- Solve **25–40 clustering and dimensionality reduction exercises** using Scikit-learn.
- Build **4–6 end-to-end unsupervised learning projects** covering clustering, visualization, and feature extraction.
- Practice comparing multiple algorithms on the same dataset and explain **why one algorithm performs better** based on data distribution, cluster shape, scalability, and evaluation metrics.