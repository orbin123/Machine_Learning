# Supervised Machine Learning Revision Roadmap

## Focus Areas
- Regression
- Classification
- Ensemble Learning
- Model Evaluation
- Hyperparameter Tuning
- Feature Engineering
- Cross Validation
- Scikit-learn Implementation
- Real-World Machine Learning Projects

---

# Week 1: Regression

## Introduction to Regression

### Topics
- What is Regression?
- Regression vs Classification
- Dependent Variable (Target)
- Independent Variables (Features)
- Real-world applications of regression

Examples:
- House Price Prediction
- Sales Forecasting
- Demand Prediction
- Salary Prediction

---

## Linear Regression

### Theory
- Simple Linear Regression
- Multiple Linear Regression
- Regression Equation
- Ordinary Least Squares (OLS)
- Cost Function

### Assumptions
- Linearity
- Independence
- Homoscedasticity
- Normality
- No Multicollinearity

### Implementation
- Train using Scikit-learn
- Predict values
- Interpret coefficients

### Evaluation Metrics
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## Polynomial Regression

### Topics
- Polynomial Features
- Choosing the polynomial degree
- Underfitting vs Overfitting
- Polynomial vs Linear Regression

### Implementation
- PolynomialFeatures
- LinearRegression pipeline

---

## Regularization

### Ridge Regression (L2)

#### Topics
- Why regularization?
- Ridge loss function
- Alpha parameter
- Preventing overfitting

#### Practice
- Implement Ridge Regression
- Tune alpha

---

### Lasso Regression (L1)

#### Topics
- L1 Regularization
- Feature Selection
- Sparse models

#### Practice
- Implement Lasso Regression
- Tune alpha

---

### ElasticNet

#### Topics
- Combining L1 and L2 regularization
- Choosing L1 ratio

---

## Practical Session

Build regression models using:
- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- ElasticNet

Compare:
- Accuracy
- RMSE
- R² Score
- Training Time

---

## Project

### House Price Prediction

Tasks:
- Data Cleaning
- Feature Engineering
- Model Building
- Hyperparameter Tuning
- Model Evaluation
- Feature Importance
- Final Prediction

---

# Week 2: Classification

## Introduction to Classification

### Topics
- Binary Classification
- Multi-class Classification
- Multi-label Classification

### Applications
- Spam Detection
- Disease Prediction
- Fraud Detection
- Customer Churn
- Sentiment Analysis

---

## Logistic Regression

### Topics
- Sigmoid Function
- Decision Boundary
- Probability Prediction
- Odds and Log Odds

### Evaluation
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score

---

## K-Nearest Neighbors (KNN)

### Topics
- Distance Metrics
  - Euclidean
  - Manhattan
- Choosing K
- Lazy Learning

### Pros
- Simple
- No training phase

### Cons
- Slow prediction
- Sensitive to scaling

---

## Decision Trees

### Topics
- Tree Structure
- Splitting Criteria
  - Gini Index
  - Entropy
- Information Gain
- Tree Depth
- Pruning

### Advantages
- Easy interpretation
- Handles nonlinear data

---

## Support Vector Machine (SVM)

### Topics
- Hyperplanes
- Support Vectors
- Margins
- Kernel Trick

### Kernels
- Linear
- Polynomial
- RBF
- Sigmoid

### Hyperparameters
- C
- Gamma
- Kernel

---

## Practical Session

Train and compare:
- Logistic Regression
- KNN
- Decision Tree
- SVM

Evaluate using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

# Week 3: Advanced Supervised Learning

## Feature Scaling

### Techniques
- StandardScaler
- MinMaxScaler
- RobustScaler

---

## Handling Imbalanced Data

### Topics
- Oversampling
- Undersampling
- SMOTE
- Class Weights

---

## Cross Validation

### Types
- Train-Test Split
- K-Fold
- Stratified K-Fold
- Leave-One-Out (LOOCV)

---

## Hyperparameter Tuning

### Grid Search

- GridSearchCV

### Random Search

- RandomizedSearchCV

### Best Practices
- Choosing parameter ranges
- Cross-validation during tuning

---

## Model Interpretation

### Topics
- Feature Importance
- Coefficients
- Permutation Importance
- SHAP (Introduction)
- LIME (Introduction)

---

## Practical Session

- Tune regression models
- Tune classification models
- Compare tuned vs default models

---

# Week 4: Ensemble Learning

## Introduction to Ensemble Methods

### Why Ensemble Learning?

- Reduce variance
- Reduce bias
- Improve accuracy

### Types

- Bagging
- Boosting
- Stacking
- Voting

---

## Random Forest

### Topics
- Decision Tree Ensemble
- Bootstrapping
- Feature Sampling
- Majority Voting

### Advantages
- High accuracy
- Robust
- Feature Importance

### Advanced Topics
- Out-of-Bag (OOB) Error
- Handling Imbalanced Data

### Hyperparameters
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features

---

## Practical Session

Build Random Forest model

Evaluate:
- Accuracy
- Feature Importance
- OOB Score

---

## Gradient Boosting

### Topics
- Sequential Learning
- Weak Learners
- Residual Learning

---

## XGBoost

### Topics
- Gradient Boosting improvements
- Regularization
- Missing value handling
- Parallel processing

### Hyperparameters
- Learning Rate
- Max Depth
- Number of Trees
- Subsample
- Colsample_bytree

---

## Other Boosting Algorithms

- AdaBoost
- LightGBM
- CatBoost

Know:
- Advantages
- Differences
- When to use each

---

## Advanced Boosting

### Topics
- Early Stopping
- Learning Rate Scheduling
- Preventing Overfitting

---

## Practical Session

Train and compare:
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

Compare:
- Accuracy
- Speed
- Feature Importance

---

# Week 5: Model Evaluation

## Regression Metrics

- MAE
- MSE
- RMSE
- R² Score
- Adjusted R²

---

## Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Specificity

---

## Confusion Matrix

Understand:
- True Positive
- False Positive
- True Negative
- False Negative

---

## ROC Curve

### Topics
- ROC Curve
- Threshold Selection
- AUC Score

---

## Precision-Recall Curve

Understand:
- Precision-Recall trade-off
- PR Curve
- Average Precision

---

## Cross Validation

Implement:
- K-Fold
- Stratified K-Fold
- Leave-One-Out

---

## Model Comparison

Compare:

### Regression
- Linear Regression
- Ridge
- Lasso
- ElasticNet

### Classification
- Logistic Regression
- KNN
- Decision Tree
- SVM

### Ensemble
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

---

# Real-World Case Studies

## Regression

- House Price Prediction
- Stock Price Prediction
- Sales Forecasting

---

## Classification

- Disease Prediction
- Credit Risk Prediction
- Fraud Detection
- Customer Churn Prediction

---

## Ensemble Learning

- Patient Outcome Prediction
- Credit Scoring
- Loan Approval

---

# End-to-End Machine Learning Projects

## Regression Project
- House Price Prediction

## Classification Project
- Heart Disease Prediction

## Ensemble Project
- Customer Churn Prediction

For each project:

- Data Cleaning
- EDA
- Feature Engineering
- Feature Selection
- Train-Test Split
- Model Training
- Hyperparameter Tuning
- Cross Validation
- Model Evaluation
- Feature Importance
- Final Prediction
- Model Interpretation

---

# Topics to Revise

## Regression
- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- ElasticNet

---

## Classification
- Logistic Regression
- KNN
- Decision Trees
- SVM

---

## Ensemble Learning
- Bagging
- Boosting
- Stacking
- Voting
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

---

## Model Optimization
- Feature Scaling
- Regularization
- Hyperparameter Tuning
- Cross Validation
- Handling Imbalanced Data

---

## Model Evaluation
- MAE
- MSE
- RMSE
- R²
- Confusion Matrix
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC
- Precision-Recall Curve

---

# Interview Preparation Checklist

Be able to explain:

- Why Linear Regression uses OLS.
- Assumptions of Linear Regression.
- Ridge vs Lasso vs ElasticNet.
- Logistic Regression vs Linear Regression.
- KNN vs Decision Trees vs SVM.
- Gini Index vs Entropy.
- Bias-Variance Tradeoff.
- Underfitting vs Overfitting.
- Regularization (L1 vs L2).
- Random Forest vs Gradient Boosting.
- XGBoost vs LightGBM vs CatBoost.
- Bagging vs Boosting vs Stacking.
- Feature Importance methods.
- Hyperparameter tuning with GridSearchCV and RandomizedSearchCV.
- Cross-validation strategies.
- Choosing the right evaluation metric for regression and classification problems.

---
