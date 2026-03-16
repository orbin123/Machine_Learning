# Telco Customer Churn ETL Pipeline & ML Model

A comprehensive **Apache Spark-based ETL pipeline** combined with **machine learning models** for predicting customer churn in the telecommunications industry. This project demonstrates best practices in data engineering, feature engineering, and model training using PySpark.

---

## 🎯 Project Overview

This project implements an end-to-end solution for:
- **Data Extraction**: Loading customer data from CSV and SQLite databases
- **Data Transformation**: Cleaning, feature engineering, and data preparation
- **Data Loading**: Storing processed data in Parquet format for efficient access
- **Machine Learning**: Training and evaluating churn prediction models
- **Model Deployment**: Saving optimized models for production use

### Business Context
Predicting customer churn helps telecommunications companies identify at-risk customers and implement retention strategies. This pipeline processes the Telco Customer Churn dataset with synthetic supplementary records for a complete data engineering demonstration.

---

## 📊 Project Structure

```
spark-etl-pipeline/
├── data/                                # Data directory
│   ├── raw/
│   │   ├── Telco-Customer-Churn.csv   # Primary customer data
│   │   └── supplementary_data.db      # SQLite database with additional records
│   ├── raw_extracted/                # Extracted raw data (Parquet format)
│   ├── transformed/                   # Transformed data after cleaning
│   ├── features_engineered/           # Data after feature engineering
│   ├── test_data/                     # Test dataset for model evaluation
│   └── telco_churn_parquet/          # Final processed data
├── models/                             # Trained ML models
│   ├── best_rf_model/                # Best Random Forest model
│   ├── lr_model/                     # Logistic Regression model
│   ├── rf_model/                     # Random Forest model
│   └── production/                   # Production-ready models
├── notebooks/                          # Jupyter notebooks
│   └── data_exploration.ipynb        # EDA and analysis
├── src/                                # Source code
│   ├── etl_pipeline.py               # Main ETL orchestration
│   ├── extract.py                    # Data extraction module
│   ├── transform.py                  # Data transformation module
│   ├── load.py                       # Data loading module
│   ├── feature_preparation.py        # Feature engineering
│   ├── model_training.py             # ML model training
│   └── ml_data_loader.py             # ML data utilities
├── output/                             # Pipeline outputs
│   └── logs/                          # Execution logs
├── generate_sql_data.py               # Utility to generate synthetic SQL data
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🛠 Prerequisites & Installation

### Requirements
- **Python 3.8+**
- **Apache Spark 3.5.1** (included via PySpark)
- **Java Runtime Environment (JRE)**

### Setup Instructions

1. **Clone the repository:**
   ```bash
   cd spark-etl-pipeline
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')"
   ```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **pyspark** | 3.5.1 | Distributed processing framework |
| **pandas** | 2.2.1 | Data manipulation & analysis |
| **scikit-learn** | 1.4.1 | Machine learning utilities |
| **matplotlib** | 3.8.3 | Data visualization |
| **seaborn** | 0.13.2 | Statistical visualization |
| **jupyter** | 1.0.0 | Interactive notebooks |

---

## 🚀 Usage Guide

### 1. Generate Synthetic Data
Create supplementary SQLite records to augment the main CSV dataset:
```bash
python generate_sql_data.py
```
This creates `data/raw/supplementary_data.db` with 20 synthetic customer records.

### 2. Run the Complete ETL Pipeline
Execute the full data processing pipeline:
```bash
python src/etl_pipeline.py
```

**Pipeline Steps:**
1. ✅ Create Spark Session
2. ✅ Extract data from CSV + SQLite
3. ✅ Transform and clean data
4. ✅ Load to Parquet format
5. ✅ Verify output integrity

### 3. Train Machine Learning Models
Train and evaluate churn prediction models:
```bash
python src/model_training.py
```

**Models trained:**
- Logistic Regression
- Random Forest Classifier
- Best performing model selection

### 4. Explore Data
Open the Jupyter notebook for exploratory data analysis:
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

---

## 🔄 ETL Pipeline Details

### Extract Phase (`src/extract.py`)
- **Source 1**: CSV file with primary customer records
- **Source 2**: SQLite database with supplementary data
- **Process**: Union both sources, standardize schema
- **Output**: Combined raw dataset in Parquet format

### Transform Phase (`src/transform.py`)
- Data type corrections
- Missing value handling
- Categorical encoding
- Feature derivation (tenure groups, spending ratios)
- Outlier detection and treatment
- Data validation and quality checks

### Load Phase (`src/load.py`)
- Save processed data in Parquet format (columnar, compressed)
- Partitioning for optimized queries
- Schema preservation
- Output verification and profiling

---

## 🤖 Machine Learning Pipeline

### Feature Engineering
**20 selected features** used for model training:
- **Demographic**: gender, SeniorCitizen, Partner, Dependents
- **Service Usage**: PhoneService, MultipleLines, InternetService
- **Subscriptions**: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account**: tenure, Contract, PaperlessBilling, PaymentMethod
- **Financial**: MonthlyCharges, TotalCharges
- **Engineered**: TenureGroup, AvgMonthlySpendRatio

### Target Variable
- **Churn**: Binary classification (Yes/No)

### Models Implemented

#### 1. Logistic Regression
- Fast, interpretable baseline model
- Suitable for binary classification
- Quick training and inference

#### 2. Random Forest Classifier
- Ensemble method for better accuracy
- Handles non-linear relationships
- Feature importance analysis
- Better generalization on imbalanced data

### Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Validation**: Train-test split (80-20) with cross-validation
- **Hyperparameter Tuning**: Grid search with cross-validator

---

## 📁 Key Files & Functions

| File | Purpose |
|------|---------|
| `etl_pipeline.py` | Main orchestration script - runs entire ETL pipeline |
| `extract.py` | Data extraction from CSV and SQLite sources |
| `transform.py` | Data cleaning, transformation, and feature engineering |
| `load.py` | Data loading to Parquet format with verification |
| `model_training.py` | Build, train, and evaluate ML models |
| `feature_preparation.py` | Feature vector assembly for ML |
| `ml_data_loader.py` | Utilities for loading data into ML pipeline |
| `generate_sql_data.py` | Generate synthetic SQLite records |

---

## 📊 Data Processing Summary

| Stage | Input | Output | Format |
|-------|-------|--------|--------|
| Raw | CSV + SQLite | Combined raw data | Parquet |
| Transformed | Raw data | Cleaned & processed | Parquet |
| Engineered | Transformed | Features prepared | Parquet |
| ML-ready | Engineered | Train/test split | Parquet |

---

## 🎓 Model Outputs

Trained models are saved in the `models/` directory:
- **Metadata**: Model configuration and parameters
- **Stages**: Individual transformation and classifier stages
- **Formats**: PySpark ML Pipeline (.parquet)

Models can be loaded and used for batch predictions or real-time inference.

---

## 📈 Performance Metrics

The pipeline includes evaluation metrics for:
- **Classification Accuracy**: Overall correctness
- **Precision & Recall**: Trade-off between false positives and false negatives
- **F1-Score**: Harmonic mean for imbalanced datasets
- **ROC-AUC**: Discrimination ability across thresholds

---

## 🔧 Configuration

Key configuration parameters in the source files:

**ETL Pipeline** (`src/etl_pipeline.py`):
```python
CSV_PATH = "data/raw/Telco-Customer-Churn.csv"
DB_PATH = "data/raw/supplementary_data.db"
OUTPUT_PATH = "data/processed/telco_churn_clean"
```

**Model Training** (`src/model_training.py`):
```python
SEED = 42  # Reproducibility
FEATURE_COLUMNS = [...]  # Selected features
TARGET_COLUMN = "Churn"
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `java.lang.OutOfMemoryError`
- **Solution**: Increase Java heap size
  ```bash
  export SPARK_DRIVER_MEMORY=4g
  export SPARK_EXECUTOR_MEMORY=4g
  ```

**Issue**: SQLite database not found
- **Solution**: Run `python generate_sql_data.py` first

**Issue**: Parquet file not found in data directory
- **Solution**: Ensure ETL pipeline completed successfully; check logs

**Issue**: Missing dependencies
- **Solution**: Reinstall requirements
  ```bash
  pip install --upgrade -r requirements.txt
  ```

---

## 📚 Technologies Used

- **Apache Spark 3.5.1**: Distributed data processing
- **PySpark**: Python API for Spark
- **Pandas**: Data analysis and manipulation
- **Scikit-learn**: ML utilities and metrics
- **SQLite**: Supplementary data storage
- **Jupyter**: Interactive analysis notebooks

---

## 🎯 Next Steps & Enhancements

- [ ] Add data drift detection for production models
- [ ] Implement hyperparameter tuning with Hyperopt
- [ ] Add real-time prediction API using Flask/FastAPI
- [ ] Implement automated model retraining pipeline
- [ ] Add data quality monitoring and alerting
- [ ] Develop customer retention recommendation engine
- [ ] Containerize with Docker for deployment
- [ ] Add comprehensive unit tests

---

## 📝 License

This project is part of a data engineering portfolio.

---

## 👤 Author

Data Engineering & ML Pipeline Development

---

## 📞 Questions?

For issues or questions about the pipeline, refer to individual module docstrings or run modules with `--help` flag where applicable.

---

**Version**: 1.0  
**Last Updated**: March 2026
