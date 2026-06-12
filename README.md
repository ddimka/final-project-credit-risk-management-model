# Credit Risk Management Model

## Project Overview

This project was developed as a final assignment for the Skillbox Machine Learning Junior program.

The objective is to build a machine learning model capable of predicting customer credit default risk based on application data and historical credit records.

### Problem Type

**Binary Classification**

Target variable:

```python
flag
```

Where:

* `0` — non-default client
* `1` — default client

---

# Business Objective

Financial institutions need reliable risk assessment tools before issuing loans.

The goal of this project is to identify high-risk borrowers by analyzing their historical credit behavior and application information.

The resulting model can be used as a component of a credit scoring system.

---

# Dataset Structure

The project uses two datasets:

### Application Data

Contains information about loan applications.

### Credit History Data

Contains multiple historical credit records for each customer.

### Key Challenge

One application (`id`) corresponds to multiple credit history records.

Therefore, the main task is to aggregate customer credit history into a single feature vector per application.

---

# Project Pipeline

## 1. Exploratory Data Analysis

Performed:

* data structure analysis;
* missing value analysis;
* target distribution analysis;
* feature inspection;
* validation of dataset consistency.

---

## 2. Feature Engineering

Feature engineering is the core part of the project.

Eight generations of aggregated features were developed and evaluated.

### Version v1

Basic credit history aggregations:

* number of loans;
* mean values;
* maximum values;
* application-level statistics.

### Version v2

Extended aggregations:

* minimum values;
* standard deviations;
* last credit characteristics.

### Version v3

Payment history features:

```text
enc_paym_0 ... enc_paym_24
```

Aggregations were calculated for all monthly payment indicators.

### Version v4

Binary risk indicators:

* zero utilization flags;
* zero delinquency indicators;
* missing activity indicators.

### Version v5

Risk-oriented features:

* credit load metrics;
* delinquency ratios;
* problematic loan shares;
* extreme risk values.

### Version v6

Full payment matrix statistics:

* mean payment behavior;
* maximum payment values;
* volatility measures.

### Version v7

Temporal payment windows and trends:

* recent payment behavior;
* historical payment behavior;
* trend calculations.

### Version v8

Advanced credit risk features:

* last loan characteristics;
* comparison of last loan vs historical behavior;
* ratio-based features;
* deviation from historical averages.

This version became the final feature set used for model selection.

---

# Model Development

The following models were trained and compared on every feature version:

* Logistic Regression
* Decision Tree
* Random Forest
* CatBoost

Each experiment was automatically stored and tracked.

---

# Evaluation Metric

Primary metric:

**ROC-AUC**

For every experiment the following metrics were monitored:

* Train ROC-AUC
* Test ROC-AUC
* Train-Test Gap

The gap was used to control overfitting.

---

# Feature Selection

After identifying the best feature version:

1. CatBoost Feature Importance was calculated.
2. Top-N feature subsets were evaluated.
3. Multiple feature sets were tested:

```text
Top-50
Top-70
Top-100
Top-150
Top-200
Top-250
```

The best-performing subset was selected for final training.

---

# Final Model

### Algorithm

CatBoostClassifier

### Parameters

```python
iterations = 700
depth = 6
learning_rate = 0.05
l2_leaf_reg = 10
auto_class_weights = "Balanced"
loss_function = "Logloss"
eval_metric = "AUC"
```

### Final Training Procedure

* Best feature version: **v8**
* Feature importance analysis
* Top-N feature selection
* Final CatBoost training

---

# Model Explainability

Feature importance was extracted from CatBoost and used to:

* rank predictors;
* remove weak features;
* build compact Top-N models;
* improve generalization.

---

# Reproducibility

The project includes:

* checkpoint system;
* feature versioning;
* experiment tracking;
* automatic result saving;
* model metadata storage.

Saved artifacts:

```text
checkpoints/
│
├── features/
├── feature_parts/
├── results/
├── importances/
├── topn/
├── final_catboost_model.pkl
├── final_model_metadata.json
└── final_features.json
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* CatBoost
* Joblib
* Matplotlib

---

# Project Structure

```text
project/
│
├── train_data/
├── train_target.csv
│
├── checkpoints/
│   ├── features/
│   ├── feature_parts/
│   ├── results/
│   ├── importances/
│   ├── topn/
│   ├── final_catboost_model.pkl
│   ├── final_model_metadata.json
│   └── final_features.json
│
└── FinalWork2_v8_en.ipynb
```

---

# Results

Final evaluation metrics are automatically calculated during training:

| Metric        | Value |
| ------------- | -------- |
| Train ROC-AUC | 0.764125 |
| Test ROC-AUC  | 0.755025 |
| Gap           | 0.0091   |

The final objective is to achieve:

```text
ROC-AUC ≥ 0.75
```

---

# Conclusion

A complete credit risk prediction pipeline was developed, including:

* credit history aggregation;
* iterative feature engineering;
* model benchmarking;
* feature importance analysis;
* feature selection;
* final CatBoost optimization.

The experiments demonstrate that feature engineering has a significantly greater impact on model quality than changing the machine learning algorithm itself.
