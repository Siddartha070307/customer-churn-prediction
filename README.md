# Customer Churn Prediction

## Overview
This project explores customer churn prediction using the Telco Customer Churn dataset and Python's Scikit-learn library. The goal is to identify customers who are likely to leave a telecom service and understand how different modeling choices affect churn detection.

## Objectives
- Explore customer data and identify patterns associated with churn.
- Clean and preprocess numerical and categorical features.
- Train a Logistic Regression baseline model.
- Address class imbalance using class weights.
- Experiment with decision thresholds to study the trade-off between precision and recall.

## Dataset
- **Source:** [Telco Customer Churn dataset — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Target:** `Churn` (`Yes` or `No`)
- **Excluded feature:** `customerID`, because it is an identifier rather than a useful predictive feature.

### Data preprocessing
- Converted `TotalCharges` to numeric values.
- Found 11 missing `TotalCharges` values, all corresponding to customers with zero tenure; filled these values with `0`.
- Used an 80/20 stratified train-test split.
- Standardized numerical features and one-hot encoded categorical features.
- Fitted preprocessing on the training data before transforming the test data.

## Exploratory Data Analysis
The initial analysis found several patterns:
- Customers who churned had an average tenure of approximately 18 months, compared with 37.6 months for customers who stayed.
- Churned customers had higher average monthly charges: approximately $74.44 versus $61.27.
- Churn rates differed substantially by contract type: approximately 42.7% for month-to-month contracts, 11.3% for one-year contracts, and 2.8% for two-year contracts.

These are observed associations and do not establish causation.

## Models and Experiments

### 1. Baseline Logistic Regression
The baseline model achieved approximately **81% accuracy**.

| Metric | No Churn | Churn |
|---|---:|---:|
| Precision | 0.85 | 0.66 |
| Recall | 0.89 | 0.56 |
| F1-score | 0.87 | 0.60 |

The baseline model missed 165 of the 374 actual churn cases in the test set.

### 2. Logistic Regression with Class Weights
Used `class_weight="balanced"` to give greater importance to the minority churn class.

| Metric | No Churn | Churn |
|---|---:|---:|
| Precision | 0.90 | 0.50 |
| Recall | 0.72 | 0.78 |
| F1-score | 0.80 | 0.61 |

Accuracy decreased to approximately 74%, but churn recall increased from 0.56 to 0.78. This experiment demonstrated why accuracy alone is insufficient for imbalanced classification problems.

### 3. Decision Threshold Tuning
Experimented with a decision threshold of `0.30` on the balanced Logistic Regression model.

| Metric | No Churn | Churn |
|---|---:|---:|
| Precision | 0.96 | 0.43 |
| Recall | 0.55 | 0.93 |
| F1-score | 0.70 | 0.59 |

At this threshold, the model detected 347 of the 374 actual churn cases but produced 461 false positives. Lowering the threshold increased churn recall while reducing precision and overall accuracy.

The appropriate threshold depends on the business cost of missed churners versus unnecessary retention interventions.

## Tools and Technologies
- Python
- Google Colab / Jupyter Notebook
- Pandas
- NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Git and GitHub

## Key Learnings
- Data cleaning and preprocessing for machine learning.
- Exploratory data analysis and feature relationships.
- Stratified train-test splitting and feature encoding.
- Logistic Regression classification.
- Class imbalance and class weighting.
- Confusion matrices, precision, recall, F1-score, and accuracy.
- Decision threshold tuning and precision-recall trade-offs.

## Next Steps
- Build a complete preprocessing and modeling pipeline.
- Compare additional classification algorithms.
- Evaluate threshold choices using training-only validation or cross-validation.
- Preserve an untouched test set for final evaluation.
- Investigate feature importance and model interpretability.
- Consider business costs when choosing a final decision threshold.

## Disclaimer
The experiments documented here are exploratory. The test set has been used for multiple comparisons, so a final model evaluation should use a properly selected threshold and an untouched test set where possible.
