# Customer Churn Prediction

## Overview
This project explores customer churn prediction using the Telco Customer Churn dataset and Python's Scikit-learn library. The goal is to identify customers who are likely to leave a telecom service and understand how different modeling choices affect churn detection.

## Objectives
- Explore customer data and identify patterns associated with churn.
- Clean and preprocess numerical and categorical features.
- Train a Logistic Regression baseline model.
- Address class imbalance using class weights.
- Select a decision threshold on a separate validation split, then report final test metrics.

## Dataset
- **Source:** [Telco Customer Churn dataset — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers (1,869 churned, 5,174 stayed; about 26.5% churn)
- **Target:** `Churn` (`Yes` or `No`)
- **Excluded feature:** `customerID`, because it is an identifier rather than a useful predictive feature.

### Data preprocessing
- Converted `TotalCharges` to numeric values.
- Found 11 missing `TotalCharges` values, all corresponding to customers with zero tenure; filled these values with `0`.
- Used an 80/20 stratified train-test split (`random_state=42`).
- Split the training portion again (75% fit / 25% validation, stratified) for threshold selection: 4,225 fit, 1,409 validation, 1,409 test rows.
- Standardized numerical features and one-hot encoded categorical features.
- Fitted preprocessing on the fit data only before transforming validation and test data.

## Exploratory Data Analysis
The initial analysis found several patterns:
- Customers who churned had an average tenure of approximately 18 months, compared with 37.6 months for customers who stayed.
- Churned customers had higher average monthly charges: approximately $74.44 versus $61.27.
- Churn rates differed substantially by contract type: approximately 42.7% for month-to-month contracts, 11.3% for one-year contracts, and 2.8% for two-year contracts.

These are observed associations and do not establish causation.

## Models and Experiments

### Early experiments (evaluated on the test set)
The three experiments below were scored on the test set. See the limitation under Final Evaluation.

**1. Baseline Logistic Regression** — approximately **81% accuracy**.

| Metric | No Churn | Churn |
|---|---:|---:|
| Precision | 0.85 | 0.66 |
| Recall | 0.89 | 0.56 |
| F1-score | 0.87 | 0.60 |

The baseline model missed 165 of the 374 actual churn cases in the test set.

**2. Logistic Regression with Class Weights** — used `class_weight="balanced"` to give greater importance to the minority churn class.

| Metric | No Churn | Churn |
|---|---:|---:|
| Precision | 0.90 | 0.50 |
| Recall | 0.72 | 0.78 |
| F1-score | 0.80 | 0.61 |

Accuracy decreased to approximately 74%, but churn recall increased from 0.56 to 0.78. This experiment showed why accuracy alone is insufficient for imbalanced classification problems.

**3. Threshold of 0.30 on the balanced model**

| Metric | No Churn | Churn |
|---|---:|---:|
| Precision | 0.96 | 0.43 |
| Recall | 0.55 | 0.93 |
| F1-score | 0.70 | 0.59 |

At this threshold, the model detected 347 of the 374 actual churn cases but produced 461 false positives. Lowering the threshold increased churn recall while reducing precision and overall accuracy.

### Threshold selection on a validation split
The class-weighted model was refit on the fit split (4,225 rows), and thresholds from 0.3 to 0.7 were compared on the validation set only:

| Threshold | Churn Precision | Churn Recall | F1-Score | False Positives | False Negatives |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.419 | 0.912 | 0.575 | 472 | 33 |
| 0.4 | 0.460 | 0.856 | 0.599 | 375 | 54 |
| **0.5** | **0.515** | **0.789** | **0.623** | **278** | **79** |
| 0.6 | 0.564 | 0.703 | 0.626 | 203 | 111 |
| 0.7 | 0.635 | 0.586 | 0.609 | 126 | 155 |

**Why threshold 0.5:** F1 is nearly flat from 0.4 to 0.6, and 0.6 is higher than 0.5 by only 0.003. Threshold 0.5 keeps churn recall much higher than 0.6 (0.789 vs 0.703; 79 vs 111 missed churners) while precision stays at 0.515. Lower thresholds raise recall further but add 375 to 472 false positives. This choice assumes that missing a customer who will churn costs more than contacting one who would have stayed. It is also the natural cutoff for a class-weighted model, since the weighting already compensates for the class imbalance.

### Final Evaluation (test set, threshold 0.5)

Confusion matrix (rows = actual, columns = predicted; order No, Yes):

|  | Predicted No | Predicted Yes |
|---|---:|---:|
| **Actual No** | 749 | 286 |
| **Actual Yes** | 81 | 293 |

| Metric | No Churn | Churn |
|---|---:|---:|
| Precision | 0.90 | 0.51 |
| Recall | 0.72 | 0.78 |
| F1-score | 0.80 | 0.61 |

Overall accuracy is 0.74 (macro-average F1 0.71). The model caught 293 of 374 churners, with 286 false alarms among 1,035 non-churners.

**Limitation:** the test set was also used in the earlier experiments above (baseline model, class-weighted model, and the 0.30 threshold check). These final test metrics are therefore **not a fully independent estimate** of performance on new customers and may be somewhat optimistic.

The appropriate threshold in practice depends on the business cost of missed churners versus unnecessary retention interventions.

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
- Stratified train/validation/test splitting and feature encoding.
- Logistic Regression classification.
- Class imbalance and class weighting.
- Confusion matrices, precision, recall, F1-score, and accuracy.
- Choosing a decision threshold on a validation set and the precision-recall trade-off.
- Why a test set that has been used for earlier comparisons gives a less reliable final estimate.

## Next Steps
- Build a complete preprocessing and modeling pipeline.
- Compare additional classification algorithms.
- Use cross-validation for model and threshold selection.
- Evaluate the final model on a fresh, untouched holdout set.
- Investigate feature importance and model interpretability.
- Consider business costs when choosing a final decision threshold.

## Disclaimer
The threshold was selected on a validation split, but the test set was used in earlier experiments, so the reported test metrics are exploratory rather than a fully independent evaluation. A final model evaluation should use a holdout set that was never used for any comparison.
