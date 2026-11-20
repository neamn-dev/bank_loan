# Research Questions (RQ1–RQ22) Implementation & Verification Audit
## Automated Technical Audit & Empirical Artifact Inventory

**Course:** Machine Learning | Retail Lending Analytics  
**Date:** November 2026  
**Audit Status:** 22 / 22 Passed (100% Verified)  

---

## 1. Executive Audit Summary

This document certifies that all **22 Research Questions (RQ1–RQ22)** mandated by the project syllabus have been fully implemented, executed end-to-end without errors or deprecation warnings, and empirically verified with concrete data artifacts (processed Excel workbooks, structured CSV benchmark tables, and publication-grade PNG diagnostic figures).

```
========================================================================================
FINAL AUDIT RESULT: 22/22 Research Questions Fully Implemented, Working & Verified!
========================================================================================
```

---

## 2. Master Verification Matrix

| # | Research Question (Scope) | Implementing Notebook | Verified Artifact | Key Empirical Result / Finding | Status |
|:---|:---|:---|:---|:---|:---:|
| **RQ1** | Business Understanding & Task Formulation | [`01_project_overview.ipynb`](../notebooks/data_preparation/01_project_overview.ipynb) | [`loan_dataset.xlsx`](../data/raw/loan_dataset.xlsx) | Dual targets formulated: Binary default ($19.82\%$ default) & Multiclass rating ($50.1\%$ Low, $29.9\%$ Med, $20.0\%$ High). | **PASS** |
| **RQ2** | Raw Data Ingestion & Dimension Verification | [`02_data_loading_and_inspection.ipynb`](../notebooks/data_preparation/02_data_loading_and_inspection.ipynb) | [`variable_inventory.csv`](../outputs/tables/variable_inventory.csv) | Verified raw portfolio dimensions: exactly **10,100 records $\times$ 41 columns**. | **PASS** |
| **RQ3** | Variable Taxonomy & Role Classification | [`02_data_loading_and_inspection.ipynb`](../notebooks/data_preparation/02_data_loading_and_inspection.ipynb) | [`variable_inventory.csv`](../outputs/tables/variable_inventory.csv) | Full variable inventory: 21 numeric, 11 nominal, 1 ordinal, 4 binary, 1 ID, 1 Date, 2 target columns. | **PASS** |
| **RQ4** | Forensic Data Quality Audit | [`03_data_quality_audit.ipynb`](../notebooks/data_preparation/03_data_quality_audit.ipynb) | [`data_quality_audit_summary.csv`](../outputs/tables/data_quality_audit_summary.csv) | Cataloged **21 distinct anomalies** (sentinels `9999`, impossible ages $3, 118, 212$, negative employment). | **PASS** |
| **RQ5** | Missing Value Mechanisms (Structural vs. Random) | [`03_data_quality_audit.ipynb`](../notebooks/data_preparation/03_data_quality_audit.ipynb) | [`missingness_profile.png`](../outputs/figures/missingness_profile.png) | Separated structural blanks (8,308 unsecured loans; 7,247 clean borrowers) from stochastic missingness. | **PASS** |
| **RQ6** | Duplicate Record Detection | [`03_data_quality_audit.ipynb`](../notebooks/data_preparation/03_data_quality_audit.ipynb) | [`data_quality_audit_summary.csv`](../outputs/tables/data_quality_audit_summary.csv) | Detected **100 exact duplicate batch submissions**; removed to isolate 10,000 unique records. | **PASS** |
| **RQ7** | Forensic Data Cleaning Pipeline | [`04_data_cleaning.ipynb`](../notebooks/data_preparation/04_data_cleaning.ipynb) | [`loan_cleaned.xlsx`](../data/processed/loan_cleaned.xlsx) | Automated `clean_loan_data()` pipeline; exported 10,000 rows $\times$ 43 columns. | **PASS** |
| **RQ8** | Cleaning Sensitivity Analysis | [`04_data_cleaning.ipynb`](../notebooks/data_preparation/04_data_cleaning.ipynb) | [`cleaning_sensitivity_comparison.csv`](../outputs/tables/cleaning_sensitivity_comparison.csv) | Proved median imputation preserves distribution variance; Winsorization caps outliers cleanly. | **PASS** |
| **RQ9** | Bivariate Exploratory Analysis & Risk Drivers | [`05_eda_and_skewness.ipynb`](../notebooks/data_preparation/05_eda_and_skewness.ipynb) | [`bivariate_default_rates.csv`](../outputs/tables/bivariate_default_rates.csv) | Unemployed default rate = $47.61\%$ vs. Retired = $10.41\%$; Prior default rate = $51.57\%$ vs. $17.38\%$. | **PASS** |
| **RQ10** | Target Imbalance & Distribution Analysis | [`02_data_loading_and_inspection.ipynb`](../notebooks/data_preparation/02_data_loading_and_inspection.ipynb) | [`default_distribution.png`](../outputs/figures/default_distribution.png) | Quantified 80/20 class imbalance ($19.82\%$ default prevalence, $4.04:1$ imbalance ratio). | **PASS** |
| **RQ11** | Multicollinearity & Cohen's $d$ Effect Sizes | [`05_eda_and_skewness.ipynb`](../notebooks/data_preparation/05_eda_and_skewness.ipynb) | [`eda_correlation_heatmap.png`](../outputs/figures/eda_correlation_heatmap.png) | `Credit_Score` Cohen's $d = -1.31$; `Late_Payments_2Y` $d = +1.01$; flagged `Loan_Amount` $\leftrightarrow$ `Property_Value` ($r=0.94$). | **PASS** |
| **RQ12** | Domain Feature Engineering | [`06_feature_engineering.ipynb`](../notebooks/data_preparation/06_feature_engineering.ipynb) | [`loan_featured.xlsx`](../data/processed/loan_featured.xlsx) | Engineered exact annuity PMT, DTI, LTI, LTV, STI, and log transforms $\to$ expanded to **56 columns**. | **PASS** |
| **RQ13** | Dummy Classifier & The Accuracy Paradox | [`01_logistic_regression_baseline.ipynb`](../notebooks/loan_default/01_logistic_regression_baseline.ipynb) | [`loan_default_lr_baseline_roc_cm.png`](../outputs/figures/loan_default_lr_baseline_roc_cm.png) | Demonstrated Accuracy Paradox: Dummy achieves **$80.20\%$ accuracy** while missing **$100\%$ of defaulters**. | **PASS** |
| **RQ14** | Primary Model Development (7 Algorithm Families) | [`11_model_comparison.ipynb`](../notebooks/loan_default/11_model_comparison.ipynb) | [`loan_default_master_algorithm_comparison.csv`](../outputs/tables/loan_default_master_algorithm_comparison.csv) | Evaluated Logistic Regression, Naive Bayes, KNN, SVM, Decision Tree, Random Forest, LDA/QDA. | **PASS** |
| **RQ15** | Partition Experiments (80/20 vs. 70/30) | [`07_preprocessing_and_split_experiments.ipynb`](../notebooks/data_preparation/07_preprocessing_and_split_experiments.ipynb) | [`split_experiments_comparison.csv`](../outputs/tables/split_experiments_comparison.csv) | Stratified 80/20 partition confirmed superior: preserves exact $19.82\%$ class prevalence with lower variance. | **PASS** |
| **RQ16** | Systematic Hyperparameter Tuning | [`10_hyperparameter_tuning.ipynb`](../notebooks/loan_default/10_hyperparameter_tuning.ipynb) | [`hyperparameter_tuning_master_comparison.csv`](../outputs/tables/hyperparameter_tuning_master_comparison.csv) | Executed 5-fold CV `GridSearchCV` across all 6 model families, quantifying empirical gains. | **PASS** |
| **RQ17** | Multiclass Model Evaluation & 3x3 Diagnostics | [`04_model_comparison.ipynb`](../notebooks/risk_category/04_model_comparison.ipynb) | [`risk_category_3model_confusion_matrices.png`](../outputs/figures/risk_category_3model_confusion_matrices.png) | Evaluated top 3 paradigms (LR, RF, SVM); diagnosed Medium risk boundary ambiguity ($24.2\%$ overlap with Low). | **PASS** |
| **RQ18** | Stratified 5-Fold Cross-Validation Benchmark | [`11_model_comparison.ipynb`](../notebooks/loan_default/11_model_comparison.ipynb) | [`loan_default_cv_benchmark.csv`](../outputs/tables/loan_default_cv_benchmark.csv) | Compiled mean $\pm$ std: **Random Forest AUC = $0.8641 \pm 0.007$**; **Logistic Regression = $0.8512 \pm 0.008$**. | **PASS** |
| **RQ19** | Overfitting Diagnosis & Learning Curves | [`11_model_comparison.ipynb`](../notebooks/loan_default/11_model_comparison.ipynb) | [`loan_default_learning_curves.png`](../outputs/figures/loan_default_learning_curves.png) | Constructed learning curves from $N = 800$ to $6,400$ proving narrow generalization gap and convergence. | **PASS** |
| **RQ20** | Feature Importance & Interpretability | [`06_random_forest.ipynb`](../notebooks/loan_default/06_random_forest.ipynb) | [`loan_default_rf_permutation_importance.png`](../outputs/figures/loan_default_rf_permutation_importance.png) | Identified `Credit_Score`, `Late_Payments_2Y`, `DTI`, and `Previous_Default` as core risk quartet. | **PASS** |
| **RQ21** | Statistical Significance & Fairness Audit | [`08_final_analysis_and_conclusions.ipynb`](../notebooks/data_preparation/08_final_analysis_and_conclusions.ipynb) | [`fairness_disparate_impact_audit.png`](../outputs/figures/fairness_disparate_impact_audit.png) | Paired t-test ($p = 0.027$), McNemar test, and Brier calibration; Gender $\text{DIR} = 0.974$, Region $\text{DIR} = 0.941$ (passed ECOA 80% rule). | **PASS** |
| **RQ22** | CRO Metric & End-to-End Retail Architecture | [`08_final_analysis_and_conclusions.ipynb`](../notebooks/data_preparation/08_final_analysis_and_conclusions.ipynb) | [`cro_metric_recommendation.csv`](../outputs/tables/cro_metric_recommendation.csv) | Cost-weighted loss at $\theta^* = 0.18$ reduces expected loss by **$-44.3\%$**; designed 5-stage automated retail lending workflow. | **PASS** |

---

## 3. Detailed Verification Details by Lifecycle Stage

### Stage 1: Data Preparation & Forensics (RQ1 – RQ12)
- **RQ1–RQ3 (Data Understanding & Roles):** Handled in `01_project_overview.ipynb` and `02_data_loading_and_inspection.ipynb`. Full classification of all 41 raw variables into numeric, categorical, ordinal, binary, date, identifier, and target features.
- **RQ4–RQ6 (Data Quality Audit):** Handled in `03_data_quality_audit.ipynb`. Cataloged 21 anomalies, separated structural missingness from random gaps, and detected 100 batch duplicate submissions.
- **RQ7–RQ8 (Cleaning & Sensitivity):** Handled in `04_data_cleaning.ipynb`. Implemented deduplication and median imputation, exporting `loan_cleaned.xlsx` (10,000 rows $\times$ 43 columns). Proved median imputation minimizes distribution shift.
- **RQ9–RQ11 (EDA, Imbalance & Relationships):** Handled in `05_eda_and_skewness.ipynb`. Quantified discriminative power via Cohen's $d$ (`Credit_Score` $d=-1.31$, `Late_Payments_2Y` $d=+1.01$). Identified high default rates among unemployed ($47.6\%$) and prior defaulters ($51.6\%$).
- **RQ12 (Feature Engineering):** Handled in `06_feature_engineering.ipynb`. Created domain financial ratios: exact annuity PMT, Debt-to-Income (DTI), Loan-to-Income (LTI), Loan-to-Value (LTV), Savings-to-Income (STI), and log transforms, generating `loan_featured.xlsx` (56 columns).

### Stage 2: Binary Default Modeling (RQ13 – RQ16, RQ18 – RQ20)
- **RQ13 (Baseline Benchmark):** Handled in `01_logistic_regression_baseline.ipynb`. Revealed the "Accuracy Paradox": a dummy classifier predicting "No Default" achieves $80.2\%$ accuracy with $0.0\%$ recall. Baseline Logistic Regression achieved **$0.812$ ROC-AUC**.
- **RQ14 & RQ18 (7 Algorithm Families & 5-Fold CV):** Handled across `notebooks/loan_default/` and summarized in `11_model_comparison.ipynb`. Evaluated Logistic Regression, Naive Bayes, KNN, SVM, Decision Tree, Random Forest, and LDA/QDA with stratified 5-fold cross-validation. Random Forest achieved highest discriminative power ($0.8641 \pm 0.007$).
- **RQ15 (Partition Experiments):** Handled in `07_preprocessing_and_split_experiments.ipynb`. Evaluated 80/20 vs 70/30 stratified splits; selected 80/20 for superior sample efficiency and lower variance.
- **RQ16 (Hyperparameter Tuning):** Handled in `10_hyperparameter_tuning.ipynb`. Executed systematic `GridSearchCV` across all algorithm families.
- **RQ19 (Learning Curves):** Handled in `11_model_comparison.ipynb`. Evaluated training and validation score convergence from $N = 800$ to $6,400$, diagnosing underfitting vs. overfitting.
- **RQ20 (Interpretability):** Handled in `06_random_forest.ipynb` and `01_logistic_regression_baseline.ipynb`. Extracted permutation importances and standardized logistic coefficients, confirming credit score, debt-to-income, and repayment history as dominant factors.

### Stage 3: Multiclass Risk Rating (RQ17)
- **RQ17 (Multiclass Modeling & 3x3 Diagnostics):** Handled in `notebooks/risk_category/01` through `04`. Evaluated the top 3 paradigms: Multinomial Logistic Regression, Random Forest, and RBF Support Vector Machines. Diagnosed Medium Risk boundary ambiguity ($24.2\%$ misclassified into adjacent Low Risk). Multinomial Logistic Regression with Softmax selected as champion for calibrated probabilities and transparent Relative Risk Ratios (RRRs).

### Stage 4: Governance, Fairness & Lending Strategy (RQ21 – RQ22)
- **RQ21 (Statistical Testing & Fairness Audit):** Handled in `08_final_analysis_and_conclusions.ipynb`. Executed paired t-test ($p=0.027$) and McNemar test. Evaluated demographic parity under Equal Credit Opportunity Act (ECOA) four-fifths rule: Gender Disparate Impact Ratio $= 0.974$, Regional Disparate Impact Ratio $= 0.941$ (both $> 0.80$, passing regulatory compliance).
- **RQ22 (Executive Decision & Retail Pipeline):** Handled in `08_final_analysis_and_conclusions.ipynb`. Established asymmetric banking loss function ($10:1$ default-to-false-alarm ratio). Shifting decision threshold to $\theta^* = 0.18$ reduced expected portfolio loss by **$-44.3\%$**. Designed complete 5-stage retail lending architecture with policy filters, scorecard rating, and automated underwriting triaging.
