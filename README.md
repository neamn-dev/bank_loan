# bank_loan

## Loan Default Risk Classification & Multiclass Risk Rating

An end-to-end, production-grade Machine Learning system for retail credit risk underwriting, combining dual classification tasks with rigorous regulatory compliance and algorithmic fairness audits.

---

## 1. Project Overview

This repository provides an evidence-based Machine Learning investigation for retail lending risk across two distinct prediction tasks:

1. **Task 1 — Loan Default Prediction (Binary Classification):**
   - **Target:** `Loan_Default` (`Yes` vs. `No`)
   - **Baseline Default Rate:** 19.82% defaulters across 10,000 unique applicants.
   - **Champion Model:** **Regularized Logistic Regression (L2)** (Test ROC-AUC: **0.8892**, PR-AUC: **0.7078**, Accuracy: **86.40%**, Generalization Gap: **0.0049**).
   - **Cost-Sensitive Cutoff:** Operating at $\theta^* = 0.18$ increases **Defaulter Recall to 88.38%**, cutting net portfolio write-offs by **44.34%** ($567,500 saved).
   - *Baseline Comparison:* On raw unengineered features, Random Forest scored 0.8652; under the full production feature pipeline, Random Forest overfits (Train AUC 0.9605 vs. Test AUC 0.8659, Gap 0.0945) with inferior probability calibration (Brier Score 0.1070 vs. 0.0969).

2. **Task 2 — Loan Risk Category Prediction (Multiclass Classification):**
   - **Target:** `Risk_Category` (`Low`: 50.10%, `Medium`: 29.90%, `High`: 20.00%)
   - **Leakage Firewall:** Strict exclusion of `Loan_Default` to simulate genuine pre-disbursement underwriting.
   - **Champion Model:** **Multinomial Logistic Regression with Softmax** (Test Accuracy: **80.75%**, Test Macro F1: **0.7880**, 5-Fold CV Macro F1: **0.7832 ± 0.0083**, Log-Loss: **0.4394**).

3. **Task 3 — Continuous Risk-Based Interest Rate Pricing (Regression Benchmark):**
   - **Target:** `Interest_Rate_Pct` (Mean: 8.51%, Std: 1.57%, Range: 4.00% to 15.10%).
   - **Champion Model:** **Ridge Regression ($L_2, \alpha=10.0$)** (Test $R^2$: **0.7129**, Test Adj $R^2$: **0.7047**, Test RMSE: **0.8416%**, 5-Fold CV $R^2$: **0.7121 ± 0.0087**, Generalization Gap: **0.0032**).
   - **Feature Selection:** Lasso ($L_1$) identified an ultra-sparse 6-feature pricing scorecard capturing 96% of full-model predictive power.

---

## 2. Repository Structure

```
bank_loan/
├── data/
│   ├── raw/
│   │   └── loan_dataset.xlsx                         # Read-only immutable raw dataset (10,100 records)
│   └── processed/
│       ├── loan_cleaned.xlsx                         # Deduplicated, audited dataset (10,000 records)
│       └── loan_featured.xlsx                        # 38 engineered features with strict target firewalls
│
├── notebooks/
│   ├── data_preparation/                             # 8 Exploratory & Preparation Notebooks
│   │   ├── 01_project_overview.ipynb                 # Environment & project structure check
│   │   ├── 02_data_loading_and_inspection.ipynb      # Variable inventory & class distributions
│   │   ├── 03_data_quality_audit.ipynb               # 21-problem forensic quality matrix
│   │   ├── 04_data_cleaning.ipynb                    # Reusable cleaning pipeline & deduplication
│   │   ├── 05_eda_and_skewness.ipynb                 # Bivariate analysis & correlation structure
│   │   ├── 06_feature_engineering.ipynb              # PMT annuity formula, DTI, LTV, log transforms
│   │   ├── 07_preprocessing_and_split_experiments.ipynb # ColumnTransformer & KNN scaling benchmark
│   │   └── 08_final_analysis_and_conclusions.ipynb   # Fairness audit (ECOA) & lending decision simulator
│   │
│   ├── loan_default/                                 # 11 Binary Default Modeling Notebooks
│   │   ├── 01_logistic_regression_baseline.ipynb     # Dummy baseline vs. Logistic Regression
│   │   ├── 02_naive_bayes.ipynb                      # Independence assumption & Laplace smoothing
│   │   ├── 03_knn.ipynb                              # Scaling sensitivity & curse of dimensionality
│   │   ├── 04_svm.ipynb                              # Linear vs. RBF kernel (C vs. gamma 2D grid)
│   │   ├── 05_decision_tree.ipynb                    # Gini/Entropy impurity & cost-complexity pruning
│   │   ├── 06_random_forest.ipynb                    # 200-tree ensemble, OOB error, MDI vs. permutation
│   │   ├── 07_discriminant_analysis.ipynb            # LDA vs. QDA & 2D canonical projection
│   │   ├── 08_lasso_ridge_logistic_experiment.ipynb  # L1, L2, Elastic Net regularization paths
│   │   ├── 09_class_imbalance_experiment.ipynb       # Balanced weights, PR curves, business cost curve
│   │   ├── 10_hyperparameter_tuning.ipynb            # Multi-algorithm GridSearchCV optimization
│   │   └── 11_model_comparison.ipynb                 # Master benchmark, learning curves, Wilcoxon tests
│   │
│   └── risk_category/                                # 4 Multiclass Risk Rating Notebooks (Top 3 Models)
│       ├── 01_logistic_regression.ipynb              # Multinomial Softmax logit & Relative Risk Ratios
│       ├── 02_random_forest.ipynb                    # Multiclass tree ensemble & OOB evaluation
│       ├── 03_svm.ipynb                              # Non-linear RBF SVM (One-vs-Rest)
│       └── 04_model_comparison.ipynb                 # 3-Way confusion matrices & Medium-tier diagnosis
│
├── outputs/
│   ├── figures/                                      # 35 high-resolution diagnostic charts (.png)
│   └── tables/                                       # 42 CSV benchmark tables and audit matrices
│
├── requirements.txt                                  # Dependency specifications
└── README.md
```

---

## 3. Key Empirical Results

### Task 1: Binary Loan Default Prediction

#### A. Final Champion Model (Engineered Pipeline — Notebook 11, Section 11)
| Metric | Regularized Logistic Regression (CHAMPION) | Random Forest Ensemble |
| :--- | :---: | :---: |
| **Test ROC-AUC** | **0.8892** | 0.8659 |
| **Test PR-AUC** | **0.7078** | 0.6508 |
| **5-Fold CV ROC-AUC** | **0.8890 ± 0.0056** | 0.8641 ± 0.0070 |
| **Test Accuracy** | **86.40%** | 84.60% |
| **Defaulter Recall ($\theta=0.50$)** | **51.01%** | 50.51% |
| **Defaulter Recall ($\theta^*=0.18$)** | **88.38%** | 88.38% |
| **Test Brier Score (Calibration)** | **0.0969 (Superior)** | 0.1070 |
| **Generalization Gap** | **0.0049 (Well-Fitted)** | 0.0945 (Moderate Overfitting) |
| **Institutional Verdict** | **Production Champion** | Overfitted on complex features |

#### B. Master 7-Algorithm Family Baseline (Pre-Engineered Features)
| Algorithm Family | 5-Fold CV ROC-AUC | Test ROC-AUC | Defaulter Recall ($\theta=0.50$) | Defaulter Recall ($\theta^*=0.18$) | Test Accuracy | Generalization Fit |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest Ensemble** | 0.8641 | 0.8652 | 50.51% | 88.38% | 83.50% | High train variance |
| **Regularized Logistic Regression** | 0.8512 | 0.8524 | 46.72% | 84.34% | 82.75% | Well-Fitted baseline |
| **Support Vector Machine (RBF)** | 0.8498 | 0.8521 | 45.96% | 85.10% | 82.60% | Well-Fitted non-linear |
| **Linear Discriminant Analysis** | 0.8450 | 0.8465 | 45.20% | 83.59% | 82.35% | High stability |
| **Decision Tree (Pruned)** | 0.8315 | 0.8351 | 43.94% | 81.82% | 81.20% | Interpretable rules |
| **K-Nearest Neighbors ($k=21$)** | 0.8210 | 0.8252 | 39.90% | 78.54% | 81.15% | Sensitive to scaling |
| **Gaussian Naive Bayes** | 0.8085 | 0.8118 | 48.99% | 76.77% | 77.80% | Poor probability calibration |
| **Dummy Baseline (Most Frequent)** | 0.5000 | 0.5000 | 0.00% | 0.00% | 80.18% | Zero-rule benchmark |

### Task 2: Multiclass Loan Risk Rating (`Risk_Category`)

| Model Paradigm | Train Accuracy | Test Accuracy | Train Macro F1 | Test Macro F1 | 5-Fold CV Macro F1 | Test Log-Loss | Fit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Multinomial Logistic Regression (CHAMPION)** | **81.21%** | **80.75%** | **0.7946** | **0.7880** | **0.7832 ± 0.0083** | **0.4394** | **Good Fit (Champion)** |
| **Support Vector Machine (RBF Kernel)** | 86.92% | 80.10% | 0.8577 | 0.7808 | 0.7736 ± 0.0090 | 0.4577 | Moderate Fit |
| **Random Forest Classifier** | 98.19% | 77.70% | 0.9819 | 0.7579 | 0.7419 ± 0.0022 | 0.5552 | Overfitting (Moderate) |
| **Dummy Baseline (Zero-Rule)** | 50.12% | 50.15% | 0.2226 | 0.2227 | 0.2222 ± 0.0000 | 17.9678 | Severe Underfitting |

### Task 3: Continuous Lending Regression Benchmark (`Interest_Rate_Pct`)

| Estimator Architecture | Train $R^2$ | Test $R^2$ | Test Adj $R^2$ | 5-Fold CV $R^2$ | Test RMSE | Test MAE | Generalization Gap | Fit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ridge Regression ($L_2, \alpha=10.0$)** | **0.7162** | **0.7129** | **0.7047** | **0.7121 ± 0.0087** | **0.8416%** | **0.6644%** | **0.0032** | **Good Fit (Champion)** |
| **Ordinary Least Squares (OLS)** | 0.7162 | 0.7129 | 0.7046 | 0.7121 ± 0.0088 | 0.8417% | 0.6644% | 0.0033 | Good Fit |
| **Lasso Regression ($L_1, \alpha=0.05$)** | 0.6851 | 0.6840 | 0.6749 | 0.6847 ± 0.0090 | 0.8830% | 0.7054% | 0.0011 | Good Fit (6 Active Features) |

### Task 4: Data Cleaning Sensitivity Analysis (RQ6)

| Cleaning Strategy | Sample Size ($N$) | Portfolio Attrition | Portfolio Default Rate | Income Skewness | Test ROC-AUC | Test PR-AUC | Methodological Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Median Imputation + 99th Capping (Baseline)** | **10,000** | **0.00%** | **19.78%** | **1.2542** | **0.8806** | **0.6910** | **Champion Pipeline (Zero Attrition)** |
| **Mean Imputation + 99th Capping** | 10,000 | 0.00% | 19.78% | 1.2281 | 0.8810 | 0.6916 | Viable; inflates imputed mean by +$5.5k |
| **Median Imputation + Outlier Deletion** | 9,185 | 8.15% | 20.12% | 0.6529 | 0.8917* | 0.7285* | **Rejected:** Severe customer attrition & selection bias |
| **Mean Imputation + Outlier Deletion** | 9,185 | 8.15% | 20.12% | 0.6529 | 0.8919* | 0.7288* | **Rejected:** Severe customer attrition & selection bias |

---

## 4. Algorithmic Fairness & Governance (ECOA)

- **Gender Disparate Impact Ratio (DIR):** **0.9744** (Comfortably exceeds the ECOA/CFPB 0.80 four-fifths threshold).
- **Regional Disparate Impact Ratio (DIR):** Minimum **0.9414** across all 5 geographical zones.
- **Verdict:** Zero discriminatory disparate impact detected.

---

## 5. Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/neamn-dev/bank_loan.git
cd bank_loan

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab / Notebooks
jupyter lab
```
