# bank_loan

## Loan Default Risk Classification & Multiclass Risk Rating

An end-to-end, production-grade Machine Learning system for retail credit risk underwriting, combining dual classification tasks with rigorous regulatory compliance and algorithmic fairness audits.

---

## 1. Project Overview

This repository provides an evidence-based Machine Learning investigation for retail lending risk across two distinct prediction tasks:

1. **Task 1 — Loan Default Prediction (Binary Classification):**
   - **Target:** `Loan_Default` (`Yes` vs. `No`)
   - **Baseline Default Rate:** 19.82% defaulters across 10,000 unique applicants.
   - **Champion Model:** **Random Forest Classifier** (Test ROC-AUC: **0.8652**).
   - **Cost-Sensitive Cutoff:** $\theta^* = 0.18$ achieves **88.38% Defaulter Recall**, cutting portfolio credit write-offs by **44.34%**.

2. **Task 2 — Loan Risk Category Prediction (Multiclass Classification):**
   - **Target:** `Risk_Category` (`Low`: 50.09%, `Medium`: 29.94%, `High`: 19.97%)
   - **Leakage Firewall:** Strict exclusion of `Loan_Default` to simulate actual loan origination.
   - **Champion Model:** **Multinomial Logistic Regression** (Test Macro F1: **0.7892**, Accuracy: **80.76%**, Log-Loss: **0.4789**).

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
├── reports/
│   ├── final_report.md                               # Complete academic research paper
│   ├── research_questions_mapping.md                 # Complete RQ1–RQ22 & Q1–Q92 evidence mapping
│   └── presentation/
│       └── presentation_slides.md                    # 11-slide university oral defense deck
│
├── requirements.txt                                  # Dependency specifications
└── README.md
```

---

## 3. Key Empirical Results

### Task 1: Binary Loan Default Prediction

| Algorithm Family | 5-Fold CV ROC-AUC | Test ROC-AUC | Defaulter Recall ($\theta=0.50$) | Defaulter Recall ($\theta^*=0.18$) | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (CHAMPION)** | **0.8641** | **0.8652** | **50.51%** | **88.38%** | **83.50%** |
| Regularized Logistic Regression | 0.8512 | 0.8524 | 46.72% | 84.34% | 82.75% |
| Support Vector Machine (RBF) | 0.8498 | 0.8521 | 45.96% | 85.10% | 82.60% |
| Linear Discriminant Analysis | 0.8450 | 0.8465 | 45.20% | 83.59% | 82.35% |
| Decision Tree (Pruned) | 0.8315 | 0.8351 | 43.94% | 81.82% | 81.20% |
| K-Nearest Neighbors ($k=25$) | 0.8210 | 0.8252 | 39.90% | 78.54% | 81.15% |
| Gaussian Naive Bayes | 0.8085 | 0.8118 | 48.99% | 76.77% | 77.80% |
| Dummy Baseline | 0.5000 | 0.5000 | 0.00% | 0.00% | 80.18% |

### Task 2: Multiclass Loan Risk Rating

| Model Paradigm | 5-Fold CV Macro F1 | Test Macro F1 | Test Accuracy | Test Log-Loss |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Logistic Regression (CHAMPION)** | **0.7892** | **0.7915** | **80.76%** | **0.4789** |
| Support Vector Machine (RBF Kernel) | 0.7768 | 0.7794 | 79.78% | 0.5124 |
| Random Forest Classifier | 0.7502 | 0.7538 | 77.20% | 0.5642 |
| Dummy Baseline | 0.2222 | 0.2222 | 50.09% | 1.0986 |

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
