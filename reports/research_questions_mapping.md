# Research Questions & Teacher Requirements Mapping Dossier
## Complete Academic Cross-Reference Matrix for Project Evaluation & Defense

**Course:** Machine Learning | University Project  
**Project:** Loan Default Risk Classification & Multiclass Risk Rating  
**Date:** November 2026  

---

## 1. Overview & Purpose
This document provides a centralized cross-reference mapping of all **22 Master Project Requirements**, **22 Research Questions (RQ1–RQ22)**, and **92 Instructor Questionnaire Items (Q1–Q92)** to their exact implementation locations across the project repository. 

To maintain clean, professional, industry-standard code readability, the notebooks have been streamlined of academic metadata banners. All specific evaluation criteria and questions are cataloged below with direct references to their underlying notebooks, data artifacts, and empirical results.

---

## 2. Master 22 Requirements & Research Questions (RQ1–RQ22)

| Requirement / RQ | Topic & Scope | Primary Notebook Location | Key Empirical Finding / Output |
| :--- | :--- | :--- | :--- |
| **Requirement 1 (RQ1)** | Business understanding, default risk definition, and stakeholder objectives. | `notebooks/data_preparation/01_project_overview.ipynb` | Formulated binary default task ($19.82\%$ default) and multiclass risk rating task ($50.1\%$ Low, $29.9\%$ Med, $20.0\%$ High). |
| **Requirement 2 (RQ2)** | Raw dataset loading, sheet inspection, dimension verification. | `notebooks/data_preparation/02_data_loading_and_inspection.ipynb` | Loaded `loan_dataset.xlsx` (10,100 rows, 41 columns). Output: `outputs/tables/variable_inventory.csv`. |
| **Requirement 3 (RQ3)** | Variable categorization (numeric, categorical, ordinal, binary, identifiers). | `notebooks/data_preparation/02_data_loading_and_inspection.ipynb` | Classified 21 numeric, 11 nominal, 1 ordinal, 4 binary, 1 ID, 1 Date, 2 target variables. |
| **Requirement 4 (RQ4)** | Systematic data quality audit for anomalies, sentinels, and impossible values. | `notebooks/data_preparation/03_data_quality_audit.ipynb` | Cataloged 21 distinct defects. Output: `outputs/tables/data_quality_audit_summary.csv`. |
| **Requirement 5 (RQ5)** | Missing-value analysis, structural blanks vs. stochastic missingness. | `notebooks/data_preparation/03_data_quality_audit.ipynb` | 8,308 missing `Property_Value` (unsecured loans); 7,247 missing `Months_Since_Delinquency` (clean borrowers). |
| **Requirement 6 (RQ6)** | Duplicate record detection and inconsistency identification. | `notebooks/data_preparation/03_data_quality_audit.ipynb` | Identified 100 exact duplicate submissions; removed to yield 10,000 unique records. |
| **Requirement 7 (RQ7)** | Forensic data cleaning decisions and dataset export. | `notebooks/data_preparation/04_data_cleaning.ipynb` | Implemented `clean_loan_data()`; standardized sentinels; exported `data/processed/loan_cleaned.xlsx`. |
| **Requirement 8 (RQ8)** | Cleaning sensitivity analysis (mean vs. median, clipping vs. dropping). | `notebooks/data_preparation/04_data_cleaning.ipynb` | Proved median imputation preserves feature distributions; output: `outputs/tables/cleaning_sensitivity_comparison.csv`. |
| **Requirement 9 (RQ9)** | Exploratory data analysis (EDA), distributions, and summary statistics. | `notebooks/data_preparation/05_eda_and_skewness.ipynb` | Unemployed default rate $= 47.61\%$; prior default rate $= 51.57\%$. Output: `outputs/tables/bivariate_default_rates.csv`. |
| **Requirement 10 (RQ10)** | Target variable distribution and class imbalance analysis. | `notebooks/data_preparation/02_data_loading_and_inspection.ipynb` | Default: $19.82\%$; Risk tiers: Low ($50.09\%$), Medium ($29.94\%$), High ($19.97\%$). |
| **Requirement 11 (RQ11)** | Feature relationships, correlation matrix, Cohen's $d$ effect sizes. | `notebooks/data_preparation/05_eda_and_skewness.ipynb` | `Credit_Score` Cohen's $d = -1.31$; `Late_Payments_2Y` $d = +1.01$. Multicollinearity: `Loan_Amount` $\leftrightarrow$ `Property_Value` $r = 0.94$. |
| **Requirement 12 (RQ12)** | Domain feature engineering (solvency and liquidity ratios). | `notebooks/data_preparation/06_feature_engineering.ipynb` | Engineered exact annuity DTI, LTI, LTV, STI, and log transforms. Output: `data/processed/loan_featured.xlsx`. |
| **Requirement 13 (RQ13)** | Zero-rule baseline benchmarks (`DummyClassifier`). | `notebooks/loan_default/01_logistic_regression_baseline.ipynb` | Quantified the "Accuracy Paradox": Dummy achieves $80.20\%$ accuracy with $0.0\%$ recall on defaulters. |
| **Requirement 14 (RQ14)** | Primary model development across distinct algorithm families. | `notebooks/loan_default/01` to `07` | Evaluated Logistic Regression, Naive Bayes, KNN, SVM, Decision Tree, Random Forest, LDA/QDA. |
| **Requirement 15 (RQ15)** | Model validation and train/test split experiments. | `notebooks/data_preparation/07_preprocessing_and_split_experiments.ipynb` | Benchmarked 80/20 vs. 70/30 stratified splits; selected 80/20 standard preserving exact $19.82\%$ class prevalence. |
| **Requirement 16 (RQ16)** | Hyperparameter tuning and search space exploration. | `notebooks/loan_default/10_hyperparameter_tuning.ipynb` | Executed GridSearchCV across all 6 model families; output: `outputs/tables/hyperparameter_tuning_master_comparison.csv`. |
| **Requirement 17 (RQ17)** | Multiclass model evaluation and 3x3 error diagnosis. | `notebooks/risk_category/01` to `04` | Evaluated top 3 paradigms; identified Medium risk boundary ambiguity ($24.2\%$ overlap with Low). |
| **Requirement 18 (RQ18)** | Stratified 5-fold cross-validation benchmarking (mean $\pm$ std). | `notebooks/loan_default/11_model_comparison.ipynb` | Random Forest achieved 5-fold CV AUC $= 0.8641 \pm 0.007$; Logistic Regression $= 0.8512 \pm 0.008$. |
| **Requirement 19 (RQ19)** | Overfitting diagnosis and learning curves. | `notebooks/loan_default/11_model_comparison.ipynb` | Constructed sample learning curves showing asymptotic convergence at $N = 6,400$. |
| **Requirement 20 (RQ20)** | Feature importance and model interpretability. | `notebooks/loan_default/06_random_forest.ipynb` & `risk_category/02` | Extracted MDI and permutation importance; top predictors: `Credit_Score`, `Late_Payments_2Y`, `Debt_to_Income`. |
| **Requirement 21 (RQ21)** | Master model comparison, reliability curves, statistical tests, fairness. | `notebooks/data_preparation/08_final_analysis_and_conclusions.ipynb` | Paired t-test ($p < 0.05$); McNemar test; Brier calibration; Gender DIR $= 0.974$, Region DIR $= 0.941$. |
| **Requirement 22 (RQ22)** | Final conclusions, CRO metric recommendation, lending pipeline design. | `notebooks/data_preparation/08_final_analysis_and_conclusions.ipynb` | CRO metric: Cost-weighted expected loss at $\theta^* = 0.18$ ($-44.3\%$ loss); designed 5-stage lending architecture. |

---

## 3. Instructor Questionnaire Items Cross-Reference (Q1–Q92)

### Section A: Problem Understanding & Baseline Mechanics (Q1–Q7)
- **Q1 (Supervised vs. Unsupervised):** `notebooks/data_preparation/01_project_overview.ipynb` — Framed as supervised learning with ground-truth historical repayment labels.
- **Q2 (Classification vs. Regression):** `notebooks/data_preparation/01_project_overview.ipynb` — Discrete targets: Binary indicator ($Y \in \{0,1\}$) and 3-tier rating ($Y \in \{\text{Low}, \text{Med}, \text{High}\}$).
- **Q3 (Accuracy Paradox):** `notebooks/loan_default/01_logistic_regression_baseline.ipynb` — Proved Dummy achieves $80.20\%$ accuracy while missing $100\%$ of defaulters.
- **Q4 (Parametric vs. Non-parametric):** `notebooks/loan_default/11_model_comparison.ipynb` — Compared fixed-form parametric models (Logistic Regression, LDA) against flexible non-parametric models (KNN, Decision Tree, Random Forest, SVM).
- **Q5 (Feature Leakage Prevention):** `notebooks/data_preparation/02_data_loading_and_inspection.ipynb` — Documented strict exclusion of post-origination columns (`Applicant_ID`, `Application_Date`) and mutual firewall between `Loan_Default` and `Risk_Category`.
- **Q6 (Bias-Variance Tradeoff):** `notebooks/loan_default/03_knn.ipynb` & `06_random_forest.ipynb` — Evaluated underfitting (high bias at large $k$ or shallow trees) vs. overfitting (high variance at $k=1$ or unpruned trees).
- **Q7 (Multiclass Strategies - OvR vs. OvO):** `notebooks/risk_category/03_svm.ipynb` — Evaluated One-vs-Rest ($K=3$ classifiers) vs. One-vs-One ($\binom{3}{2}=3$ pairwise tournament classifiers).

### Section B: Data Forensics & Quality Remediation (Q8–Q20)
- **Q8 (Missingness Impact):** `notebooks/data_preparation/03_data_quality_audit.ipynb` — Audited scikit-learn estimators failing on unhandled `np.nan`.
- **Q9 (Structural Missingness):** `notebooks/data_preparation/03_data_quality_audit.ipynb` — Proved `Property_Value` missingness strictly aligns with non-mortgage loans ($8,308$ records).
- **Q10 (Hidden Missing Tokens):** `notebooks/data_preparation/03_data_quality_audit.ipynb` — Discovered 614 string `'N/A'` occurrences in numeric `Savings_Balance`.
- **Q11 (Sentinel Value Detection):** `notebooks/data_preparation/03_data_quality_audit.ipynb` — Identified `Credit_Score == 9999` (6 rows) as missingness sentinel.
- **Q12 (Impossible Demographics):** `notebooks/data_cleaning.ipynb` — Identified impossible ages ($3, 118, 212$) and converted to `np.nan`.
- **Q13 (Impossible Signs):** `notebooks/data_cleaning.ipynb` — Remediated negative employment years ($-1.0$).
- **Q14 (Unit Inconsistencies):** `notebooks/data_cleaning.ipynb` — Converted fractional interest rates ($<1.0$) to percentage scale ($\times 100$).
- **Q15 (Valid Overdrafts):** `notebooks/data_cleaning.ipynb` — Retained 135 negative checking balances as authorized overdraft facilities.
- **Q16 (Duplicate Types):** `notebooks/data_quality_audit.ipynb` — Differentiated 100 exact system duplicates from legitimate repeated applicants.
- **Q17 (Cleaning Sensitivity):** `notebooks/data_cleaning.ipynb` — Compared median imputation against mean and outlier removal.
- **Q18–Q20 (Categorical Hygiene & Cardinality):** `notebooks/data_loading_and_inspection.ipynb` — Verified low-cardinality nominal features and consistent case formatting.

### Section C: Feature Engineering & Preprocessing (Q21–Q32)
- **Q21 (One-Hot Encoding & Dummy Trap):** `notebooks/data_preparation/06_feature_engineering.ipynb` — Applied `drop='first'` resulting in 34 clean model features.
- **Q22 (Ordinal Encoding):** `notebooks/data_preparation/07_preprocessing_and_split_experiments.ipynb` — Encoded `Education_Level` with strict monotonic hierarchy.
- **Q23 (Feature Scaling Necessity):** `notebooks/data_preparation/07_preprocessing_and_split_experiments.ipynb` — Demonstrated KNN test AUC surging from $0.5990$ to $0.8075$ upon standard scaling.
- **Q24 (Employment & Default):** `notebooks/data_preparation/05_eda_and_skewness.ipynb` — Quantified Unemployed default rate at $47.61\%$ vs. Retired at $10.41\%$.
- **Q25 (Credit Utilization Cliffs):** `notebooks/data_preparation/05_eda_and_skewness.ipynb` — Documented non-linear risk surge above $70\%$ credit utilization.
- **Q26 (Actuarial Debt-to-Income):** `notebooks/data_preparation/06_feature_engineering.ipynb` — Implemented exact monthly annuity installment PMT formula.
- **Q27 (Right-Skew Transformations):** `notebooks/data_preparation/06_feature_engineering.ipynb` — Evaluated $\ln(1+x)$ transformations on income and loan amount.
- **Q28 (Multicollinearity Diagnosis):** `notebooks/data_preparation/05_eda_and_skewness.ipynb` — Identified `Loan_Amount` $\leftrightarrow$ `Property_Value` collinearity ($r = 0.9415$).
- **Q29 (Cohen's d Effect Sizes):** `notebooks/data_preparation/05_eda_and_skewness.ipynb` — `Credit_Score` ($d = -1.31$) and `Late_Payments_2Y` ($d = +1.01$).
- **Q30 (Prior Default Association):** `notebooks/data_preparation/05_eda_and_skewness.ipynb` — Prior default applicants default at $51.57\%$ vs. $17.38\%$ for first-time applicants.
- **Q31 (Date Feature Extraction):** `notebooks/data_preparation/06_feature_engineering.ipynb` — Extracted application quarter and seasonality indicators.
- **Q32 (Information Leakage in Preprocessing):** `notebooks/data_preparation/07_preprocessing_and_split_experiments.ipynb` — Fitted all imputers and scalers strictly on training folds.

### Section D: Logistic Regression, Regularization & Imbalance (Q33–Q42)
- **Q33 (Balanced Class Weighting):** `notebooks/loan_default/09_class_imbalance_experiment.ipynb` — Proved `class_weight='balanced'` increases defaulter recall from $44.9\%$ to $76.5\%$.
- **Q34 (Class Imbalance Failure Modes):** `notebooks/loan_default/09_class_imbalance_experiment.ipynb` — Analyzed boundary shift towards majority class under empirical priors.
- **Q35 (Sigmoid Function Derivation):** `notebooks/loan_default/01_logistic_regression_baseline.ipynb` — Derived $\sigma(z) = \frac{1}{1 + e^{-z}}$ mapping $\mathbb{R} \to [0, 1]$.
- **Q36 (Log-Odds to Probability Calculation):** `notebooks/loan_default/01_logistic_regression_baseline.ipynb` — Calculated $z = -1.35 \implies \text{Odds} = 0.2592 \implies p = 20.59\%$.
- **Q37 (Odds Ratio Interpretation):** `notebooks/loan_default/01_logistic_regression_baseline.ipynb` — Interpreted $\beta = 0.35 \implies \text{OR} = e^{0.35} = 1.4191$ ($+41.91\%$ default odds per late payment).
- **Q38 (L1 Lasso vs. L2 Ridge Regularization):** `notebooks/loan_default/08_lasso_ridge_logistic_experiment.ipynb` — Evaluated sparsity paths; L1 drove 39 of 44 coefficients to zero at $C=0.01$.
- **Q39 (Threshold Tuning for $\ge 75\%$ Recall):** `notebooks/loan_default/09_class_imbalance_experiment.ipynb` — Derived operating cutoff $\theta = 0.28$ to achieve $75\%$ recall.
- **Q40 (Logistic Regression Assumptions):** `notebooks/loan_default/01_logistic_regression_baseline.ipynb` — Documented linear log-odds, lack of perfect multicollinearity, and independent observations.
- **Q41 (Multinomial Softmax Theory):** `notebooks/risk_category/01_logistic_regression.ipynb` — Formulated $P(y=k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_j e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$ and Relative Risk Ratios.
- **Q42 (Top Logistic Coefficients):** `notebooks/loan_default/01_logistic_regression_baseline.ipynb` — Identified `Credit_Score` ($\beta = -0.748$) and `Debt_to_Income` ($\beta = +0.448$) as top predictors.

### Section E: Algorithm Families & Theory (Q43–Q77)
- **Q43–Q48 (Naive Bayes):** `notebooks/loan_default/02_naive_bayes.ipynb` — Gaussian likelihoods, Laplace smoothing, conditional independence violations, and overconfidence pathology.
- **Q49–Q54 (K-Nearest Neighbors):** `notebooks/loan_default/03_knn.ipynb` — Standardized Euclidean distance, optimal $k=21$, distance weighting, curse of dimensionality experiment.
- **Q55–Q64 (Support Vector Machines):** `notebooks/loan_default/04_svm.ipynb` — Soft margin $C$, hinge loss derivation, RBF kernel computation $K(\mathbf{x},\mathbf{z}) = 0.0821$, Platt scaling, and $O(N^3)$ computational scaling.
- **Q65–Q70 (Decision Trees):** `notebooks/loan_default/05_decision_tree.ipynb` — Hand Gini calculation ($I_G = 0.4200$), entropy, split gain ($+0.1408$), pre-pruning vs. post-pruning `ccp_alpha`.
- **Q71–Q72 (Random Forest Ensemble):** `notebooks/loan_default/06_random_forest.ipynb` — Bagging variance reduction, 20-tree bootstrap disagreement experiment on 50 test applicants, OOB error.
- **Q73–Q77 (Discriminant Analysis - LDA/QDA):** `notebooks/loan_default/07_discriminant_analysis.ipynb` — Gaussian discriminant rules, collinearity matrix singularity fix (`reg_param=0.3`), 2D canonical projection (LD1/LD2), Box's M test.

### Section F: Model Evaluation, Selection & Production (Q78–Q92)
- **Q78 (Manual Metrics Calculation):** `notebooks/loan_default/11_model_comparison.ipynb` — Derived Accuracy ($83.5\%$), Precision ($68.9\%$), Recall ($46.6\%$), Specificity ($94.0\%$), F1 ($0.5561$), MCC ($0.4725$).
- **Q79 (Dummy Baseline Benchmarking):** `notebooks/loan_default/11_model_comparison.ipynb` — Benchmarked all algorithms against zero-rule majority class prediction.
- **Q80 (5-Fold Cross-Validation Summary Table):** `notebooks/loan_default/11_model_comparison.ipynb` — Compiled mean and standard deviation table for all 7 algorithm families.
- **Q81 (Unified ROC & PR Curves):** `notebooks/loan_default/11_model_comparison.ipynb` — Plotted multi-model ROC and Precision-Recall curves.
- **Q82 (PR-AUC vs. ROC-AUC):** `notebooks/loan_default/09_class_imbalance_experiment.ipynb` — Demonstrated PR-AUC superiority under severe class imbalance.
- **Q83 (Cost-Weighted Financial Loss Optimization):** `notebooks/loan_default/09_class_imbalance_experiment.ipynb` — Optimized $10:1$ loss function, identifying $\theta^* = 0.18$ (saving $44.34\%$ in losses).
- **Q84 (Macro vs. Micro vs. Weighted F1):** `notebooks/risk_category/04_model_comparison.ipynb` — Proved Macro F1 prevents minority class performance from being masked by majority class.
- **Q85 (Calibration Curves & Brier Score):** `notebooks/loan_default/11_model_comparison.ipynb` — Evaluated Brier reliability curves (Logistic Regression Brier $= 0.1082$ vs. Naive Bayes $= 0.1841$).
- **Q86 (Learning Curves):** `notebooks/loan_default/11_model_comparison.ipynb` — Analyzed sample size scaling behavior from $N = 800$ to $6,400$.
- **Q87 (Statistical Significance Tests):** `notebooks/loan_default/11_model_comparison.ipynb` — Paired t-test ($t = 3.41, p = 0.027$) and McNemar test confirming Random Forest dominance.
- **Q88 (Hyperparameter Search Comparison):** `notebooks/loan_default/10_hyperparameter_tuning.ipynb` — Documented baseline vs. tuned metrics across all models.
- **Q89 (Chief Risk Officer Metric Recommendation):** `notebooks/data_preparation/08_final_analysis_and_conclusions.ipynb` — Defended Cost-Weighted Financial Loss at $\theta^* = 0.18$.
- **Q90 (Complete 5-Stage Lending Architecture):** `notebooks/data_preparation/08_final_analysis_and_conclusions.ipynb` — Designed production underwriting workflow with policy gates, rating assignment, scoring, and triaging.
- **Q91 (Demographic Fairness Audit):** `notebooks/data_preparation/08_final_analysis_and_conclusions.ipynb` — Verified Gender DIR ($0.974$) and Regional DIR ($0.941$) compliant with ECOA 80% rule.
- **Q92 (Master Comparative Taxonomy):** `notebooks/loan_default/11_model_comparison.ipynb` — Synthesized mathematical assumptions, scaling requirements, interpretability, and production recommendations across all algorithms.
