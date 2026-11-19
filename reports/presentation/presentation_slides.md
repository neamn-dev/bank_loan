# Loan Default Risk Classification & Multiclass Risk Rating
## University Project Presentation & Oral Defense Slide Deck

**Student / Data Science Team:** Loan Default ML Research Team  
**Course:** Machine Learning | University Capstone  
**Target Date:** November 2026  

---

# Slide 1: Title & Project Overview

### Project Title
**Loan Default Risk Classification & Multiclass Risk Rating**  
*An Evidence-Based Machine Learning Investigation for Retail Banking*

### Dual Objectives
1. **Task 1 (Binary Classification):** Predict whether an approved borrower will default (`Loan_Default`: Yes vs. No).
2. **Task 2 (Multiclass Classification):** Predict the applicant's risk tier (`Risk_Category`: Low, Medium, High).

### Methodological Rigor
- **Target Separation Firewall:** Zero leakage between underwriting rating and realized default.
- **Empirical Reproducibility:** 10,000 unique applicants, 100% executable notebooks, strictly no fabricated metrics.

---

# Slide 2: Business Problem & Cost Asymmetry

### The Banking Problem
- Lending represents the primary revenue engine of commercial banks, but bad debt write-offs threaten bank capital solvency.
- **Cost Asymmetry in Credit Risk:**
  - **False Negative (Missed Default):** Loss of principal ($\approx 60\%\text{--}100\%$ of loan amount).
  - **False Positive (False Alarm):** Lost origination fee and margin ($\approx 5\%\text{--}10\%$).
  - **Empirical Ratio:** A False Negative is **$10\times$ more costly** than a False Positive!

### Why Accuracy Fails
- The portfolio default rate is $19.82\%$.
- A naive dummy model predicting "No Default" for everyone achieves **$80.18\%$ accuracy**, but incurs a $100\%$ loss on defaulters!

---

# Slide 3: Forensic Data Quality Audit (21 Defects Resolved)

Our systematic audit identified **21 cataloged data defects** in the raw dataset:
1. **100 Duplicate Rows:** Removed cleanly; preserved 10,000 unique records.
2. **Sentinel Values:** `Credit_Score = 9999` (6 rows) identified and imputed via training median.
3. **Corrupted Tokens:** `'N/A'` string tokens inside numeric `Savings_Balance` (614 rows).
4. **Impossible Demographics:** Age values of 3, 118, 212 filtered out.
5. **Structural Blanks:**
   - 8,308 missing `Property_Value` mapped to unsecured/auto loans via `Is_Home_Loan` flag.
   - 7,247 missing `Months_Since_Last_Delinquency` mapped to clean borrowers via `Ever_Delinquent` flag.
6. **Legitimate Negative Checking:** 135 negative checking balances verified as authorized overdraft facilities and retained.

---

# Slide 4: Feature Engineering & Preprocessing Architecture

### Domain-Specific Financial Ratios Engineered
- **`Debt_to_Income` (DTI):** Derived via exact annuity loan installment formula:
  $$\text{PMT} = \frac{L \cdot r \cdot (1+r)^n}{(1+r)^n - 1}$$
- **`Loan_to_Income` (LTI)** & **`Loan_to_Value` (LTV)**
- **Logarithmic Transforms:** $\ln(1 + \text{Income})$, $\ln(1 + \text{Loan})$, $\ln(1 + \text{Savings})$.

### Leak-Free ColumnTransformer Pipeline
- **Numeric (22 features):** Median Imputation + `StandardScaler`.
- **Nominal (7 features):** Mode Imputation + `OneHotEncoder(drop='first')`.
- **Ordinal (`Education_Level`):** Hierarchical `OrdinalEncoder`.
- **Binary (8 features):** Mode Imputation + `OrdinalEncoder`.
- **Stratified 80/20 Partition:** 8,000 train / 2,000 test preserving exact $19.82\%$ class prevalence.

---

# Slide 5: Task 1 — Binary Model Comparison

| Algorithm Family | 5-Fold CV ROC-AUC | Test ROC-AUC | Defaulter Recall ($\theta=0.50$) | Defaulter Recall ($\theta^*=0.18$) | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (CHAMPION)** | **$0.8641$** | **$0.8652$** | **$50.51\%$** | **$88.38\%$** | **$83.50\%$** |
| Regularized Logistic Regression | $0.8512$ | $0.8524$ | $46.72\%$ | $84.34\%$ | $82.75\%$ |
| Support Vector Machine (RBF) | $0.8498$ | $0.8521$ | $45.96\%$ | $85.10\%$ | $82.60\%$ |
| Linear Discriminant Analysis | $0.8450$ | $0.8465$ | $45.20\%$ | $83.59\%$ | $82.35\%$ |
| Cost-Complexity Pruned Decision Tree | $0.8315$ | $0.8351$ | $43.94\%$ | $81.82\%$ | $81.20\%$ |
| K-Nearest Neighbors ($k=21$) | $0.8210$ | $0.8252$ | $39.90\%$ | $78.54\%$ | $81.15\%$ |
| Gaussian Naive Bayes | $0.8085$ | $0.8118$ | $48.99\%$ | $76.77\%$ | $77.80\%$ |
| Dummy Baseline | $0.5000$ | $0.5000$ | $0.00\%$ | $0.00\%$ | $80.18\%$ |

### Key Insight
- Random Forest is the undisputed champion ($p < 0.05$ under paired t-test).

---

# Slide 6: Task 1 — Cost-Sensitive Optimization

### Defaulter Loss Minimization
- At standard cutoff $\theta = 0.50$, the bank misses **$49.5\%$ of defaulters**.
- Minimizing total expected portfolio cost under $10:1$ loss asymmetry:
  $$\mathcal{C}(\theta) = 10 \cdot \text{FN} + 1 \cdot \text{FP}$$
- Optimal threshold identified at **$\theta^* = 0.18$**:
  - Defaulter Recall increases from **$50.5\%$ to $88.4\%$**.
  - Net portfolio losses reduced by **$44.34\%$**!

---

# Slide 7: Task 2 — Multiclass Risk Rating (Top 3 Models)

We trained the 3 most appropriate and powerful learning paradigms for 3-class risk grading:

| Model Paradigm | 5-Fold CV Macro F1 | Test Macro F1 | Test Accuracy | Test Log-Loss |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Logistic Regression (CHAMPION)** | **$0.7892$** | **$0.7915$** | **$80.76\%$** | **$0.4789$** |
| Support Vector Machine (RBF Kernel) | $0.7768$ | $0.7794$ | $79.78\%$ | $0.5124$ |
| Random Forest Classifier | $0.7502$ | $0.7538$ | $77.20\%$ | $0.5642$ |
| Dummy Baseline (Most Frequent) | $0.2222$ | $0.2222$ | $50.09\%$ | $1.0986$ |

### Why Multinomial Logistic Regression Won Task 2
1. **Peak Discrimination:** Highest Macro F1 ($0.7915$) and lowest Log-Loss ($0.4789$).
2. **Probability Calibration:** Superior calibration compared to tree and kernel methods.
3. **Regulatory Explainability:** Exact Relative Risk Ratios (RRRs) for Basel II / ECOA compliance.

---

# Slide 8: The Boundary Ambiguity of `Medium` Risk

### Error Analysis Across Rating Categories
- **Low Risk:** $91.4\%$ Recall (pristine credit history, high liquidity).
- **High Risk:** $81.2\%$ Recall (clear distress signals: late payments, high DTI).
- **Medium Risk:** **$64.8\%$ Recall** ($24.2\%$ misclassified as Low; $11.0\%$ misclassified as High).

### Root Cause
Medium risk borrowers occupy a continuous, overlapping spectrum (e.g. good credit score but elevated debt ratio, or clean history but short employment). They represent genuine borderline credit decisions that require human underwriting escalation!

---

# Slide 9: Demographic Fairness & Bias Audit (Q91)

We conducted an empirical bias audit on the test set under the CFPB / ECOA four-fifths rule:

| Protected Attribute | Sub-Group | Approval Rate | Defaulter Recall | Disparate Impact Ratio (DIR) | Regulatory Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Gender** | Male (Privileged) | $62.4\%$ | $88.1\%$ | $1.000$ | Baseline |
| **Gender** | Female (Unprivileged) | $60.8\%$ | $88.7\%$ | **$0.9744$** | **PASS ($\ge 0.80$)** |
| **Region** | North | $63.1\%$ | $88.5\%$ | $1.000$ | Baseline |
| **Region** | South | $59.4\%$ | $87.9\%$ | **$0.9414$** | **PASS ($\ge 0.80$)** |
| **Region** | East | $61.5\%$ | $88.6\%$ | **$0.9746$** | **PASS ($\ge 0.80$)** |
| **Region** | West | $62.7\%$ | $88.4\%$ | **$0.9937$** | **PASS ($\ge 0.80$)** |

**Verdict:** Zero unlawful disparate impact detected across both gender and geography.

---

# Slide 10: Executive Recommendations & 5-Stage Lending Pipeline

### Recommendation to CRO (Q89)
Adopt **Cost-Weighted Expected Loss at $\theta^* = 0.18$**. It directly ties model predictions to dollar P&L outcomes, saving $44.3\%$ in write-offs.

### End-to-End Decision Architecture (Q90)
1. **Stage 1: Hard Policy Gate** (KYC, age $\in [18, 75]$, employment).
2. **Stage 2: Multiclass Rating Tier** (Multinomial Logistic Model).
3. **Stage 3: Default Risk Probability** (Random Forest Model, $p = P(Y=1)$).
4. **Stage 4: Automated Triaging:**
   - **Green ($p < 0.10$):** Instant auto-approval ($43.2\%$ of applicants).
   - **Amber ($0.10 \le p < 0.25$ or Medium Risk):** Senior underwriter review ($21.6\%$).
   - **Red ($p \ge 0.25$):** Automated adverse action with key reason codes ($16.8\%$).
5. **Stage 5: Risk-Based Pricing:** $\text{APR} = \text{Base} + \frac{p \cdot \text{LGD}}{1 - p} + \text{Margin}$.

---

# Slide 11: Production Artifacts & Defensibility

### Project Deliverables Summary:
- **Champion Production Pipelines:**
  - **Binary Default Champion:** Random Forest Pipeline (Test ROC-AUC: 0.8652, Cost-Optimal Threshold $\theta^* = 0.18$)
  - **Multiclass Risk Champion:** Multinomial Logistic Regression Pipeline (Test Macro F1: 0.7892, Accuracy: 80.76%)
- **Data Assets:**
  - `data/processed/loan_cleaned.xlsx` (10,000 unique records)
  - `data/processed/loan_featured.xlsx` (56 features)
- **Executable Notebook Suite:**
  - Core exploratory & preprocessing series: `notebooks/data_preparation/01` to `07`
  - Complete Binary Default series: `notebooks/loan_default/01` to `11`
  - Curated Multiclass Risk series: `notebooks/risk_category/01` to `04`
  - Synthesis, Fairness & Governance: `notebooks/data_preparation/08`
- **Comprehensive Academic Report:** `reports/final_report.md`

**Thank you! Ready for Questions and Defense.**
