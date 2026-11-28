# Data Cleaning & Preprocessing Documentation

## 1. Raw Dataset Overview
- **Source:** Kaggle — LendingClub Accepted Loans (`accepted_2007_to_2018Q4.csv.gz`)
- **Raw Size:** 374.4 MB (gzip-compressed CSV)
- **Raw Dimensions:** **2,260,701 rows × 151 columns** (one row per loan application, spanning 2007–2018 Q4)
- **Task:** Binary classification — predicting whether a loan applicant will default (`target = 1`) or fully repay (`target = 0`) using **only information available at the time of application (Day 0)**.

---

## 2. Transformation Funnel (Raw → Clean Parquet)

| Stage | Action Taken | Rows Remaining | Columns Remaining |
| :--- | :--- | ---: | ---: |
| **0. Raw File** | Loaded schema & metadata from `accepted_2007_to_2018Q4.csv.gz` | 2,260,701 | 151 |
| **1. Leakage & Metadata Removal** | Excluded 41 post-origination leakage columns + 8 ID/constant/free-text columns via `usecols` | 2,260,701 | 102 |
| **2. Target Resolution & Year Window** | Kept only finished loans (`Fully Paid`, `Charged Off`, `Default`) issued in **2015–2016** | 668,651 | 104 *(incl. temp date cols)* |
| **3. Target Encoding & Text Parsing** | Encoded `target`, parsed `term`, `emp_length`, and `earliest_cr_line` → `credit_history_months`; dropped raw strings | 668,651 | 102 |
| **4. Missingness Handling** | Dropped 16 structural `>70%` empty co-borrower columns; added 5 explicit delinquency `_is_missing` flags; preserved remaining `NaN`s for fold-safe pipeline imputation | 668,651 | 91 |
| **5. Stratified Sampling** | Stratified sample by `(issue_year, target)` to 200,000 borrowers | **200,000** | **91** *(89 features + `target` + `issue_year`)* |

**Saved Artifact:** `lending_clean.parquet` (compressed columnar Parquet format preserving exact data types).

---

## 3. Target Variable Definition & Filtering Unresolved Loans

In the raw 2,260,701 rows, `loan_status` contained 10 distinct values:

| Raw `loan_status` | Count | Handling | Reason |
| :--- | ---: | :--- | :--- |
| `Fully Paid` | 1,076,751 | **Keep → `target = 0`** | Confirmed non-default outcome |
| `Charged Off` | 268,559 | **Keep → `target = 1`** | Confirmed default / write-off |
| `Default` | 40 | **Keep → `target = 1`** | Confirmed default |
| `Current` | 878,317 | **Drop** | Active loan at end of 2018; final outcome unknown |
| `Late (31-120 days)` / `Late (16-30 days)` / `In Grace Period` | 34,252 | **Drop** | Unresolved mid-delinquency; outcome not yet finalized |
| Legacy credit-policy statuses & `NaN` | 2,782 | **Drop** | Legacy 2007–2010 exceptions / missing labels |

---

## 4. Data Leakage Prevention & Metadata Removal (49 Columns Dropped)

Real-world lending archives record variables months or years *after* a loan is approved. Including them leaks the target outcome from the future. All 49 columns below were excluded before reading data into memory:

### A. Post-Origination Leakage Columns (41 Dropped)
- **Funding & Loan Status Post-Approval (4):** `funded_amnt`, `funded_amnt_inv`, `pymnt_plan`, `disbursement_method`
- **Repayment & Principal Tracking (7):** `out_prncp`, `out_prncp_inv`, `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`
- **Post-Default Collections & Recoveries (2):** `recoveries`, `collection_recovery_fee`
- **Post-Origination Payment & Credit Pull Updates (6):** `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`, `last_credit_pull_d`, `last_fico_range_high`, `last_fico_range_low`
- **Debt Settlement Fields (7):** `debt_settlement_flag`, `debt_settlement_flag_date`, `settlement_status`, `settlement_date`, `settlement_amount`, `settlement_percentage`, `settlement_term`
- **Hardship Plan Fields (15):** `hardship_flag`, `hardship_type`, `hardship_reason`, `hardship_status`, `deferral_term`, `hardship_amount`, `hardship_start_date`, `hardship_end_date`, `payment_plan_start_date`, `hardship_length`, `hardship_dpd`, `hardship_loan_status`, `orig_projected_additional_accrued_interest`, `hardship_payoff_balance_amount`, `hardship_last_payment_amount`

### B. Non-Predictive Identifiers, Constants & Free Text (8 Dropped)
- **Identifiers / Empty / Constants (4):** `id` (unique key), `member_id` (100% `NaN`), `url` (link), `policy_code` (constant value `1`)
- **Unstructured / Redundant Text (4):** `desc` (phased out prior to 2015–2016), `emp_title` (unstandardized high-cardinality free text), `title` (redundant with structured `purpose`), `zip_code` (masked 3-digit prefix with ~900 sparse categories; geography retained via `addr_state`)

> **Note on Underwriting Features:** Pre-origination features assigned at application time (`grade`, `sub_grade`, `int_rate`, `installment`) were retained as valid Day-0 features, and can be evaluated separately in Feature Ablation studies (Model performance *with* vs. *without* LendingClub's proprietary risk grades).

---

## 5. Feature Parsing (Text & Dates → Numeric)

| Raw Column(s) | Engineered Feature | Transformation |
| :--- | :--- | :--- |
| `term` (`" 36 months"`, `" 60 months"`) | `term_months` | Extracted numeric integer (`36.0`, `60.0`) |
| `emp_length` (`"< 1 year"`, `"10+ years"`, etc.) | `emp_length_years` | Mapped `< 1 year` → `0`, `10+ years` → `10`, extracted digits; preserved `NaN` |
| `issue_d`, `earliest_cr_line` | `credit_history_months` | Computed `(issue_d - earliest_cr_line)` in months at origination |

Raw string columns (`term`, `emp_length`, `earliest_cr_line`, `issue_d`, `issue_dt`) were dropped after extraction.

---

## 6. Cautious Missing-Value Strategy

Missing values in credit data are not random noise; they were categorized into four distinct types rather than blindly dropped or filled:

1. **Informative Missingness — Preserved with Explicit Indicator Flags (5 columns protected):**
   - Columns: `mths_since_last_delinq`, `mths_since_last_record`, `mths_since_last_major_derog`, `mths_since_recent_bc_dlq`, `mths_since_recent_revol_delinq`.
   - **Rationale:** A `NaN` in "months since last delinquency" typically means the borrower has **never** been delinquent. Filling with the column median destroys that signal. These columns were protected from the `>70%` drop threshold and paired with binary `*_is_missing` flags (`1` = missing / clean history, `0` = recorded event).
2. **Structural Missingness — Dropped (16 columns):**
   - Columns with `>70%` missingness in 2015–2016 (`annual_inc_joint`, `dti_joint`, `verification_status_joint`, `revol_bal_joint`, and 12 `sec_app_*` co-borrower metrics) are structurally blank for ~99% of rows because individual applications have no secondary applicant. Retaining `application_type` preserves the Individual vs. Joint distinction without 16 near-empty columns.
3. **Time-Gated Missingness — Flagged for Pipeline Review:**
   - Credit-bureau fields introduced in December 2015 (e.g., `open_acc_6m`, `il_util`, `inq_last_12m`) are largely `NaN` in Jan–Nov 2015 and populated in 2016. During model preprocessing, missingness indicators must **not** be added to these time-gated columns so they do not act as a proxy for `issue_year`.
4. **Remaining Missingness — Deferred to Fold-Safe Pipeline Imputation:**
   - All other `NaN`s (including informative missingness in `emp_length_years` and `mths_since_recent_inq`) were intentionally left un-imputed in `lending_clean.parquet`. Imputation (median/mode + selective missingness indicators) is fitted **strictly inside the training split** via `scikit-learn` `ColumnTransformer` to prevent train-test data leakage.

---

## 7. Cohort Selection & Sampling Justification

### Why 2015–2016 Resolved Loans (668,651 Borrowers)?
- **Avoids legacy sparsity (2007–2014):** Early LendingClub years lacked modern credit-bureau attributes, leaving dozens of features systematically blank.
- **Avoids right-censoring bias (2017–2018):** Because standard loan terms are 36 or 60 months and the dataset ends in 2018 Q4, most 2017–2018 loans were still `Current`. The small fraction that had already resolved by late 2018 is disproportionately skewed toward early defaults.
- **Strong volume & realistic default rates:**
  - **2015:** 375,546 resolved loans (**20.19%** default rate)
  - **2016:** 293,105 resolved loans (**23.29%** default rate)

### Why Stratified Sampling to 200,000 Borrowers?
- **Exceeds project requirements:** **200,000 borrowers** (2× the 100,000 minimum) and **89 clean pre-origination features** (well above the 50-feature minimum).
- **Computational feasibility:** Enables fair, systematic benchmarking of Decision Trees, Random Forests, PCA pipelines, multiple Boosting algorithms, ablation studies, error analysis, and SHAP explainability within hardware/time limits without memory crashes.
- **Exact distribution preservation:** Stratified on the composite key `(issue_year, target)` (`random_state=42`):
  - **2015:** 112,329 rows (56.16%)
  - **2016:** 87,671 rows (43.84%)
  - **Overall Default Rate:** **21.54%** (~43,080 defaults vs. ~156,920 fully paid — ~3.64:1 class imbalance).

---

## 8. Current State & Safeguards Before Modeling
- **Ready Artifact:** `data/lending_clean.parquet` — `(200,000, 91)`
- **Mandatory pre-training rules:**
  1. Drop `issue_year` from feature matrix `X` before fitting models (use it only for temporal splitting or drift analysis).
  2. Fit all imputers, encoders, scalers, and `PCA` strictly on `X_train` inside a `Pipeline`.
  3. Handle the 21.54% class imbalance via **class weights** and **cost-sensitive probability threshold tuning** rather than blind SMOTE oversampling, preserving calibrated default probabilities.
  4. Verify leak-free performance: realistic ROC-AUC on clean LendingClub data is ~0.70–0.74 (anything `>0.90` indicates an undetected leakage column).