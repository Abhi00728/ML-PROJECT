# LendingClub Loan Default Prediction

The cleaned dataset is ready and pushed directly to the repo at `data/lending_clean.parquet`—nobody needs to download the raw Kaggle file or re-run the cleaning script.

---

## 1. Dataset Snapshot (`data/lending_clean.parquet`)
* **Shape:** `200,000 rows × 91 columns` (89 features + `target` + `issue_year`)
* **Cohort:** Stratified sample of resolved 2015–2016 loans (`2015`: 112,329 | `2016`: 87,671)
* **Target (`target`):** `0` = Fully Paid (78.46%) | `1` = Charged Off / Default (21.54%)
* **Leakage & Cleaning:** Removed 49 post-origination leakage/metadata columns and 16 structural `>70%` empty joint-applicant columns; added 5 explicit delinquency `_is_missing` flags. Full audit is in `docs/DATA_CLEANING.md`.

---

## 2. Quick Local Setup

```bash
# Pull the latest dataset and docs
git pull origin main

# Install required packages (pyarrow is required to read .parquet)
pip install pandas pyarrow scikit-learn xgboost lightgbm catboost shap matplotlib seaborn streamlit joblib
```

Load the dataset in ~1–2 seconds:

```python
import pandas as pd

df = pd.read_parquet("data/lending_clean.parquet")
print(df.shape, "Default rate:", round(df["target"].mean() * 100, 2), "%")
```

---

## 3. Mandatory Rules Before Training Models

1. **Drop `issue_year` from `X`:** Keep it only for train/test splitting or cohort analysis—never feed it as a model feature.
2. **Drop these 14 Dec-2015 schema-change columns** (they are ~95% missing in 2015 and ~0% missing in 2016, which acts as a hidden year proxy):
   ```python
   SCHEMA_SHIFT_COLS = [
       "total_bal_il", "open_il_24m", "open_il_12m", "open_act_il",
       "open_acc_6m", "open_rv_12m", "inq_fi", "max_bal_bc",
       "inq_last_12m", "total_cu_tl", "open_rv_24m", "all_util",
       "mths_since_rcnt_il", "il_util",
   ]
   ```
   *(After dropping these 14 + `issue_year` + `target`, we still have **75 clean features**—well above the 50-feature minimum.)*
3. **Fit preprocessing strictly on `X_train`:** Remaining `NaN`s were intentionally left un-imputed. Fit all imputers, encoders, scalers, and `PCA` inside a `scikit-learn` `Pipeline` on `X_train` only. Add `_is_missing` indicators for `emp_length_years` and `mths_since_recent_inq`.
4. **Class imbalance (~21.5% default):** Prefer `class_weight="balanced"` / `scale_pos_weight` and probability threshold tuning over blind SMOTE so predicted default probabilities stay calibrated.
5. **Leakage sanity check:** Realistic ROC-AUC on clean LendingClub data is **~0.70–0.74**. Anything `>0.90` means a leakage column slipped in.

---

## 4. Repo Structure & Git Etiquette

```text
ML-PROJECT/
├── data/lending_clean.parquet   # Tracked clean dataset (other files & CSVs are ignored)
├── docs/DATA_CLEANING.md        # Full cleaning & leakage audit for report/viva
├── notebooks/                   # EDA & exploration notebooks
├── src/                         # Shared preprocessing, training, and analysis scripts
└── app/                         # Streamlit UI connected to the trained model
```

* **Always pull before working and before pushing:** `git pull origin main`
* **Do not force-add raw CSVs or `.pkl`/`.joblib` model files:** `.gitignore` already blocks everything in `data/` except `data/*.parquet`.
* **Avoid merge conflicts:** Put your experiments in separate files under `src/` or `notebooks/` rather than editing the same file simultaneously.
