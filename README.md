# Water Pump Status Classifier

Predicting the operational status of water pumps across Tanzania — **functional**, **functional needs repair**, or **non-functional** — using machine learning on real-world sensor and survey data.

## Problem

In Tanzania, water pumps are often the only source of clean water for entire communities, but with tens of thousands of pumps spread across the country, manually tracking which ones are failing isn't feasible. This project builds a classifier that flags at-risk pumps from their characteristics (location, age, management, water output, etc.) so maintenance can be proactive instead of reactive.

## Dataset

~59,400 pumps, 40 raw features (GPS coordinates, extraction type, installer, funder, water quantity, region, construction year, etc.) from a public competition dataset (DrivenData's "Pump it Up" challenge).

## Data Cleaning & Feature Engineering

- **Outliers:** IQR clipping on `population` and `amount_tsh`
- **Skew:** log-transformed `population` and `amount_tsh` to approximate normal distributions
- **Missing values:** categorical → `'Unknown'`, numerical → column median
- **Bad zeros:** invalid `0,0` coordinates and zero population replaced with column medians
- **Dropped** redundant/irrelevant columns (e.g. `recorded_by`, `permit`, duplicate grouping columns), leaving 28 features
- **Class imbalance:** `functional needs repair` was underrepresented — balanced with SMOTE
- **New features:** `pump_age` (from construction year), `year_recorded` / `month_recorded`
- **Encoding:** categorical columns via `OrdinalEncoder`

## Model

**CatBoostClassifier** (primary model)
- 1,000 iterations, learning rate 0.1, depth 10, `auto_class_weights='Balanced'`
- Early stopping at iteration 606 (best validation TotalF1 = 0.856)
- Trained on SMOTE-balanced data, evaluated on a 20% held-out test set (19,356 samples)

A secondary variant was trained on 14 features selected via CatBoost's recursive SHAP-based feature selection — it converged slower and slightly underperformed the full-feature model (TotalF1 0.820 vs 0.856), confirming CatBoost handles irrelevant features well on its own.

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Functional | 0.81 | 0.88 | 0.84 |
| Functional needs repair | 0.89 | 0.91 | 0.90 |
| Non-functional | 0.88 | 0.78 | 0.83 |

**Overall accuracy: 86%** | Macro avg precision/recall/F1: 0.86

## Tools

Python, Pandas, NumPy, Scikit-learn, CatBoost, imbalanced-learn (SMOTE), Matplotlib, Seaborn

## Team

Group project — built with Gamal Alaa El Ansary, Ali Ahmed

## Files

- `1_Waterpump_code.ipynb` — full pipeline: EDA, cleaning, feature engineering, model training/evaluation
- `2_waterpump_report.docx` — written report
- `3_submission.csv` — final competition submission
