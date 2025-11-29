# Nepal Earthquake Building Damage — ML Prototype

### Summary
-------
This repository contains a complete workflow for building a binary classifier that predicts whether a building sustained severe earthquake damage. Work includes data extraction from a local SQLite dataset, preprocessing, one‑hot encoding, model training with XGBoost, hyperparameter tuning (GridSearch / RandomizedSearch), artifact serialization, and a Streamlit dashboard with a single-record UI and a bulk CSV upload page.

### Files of interest
-----------------
- notebook.ipynb — end-to-end exploratory analysis, preprocessing, modeling, hyperparameter search and artifact creation.
- nepal_app.py — Streamlit app (main interactive page + integrated bulk-upload page).
- bulk_upload.py — optional standalone Streamlit page for CSV bulk predictions (if used separately).
- nepal_eq.db — SQLite dataset (id_map, building_structure, building_damage tables).
- nepal_artifacts.pkl — serialized artifacts saved by the notebook (model, encoder, column lists, feature order, threshold).
- requirements.txt — Python dependencies used during development.

### Environment & installation
--------------------------
Recommended: create a new virtual environment on Windows:

1. Create & activate venv
   - PowerShell:
     ```
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - Command Prompt:
     ```
     python -m venv .venv
     .\.venv\Scripts\activate
     ```

2. Install dependencies
   ```
   pip install -r requirements.txt"
   ```

### Running the notebook
--------------------
Open `notebook.ipynb` in JupyterLab / Jupyter Notebook or VS Code and run the cells in order. The notebook:
- Connects to `nepal_eq.db` via sqlite3
- Joins `id_map`, `building_structure`, `building_damage` for a selected district
- Creates a binary target column `severe_damage` (e.g. using damage_grade)
- Encodes categorical features with OneHotEncoder and constructs `X_final`
- Splits data, applies SMOTE, trains XGBoost models (model, model2, model3)
- Saves artifacts to `nepal_artifacts.pkl`

Important artifact note: the notebook currently saves `"threshold": [0.3, 0.4, 0.5, 0.6]` (a list). This causes broadcasting errors when comparing model probabilities to threshold arrays in the Streamlit app. See Troubleshooting / Fixes below.

### Streamlit dashboard
-------------------
Launch the app:
```
cd to the file folder
streamlit run nepal_app.py
```

Main features:
- Single-record prediction UI that uses the same preprocessing pipeline (OneHotEncoder + numeric columns).
- Bulk CSV upload page (integrated function `bulk_upload_page()` or standalone `bulk_upload.py`) that:
  - Uploads a CSV with the same training columns
  - Preprocesses rows, aligns columns to the saved `feature_columns`
  - Produces probability (severe_proba) and binary prediction (severe_pred)
  - Allows downloading the predictions CSV

### How inference works
-------------------
1. Load `nepal_artifacts.pkl` which must include:
   - "model" : trained XGBClassifier
   - "ohe" : fitted OneHotEncoder
   - "cat_cols", "numeric_cols" : lists used at training
   - "feature_columns" : exact training column order
   - "threshold" : preferred scalar threshold (recommended)

2. Preprocess new input:
   - Convert numeric cols to numeric dtype
   - Transform categorical cols using saved `ohe`
   - Concatenate and align columns with `feature_columns` (add missing columns with zeros)
   - Call model.predict_proba(X)[:, 1] and apply threshold to produce binary label

### Troubleshooting
---------------
- ValueError: operands could not be broadcast together with shapes (N,) (4,)
  - Cause: comparing probability array of shape (N,) with a threshold list of shape (4,)
  - Fix: save a single float threshold in artifacts, or update Streamlit to let users pick a single threshold. Example change to notebook before saving artifacts:
    ```
    artifacts['threshold'] = 0.4   # scalar, not list
    ```
  - Or in the app, coerce artifact threshold to scalar or present a selectbox if it's a list.

- Feature mismatch / KeyError on upload:
  - Ensure uploaded CSV contains all `numeric_cols` + `cat_cols` used in training. The app fills missing one-hot columns with zeros but requires the original raw columns present.

### Reproducing artifacts (example snippet)
--------------------------------------
If you want to re-save artifacts with a single threshold, run in the notebook:

artifacts = {
    "model": model3,
    "ohe": ohe,
    "cat_cols": list(cat_cols),
    "numeric_cols": list(numeric_cols),
    "feature_columns": list(X_final.columns),
    "threshold": 0.4   # single float
}
with open('nepal_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

### Best practices & notes
* Always keep the encoder (ohe) and feature order with the model to guarantee consistent preprocessing.
* Choose threshold using validation set metrics (ROC/PR curves) rather than ad-hoc values.
* For production, consider using joblib or a model-serving framework (FastAPI, BentoML) and avoid pickling whole models+encoders in a single file if security or cross-version compatibility is a concern.

### next steps
* To integrate additional districts, modify the notebook SQL WHERE clause or add a UI control to select district IDs.
* For model explainability add SHAP plots to the app.