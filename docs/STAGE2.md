# Stage 2 — cleaning, tracked training and v1

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.helpline.train
.\.venv\Scripts\python.exe -m pytest -q
```

Raw data must be restored from this group's assignment ZIP into `data/raw/`.
Do not download another group's data or save originals through Excel.
The script checks their original SHA-256 hashes before using them.

`cleaning.py` writes `data/processed/train_clean.csv` separately. It removes
the post-event phone-bill column, expands language aliases, replaces -999 waits
with missing values, marks issue_count missing during March 1–20, and divides
the 12 confirmed upper-cluster targets by ten. Exact duplicate removal is
defensive; Stage 1 found none. No raw file is changed.

The cleaned frame keeps missing values. Each candidate pipeline learns numeric
medians and missing indicators only from its training portion. Categories are
one-hot encoded. Validation is chronological, with whole days kept together:
798 fitting rows and 202 validation rows on this dataset. Four settings compare
two ridge penalties, histogram boosting, and a random forest. Later data is
not used for fitting, selecting settings, or computing validation scores.

The winning setting is refitted on all 1,000 cleaned training rows, linked to
its original experiment run, registered as version 1, and assigned `champion`.
Every run logs `data_sha256` and an MAE metric in `sqlite:///mlflow.db`.
`reports/training.json` explains the split and all candidate results.

Commit `mlflow.db`, `models/model.joblib`, the entire `models/registry/v1/`
directory, `reports/training.json`, and the updated `reports/findings.yaml`.
The registry source is a relative path under `models/registry/`, so it works
after another member clones the repository and runs commands from its root.
Optional duplicate experiment artifacts in `mlartifacts/` or `mlruns/` stay
ignored. The portable registry files must NOT be ignored.

Rerunning training preserves an existing registry and exports its champion;
it does not create duplicate versions or overwrite the experiment history.
If an interrupted run left partial artifacts, stop and inspect the error rather
than deleting the database or forcing a new registry history.

Tests use synthetic frames, not the raw CSVs. The Stage 1 and Stage 2 suite
contains six tests. API TestClient coverage is added in Stage 3.

Optional tracking UI:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Close the UI before committing the database. Tracked artifacts are this group's
trusted generated models; do not load arbitrary downloaded pickle/joblib files.

Reference: [MLflow scikit-learn persistence](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html).
