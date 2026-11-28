# Stage 4 â€” drift, retraining, final predictions and submission

Pull Stage 3 first. Restore this group's original train/later files to `data/raw/`
if your checkout lacks them. Do not copy any teammate's `.git` or `.venv` folder.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.helpline.monitor
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.helpline.verify_final
```

The script loads version 1 explicitly, even if a previous run promoted v2.
It computes PSI for all numeric columns in the cleaned training frame (the six
features plus the target), and records the six required feature values in the
findings template. The post-event column has already been dropped. The full
numeric PSI and missing-value rates are retained in `reports/monitoring.json`.

PSI uses ten equal-width bins derived from training values, with infinite outer
edges to include later observations outside the training range. Missing values
are excluded; zero shares use epsilon 1e-6 followed by renormalization.
The formula is sum((later_share - train_share) * log(later_share / train_share)).
The P13 notebook was not supplied: compare this documented binning/smoothing
convention with the lab before final submission. If P13 specifies a different
convention, change the PSI function and rerun to regenerate consistent findings.
Do not copy sample PSI values into the form without running your chosen method.

v1's MAE is measured for every team over all 300 later rows. Sorting by date and
then record_id gives a reproducible 210-row retraining subset and a 90-row check
subset. IDs for both subsets are saved in the report. v2 keeps the v1-selected model settings and gives recent rows extra weight.
The recent-row weight is selected from 1, 3, 10, 30 using two forward-validation
folds entirely within the first 210 later rows: fit the first 90 and validate
rows 90–149, then fit the first 150 and validate rows 150–209. All 1,000 original
rows are included in each fit with weight 1. Selection requires at least 10%
better pooled validation MAE than weight 1, improvement in both folds, and no
team's pooled MAE worsening by more than 0.25 minutes; otherwise weight 1 is kept.
`reports/retraining_selection.json` records every candidate and the selected weight.
The final v2 is refitted on all 1,000 cleaned original rows plus the first 210
later rows with those weights. The final 90 rows are not inputs to this selection.
This makes recent evidence more influential without discarding original rows.

Both versions are evaluated on those same 90 rows. v2 becomes champion only
when its MAE is at most 0.88 * v1 MAE (at least a 12% reduction). Both registered
versions remain available even if v1 is kept. Reruns reuse existing v2 when its recorded recipe matches, rather than creating
extra versions. If you already ran an older Stage 4 package, this script stops
instead of silently replacing v2. Preserve your database and ask for a reviewed
migration; do not delete or overwrite committed registry history.

All 250 holdout predictions come from `models:/G247_MP061_Helpline@champion`.
The output columns are exactly `record_id,prediction`, in original holdout order.
`models/model.joblib` is refreshed from that same alias. The final verifier
compares registry, service-model and CSV predictions numerically.

Commit the updated `mlflow.db`, `models/model.joblib`, `reports/findings.yaml`,
new `models/registry/v2/`, `reports/monitoring.json`, `reports/retraining_selection.json`, and
`predictions/holdout_predictions.csv` along with the Stage 4 source/config/docs.
Never overwrite Stage 2's `reports/training.json` or v1 model with new results.
The complete test suite contains 19 tests and uses no assignment CSVs.

## Final external checks

- The public repository opens while signed out.
- All four members have their own genuine contributions/commits.
- The most recent Actions run is green before submitting.
- Rebuild the Docker image if serving the newly promoted champion.
- Every member understands the pipeline and can explain their work honestly.
- EVERY member submits the same repository link using their Bennett account.
- Finish by November 29, 2026, 11:59 PM; do not assume a later push is included.

The local verifier cannot check GitHub Actions, collaborator acceptance, private
submission forms, holdout ground-truth MAE, or the marker's scoring. None of
those external outcomes are claimed by the handoff's local evidence.

Reference: [SAS discussion of PSI](https://support.sas.com/resources/papers/proceedings10/288-2010.pdf).

## Why this revision was adopted

Training-only forward validation reduced MAE from 3.4052 (equal weights) to
1.2827 (recent-row weight 30). The subsequent locked comparison reduced the
90-row check MAE from 2.6873 for the prior v2 to 1.0451 for the revised v2.
Gamma improved from 6.0048 to 1.1513 minutes. Alpha and Beta rose slightly,
from 0.6708/0.9920 to 0.8996/1.1811 respectively; overall error fell substantially.
The check set had already been examined during the earlier pipeline review,
so this is not presented as a new untouched test or a guarantee of hidden
holdout performance. Candidate selection itself uses only the first 210 rows.
