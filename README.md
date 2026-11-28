# G247 — MP061: Helpline Call Length (Indore)

SCSE3040 MLOps mini project. Predict `handle_min` and evaluate using mean
absolute error (MAE), measured in minutes.

## Members

| Enrolment number | Name from GROUP.md |
| --- | --- |
| S24CSEU1940 | Aarushichoudhary |
| S24CSEU2555 | Chhavi. |
| S24CSEU0593 | TanishkaSingh |
| S24CSEU1663 | ShwaasWahalA |

## Local setup (PowerShell)

Run commands from the repository root, the inner `G247_MP061` directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\dvc.exe status
```

In VS Code, select `.venv\Scripts\python.exe` as the Python interpreter.
Direct executable paths avoid requiring PowerShell activation-policy changes.
The current requirements cover Stage 4; add pinned libraries as stages need them.
The Stage 3 Docker image must use `python:3.12-slim` per the assignment.

## Data locations and preservation

- `data/*.csv`: untouched extracted originals, ignored by Git.
- `data/raw/train.csv` and `data/raw/later.csv`: byte-identical copies tracked
  by DVC; Git stores only their `.dvc` metadata.
- `data/raw/holdout_features.csv`: byte-identical holdout copy, tracked by Git
  as specified in the handbook layout. `.gitattributes` disables CSV newline conversion.
- `data/processed/`: future cleaning outputs; never overwrite raw data.
- `reports/raw_data_sha256.json`: SHA-256 fingerprints of all three originals.
- `reports/findings.yaml`: working findings form. Root `findings.yaml` remains
  the original template. Only the data section is currently filled.

No DVC remote is configured yet. A clone alone will not contain training/later
data. Each teammate can copy those files from this group's original ZIP into
`data/raw/` and verify their hashes, or use `dvc pull` once shared DVC storage
is configured. Do not share another group's data or copy `.git` folders.

## Stage boundaries and handoffs

1. Stage 1: setup, DVC, seven data checks and data findings; initial member commits.
2. Stage 2: cleaning, v1 training, MLflow registry and tests; second member commits.
3. Stage 3: API, Docker, CI, Kubernetes and logging; third member commits.
4. Stage 4: drift, retraining, promotion, predictions and submission; fourth member commits.

Implementation through Stage 4 is present. Run training in your own checkout
to generate tracking/model outputs. Run monitoring and the final verifier before submission.
Later-stage code must be handed off separately with an exact changed-file list
and test instructions. Each member pulls the latest shared repository, applies,
reviews, tests and understands their changes, then commits and pushes using
their own account and checkout. Review exact files and commit messages before
committing. Never use another member's credentials or falsify commit authorship.

Shared public repository: https://github.com/Tancy0818/MP061-Helpline-Call-Length
Confirm that all three collaborator invitations have been accepted.
Each member must submit the same repository URL separately by November 29,
2026, 11:59 PM, and be able to explain the whole project in the viva.

## Run and understand Stage 1

```powershell
.\.venv\Scripts\python.exe src\helpline\explore.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\dvc.exe status
```

`params.yaml` supplies paths and analysis settings, loaded using `yaml.safe_load`.
`src/helpline/data_checks.py` implements hash verification, daily sensor checks,
upper-tail target screening, and diagnostic language normalization.
`src/helpline/explore.py` combines those checks, prints evidence, saves
`reports/stage1_exploration.txt`, and updates only `data_issues` in
`reports/findings.yaml`. It preserves tracking/monitoring values on later reruns.
No cleaned CSV or model is created in this stage. Temporary diagnostic Series
do not change any source file. SHA-256 verification runs before and after analysis.

### Findings on the original 1,000 training rows

| Issue | Result and evidence |
| --- | --- |
| Leaking column | `duration_from_phone_bill` is known after the call and absent from holdout. Its correlation with the target on the 988 non-typo rows is approximately 0.9990. |
| Unit mixup | Absent: the six predictors have overlapping team ranges and comparable medians; no team exhibits an alternative-unit scale. Invalid wait sentinels are excluded from these diagnostic summaries only. |
| Sentinel value | `queue_wait_sec` contains `-999` in 58 rows. Negative waiting time violates the client's stated rule. |
| Duplicate rows | Absent: zero extra full-row copies and zero duplicate record IDs. The findings count is null because the template requires null details for absent issues. |
| Stuck sensor | All 113 rows from March 1 through March 20, 2026 have `issue_count = 2`. Adjacent days contain multiple values. |
| Target typos | 12 targets form a separated upper cluster from 66 to 241.3; the other 988 are at most 28. Dividing the upper cluster by 10 agrees with the diagnostic phone bill to within 0.21 minutes. |
| Spelling variants | `language` has 18 literal labels representing three real categories. Trimming, lowercasing and expanding `hin.`, `eng.`, `reg.` normalizes 150 rows. |

The phone-bill column is used only to investigate data errors, never as a
prediction feature. The target screening does not assume that all large targets
are mistakes: a separated cluster is checked against the independent phone-bill
record. The checks are tailored to this assignment's evidence, not universal
detectors for all possible datasets.

The sensor check works on complete calendar days. A matching row at the end of
February 28 must not extend the fault interval because other rows that day differ.
Four synthetic tests cover this boundary, missing calendar days, upper-tail gaps,
and non-mutating language normalization. They do not require raw CSV files and
can also be collected by pytest when it is introduced in Stage 2. Stage 3 adds the required API TestClient tests.

See `STAGE1_REVIEW.md` for the exact proposed first-commit files and the next
member's checkout instructions. Git history should record actual completed work;
there is no need to manufacture multiple progress commits for this preparation.

## Stage 2

See [Stage 2 instructions](docs/STAGE2.md) for cleaning, training, registry, and tests.

## Stage 3

See [Stage 3 instructions](docs/STAGE3.md) for the API, Docker, CI, and Kubernetes.

## Stage 4

See [Stage 4 instructions](docs/STAGE4.md) for PSI, retraining, predictions, and submission.
