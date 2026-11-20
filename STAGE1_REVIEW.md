# Stage 1 review and first commit

All paths below are relative to the inner `G247_MP061` repository root.
No commit or push has been made by the assistant. Nothing is staged.
The original CSVs have been preserved and verified by SHA-256.

## Proposed commit

`Complete Stage 1 setup and verified data-quality findings`

Review these exact files before staging:

```text
.dvc/.gitignore
.dvc/config
.dvcignore
.gitattributes
.gitignore
GROUP.md
PROBLEM.md
README.md
STAGE1_REVIEW.md
STUDENT_HANDBOOK.md
data/processed/.gitkeep
data/raw/.gitignore
data/raw/holdout_features.csv
data/raw/later.csv.dvc
data/raw/train.csv.dvc
findings.yaml
params.yaml
reports/findings.yaml
reports/raw_data_sha256.json
reports/stage1_exploration.txt
requirements.txt
src/helpline/__init__.py
src/helpline/data_checks.py
src/helpline/explore.py
tests/test_data_checks.py
```

The assignment documents and root findings template are unchanged references.
The holdout CSV is intentionally Git-trackable as specified by the handbook.
The original CSVs in `data/`, training/later copies in `data/raw/`, `.venv/`,
and `.dvc/cache/` must not be staged. Do not use `git add -f` on those files.
Do not stage later-stage files under this commit.

## Local verification

```powershell
.\.venv\Scripts\python.exe src\helpline\explore.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\dvc.exe status
git status --short --untracked-files=all
```

Review the evidence report and data findings. The four tests should pass;
DVC should say data and pipelines are up to date. The later-stage fields in
findings are intentionally null. This is not a final-submission-ready repository.

After reviewing, stage only the listed files, then inspect
`git diff --cached --stat` and `git diff --cached`. Confirm the commit message
before committing. Use your own Git identity and account. No author overrides,
backdated commits, teammate credentials, or copied `.git` folders are needed.

## GitHub work still required

Create or identify the group's one public repository. If creating it for this
existing local repository, leave GitHub's automatic README/license/gitignore
options unchecked to avoid an unrelated initial history. Add all three teammates
as collaborators and have them accept their invitations. Set the remote only
after confirming the exact repository URL. Commit and push after file review.
Verify the public URL and DVC metadata are visible on GitHub.

No remote URL or member usernames have been supplied to this checkout yet.
The local preparation alone does not finish the shared-repository requirement.

## Stage 2 member handoff

1. Clone the shared repository into your own checkout using your own account.
   If you already cloned it, finish or safely set aside your local work and run
   `git pull --ff-only` on the shared main branch.
2. Create your own `.venv` and install `requirements.txt` using the README commands.
3. Since no DVC remote is configured, copy this group's original `train.csv` and
   `later.csv` from the assignment ZIP to `data/raw/` without opening/saving them
   through Excel. The holdout file is already in Git.
4. Run the exploration command; its saved SHA-256 manifest checks these copies.
   Run `dvc add data/raw/train.csv data/raw/later.csv` to populate your local
   DVC cache. The resulting `.dvc` metadata should match the committed files.
5. Run the four tests and read the evidence before designing cleaning rules.
6. No Stage 2 implementation is included yet. A later Stage 2 handoff must list
   its exact changed files and test commands. Apply only those changes, review,
   test and understand them, then commit and push from your own checkout/account.

All members should describe their actual work and AI assistance accurately and
be prepared to explain every part in the viva.
