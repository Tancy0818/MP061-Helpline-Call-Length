# Stage 3 — service, container, CI and Kubernetes

Pull the completed Stage 2 commit first. Tests don't need raw data. Running the
real API requires the committed `models/model.joblib`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Keep that terminal open. In a second terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$body = '{"queue_wait_sec":120,"agent_exp_months":24,"previous_calls":3,"issue_count":2,"hour":14,"caller_age":35,"language":"hindi","team":"Team Alpha"}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body
```

The prediction is returned as `{"prediction": <minutes>}`. Stop the server
with Ctrl+C when finished. The schema forbids extra fields; the exact eight
prediction-time fields are required. `previous_calls > 20` and negative waiting
times fail validation with HTTP 422 before the model is called. Exact boundary
values of 20 previous calls and zero wait are allowed. Tests cover those rules,
missing/extra fields, predictions, health, and JSON logging without customer data.
The Stage 1–3 test suite contains 13 tests.

Prediction logs use Python logging, a JSON formatter and stdout. They contain
an event, request ID, status and latency, never the request body or caller age.

## Docker

Start the installed Docker Desktop application. `docker version` must show a
Server section. No Docker installation is needed if it is already present.

```powershell
docker build -t mp061-api:1.0 .
docker run --rm --name mp061-local -p 8000:8000 mp061-api:1.0
```

Repeat the two HTTP requests from another terminal. Stop the container with
Ctrl+C. If a local uvicorn process uses port 8000, stop it first.
The image uses Python 3.12-slim, installs dependencies before copying code,
exposes 8000, and starts uvicorn on 0.0.0.0:8000. No datasets enter the image.
Rebuild this same image after Stage 4 changes the champion model.

## CI and Kubernetes

The workflow begins with the literal key `on:` and runs pytest on every push
and pull request using Python 3.12. After pushing, open the Actions tab and
confirm the actual run is green. A local passing test run is not a substitute
for that GitHub check. If CI fails, inspect its logs, fix the relevant files,
rerun locally, and commit/push the fix from your own account.

The Deployment is `mp061-api`, the Service is `mp061-svc`, the image is
`mp061-api:1.0`, and there are two replicas. Service port 80 forwards to 8000.
Both probes use `/health`; readiness starts after 5 seconds and liveness after
30 seconds. Labels and selectors match. No cluster is required by the handbook.
If deploying locally, load/build the image in your cluster's image store before
applying the files; the local image is not published to a registry by this task.
