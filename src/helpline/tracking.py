"""SQLite tracking and portable, Git-tracked registry model artifacts."""
import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from .common import ROOT


def connect(settings):
    os.chdir(ROOT)
    # Only load the group's trusted, locally generated cloudpickle models.
    os.environ["MLFLOW_ALLOW_PICKLE_DESERIALIZATION"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(settings["experiment_name"])
    return MlflowClient()


def register_pipeline(model, version, run_id, sample, settings):
    client = MlflowClient()
    name = settings["registered_model_name"]
    names = {item.name for item in client.search_registered_models()}
    if name not in names:
        client.create_registered_model(name)
    existing = list(client.search_model_versions(f"name='{name}'"))
    if len(existing) != version - 1:
        raise ValueError("Unexpected registry history; do not overwrite or renumber versions.")
    model_dir = ROOT / "models" / "registry" / f"v{version}"
    if model_dir.exists():
        raise FileExistsError(f"Partial model output exists: {model_dir}; inspect before retrying.")
    sample = sample.copy()
    numeric = sample.select_dtypes(include="number").columns
    sample[numeric] = sample[numeric].astype(float)
    mlflow.sklearn.save_model(
        model, str(model_dir), serialization_format="cloudpickle",
        signature=infer_signature(sample, model.predict(sample)),
        pip_requirements=["scikit-learn==1.9.0", "numpy==2.4.6", "scipy==1.17.1",
                          "pandas==3.0.5", "cloudpickle==3.1.2"],
    )
    client.log_artifacts(run_id, str(model_dir), artifact_path=f"model_v{version}")
    # Relative source + tracked artifacts keep models:/name@champion portable.
    registered = client.create_model_version(
        name, source=model_dir.relative_to(ROOT).as_posix(), run_id=run_id,
        description=f"MP061 v{version}; relative source resolves from repository root.",
    )
    if int(registered.version) != version:
        raise RuntimeError("Unexpected version number")
    return registered


def load_registered(name, version_or_alias):
    return mlflow.sklearn.load_model(f"models:/{name}{version_or_alias}")


def export_champion(settings):
    model = load_registered(settings["registered_model_name"], "@champion")
    destination = ROOT / "models/model.joblib"
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return model
