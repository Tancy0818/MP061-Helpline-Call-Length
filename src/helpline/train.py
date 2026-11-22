"""Stage 2: chronological validation, tracked experiments and registered v1."""
import hashlib
import json
import mlflow
from sklearn.metrics import mean_absolute_error
from .common import ROOT, config, findings, write_findings, write_json
from .cleaning import load_clean_training
from .modeling import build_pipeline
from .tracking import connect, register_pipeline, export_champion


def main():
    cfg = config()
    settings = cfg["training"]
    data = load_clean_training().sort_values(["date", "record_id"], kind="stable")
    client = connect(settings)
    name = settings["registered_model_name"]
    versions = list(client.search_model_versions(f"name='{name}'"))
    if versions:
        if not (ROOT / "reports/training.json").exists():
            raise RuntimeError("Registry already exists without its report; inspect this checkout.")
        export_champion(settings)
        print("Existing registry preserved; v1 is not retrained. See reports/training.json.")
        return
    features = settings["numeric_features"] + settings["categorical_features"]
    dates = sorted(data.date.unique())
    cutoff = dates[int(len(dates) * settings["train_fraction"])]
    fit = data[data.date < cutoff]
    validation = data[data.date >= cutoff]
    sha = hashlib.sha256((ROOT / cfg["paths"]["raw"] / "train.csv").read_bytes()).hexdigest()
    scores = []
    baseline = float(mean_absolute_error(validation.handle_min,
                                         [fit.handle_min.median()] * len(validation)))
    for candidate in settings["candidates"]:
        pipeline = build_pipeline(candidate, settings)
        with mlflow.start_run(run_name=f"v1_{candidate['name']}") as run:
            mlflow.log_params({"data_sha256": sha, "candidate": candidate["name"],
                               "model_kind": candidate["kind"], "seed": settings["seed"],
                               "split": "chronological_train_only", "cutoff_date": cutoff,
                               "fit_rows": len(fit), "validation_rows": len(validation),
                               **candidate["params"]})
            pipeline.fit(fit[features], fit.handle_min)
            mae = float(mean_absolute_error(validation.handle_min, pipeline.predict(validation[features])))
            mlflow.log_metrics({"mae_validation": mae, "mae_median_baseline": baseline})
            mlflow.set_tag("stage", "2")
            scores.append({"candidate": candidate, "mae": mae, "run_id": run.info.run_id})
    best = min(scores, key=lambda row: row["mae"])
    final = build_pipeline(best["candidate"], settings)
    final.fit(data[features], data.handle_min)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_param("final_fit_rows", len(data))
        mlflow.set_tag("final_fit", "Refit best setting on all cleaned train; later.csv never used.")
        register_pipeline(final, 1, best["run_id"], data[features].head(5), settings)
    client.set_registered_model_alias(name, "champion", "1")
    export_champion(settings)
    report = {"cutoff_date": cutoff, "fit_rows": len(fit), "validation_rows": len(validation),
              "data_sha256": sha, "baseline_mae": baseline, "runs": scores,
              "best_candidate": best["candidate"], "best_run_id": best["run_id"],
              "validation_mae": best["mae"], "final_fit_rows": len(data)}
    write_json("reports/training.json", report)
    form = findings()
    form["tracking"] = {"experiment_name": settings["experiment_name"],
                        "best_run_id": best["run_id"], "registered_model_name": name,
                        "champion_version": 1}
    write_findings(form)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
