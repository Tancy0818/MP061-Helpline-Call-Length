"""Stage 4: drift, group errors, chronological retraining, and final predictions."""
import hashlib
import json
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from .common import ROOT, config, findings, write_findings, write_json
from .cleaning import load_clean_training
from .retraining import fit_recent_weight, select_recency_weight
from .tracking import connect, load_registered, register_pipeline, export_champion


def population_stability_index(reference, current, bins=10, epsilon=1e-6):
    """Ten equal-width reference bins, shared edges, inclusive outer tails.

    Missing values are excluded and their rates reported separately. Empty-bin
    probabilities are floored then renormalized. Constant references use a
    small finite span. No bin edge is estimated from the later sample.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if not len(reference) or not len(current):
        raise ValueError("PSI needs nonempty finite samples")
    low, high = float(reference.min()), float(reference.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    edges = np.linspace(low, high, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)
    expected = np.maximum(expected, epsilon)
    actual = np.maximum(actual, epsilon)
    expected, actual = expected / expected.sum(), actual / actual.sum()
    score = float(np.sum((actual - expected) * np.log(actual / expected)))
    return score


def promotion_decision(mae_v1, mae_v2, required_improvement=0.12):
    # A zero-error incumbent cannot be strictly improved by 12 percent.
    return "promote" if mae_v1 > 0 and mae_v2 <= (1 - required_improvement) * mae_v1 else "keep"


def chronological_split(later, fraction=0.7):
    ordered = later.assign(date=pd.to_datetime(later.date, errors="raise"))
    ordered = ordered.sort_values(["date", "record_id"], kind="stable")
    boundary = int(len(ordered) * fraction)
    if boundary < 1 or boundary >= len(ordered):
        raise ValueError("Both the retraining and check subsets must be nonempty")
    return ordered.iloc[:boundary].copy(), ordered.iloc[boundary:].copy()


def main():
    cfg = config()
    settings, monitoring = cfg["training"], cfg["monitoring"]
    clean = load_clean_training()
    later = pd.read_csv(ROOT / cfg["paths"]["raw"] / "later.csv")
    holdout = pd.read_csv(ROOT / cfg["paths"]["raw"] / "holdout_features.csv")
    training = json.loads((ROOT / "reports/training.json").read_text())
    client = connect(settings)
    name = settings["registered_model_name"]
    v1 = load_registered(name, "/1")
    features = settings["numeric_features"] + settings["categorical_features"]
    all_psi = {column: population_stability_index(clean[column], later[column],
               monitoring["bins"], monitoring["epsilon"])
               for column in clean.select_dtypes(include="number").columns}
    feature_psi = {column: all_psi[column] for column in settings["numeric_features"]}
    group_mae = {str(team): float(mean_absolute_error(rows.handle_min, v1.predict(rows[features])))
                 for team, rows in later.groupby("team")}
    early, check = chronological_split(later, monitoring["later_train_fraction"])
    # Use only the eight predictors and target; the post-event column never enters X.
    selection = select_recency_weight(clean, early, training["best_candidate"], settings,
                                      monitoring["recency_selection"])
    weight = selection["selected_weight"]
    write_json("reports/retraining_selection.json", selection)
    versions = {int(version.version): version for version in client.search_model_versions(f"name='{name}'")}
    if set(versions) - {1, 2}:
        raise ValueError("Unexpected extra versions; inspect registry history before continuing")
    if 2 in versions:
        run_id = versions[2].run_id
        recorded = client.get_run(run_id).data.params
        if recorded.get("recency_selection_method") != "forward_validation_v1" or float(recorded.get("recent_row_weight", 0)) != weight:
            raise RuntimeError("Existing v2 uses a different recipe. Preserve the database and request a reviewed migration; do not overwrite registry history.")
        v2 = load_registered(name, "/2")
    else:
        sha = hashlib.sha256((ROOT / cfg["paths"]["raw"] / "train.csv").read_bytes()).hexdigest()
        with mlflow.start_run(run_name="v2_chronological_retrain") as run:
            run_id = run.info.run_id
            mlflow.log_params({"data_sha256": sha, "stage": 4, "original_rows": len(clean),
                               "later_fit_rows": len(early), "later_check_rows": len(check),
                               "candidate": training["best_candidate"]["name"],
                               "recent_row_weight": weight,
                               "recency_selection_method": "forward_validation_v1",
                               "promotion_minimum_improvement": monitoring["required_improvement"],
                               **training["best_candidate"]["params"]})
            for result in selection["results"]:
                mlflow.log_metric(f"mae_inner_weight_{result['weight']}", result["mae"])
            v2 = fit_recent_weight(clean, early, training["best_candidate"], settings, weight)
            mlflow.log_metric("mae_check", float(mean_absolute_error(check.handle_min, v2.predict(check[features]))))
            register_pipeline(v2, 2, run_id, clean[features].head(5), settings)
    mae1 = float(mean_absolute_error(check.handle_min, v1.predict(check[features])))
    mae2 = float(mean_absolute_error(check.handle_min, v2.predict(check[features])))
    decision = promotion_decision(mae1, mae2, monitoring["required_improvement"])
    champion_version = 2 if decision == "promote" else 1
    client.log_metric(run_id, "mae_v1_on_check", mae1)
    client.log_metric(run_id, "mae_v2_on_check", mae2)
    client.set_tag(run_id, "promotion_decision", decision)
    client.set_registered_model_alias(name, "champion", str(champion_version))
    champion = export_champion(settings)
    predictions = pd.DataFrame({"record_id": holdout.record_id,
                                "prediction": champion.predict(holdout[features])})
    if not predictions.record_id.is_unique or not np.isfinite(predictions.prediction).all():
        raise ValueError("Invalid holdout predictions")
    destination = ROOT / "predictions/holdout_predictions.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    form = findings()
    form["tracking"]["champion_version"] = champion_version
    form["monitoring"] = {
        "psi": feature_psi, "most_drifted_feature": max(feature_psi, key=feature_psi.get),
        "mae_by_group": group_mae, "worst_served_group": max(group_mae, key=group_mae.get),
        "mae_v1_on_check": mae1, "mae_v2_on_check": mae2, "decision": decision,
    }
    write_findings(form)
    report = {**form["monitoring"], "psi_all_clean_numeric_columns": all_psi,
              "psi_method": "10 equal-width reference bins; infinite outer edges; missing excluded; epsilon renormalized",
              "missing_rates_clean_train": clean[settings["numeric_features"]].isna().mean().to_dict(),
              "later_fit_rows": len(early), "later_check_rows": len(check),
              "later_fit_record_ids": early.record_id.tolist(),
              "later_check_record_ids": check.record_id.tolist(),
              "v2_run_id": run_id, "champion_version": champion_version,
              "selected_recent_weight": weight,
              "selection_report": "reports/retraining_selection.json",
              "relative_mae_improvement": (mae1 - mae2) / mae1 if mae1 > 0 else 0.0,
              "prediction_rows": len(predictions)}
    write_json("reports/monitoring.json", report)
    print(json.dumps({key: value for key, value in report.items()
                      if not key.endswith("record_ids")}, indent=2))


if __name__ == "__main__":
    main()
