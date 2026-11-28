"""Read-only local final checks; GitHub CI and member submissions are external."""
import json
import joblib
import numpy as np
import pandas as pd
from .common import ROOT, config, findings
from .data_checks import verify_hashes
from .tracking import connect, load_registered


def main():
    cfg, form = config(), findings()
    verify_hashes(ROOT, ROOT / cfg["paths"]["raw"], ROOT / cfg["paths"]["hash_manifest"])
    client = connect(cfg["training"])
    name = cfg["training"]["registered_model_name"]
    versions = client.search_model_versions(f"name='{name}'")
    assert {int(version.version) for version in versions} == {1, 2}, "Need v1 and v2"
    champion = client.get_model_version_by_alias(name, "champion")
    assert int(champion.version) == int(form["tracking"]["champion_version"])
    runs = client.search_runs([client.get_experiment_by_name(cfg["training"]["experiment_name"]).experiment_id])
    assert len(runs) >= 4
    for run in runs:
        assert run.data.params.get("data_sha256")
        assert any("mae" in key.lower() for key in run.data.metrics)
    for key, issue in form["data_issues"].items():
        assert isinstance(issue["present"], bool), key
        if issue["present"]:
            assert all(value is not None for value in issue.values()), key

    def no_null(value):
        if isinstance(value, dict):
            return all(no_null(item) for item in value.values())
        return value is not None

    assert no_null(form["tracking"]) and no_null(form["monitoring"])
    from .monitor import promotion_decision
    monitoring = form["monitoring"]
    assert monitoring["decision"] == promotion_decision(
        monitoring["mae_v1_on_check"], monitoring["mae_v2_on_check"], cfg["monitoring"]["required_improvement"])
    assert int(champion.version) == (2 if monitoring["decision"] == "promote" else 1)
    holdout = pd.read_csv(ROOT / "data/raw/holdout_features.csv")
    predictions = pd.read_csv(ROOT / "predictions/holdout_predictions.csv")
    assert predictions.columns.tolist() == ["record_id", "prediction"]
    assert predictions.record_id.tolist() == holdout.record_id.tolist()
    assert len(predictions) == 250 and predictions.record_id.is_unique
    features = cfg["training"]["numeric_features"] + cfg["training"]["categorical_features"]
    registered = load_registered(name, "@champion")
    service = joblib.load(ROOT / "models/model.joblib")
    expected = registered.predict(holdout[features])
    np.testing.assert_allclose(predictions.prediction, expected)
    np.testing.assert_allclose(service.predict(holdout[features]), expected)
    print(json.dumps({"local_checks": "passed", "runs": len(runs), "versions": len(versions),
                      "champion": champion.version, "prediction_rows": len(predictions)}, indent=2))
    print("Still verify on GitHub: green Actions run, public access, member commits and each member's form.")


if __name__ == "__main__":
    main()
