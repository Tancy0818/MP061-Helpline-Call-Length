"""Choose recency weighting using only the permitted first 70% of later data."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from .modeling import build_pipeline


def fit_recent_weight(clean, recent, candidate, settings, weight):
    if weight <= 0:
        raise ValueError("All training observations must have positive weight")
    features = settings["numeric_features"] + settings["categorical_features"]
    combined = pd.concat([clean, recent], ignore_index=True)
    weights = np.r_[np.ones(len(clean)), np.full(len(recent), weight)]
    model = build_pipeline(candidate, settings)
    model.fit(combined[features], combined.handle_min, regressor__sample_weight=weights)
    return model


def select_recency_weight(clean, early, candidate, settings, search):
    """Forward validation inside early only; the final check set is not an input."""
    early = early.sort_values(["date", "record_id"], kind="stable")
    features = settings["numeric_features"] + settings["categorical_features"]
    if 1 not in search["weights"]:
        raise ValueError("Include weight 1 as the unchanged retraining baseline")
    for fit_end, validation_end in search["folds"]:
        if not 0 < fit_end < validation_end <= len(early):
            raise ValueError("Every fold must stay within the allowed early rows")
    results = []
    for weight in search["weights"]:
        folds, evaluated = [], []
        for fit_end, validation_end in search["folds"]:
            fit, validate = early.iloc[:fit_end], early.iloc[fit_end:validation_end]
            model = fit_recent_weight(clean, fit, candidate, settings, weight)
            predicted = model.predict(validate[features])
            rows = validate[["record_id", "team", "handle_min"]].copy()
            rows["prediction"] = predicted
            evaluated.append(rows)
            folds.append({"fit_recent_rows": fit_end, "validation_rows": len(validate),
                          "validation_ids": validate.record_id.tolist(),
                          "mae": float(mean_absolute_error(validate.handle_min, predicted))})
        pooled = pd.concat(evaluated)
        results.append({"weight": weight, "folds": folds,
                        "mae": float(mean_absolute_error(pooled.handle_min, pooled.prediction)),
                        "mae_by_team": {str(team): float(mean_absolute_error(rows.handle_min, rows.prediction))
                                        for team, rows in pooled.groupby("team")}})
    baseline = next(row for row in results if row["weight"] == 1)
    winner = min(results, key=lambda row: (row["mae"], row["weight"]))
    improves = (
        winner["mae"] <= baseline["mae"] * (1 - search["minimum_validation_gain"])
        and all(a["mae"] < b["mae"] for a, b in zip(winner["folds"], baseline["folds"]))
        and all(winner["mae_by_team"][team] <= score + search["maximum_team_regression"]
                for team, score in baseline["mae_by_team"].items())
    )
    selected = winner["weight"] if improves else 1
    return {"selected_weight": selected, "adoption_gate_passed": bool(improves),
            "results": results, "allowed_recent_rows": len(early),
            "selection_uses_final_check": False,
            "method": "Forward validation within first 210 later rows; original rows retain weight 1"}
