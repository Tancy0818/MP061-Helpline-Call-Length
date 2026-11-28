import numpy as np
import pandas as pd
import pytest
import helpline.retraining as retraining


SETTINGS = {"numeric_features": ["value"], "categorical_features": []}
SEARCH = {"weights": [1, 3], "folds": [[2, 4], [4, 6]],
          "minimum_validation_gain": 0.1, "maximum_team_regression": 0.25}


def frames():
    clean = pd.DataFrame({"value": [-2, -1], "handle_min": [1., 1.]})
    early = pd.DataFrame({"value": list(range(6)), "handle_min": [3.] * 6,
                          "record_id": list("abcdef"), "team": ["Team Alpha"] * 6,
                          "date": pd.date_range("2026-07-01", periods=6)})
    return clean, early


def test_validation_rows_never_enter_fold_fitting_and_no_improvement_keeps_baseline(monkeypatch):
    calls = []

    class Dummy:
        def fit(self, x, y, **kwargs):
            calls.append((x.value.tolist(), kwargs["regressor__sample_weight"].tolist()))
            return self

        def predict(self, x):
            return np.full(len(x), 3.)

    monkeypatch.setattr(retraining, "build_pipeline", lambda *args: Dummy())
    clean, early = frames()
    before = early.copy(deep=True)
    result = retraining.select_recency_weight(clean, early, {}, SETTINGS, SEARCH)
    assert result["selected_weight"] == 1  # Both weights tie: no evidence to change.
    assert result["selection_uses_final_check"] is False
    assert [entry[0] for entry in calls] == [[-2, -1, 0, 1], [-2, -1, 0, 1, 2, 3]] * 2
    assert calls[2][1] == [1., 1., 3., 3.]
    pd.testing.assert_frame_equal(early, before)


def test_fold_cannot_reach_outside_allowed_training_rows():
    clean, early = frames()
    with pytest.raises(ValueError, match="allowed early rows"):
        retraining.select_recency_weight(clean, early, {}, SETTINGS,
                                         {**SEARCH, "folds": [[4, 7]]})


def test_recency_weight_must_not_remove_original_training_rows():
    clean, early = frames()
    with pytest.raises(ValueError, match="positive weight"):
        retraining.fit_recent_weight(clean, early, {}, SETTINGS, 0)
