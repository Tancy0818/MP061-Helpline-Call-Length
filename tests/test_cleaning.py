import pandas as pd
import numpy as np
from helpline.cleaning import clean_training
from helpline.common import config
from helpline.modeling import build_pipeline


def test_repairs_are_separate_and_raw_input_is_unchanged():
    raw = pd.DataFrame({"date": ["2026-03-01", "2026-04-01"],
                        "language": [" hin. ", "English"], "queue_wait_sec": [-999, 120],
                        "issue_count": [2, 3], "handle_min": [100., 10.],
                        "duration_from_phone_bill": [10., 10.]})
    before = raw.copy(deep=True)
    cleaned = clean_training(raw, config()["cleaning"])
    pd.testing.assert_frame_equal(raw, before)
    assert "duration_from_phone_bill" not in cleaned
    assert cleaned.handle_min.tolist() == [10., 10.]
    assert cleaned.language.tolist() == ["hindi", "english"]
    assert pd.isna(cleaned.loc[0, "queue_wait_sec"])
    assert pd.isna(cleaned.loc[0, "issue_count"])
    assert cleaned.loc[1, "issue_count"] == 3


def test_imputation_is_fitted_on_training_only_and_unknown_categories_work():
    settings = config()["training"]
    rows = pd.DataFrame({name: [1., 2., 3., 4.] for name in settings["numeric_features"]})
    rows["language"] = ["hindi"] * 4
    rows["team"] = ["Team Alpha"] * 4
    model = build_pipeline(settings["candidates"][0], settings).fit(rows, [5., 6., 7., 8.])
    numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
    before = numeric.named_steps["impute"].statistics_.copy()
    future = rows.iloc[:1].copy()
    future["queue_wait_sec"] = np.nan
    future["language"] = "unseen"
    assert np.isfinite(model.predict(future)).all()
    np.testing.assert_array_equal(before, numeric.named_steps["impute"].statistics_)
