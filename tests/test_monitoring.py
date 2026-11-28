import numpy as np
import pandas as pd
import pytest
from helpline.monitor import chronological_split, population_stability_index, promotion_decision


def test_psi_identical_distributions_zero_and_tails_included():
    reference = np.arange(100)
    assert population_stability_index(reference, reference) == pytest.approx(0)
    assert population_stability_index(reference, reference + 1000) > 1
    assert np.isfinite(population_stability_index([2, 2, np.nan], [2, 2]))
    with pytest.raises(ValueError):
        population_stability_index([np.nan], [1])


def test_promotion_threshold_includes_exactly_twelve_percent():
    assert promotion_decision(10, 8.8) == "promote"
    assert promotion_decision(10, 8.8001) == "keep"
    assert promotion_decision(0, 0) == "keep"


def test_chronological_split_has_no_overlap_and_does_not_mutate():
    data = pd.DataFrame({"record_id": [str(i) for i in range(10)],
                         "date": pd.date_range("2026-07-01", periods=10).astype(str)})
    data = data.iloc[::-1].copy()
    before = data.copy(deep=True)
    early, check = chronological_split(data)
    assert len(early) == 7 and len(check) == 3
    assert not set(early.record_id).intersection(check.record_id)
    assert early.date.max() <= check.date.min()
    pd.testing.assert_frame_equal(data, before)
