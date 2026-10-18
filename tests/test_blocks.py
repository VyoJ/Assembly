import numpy as np
import pandas as pd
import pytest

from ml.models.calibrate import calibrate_threshold
from ml.models.features import window_features
from ml.models.splits import internal_split


@pytest.fixture
def sensors():
    # two units, unit 1 rising linearly, unit 2 constant
    return pd.DataFrame({
        "id": [f"train-{i}" for i in range(8)],
        "group": [1, 1, 1, 1, 2, 2, 2, 2],
        "cycle": [1, 2, 3, 4, 1, 2, 3, 4],
        "temp": [10.0, 12.0, 14.0, 16.0, 5.0, 5.0, 5.0, 5.0],
    })


def test_window_features_stay_inside_each_group(sensors):
    f = window_features(sensors, "group", window=2, stats=("mean", "std", "slope", "delta", "last"), cols=["temp"])
    assert list(f.index) == list(sensors.index)
    # unit 2's first row must not see unit 1's last value
    assert f.loc[4, "temp__mean_w2"] == 5.0
    assert f.loc[3, "temp__mean_w2"] == 15.0
    assert f.loc[3, "temp__slope_w2"] == pytest.approx(2.0)
    assert f.loc[6, "temp__slope_w2"] == 0.0
    assert f.loc[3, "temp__delta_w2"] == 2.0
    assert f["temp__std_w2"].notna().all()


def test_window_features_never_use_future_rows(sensors):
    full = window_features(sensors, "group", window=3, cols=["temp"])
    cut = window_features(sensors.iloc[[0, 1, 4, 5]], "group", window=3, cols=["temp"])
    assert np.allclose(full.loc[[0, 1, 4, 5]].to_numpy(), cut.to_numpy())


def test_window_features_default_columns_and_fft(sensors):
    f = window_features(sensors, "group", window=4, stats=("mean",), fft_bands=2)
    assert "cycle__mean_w4" in f and "temp__mean_w4" in f
    assert not any(c.startswith(("id", "group")) for c in f.columns)
    assert "temp__fft0_w4" in f


def test_internal_split_by_group_has_no_overlap():
    X = pd.DataFrame({"group": np.repeat(np.arange(10), 5), "x": np.arange(50)})
    X_tr, y_tr, X_ho, _y_ho = internal_split(X, np.arange(50), frac=0.2)
    assert not set(X_tr["group"]) & set(X_ho["group"])
    assert X_ho["group"].nunique() == 2 and len(X_tr) == len(y_tr)


def test_internal_split_by_time_holds_out_the_end_of_each_group():
    X = pd.DataFrame({"group": np.repeat([1, 2], 10), "t": np.tile(np.arange(10), 2)})
    _, _, X_ho, _ = internal_split(X, np.zeros(20), frac=0.3, by_time=True)
    assert sorted(X_ho["t"].unique()) == [7, 8, 9]


def test_calibrate_respects_false_alarm_budget():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(900), np.ones(100)]
    scores = np.r_[rng.normal(0, 1, 900), rng.normal(2, 1, 100)]
    thr = calibrate_threshold(scores, labels, max_far=0.05)
    alarms = scores >= thr
    far = alarms[labels == 0].mean()
    assert far <= 0.05
    # a slightly lower threshold would break the budget (we used all of it)
    assert (scores[labels == 0] >= np.nextafter(thr, -np.inf)).mean() <= 0.05 + 1 / 900


def test_calibrate_min_recall_and_f1_fallback():
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    labels = np.array([0, 0, 1, 0, 1, 1])
    assert calibrate_threshold(scores, labels, min_recall=1.0) == 0.3
    assert 0.1 < calibrate_threshold(scores, labels) <= 0.9
    with pytest.raises(ValueError):
        calibrate_threshold(scores, np.ones(6))
