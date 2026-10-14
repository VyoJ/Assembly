"""Our data side + Aryan's runner and scorer, for real: catalog -> prepare.py -> write_dataset ->
train.py (loaders) -> predictions -> scoring vs the hidden labels.

validators.validate is stubbed (not implemented yet) and train.py writes its predictions directly in the
ML_PLAN.md layout, because harness.finish isn't implemented yet either.
"""

import json
from pathlib import Path

import pytest

from core.config import settings
from core.contracts import TaskSpec, ValidationReport
from ml import runner, validators, workspace

try:
    import xgboost  # noqa: F401
except Exception as e:  # noqa: BLE001 - on macOS a missing libomp raises XGBoostError, not ImportError
    pytest.skip(f"xgboost unavailable: {e!r:.80}", allow_module_level=True)
ROOT = Path(__file__).resolve().parents[1]

WRITE_PREDS = """
exp = Path(os.environ["ASSEMBLY_EXP_DIR"])
for split, Xs in (("val", harness.load_val_inputs()), ("test", harness.load_test_inputs())):
    sids = harness.ids(split)
    predictions(sids, Xs).to_parquet(exp / f"{split}_predictions.parquet", index=False)
(exp / "timings.json").write_text(json.dumps({"p50_latency_ms": 0.1, "p95_latency_ms": 0.2}))
"""

TRAIN_RUL = """
import json, os
from pathlib import Path
import pandas as pd, xgboost as xgb, harness
from ml.models.features import window_features

streams = harness.task()["input_streams"]
def feats(X, groups):
    return pd.concat([X[["time"]], window_features(X.assign(group=groups.values), "group", 30, cols=streams)], axis=1)

X, y = harness.load_train()
model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05).fit(feats(X, harness.groups("train")), y)
def predictions(sids, Xs):
    split = sids[0].split("-")[0]
    return pd.DataFrame({"sample_id": sids, "pred": model.predict(feats(Xs, harness.groups(split)))})
""" + WRITE_PREDS

TRAIN_BEARING = """
import json, os
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb, harness

def feats(W):
    spec = np.abs(np.fft.rfft(W - W.mean(1, keepdims=True), axis=1))
    bands = np.stack([b.mean(1) for b in np.array_split(spec, 32, axis=1)], 1)
    return np.c_[np.sqrt((W ** 2).mean(1)), np.abs(W).max(1), np.log1p(bands)]

X, y = harness.load_train()
model = xgb.XGBClassifier(n_estimators=200, max_depth=4).fit(feats(X), y.to_numpy())
def predictions(sids, Xs):
    p = model.predict_proba(feats(Xs))
    return pd.DataFrame({"sample_id": sids, **{f"p_{i}": p[:, i] for i in range(p.shape[1])}})
""" + WRITE_PREDS


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")
    monkeypatch.setattr(validators, "validate", lambda ws: ValidationReport(ok=True, checks={}, data_card="# stub"))


def _run(spec_name: str, train_py: str):
    spec = TaskSpec(**json.loads((ROOT / "specs" / f"{spec_name}.json").read_text()))
    ws = workspace.create_workspace("it", spec)
    report = runner.run_prepare(ws, (ROOT / "specs" / f"{spec_name}_prepare.py").read_text())
    assert report.ok, report
    return runner.run_experiment(ws, train_py, {"description": "integration"})


@pytest.mark.skipif(not (settings.data_dir / "cmapss" / "train_FD001.txt").exists(), reason="C-MAPSS not downloaded")
def test_cmapss_table_scores_end_to_end():
    result = _run("rul_example", TRAIN_RUL)
    assert result.status == "keep", result.log_tail
    assert result.metrics["rmse"] < 18 and result.kpis_passed["rmse <= 18"]


@pytest.mark.skipif(not (settings.data_dir / "cwru" / "97.mat").exists(), reason="CWRU not downloaded")
def test_cwru_raw_signal_scores_end_to_end():
    result = _run("bearing_fault_example", TRAIN_BEARING)
    assert result.status == "keep", result.log_tail
    assert 0.4 < result.metrics["macro_f1"] <= 1.0
