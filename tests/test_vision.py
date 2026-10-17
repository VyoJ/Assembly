"""Vision path: ml.models.vision + recipes/image_anomaly.py through the real runner, final test and package.

The synthetic test needs no downloads (random-init backbone); the MVTec one runs the reference prepare.py and the
recipe exactly as the agent would read them."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from core.config import settings
from core.contracts import KPI, TaskSpec
from ml import package, runner, workspace
from ml.models.vision import ImageEmbeddingClassifier, PatchKNNAnomaly

ROOT = Path(__file__).resolve().parents[1]
HAS_MVTEC = (settings.data_dir / "mvtec" / "metal_nut" / "test").is_dir()
HAS_WEIGHTS = (settings.data_dir / "weights" / "resnet18.pth").exists()


@pytest.fixture(autouse=True)
def _runs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")


def _parts(n, seed, defect_rate=0.3):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 48, 48, 3)) * 60).astype(np.uint8)
    y = (rng.random(n) < defect_rate).astype(int)
    for i in np.flatnonzero(y):
        r, c = rng.integers(4, 30, 2)
        X[i, r:r + 12, c:c + 12] = 255  # bright scratch
    return X, y


def test_models_take_paths_and_decoded_batches_alike(tmp_path):
    from PIL import Image

    X, y = _parts(40, 0)
    paths = []
    for i, img in enumerate(X):
        paths.append(str(tmp_path / f"{i}.png"))
        Image.fromarray(img).save(paths[-1])
    import pandas as pd

    df = pd.DataFrame({"path": paths})
    m = PatchKNNAnomaly(image_size=64, memory_size=2000, random_state=0).fit(df, y)
    s_df, s_arr = m.predict_proba(df)[:, 1], m.predict_proba(X.astype(np.float32))[:, 1]
    assert np.allclose(s_df, s_arr, atol=1e-5) and ((s_df >= 0) & (s_df <= 1)).all()
    assert s_df[y == 1].mean() > s_df[y == 0].mean()  # even a random-init CNN sees a bright square
    gray = X[..., 0]  # (N, H, W) works too
    assert m.predict_proba(gray[:3]).shape == (3, 2)
    c = ImageEmbeddingClassifier(image_size=64).fit(df, y)
    assert c.predict_proba(X[:5]).shape == (5, 2)


SYNTH_PREPARE = """
import harness, numpy as np
from pathlib import Path
from PIL import Image
rng = np.random.default_rng(3)
out = Path("raw_images"); out.mkdir(exist_ok=True)
def make(n, start):
    paths, ys = [], []
    for i in range(n):
        img = (rng.random((48, 48, 3)) * 60).astype("uint8")
        defect = int(rng.random() < 0.35)
        if defect:
            r, c = rng.integers(4, 30, 2); img[r:r + 12, c:c + 12] = 255
        p = out / f"part{start + i}.png"; Image.fromarray(img).save(p)
        paths.append(str(p)); ys.append(defect)
    return paths, ys
harness.write_dataset(make(80, 0), make(30, 1000), make(30, 2000), meta={"split": "random"})
"""


def _synthetic_package():
    spec = TaskSpec(id="synth_defect", name="synthetic defects", rationale="", framing="", input_streams=[],
                    label_sources=[], target_definition="", prediction_format="binary_score",
                    primary_metric="auroc", metrics=["auroc", "recall", "false_alarm_rate", "p95_latency_ms"],
                    kpis=[KPI(metric="auroc", op=">=", threshold=0.7)], split_strategy="random", time_budget_s=120)
    ws = workspace.create_workspace("synth-vision", spec)
    assert runner.run_prepare(ws, SYNTH_PREPARE).ok
    recipe = (ROOT / "src" / "ml" / "recipes" / "image_anomaly.py").read_text()
    # No downloads in CI: same recipe with a random-init backbone at a small image size.
    recipe = recipe.replace('harness.weights("resnet18.pth")', "None").replace(
        "memory_size=10000)", "memory_size=3000, image_size=64)")
    result = runner.run_experiment(ws, recipe, {"description": "recipe, random-init backbone"})
    assert result.status == "keep", result.log_tail
    assert runner.final_test(ws).test_metrics["auroc"] >= 0.7
    return spec, package.package(ws)


def test_synthetic_image_task_end_to_end_with_the_recipe():
    _, pkg = _synthetic_package()
    pred = package.load_predictor(pkg)
    path = pred.replay(0)["path"]
    out = pred.predict({"image_path": path})
    assert 0 <= out["score"] <= 1 and isinstance(out["alarm"], bool)
    assert {pred.truth(pred.replay(i)["sample_id"]) for i in range(pred.n_replay)} == {0, 1}


async def test_live_camera_replays_held_out_images():
    from fastapi.testclient import TestClient

    from api import deployments
    from api.main import app
    from core import registry
    from core.contracts import Deployment

    spec, pkg = _synthetic_package()
    pred = package.load_predictor(pkg)
    dep = Deployment(id="dep_cam", run_id="r", task_ids=[spec.id], exp_ids=["exp_000"],
                     package_paths={spec.id: pkg}, endpoint="/api/deployments/dep_cam")
    registry.deployments["dep_cam"] = registry.LiveDeployment(dep, {spec.id: spec}, {spec.id: pred})
    try:
        response = await deployments.live_predictions("dep_cam", hz=50)
        events = []
        async for event in response.body_iterator:
            events.append(json.loads(event["data"])["predictions"][spec.id])
            if len(events) == 3:
                break
        sids = [e["sample_id"] for e in events]
        assert sids == [pred.replay(i)["sample_id"] for i in range(3)]  # held-out test images, in order
        for e in events:
            assert e["outcome"] in ("caught", "missed", "false_alarm", "ok") and e["truth"] in (0, 1)
            assert e["frame_url"] == f"/api/deployments/dep_cam/frames/{spec.id}/{e['sample_id']}"

        client = TestClient(app)
        r = client.get(events[0]["frame_url"])
        assert r.status_code == 200 and r.content == Path(pred.replay_paths()[sids[0]]).read_bytes()
        assert client.get(f"/api/deployments/dep_cam/frames/{spec.id}/..%2F..%2Fetc%2Fpasswd").status_code == 404
        assert client.get(f"/api/deployments/dep_cam/frames/{spec.id}/train-000000").status_code == 404
    finally:
        registry.deployments.pop("dep_cam", None)


@pytest.mark.skipif(not (HAS_MVTEC and HAS_WEIGHTS), reason="MVTec metal_nut or resnet18 weights not downloaded")
def test_metal_nut_end_to_end():
    spec = TaskSpec(**json.loads((ROOT / "specs" / "metal_nut_example.json").read_text()))
    ws = workspace.create_workspace("metal-nut", spec)
    report = runner.run_prepare(ws, (ROOT / "specs" / "metal_nut_example_prepare.py").read_text())
    assert report.ok, report.messages
    result = runner.run_experiment(ws, (ROOT / "src" / "ml" / "recipes" / "image_anomaly.py").read_text(),
                                   {"description": "patchcore-lite"})
    assert result.status == "keep" and result.all_kpis_passed and result.metrics["auroc"] > 0.95, result.log_tail
    final = runner.final_test(ws)
    assert final.passed and final.test_metrics["auroc"] > 0.95, final.test_metrics

    pkg = Path(package.package(ws))
    assert sum(f.stat().st_size for f in (pkg / "model").iterdir()) < 40e6  # backbone trimmed to layer3
    pred = package.load_predictor(str(pkg))
    sample = pred.replay(1)
    live = pred.predict({"image_path": sample["path"]})  # the model really runs on the image
    stored = pred.predict({"replay": 1})
    assert abs(live["score"] - stored["score"]) < 1e-3 and live["alarm"] == stored["alarm"]


def test_image_task_runs_in_a_remote_box(tmp_path, monkeypatch):
    """Images reach the box with data/ (relative paths), the vision code with src/ml, and the run comes back."""
    import os
    import sys

    from test_executors import DirExecutor

    from ml.executors import base

    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    fake = DirExecutor(tmp_path / "remote")
    monkeypatch.setattr(settings, "executor", "fake")
    monkeypatch.setitem(base._instances, "fake", fake)

    spec = TaskSpec(id="cam", name="cam", rationale="", framing="", input_streams=[], label_sources=[],
                    target_definition="", prediction_format="binary_score", primary_metric="auroc",
                    metrics=["auroc"], kpis=[KPI(metric="auroc", op=">=", threshold=0.7)], split_strategy="random",
                    time_budget_s=120)
    ws = workspace.create_workspace("remote-cam", spec)
    assert runner.run_prepare(ws, SYNTH_PREPARE).ok
    ws = workspace.freeze(ws)
    train = """
import harness
from ml.models.vision import PatchKNNAnomaly
X, y = harness.load_train()
assert all(p.startswith(str(harness.DATA_DIR)) for p in X["path"]), "paths must resolve inside the box"
model = PatchKNNAnomaly(image_size=64, memory_size=2000).fit(X, y)
harness.save_model(model)
harness.finish(lambda X: model.predict_proba(X)[:, 1], threshold=model.threshold_for_far(0.05))
"""
    result = runner.run_experiment(ws, train, {"description": "remote patchknn"})
    assert result.status == "keep", result.log_tail
    assert "[fake] running exp_000/train.py remotely" in (Path(ws.path) / "exp_000" / "run.log").read_text()
    assert (Path(fake.remote_root) / "runs" / "remote-cam" / "cam" / "data" / "files" / "train").is_dir()
    runner.release(ws)
