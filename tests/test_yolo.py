"""Defect boxes with Ultralytics YOLO: validators, recipes/yolo_boxes.py through the runner, package, live outcome.

The synthetic test trains for ~40 s on tiny images; the MVTec one (~7 min on a laptop CPU) only runs with
ASSEMBLY_SLOW_TESTS=1."""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("ultralytics")

from core.config import settings
from core.contracts import KPI, TaskSpec
from ml import package, runner, validators, workspace

ROOT = Path(__file__).resolve().parents[1]
HAS_YOLO = (settings.data_dir / "weights" / "yolo11n.pt").exists()
HAS_MVTEC = (settings.data_dir / "mvtec" / "metal_nut" / "test").is_dir()
RECIPE = (ROOT / "src" / "ml" / "recipes" / "yolo_boxes.py").read_text()

SYNTH_PREPARE = """
import harness, numpy as np
from pathlib import Path
from PIL import Image
rng = np.random.default_rng(1)
out = Path("raw"); out.mkdir(exist_ok=True)
def make(n, start):
    paths, boxes = [], []
    for i in range(n):
        img = (rng.random((64, 64, 3)) * 50).astype("uint8"); b = []
        if rng.random() < 0.5:
            x, y = rng.integers(4, 40, 2); img[y:y + 16, x:x + 16] = 255; b = [[int(x), int(y), int(x) + 16, int(y) + 16, 0]]
        p = out / f"img{start + i}.png"; Image.fromarray(img).save(p); paths.append(str(p)); boxes.append(b)
    return paths, boxes
harness.write_dataset(make(40, 0), make(12, 100), make(12, 200), meta={"split": "random", "classes": ["defect"]})
"""


@pytest.fixture(autouse=True)
def _runs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")


def _spec(budget=60):
    return TaskSpec(id="boxes", name="boxes", rationale="", framing="", input_streams=[], label_sources=[],
                    target_definition="", prediction_format="boxes", primary_metric="map50",
                    metrics=["map50", "image_recall", "image_far"],
                    kpis=[KPI(metric="image_recall", op=">=", threshold=0.0)], split_strategy="random",
                    time_budget_s=budget)


def test_validators_check_box_labels():
    ws = workspace.create_workspace("box-val", _spec())
    assert runner.run_prepare(ws, SYNTH_PREPARE).ok
    hidden = workspace.labels_dir(ws.run_id, ws.task_id) / "val_labels.json"
    labels = json.loads(hidden.read_text())
    sid = next(iter(labels))
    hidden.write_text(json.dumps({**labels, sid: [[10, 10, 5, 20, 0]]}))  # x2 < x1
    report = validators.validate(ws)
    assert not report.ok and not report.checks["format"]
    hidden.write_text(json.dumps({**labels, sid: [[1, 1, 9, 9, 3]]}))  # class 3 of 1
    assert not validators.validate(ws).checks["format"]
    hidden.write_text(json.dumps({k: [] for k in labels}))  # nothing to detect
    assert not validators.validate(ws).checks["label_sanity"]


@pytest.mark.skipif(not HAS_YOLO, reason="yolo11n.pt not downloaded (download_data.py --only yolo)")
def test_yolo_recipe_end_to_end_with_live_outcome():
    from api import deployments

    ws = workspace.create_workspace("box-e2e", _spec())
    assert runner.run_prepare(ws, SYNTH_PREPARE).ok
    result = runner.run_experiment(ws, RECIPE.replace("IMGSZ = 320", "IMGSZ = 64"), {"description": "yolo"})
    assert result.status == "keep", result.log_tail
    assert runner.final_test(ws).test_metrics["map50"] >= 0.0
    pred = package.load_predictor(package.package(ws))
    assert pred.model.kind == "ultralytics"
    tick = deployments._replay_tick("dep", "boxes", pred, 0)
    assert isinstance(tick["boxes"], list) and tick["truth"] is not None
    assert tick["outcome"] in ("caught", "missed", "false_alarm", "ok") and tick["frame_url"].endswith(tick["sample_id"])


@pytest.mark.skipif(not (HAS_YOLO and HAS_MVTEC and os.environ.get("ASSEMBLY_SLOW_TESTS")),
                    reason="slow: set ASSEMBLY_SLOW_TESTS=1 (needs MVTec metal_nut and yolo11n.pt)")
def test_metal_nut_boxes_end_to_end():
    spec = TaskSpec(**json.loads((ROOT / "specs" / "metal_nut_boxes_example.json").read_text()))
    ws = workspace.create_workspace("nut-boxes", spec)
    assert runner.run_prepare(ws, (ROOT / "specs" / "metal_nut_boxes_example_prepare.py").read_text()).ok
    result = runner.run_experiment(ws, RECIPE, {"description": "yolo11n"})
    assert result.status == "keep" and result.all_kpis_passed, result.log_tail
    final = runner.final_test(ws)
    assert final.passed and final.test_metrics["map50"] > 0.5, final.test_metrics
