import importlib.util
import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config import settings

ROOT = Path(__file__).resolve().parents[1]
HAS_CMAPSS = (settings.data_dir / "cmapss" / "train_FD001.txt").exists()
HAS_CWRU = (settings.data_dir / "cwru" / "97.mat").exists()


def _workspace(tmp_path, monkeypatch, spec: dict):
    ws, labels = tmp_path / "ws", tmp_path / "_labels"
    ws.mkdir()
    (ws / "task.json").write_text(json.dumps(spec))
    shutil.copy(ROOT / "src" / "ml" / "harness" / "harness.py", ws / "harness.py")
    monkeypatch.setenv("ASSEMBLY_WORKSPACE", str(ws))
    monkeypatch.setenv("ASSEMBLY_LABELS_DIR", str(labels))
    mod_spec = importlib.util.spec_from_file_location("harness", ws / "harness.py")
    harness = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(harness)
    monkeypatch.setitem(sys.modules, "harness", harness)
    return ws, labels, harness


def _spec(name: str) -> dict:
    return json.loads((ROOT / "specs" / name).read_text())


def _check_hidden(ws: Path, labels: Path, harness, split: str):
    hidden = pd.read_parquet(labels / f"{split}.parquet")
    assert hidden["id"].tolist() == harness.ids(split)
    assert "y" not in pd.read_parquet(ws / "data" / f"{split}_inputs.parquet").columns
    return hidden


@pytest.mark.skipif(not HAS_CMAPSS, reason="C-MAPSS not downloaded")
def test_table_dataset_round_trip(tmp_path, monkeypatch):
    ws, labels, harness = _workspace(tmp_path, monkeypatch, _spec("rul_example.json"))
    runpy.run_path(str(ROOT / "specs" / "rul_example_prepare.py"))
    X, y = harness.load_train()
    assert {"id", "group"} <= set(X.columns) and len(X) == len(y) and y.max() <= 125
    for split in ("val", "test"):
        _check_hidden(ws, labels, harness, split)
        assert not set(harness.groups(split)) & set(X["group"])  # no machine in two splits
    meta = json.loads((ws / "data" / "meta.json").read_text())
    assert meta["input_kind"]["train"] == "table" and sum(meta["sizes"].values()) == 20631


@pytest.mark.skipif(not HAS_CWRU, reason="CWRU not downloaded")
def test_array_dataset_from_raw_signals(tmp_path, monkeypatch):
    ws, labels, harness = _workspace(tmp_path, monkeypatch, _spec("bearing_fault_example.json"))
    runpy.run_path(str(ROOT / "specs" / "bearing_fault_example_prepare.py"))
    X, y = harness.load_train()
    assert isinstance(X, np.ndarray) and X.shape[1:] == (2048,) and len(X) == len(y)
    assert set(y) == {"normal", "inner_race", "ball", "outer_race"}
    Xv = harness.load_val_inputs()
    hidden = _check_hidden(ws, labels, harness, "val")
    assert len(Xv) == len(hidden)
    assert not set(harness.groups("val")) & set(harness.groups("train"))


def test_files_dataset_hides_labels_in_paths(tmp_path, monkeypatch):
    from PIL import Image

    src = tmp_path / "src"
    paths, masks = [], []
    for i, label in enumerate(["good", "scratch", "good", "scratch"]):
        (src / label).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(src / label / f"{i}.png")
        paths.append(str(src / label / f"{i}.png"))
        mask = src / f"{i}_mask.png"
        Image.new("L", (4, 4)).save(mask)
        masks.append(str(mask) if label != "good" else None)

    spec = {"input_streams": [], "label_sources": []}
    ws, labels, harness = _workspace(tmp_path, monkeypatch, spec)
    boxes = [[[0, 0, 2, 2, 1.0, 0]], []]
    harness.write_dataset((paths[:2], masks[:2]), (paths[2:3], boxes[:1]), (paths[3:], boxes[1:]))

    X, y = harness.load_train()
    assert all(Path(p).exists() and "scratch" not in p and "good" not in p for p in X["path"])
    assert pd.isna(y.iloc[0]) and Path(y.iloc[1]).exists()  # good part: no mask
    Xv = harness.load_val_inputs()
    assert Path(Xv["path"].iloc[0]).name == "val-000000.png"
    hidden = pd.read_parquet(labels / "val.parquet")
    assert json.loads(hidden["y"].iloc[0]) == boxes[0]
    assert not list((ws / "data").rglob("*mask*"))


def test_write_dataset_refuses_without_labels_dir(tmp_path, monkeypatch):
    _, _, harness = _workspace(tmp_path, monkeypatch, {"input_streams": [], "label_sources": []})
    monkeypatch.delenv("ASSEMBLY_LABELS_DIR")
    with pytest.raises(RuntimeError):
        harness.write_dataset(([[1.0]], [1]), ([[1.0]], [1]), ([[1.0]], [1]))


def test_streams_outside_the_task_are_blocked(tmp_path, monkeypatch):
    _, _, harness = _workspace(tmp_path, monkeypatch, _spec("rul_example.json"))
    with pytest.raises(KeyError):
        harness.read_stream("hvac.ambient_temp")
