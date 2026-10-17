"""harness.py data side against the ML_PLAN.md layout, in a throwaway workspace (no runner needed)."""

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


def _no_labels_in_workspace(ws: Path, split: str):
    assert not list((ws / "data").glob(f"{split}_labels*"))
    assert "y" not in pd.read_parquet(ws / "data" / f"{split}_inputs.parquet").columns


@pytest.mark.skipif(not HAS_CMAPSS, reason="C-MAPSS not downloaded")
def test_table_dataset_matches_the_seam(tmp_path, monkeypatch):
    ws, labels, harness = _workspace(tmp_path, monkeypatch, _spec("rul_example.json"))
    runpy.run_path(str(ROOT / "specs" / "rul_example_prepare.py"))
    X, y = harness.load_train()
    assert X.index.name == "sample_id" and "sample_id" not in X.columns
    assert y.index.equals(X.index) and y.max() <= 125
    for split in ("val", "test"):
        _no_labels_in_workspace(ws, split)
        hidden = pd.read_parquet(labels / f"{split}_labels.parquet")
        assert list(hidden.columns) == ["sample_id", "y"] and hidden["sample_id"].tolist() == harness.ids(split)
        assert not set(harness.groups(split)) & set(harness.groups("train"))  # no machine in two splits
    meta = json.loads((ws / "data" / "meta.json").read_text())
    assert meta["prediction_format"] == "scalar" and meta["input_kind"]["train"] == "table"
    assert sum(meta["sizes"].values()) == 20631


@pytest.mark.skipif(not HAS_CWRU, reason="CWRU not downloaded")
def test_array_dataset_from_raw_signals(tmp_path, monkeypatch):
    ws, labels, harness = _workspace(tmp_path, monkeypatch, _spec("bearing_fault_example.json"))
    runpy.run_path(str(ROOT / "specs" / "bearing_fault_example_prepare.py"))
    X, y = harness.load_train()
    assert isinstance(X, np.ndarray) and X.shape[1:] == (2048,) and len(X) == len(y)
    assert harness.classes() == ["ball", "inner_race", "normal", "outer_race"]
    assert set(y) == {0, 1, 2, 3}  # class indices; names via classes()
    assert len(harness.load_val_inputs()) == len(pd.read_parquet(labels / "val_labels.parquet"))
    _no_labels_in_workspace(ws, "val")
    assert not set(harness.groups("val")) & set(harness.groups("train"))


def test_files_and_masks_hide_labels(tmp_path, monkeypatch):
    from PIL import Image

    src = tmp_path / "src"
    paths, masks = [], []
    for i, label in enumerate(["good", "scratch", "good", "scratch"]):
        (src / label).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(src / label / f"{i}.png")
        paths.append(str(src / label / f"{i}.png"))
        mask = src / label / f"{i}_mask.png"
        Image.new("L", (4, 4)).save(mask)
        masks.append(str(mask) if label != "good" else None)

    ws, labels, harness = _workspace(tmp_path, monkeypatch, {"input_streams": [], "label_sources": [],
                                                             "prediction_format": "masks"})
    harness.write_dataset((paths[:2], masks[:2]), (paths[2:3], masks[2:3]), (paths[3:], masks[3:]))

    X, y = harness.load_train()
    assert all(Path(p).exists() and "scratch" not in p and "good" not in p for p in X["path"])
    assert pd.isna(y.iloc[0]) and Path(y.iloc[1]).exists() and "scratch" not in y.iloc[1]  # good: no mask
    assert Path(harness.load_val_inputs()["path"].iloc[0]).name == "val-000000.png"
    test_masks = json.loads((labels / "test_labels.json").read_text())
    assert Path(test_masks["test-000000"]).exists() and str(labels) in test_masks["test-000000"]
    assert not (ws / "data" / "val_labels.json").exists() and not (ws / "data" / "test_labels.json").exists()
    assert not (ws / "data" / "files_y" / "test").exists()  # test masks only in the hidden dir


def test_boxes_labels_are_a_json_map(tmp_path, monkeypatch):
    _, labels, harness = _workspace(tmp_path, monkeypatch, {"input_streams": [], "label_sources": [],
                                                            "prediction_format": "boxes"})
    X = np.zeros((2, 4, 4, 3))
    boxes = [[[0, 0, 2, 2, 0]], []]
    harness.write_dataset((X, boxes), (X, boxes), (X, boxes))
    assert json.loads((labels / "val_labels.json").read_text()) == {"val-000000": boxes[0], "val-000001": []}
    _, y = harness.load_train()
    assert y.tolist() == boxes


def test_write_dataset_refuses_without_labels_dir(tmp_path, monkeypatch):
    _, _, harness = _workspace(tmp_path, monkeypatch, {"input_streams": [], "label_sources": []})
    monkeypatch.delenv("ASSEMBLY_LABELS_DIR")
    with pytest.raises(RuntimeError):
        harness.write_dataset(([[1.0]], [1]), ([[1.0]], [1]), ([[1.0]], [1]))


def test_streams_outside_the_task_are_blocked(tmp_path, monkeypatch):
    _, _, harness = _workspace(tmp_path, monkeypatch, _spec("rul_example.json"))
    with pytest.raises(KeyError):
        harness.read_stream("hvac.ambient_temp")


def test_groups_without_split_default_to_a_group_split(tmp_path, monkeypatch):
    ws, _, harness = _workspace(tmp_path, monkeypatch, {"prediction_format": "scalar"})
    part = lambda n, g: ((pd.DataFrame({"a": np.arange(n, dtype=float)}), list(range(n))), [g] * n)
    (tr, gtr), (va, gva), (te, gte) = part(6, "u1"), part(3, "u2"), part(3, "u3")
    harness.write_dataset(tr, va, te, groups={"train": gtr, "val": gva, "test": gte})
    assert json.loads((ws / "data" / "meta.json").read_text())["split"] == "group"
    harness.write_dataset(tr, va, te, groups={"train": gtr, "val": gva, "test": gte}, meta={"split": "temporal"})
    assert json.loads((ws / "data" / "meta.json").read_text())["split"] == "temporal"  # explicit choice wins
