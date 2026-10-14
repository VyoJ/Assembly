import importlib.util
import json
import runpy
import shutil
from pathlib import Path

import pytest

from core.config import settings

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (settings.data_dir / "cmapss" / "train_FD001.txt").exists(),
    reason="C-MAPSS not downloaded (scripts/download_data.py --only cmapss)",
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws, labels = tmp_path / "ws", tmp_path / "_labels"
    ws.mkdir()
    shutil.copy(ROOT / "specs" / "rul_example.json", ws / "task.json")
    shutil.copy(ROOT / "src" / "ml" / "harness" / "harness.py", ws / "harness.py")
    monkeypatch.setenv("ASSEMBLY_WORKSPACE", str(ws))
    monkeypatch.setenv("ASSEMBLY_LABELS_DIR", str(labels))
    monkeypatch.syspath_prepend(str(ws))
    spec = importlib.util.spec_from_file_location("harness", ws / "harness.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    monkeypatch.setitem(__import__("sys").modules, "harness", harness)
    return ws, labels, harness


def test_prepare_writes_hidden_labels_and_loaders_read_back(workspace):
    ws, labels, harness = workspace
    runpy.run_path(str(ROOT / "specs" / "rul_example_prepare.py"))

    X, y = harness.load_train()
    assert {"id", "group"} <= set(X.columns) and "y" not in X.columns
    assert len(X) == len(y) and y.max() <= 125

    for split in ("val", "test"):
        inputs = getattr(harness, f"load_{split}_inputs")()
        hidden = __import__("pandas").read_parquet(labels / f"{split}.parquet")
        assert "y" not in inputs.columns
        assert list(inputs["id"]) == list(hidden["id"]) and inputs["id"].is_unique
        assert not (set(inputs["group"]) & set(X["group"]))  # no unit in two splits

    meta = json.loads((ws / "data" / "meta.json").read_text())
    assert meta["split"] == "group" and sum(meta["sizes"].values()) == 20631
    assert not any((ws / "data").glob("*val.parquet"))  # labels never inside the workspace


def test_write_dataset_refuses_without_labels_dir(workspace, monkeypatch):
    _, _, harness = workspace
    monkeypatch.delenv("ASSEMBLY_LABELS_DIR")
    with pytest.raises(RuntimeError):
        harness.write_dataset(([[1]], [1]), ([[1]], [1]), ([[1]], [1]))


def test_streams_outside_the_task_are_blocked(workspace):
    _, _, harness = workspace
    with pytest.raises(KeyError):
        harness.read_stream("hvac.ambient_temp")
