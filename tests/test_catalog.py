import numpy as np
import pytest

from core.config import settings
from ml import catalog

HAS_CMAPSS = (settings.data_dir / "cmapss" / "train_FD001.txt").exists()
HAS_CWRU = (settings.data_dir / "cwru" / "97.mat").exists()


def _ids(kind: str) -> list[str]:
    streams, _ = catalog.list_catalog()
    sources = catalog._raw()["sources"]
    return [s.id for s in streams if sources[s.source.split(":")[0]]["kind"] == kind]


@pytest.mark.skipif(not HAS_CMAPSS, reason="C-MAPSS not downloaded")
def test_every_table_stream_reads():
    for sid in _ids("table") + _ids("synthetic"):
        df = catalog.read_stream(sid)
        assert list(df.columns) == ["entity", "time", sid]
        assert len(df) > 0 and df[sid].notna().all()


@pytest.mark.skipif(not HAS_CMAPSS, reason="C-MAPSS not downloaded")
def test_wide_table_and_events_line_up():
    ids = _ids("table")
    wide = catalog.read_streams(ids)
    assert list(wide.columns) == ["entity", "time", *ids]
    events = catalog.read_labels("maintenance_log")
    assert list(events.columns) == ["entity", "event_time"]
    assert events["entity"].nunique() == wide["entity"].nunique() == 100
    last = wide.groupby("entity")["time"].max().reset_index(drop=True)
    assert (last == events.sort_values("entity")["event_time"].reset_index(drop=True)).all()


@pytest.mark.skipif(not HAS_CMAPSS, reason="C-MAPSS not downloaded")
def test_distractors_are_deterministic_and_unrelated():
    a = catalog.read_stream("hvac.ambient_temp")
    assert a.equals(catalog.read_stream("hvac.ambient_temp"))
    df = a.merge(catalog.read_labels("maintenance_log"), on="entity")
    assert abs(df["hvac.ambient_temp"].corr(df["event_time"] - df["time"])) < 0.1


@pytest.mark.skipif(not HAS_CWRU, reason="CWRU not downloaded")
def test_signal_stream_and_record_labels():
    rec = catalog.read_stream("motor4.vibration_drive_end")
    assert {"record_id", "path", "channel", "sample_hz"} <= set(rec.columns) and len(rec) == 16
    sig = catalog.load_signal(rec["path"].iloc[0], rec["channel"].iloc[0])
    assert sig.ndim == 1 and len(sig) > 100_000 and np.isfinite(sig).all()
    labels = catalog.read_labels("bearing_inspections")
    assert set(labels["fault"]) == {"normal", "inner_race", "ball", "outer_race"}
    assert set(labels["record_id"]) == set(rec["record_id"])
    assert not rec["record_id"].str.contains(r"/\d+$").any()  # not the CWRU file number
    assert "16 recordings" in catalog.peek_stream("motor4.vibration_drive_end", n=2)


def test_cwru_file_with_two_recordings_picks_its_own(tmp_path):
    from scipy.io import savemat

    path = tmp_path / "99.mat"
    savemat(path, {"X098_DE_time": np.zeros((5, 1)), "X099_DE_time": np.ones((5, 1))})
    assert catalog.load_signal(str(path), "DE").tolist() == [1.0] * 5


@pytest.fixture
def fake_mvtec(tmp_path, monkeypatch):
    from PIL import Image

    root = tmp_path / "mvtec" / "metal_nut"
    for split, label, n in (("train", "good", 3), ("test", "good", 1), ("test", "scratch", 2)):
        (root / split / label).mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (8, 8)).save(root / split / label / f"{i:03d}.png")
            if label != "good":
                (root / "ground_truth" / label).mkdir(parents=True, exist_ok=True)
                Image.new("L", (8, 8)).save(root / "ground_truth" / label / f"{i:03d}_mask.png")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return root


def test_image_stream_and_annotations(fake_mvtec):
    idx = catalog.read_stream("line3.cam_top")
    assert len(idx) == 6 and sorted(idx["frame_idx"]) == list(range(6))
    assert not idx["record_id"].str.contains("good|scratch|train|test").any()  # ids don't leak labels
    assert catalog.load_image(idx["path"].iloc[0]).shape == (8, 8, 3)
    ann = catalog.read_labels("qc_inspections")
    assert ann["defective"].sum() == 2
    assert ann.loc[ann["defective"], "mask_path"].notna().all()
    assert ann.loc[~ann["defective"], "mask_path"].isna().all()
    assert "6 images" in catalog.peek_stream("line3.cam_top")
