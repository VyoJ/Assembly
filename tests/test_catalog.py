import pytest

from core.config import settings
from ml import catalog

pytestmark = pytest.mark.skipif(
    not (settings.data_dir / "cmapss" / "train_FD001.txt").exists(),
    reason="C-MAPSS not downloaded (scripts/download_data.py --only cmapss)",
)


def _sensor_ids() -> list[str]:
    streams, _ = catalog.list_catalog()
    return [s.id for s in streams if s.modality != "image"]


def test_every_sensor_stream_reads():
    for sid in _sensor_ids():
        df = catalog.read_stream(sid)
        assert list(df.columns) == ["unit", "cycle", sid]
        assert len(df) > 0 and df[sid].notna().all()


def test_wide_table_and_labels_line_up():
    ids = _sensor_ids()
    wide = catalog.read_streams(ids)
    assert list(wide.columns) == ["unit", "cycle", *ids]
    labels = catalog.read_labels("maintenance_log")
    assert list(labels.columns) == ["unit", "failure_cycle"]
    assert labels["unit"].nunique() == wide["unit"].nunique() == 100
    last = wide.groupby("unit")["cycle"].max().reset_index(drop=True)
    assert (last == labels.sort_values("unit")["failure_cycle"].reset_index(drop=True)).all()


def test_distractors_are_deterministic_and_unrelated():
    a = catalog.read_stream("hvac.ambient_temp")
    b = catalog.read_stream("hvac.ambient_temp")
    assert a.equals(b)
    labels = catalog.read_labels("maintenance_log")
    df = a.merge(labels, on="unit")
    rul = df["failure_cycle"] - df["cycle"]
    assert abs(df["hvac.ambient_temp"].corr(rul)) < 0.1


def test_peek_stream():
    text = catalog.peek_stream("press3.core_speed", n=3)
    assert "press3.core_speed" in text and "100 units" in text
