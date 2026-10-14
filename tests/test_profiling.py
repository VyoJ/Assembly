"""Synthetic tests of real discovery profiling; no datasets or ML mocks required."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.contracts import StreamInfo, StreamProfile, TaskSpec
from ml import catalog, profiling


@pytest.fixture
def spec():
    return TaskSpec(
        id="sensor_task", name="Sensor task", rationale="Prevent downtime", framing="Predict failures",
        input_streams=["sensor"], label_sources=["maintenance_log", "hidden_val", "hidden_test"],
        target_definition="Not executable: target construction belongs to prepare.py",
        prediction_format="scalar", primary_metric="rmse", metrics=["rmse"], kpis=[],
        split_strategy="by engine", time_budget_s=60,
    )


def _stream(stream_id="sensor", modality="temperature"):
    return StreamInfo(id=stream_id, name=stream_id, modality=modality, unit="degC", source="synthetic:test")


def _frame(values, column="sensor"):
    return pd.DataFrame({"unit": [1] * len(values), "cycle": range(1, len(values) + 1), column: values})


def test_normal_varying_sensor(monkeypatch, spec):
    frame = _frame([2.0, 4.0, 6.0])
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    stats = profiling._numeric_stats(frame["sensor"])
    assert stats == {
        "total": 3, "missing": 0, "missing_percent": 0.0, "infinite": 0, "non_finite": 0, "finite": 3,
        "mean": 4.0, "std": pytest.approx(np.sqrt(8 / 3)), "min": 2.0, "max": 6.0,
        "exactly_constant": False, "approximately_constant": False,
    }
    profile, = profiling.profile_streams(spec, [_stream()])
    assert profile.keep and profile.relevance == 1.0
    assert "Varying finite readings" in profile.reason
    assert "population std" in profile.reason
    assert "Predictive relevance is unassessed" in profile.reason


@pytest.mark.parametrize("level", [0.0, 5.0, -5.0, 1e-12, 1e12])
def test_exactly_constant_sensor(monkeypatch, spec, level):
    frame = _frame([level] * 3)
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    stats = profiling._numeric_stats(frame["sensor"])
    assert stats["exactly_constant"] and stats["approximately_constant"]
    assert stats["mean"] == pytest.approx(level) and stats["std"] == 0.0
    profile, = profiling.profile_streams(spec, [_stream(modality="pressure")])
    assert not profile.keep and profile.relevance == 1.0  # quality is not predictive utility
    assert "Exactly constant finite readings" in profile.reason


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12, -1e12])
@pytest.mark.parametrize("relative_change, near", [(1e-8, True), (1e-3, False)])
def test_near_constancy_is_scale_aware(monkeypatch, spec, scale, relative_change, near):
    frame = _frame([scale, scale * (1 + relative_change), scale * (1 - relative_change)])
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    stats = profiling._numeric_stats(frame["sensor"])
    assert not stats["exactly_constant"]
    assert stats["approximately_constant"] is near
    profile, = profiling.profile_streams(spec, [_stream()])
    assert profile.keep is not near
    if near:
        assert "relative span <= 1e-6" in profile.reason


def test_small_signal_near_zero_is_not_constant(monkeypatch, spec):
    frame = _frame([0.0, 1e-12, -1e-12])
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    stats = profiling._numeric_stats(frame["sensor"])
    assert not stats["approximately_constant"]
    assert profiling.profile_streams(spec, [_stream()])[0].keep


@pytest.mark.parametrize(
    "values, missing, infinite",
    [([2.0, np.nan, 4.0, None], 2, 0), ([2.0, np.inf, 4.0, -np.inf], 0, 2),
     ([2.0, np.nan, np.inf, 4.0, -np.inf], 1, 2)],
)
def test_missing_and_infinite_values(monkeypatch, spec, values, missing, infinite):
    frame = _frame(values)
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    stats = profiling._numeric_stats(frame["sensor"])
    assert stats["missing"] == missing and stats["infinite"] == infinite
    assert stats["non_finite"] == missing + infinite
    assert stats["finite"] == 2
    assert stats["missing_percent"] == pytest.approx(100 * missing / len(values))
    assert stats["mean"] == 3.0 and stats["std"] == pytest.approx(1.0)
    assert stats["min"] == 2.0 and stats["max"] == 4.0
    profile, = profiling.profile_streams(spec, [_stream()])
    assert profile.keep and profile.relevance == pytest.approx(2 / len(values))
    assert f"missing={missing}" in profile.reason and f"infinite={infinite}" in profile.reason


@pytest.mark.parametrize("values", [[], [None, None], [np.inf, -np.inf], [np.nan, np.inf, -np.inf]])
def test_empty_or_all_invalid_observations(monkeypatch, spec, values):
    frame = _frame(values)
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    stats = profiling._numeric_stats(frame["sensor"])
    assert stats["total"] == len(values) and stats["finite"] == 0
    assert stats["non_finite"] == len(values)
    assert stats["missing_percent"] == (0.0 if not values else 100 * stats["missing"] / len(values))
    assert all(stats[key] is None for key in ("mean", "std", "min", "max"))
    assert not stats["exactly_constant"] and not stats["approximately_constant"]
    profile, = profiling.profile_streams(spec, [_stream()])
    assert not profile.keep and profile.relevance == 0.0
    assert "No usable finite observations" in profile.reason and "mean=unavailable" in profile.reason


@pytest.mark.parametrize("values", [[7.0], [np.nan, 7.0, np.inf]])
def test_one_finite_observation_does_not_prove_constancy(monkeypatch, spec, values):
    frame = _frame(values)
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    stats = profiling._numeric_stats(frame["sensor"])
    assert stats["mean"] == stats["min"] == stats["max"] == 7.0 and stats["std"] == 0.0
    assert not stats["exactly_constant"] and not stats["approximately_constant"]
    profile, = profiling.profile_streams(spec, [_stream()])
    assert profile.keep and profile.relevance == pytest.approx(1 / len(values))
    assert "insufficient evidence to assess constancy" in profile.reason


@pytest.mark.parametrize("dtype", ["Float64", "Int64"])
def test_nullable_numeric_types(monkeypatch, spec, dtype):
    frame = _frame(pd.Series([2, pd.NA, 4], dtype=dtype))
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    profile, = profiling.profile_streams(spec, [_stream()])
    assert profile.keep and profile.relevance == pytest.approx(2 / 3)
    assert "missing=1 (33.3%)" in profile.reason


@pytest.mark.parametrize("values", [[1e308, -1e308], [1e-300, -1e-300], [1e308, 1e308]])
def test_statistics_avoid_overflow_and_underflow(values):
    with np.errstate(over="raise", invalid="raise"):
        stats = profiling._numeric_stats(pd.Series(values))
    assert np.isfinite(stats["mean"]) and np.isfinite(stats["std"])
    assert stats["std"] == pytest.approx(0.0 if values[0] == values[1] else abs(values[0]), abs=0.0)


def test_small_variation_around_large_offset_keeps_standard_deviation():
    values = pd.Series([1e12, 1e12 + 0.01])
    stats = profiling._numeric_stats(values)
    assert not stats["exactly_constant"] and stats["approximately_constant"]
    assert stats["std"] == pytest.approx((values.iloc[1] - values.iloc[0]) / 2, abs=0.0)


@pytest.mark.parametrize(
    "values",
    [["2", "4"], [2.0, "broken", 4.0], [True, False], [1 + 2j, 3 + 4j],
     pd.to_datetime(["2026-01-01", "2026-01-02"])],
)
def test_invalid_numeric_types_are_rejected_without_coercion(monkeypatch, spec, values):
    frame = _frame(values)
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    with pytest.raises(ValueError, match="real numeric sensor dtype"):
        profiling._numeric_stats(frame["sensor"])
    profile, = profiling.profile_streams(spec, [_stream()])
    assert not profile.keep and profile.relevance == 0.0
    assert "no coercion performed" in profile.reason


@pytest.mark.parametrize("column", ["sensor", "value"])
def test_named_and_legacy_columns(monkeypatch, spec, column):
    frame = _frame([2.0, 4.0], column)
    calls = []

    def read(stream_id):
        calls.append(stream_id)
        return frame

    monkeypatch.setattr(catalog, "read_stream", read)
    profile, = profiling.profile_streams(spec, [_stream()])
    assert calls == ["sensor"] and profile.stream_id == "sensor" and profile.keep


def test_named_column_is_authoritative(monkeypatch, spec):
    frame = _frame([5.0, 5.0]).assign(value=[1.0, 2.0])
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    assert not profiling.profile_streams(spec, [_stream()])[0].keep


@pytest.mark.parametrize(
    "frame, message",
    [(pd.DataFrame(), "unit/cycle"), (_frame([1, 2]).drop(columns="unit"), "unit/cycle"),
     (_frame([1, 2]).drop(columns="cycle"), "unit/cycle"),
     (_frame([1, 2]).drop(columns="sensor"), "missing sensor column"),
     (_frame([1, 2], "unrelated"), "missing sensor column"),
     (_frame([1, 2], "value").assign(unrelated=[3, 4]), "missing sensor column"),
     (pd.DataFrame([[1, 1, 2, 3]], columns=["unit", "cycle", "sensor", "sensor"]), "flat and unique"),
     (_frame([1, 2]).assign(cycle=[1, 1]), "non-missing and unique"),
     (_frame([1, 2]).assign(unit=[1, None]), "non-missing and unique")],
)
def test_invalid_or_missing_columns(monkeypatch, spec, frame, message):
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    profile, = profiling.profile_streams(spec, [_stream()])
    assert not profile.keep and profile.relevance == 0.0
    assert message in profile.reason


@pytest.mark.parametrize("frame", [None, [1, 2], {"unit": [1], "cycle": [1], "sensor": [2]}])
def test_loader_must_return_dataframe(monkeypatch, spec, frame):
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    profile, = profiling.profile_streams(spec, [_stream()])
    assert not profile.keep and "pandas DataFrame" in profile.reason


def test_stream_id_must_not_select_a_key(monkeypatch, spec):
    monkeypatch.setattr(catalog, "read_stream", lambda sid: _frame([1.0, 2.0]))
    profile, = profiling.profile_streams(spec, [_stream("cycle")])
    assert not profile.keep and "stream_id must not be a unit/cycle key" in profile.reason


def test_multilevel_columns_are_rejected(monkeypatch, spec):
    frame = _frame([1, 2])
    frame.columns = pd.MultiIndex.from_tuples([(column, "reading") for column in frame.columns])
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    profile, = profiling.profile_streams(spec, [_stream()])
    assert not profile.keep and "flat and unique" in profile.reason


def test_multiple_streams_preserve_order_ids_and_contract(monkeypatch, spec):
    frames = {"temperature": _frame([20, 22, 24], "temperature"),
              "pressure": _frame([5, 5, 5], "pressure"),
              "vibration": _frame([1, np.nan, np.inf, 3], "vibration")}
    streams = [_stream("vibration", "vibration"), _stream("pressure", "pressure"), _stream("temperature")]
    monkeypatch.setattr(catalog, "read_stream", frames.__getitem__)
    profiles = profiling.profile_streams(spec, streams)
    assert [p.stream_id for p in profiles] == [s.id for s in streams]
    assert [p.keep for p in profiles] == [True, False, True]
    assert [p.relevance for p in profiles] == [0.5, 1.0, 1.0]
    for profile in profiles:
        assert isinstance(profile, StreamProfile) and np.isfinite(profile.relevance)
        assert set(profile.model_dump()) == {"stream_id", "relevance", "keep", "reason"}
        assert StreamProfile.model_validate_json(profile.model_dump_json()) == profile
    assert profiling.profile_streams(spec, streams) == profiles  # deterministic


def test_empty_stream_list_does_not_load(monkeypatch, spec):
    def forbidden(*args, **kwargs):
        pytest.fail("empty candidates must not load data")

    monkeypatch.setattr(catalog, "read_stream", forbidden)
    assert profiling.profile_streams(spec, []) == []


@pytest.mark.parametrize(
    "error", [NotImplementedError("pending"), FileNotFoundError("missing source"),
              KeyError("unknown sensor"), OSError("unreadable source"), ImportError("loader")],
)
def test_unavailable_loader_or_source_retains_unassessed_stream(monkeypatch, spec, error):
    def read(stream_id):
        if stream_id == "missing":
            raise error
        return _frame([1, 2], stream_id)

    monkeypatch.setattr(catalog, "read_stream", read)
    missing, good = profiling.profile_streams(spec, [_stream("missing"), _stream("good")])
    assert missing.keep and missing.relevance == 0.0
    assert type(error).__name__ in missing.reason and "quality unassessed" in missing.reason
    assert good.keep and good.relevance == 1.0


def test_invalid_source_is_reported(monkeypatch, spec):
    def read(stream_id):
        raise ValueError("unknown source")

    monkeypatch.setattr(catalog, "read_stream", read)
    profile, = profiling.profile_streams(spec, [_stream()])
    assert not profile.keep and "Invalid stream source: unknown source" in profile.reason


def test_unexpected_loader_bug_is_not_hidden(monkeypatch, spec):
    def read(stream_id):
        raise RuntimeError("loader bug")

    monkeypatch.setattr(catalog, "read_stream", read)
    with pytest.raises(RuntimeError, match="loader bug"):
        profiling.profile_streams(spec, [_stream()])


@pytest.mark.parametrize("modality", ["image", "video", "log"])
def test_unsupported_modality_is_retained_without_fabricated_stats(monkeypatch, spec, modality):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported modalities must not load numeric data")

    monkeypatch.setattr(catalog, "read_stream", forbidden)
    profile, = profiling.profile_streams(spec, [_stream("unsupported", modality)])
    assert profile.keep and profile.relevance == 0.0 and profile.stream_id == "unsupported"
    assert "not yet implemented" in profile.reason and modality in profile.reason
    assert "quality unassessed" in profile.reason and "mean=" not in profile.reason


@pytest.mark.parametrize("modality", ["temperature", "pressure", "vibration", "plc", "other"])
def test_numeric_modalities(monkeypatch, spec, modality):
    monkeypatch.setattr(catalog, "read_stream", lambda sid: _frame([1, 2]))
    profile, = profiling.profile_streams(spec, [_stream(modality=modality)])
    assert profile.keep and "observations=2" in profile.reason


def test_no_label_access_target_construction_or_file_writes(monkeypatch, spec):
    def forbidden(*args, **kwargs):
        pytest.fail("profiling must not access labels or create artifacts")

    calls = []

    def read(stream_id):
        calls.append(stream_id)
        return _frame([1, 2])

    monkeypatch.setattr(catalog, "read_stream", read)
    monkeypatch.setattr(catalog, "read_labels", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    original = profiling.profile_streams(spec, [_stream()])
    different_target = spec.model_copy(update={
        "target_definition": "completely different", "label_sources": [],
    })
    assert profiling.profile_streams(different_target, [_stream()]) == original
    assert calls == ["sensor", "sensor"]


@pytest.mark.parametrize("values", [[1, 2, 3], [1.0, np.nan, np.inf, 3.0], [1.0, "invalid"], [None, None]])
def test_source_dataframe_is_unchanged(monkeypatch, spec, values):
    frame = _frame(values)
    frame.index = pd.Index(range(10, 10 + len(frame)), name="original_index")
    frame.attrs["source"] = "test"
    before = frame.copy(deep=True)
    monkeypatch.setattr(catalog, "read_stream", lambda sid: frame)
    profiling.profile_streams(spec, [_stream()])
    pd.testing.assert_frame_equal(frame, before)
    assert frame.attrs == before.attrs


def test_api_dispatches_to_real_implementation_when_mock_disabled(monkeypatch, spec):
    from core.config import settings
    from ml import api, mock

    def forbidden(*args, **kwargs):
        pytest.fail("real profiling must not use the ML mock")

    monkeypatch.setattr(settings, "ml_mock", False)
    monkeypatch.setattr(mock, "profile_streams", forbidden)
    monkeypatch.setattr(catalog, "read_stream", lambda sid: _frame([1, 2]))
    assert api.profile_streams(spec, [_stream()]) == profiling.profile_streams(spec, [_stream()])


@pytest.mark.parametrize("problem", [None, "duplicate", "null", "missing_sensor", "key_sensor"])
def test_current_entity_time_tables(monkeypatch, spec, problem):
    frame = _frame([1.0, 2.0, 3.0]).rename(columns={"unit": "entity", "cycle": "time"})
    sid = "sensor"
    if problem == "duplicate":
        frame["time"] = 1
    elif problem == "null":
        frame.loc[0, "entity"] = None
    elif problem == "missing_sensor":
        frame = frame.drop(columns="sensor")
    elif problem == "key_sensor":
        sid = "time"
    monkeypatch.setattr(catalog, "read_stream", lambda _: frame)
    before = frame.copy(deep=True)
    profile, = profiling.profile_streams(spec, [_stream(sid)])
    assert profile.keep == (problem is None)
    assert profile.relevance == (1.0 if problem is None else 0.0)
    pd.testing.assert_frame_equal(frame, before)
    if problem is None:
        with pytest.raises(ValueError, match="missing sensor column"):
            profiling._sensor_values(frame.rename(columns={"sensor": "value"}), "sensor")


@pytest.mark.parametrize("problem", [None, "duplicate", "null"])
def test_recording_indices_are_retained_without_profiling_metadata(monkeypatch, spec, problem):
    frame = pd.DataFrame({"record_id": ["record-1", "record-2"], "entity": ["motor", "motor"],
                          "path": ["/unused/1.mat", "/unused/2.mat"], "channel": ["DE", "DE"],
                          "rpm": [1797, 1750], "sample_hz": [48000, 12000]})
    if problem == "duplicate":
        frame["record_id"] = "same"
    elif problem == "null":
        frame.loc[0, "path"] = None
    monkeypatch.setattr(catalog, "read_stream", lambda _: frame)
    profile, = profiling.profile_streams(spec, [_stream("motor4.vibration_drive_end", "vibration")])
    assert profile.keep == (problem is None)
    assert profile.relevance == 0.0
    if problem is None:
        assert "quality unassessed" in profile.reason and "mean=" not in profile.reason
        assert "unused" not in profile.reason


def test_real_catalog_current_table_reader(tmp_path, monkeypatch, spec):
    if not hasattr(catalog, "TABLE_KEYS"):
        pytest.skip("current table reader absent on this branch")
    from core.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    folder = tmp_path / "cmapss"
    folder.mkdir()
    rows = [[1, i, 0, 0, 0, *[j + i for j in range(21)]] for i in range(1, 5)]
    (folder / "train_FD001.txt").write_text("\n".join(" ".join(map(str, row)) for row in rows))
    catalog._table.cache_clear()
    try:
        stream = next(s for s in catalog.list_catalog()[0] if s.id == "press3.lpc_outlet_temp")
        assert list(catalog.read_stream(stream.id).columns) == ["entity", "time", stream.id]
        profile, = profiling.profile_streams(spec, [stream])
        assert profile.keep and profile.relevance == 1.0 and "mean=3.5" in profile.reason
    finally:
        catalog._table.cache_clear()


def test_real_catalog_recordings_do_not_become_numeric_sensor_tables(tmp_path, monkeypatch, spec):
    if not hasattr(catalog, "TABLE_KEYS"):
        pytest.skip("recording reader absent on this branch")
    from core.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "cwru").mkdir()
    stream = next(s for s in catalog.list_catalog()[0] if s.id == "motor4.vibration_drive_end")
    frame = catalog.read_stream(stream.id)
    assert {"record_id", "entity", "path", "channel", "sample_hz"} <= set(frame)
    assert "fault" not in frame and "fault_size_in" not in frame
    profile, = profiling.profile_streams(spec, [stream])
    assert profile.keep and profile.relevance == 0.0 and "quality unassessed" in profile.reason
    assert "mean=" not in profile.reason
