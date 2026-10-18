"""Stream previews show the real data behind a stream, and say so when values are generated."""

from fastapi.testclient import TestClient

from api import streams
from api.main import app


def test_preview_falls_back_to_simulated_values_when_data_is_missing(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("data/cmapss/train_FD001.txt")

    monkeypatch.setattr(streams.catalog, "read_stream", missing)
    r = TestClient(app).get("/api/streams/press3.fan_speed/preview").json()
    assert r["synthetic"] is True and "not available" in r["note"] and r["points"]


def test_unknown_stream_is_404():
    c = TestClient(app)
    assert c.get("/api/streams/nope/preview").status_code == 404
    assert c.get("/api/streams/press3.fan_speed/images/0").status_code == 404  # not an image stream
