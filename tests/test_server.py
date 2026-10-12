import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from emotion_recognition.server import create_app  # noqa: E402


def jpeg(w=320, h=240) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 128, np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(fake_recognizer):
    return TestClient(create_app(fake_recognizer))


def test_index_and_info(client):
    assert "<title>" in client.get("/").text
    info = client.get("/api/info").json()
    assert info["model"] == "hsemotion" and "happy" in info["labels"]


def test_analyze(client):
    r = client.post("/api/analyze", content=jpeg())
    assert r.status_code == 200
    body = r.json()
    assert (body["width"], body["height"]) == (320, 240)
    assert body["faces"][0]["emotion"] == "happy"
    assert client.post("/api/analyze", content=b"not an image").status_code == 400


def test_websocket_stream_keeps_track_ids(client):
    with client.websocket_connect("/ws") as ws:
        ids = []
        for _ in range(3):
            ws.send_bytes(jpeg())
            msg = ws.receive_json()
            assert msg["faces"] and msg["latency_ms"] >= 0
            ids.append(msg["faces"][0]["track_id"])
        assert ids == [1, 1, 1]

        ws.send_bytes(b"garbage")
        assert "error" in ws.receive_json()
