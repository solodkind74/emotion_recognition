import numpy as np
from conftest import FakeRecognizer, make_face

from emotion_recognition.draw import draw_results
from emotion_recognition.pipeline import EmotionStream


def test_stream_returns_smoothed_results():
    rec = FakeRecognizer("vit")
    stream = EmotionStream(rec, smoothing=0.5)
    (first,) = stream.process(np.zeros((480, 640, 3), np.uint8))
    assert first.track_id == 1
    assert first.box == (100, 80, 120, 150)
    assert first.emotion == "happy" and first.confidence == 1.0
    assert set(first.probabilities) == set(rec.labels)

    rec.probs = np.eye(7, dtype=np.float32)[[rec.labels.index("sad")]]
    (second,) = stream.process(np.zeros((480, 640, 3), np.uint8))
    assert second.track_id == 1
    assert second.probabilities["happy"] == 0.5 and second.probabilities["sad"] == 0.5


def test_no_faces():
    rec = FakeRecognizer("vit", faces=[], probs=np.zeros((0, 7), np.float32))
    assert EmotionStream(rec).process(np.zeros((10, 10, 3), np.uint8)) == []


def test_results_serialise_and_draw():
    rec = FakeRecognizer("hsemotion", faces=[make_face(), make_face(x=300)])
    frame = np.zeros((480, 640, 3), np.uint8)
    results = EmotionStream(rec).process(frame)
    assert [r.track_id for r in results] == [1, 2]
    d = results[0].to_dict()
    assert d["emotion"] == "happy" and isinstance(d["box"], tuple)
    out = draw_results(frame, results, hud="test")
    assert out.shape == frame.shape and out.any() and not frame.any()
