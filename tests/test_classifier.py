import cv2
import numpy as np
import pytest
from conftest import make_face

from emotion_recognition import classifier as clf
from emotion_recognition.classifier import EmotionClassifier, alignment_matrix, crop_face, softmax


def apply(m, pts):
    pts = np.asarray(pts, np.float64)
    return pts @ m[:, :2].T + m[:, 2]


@pytest.mark.parametrize("dy", [-30, -10, 0, 15, 40])
def test_alignment_levels_the_eyes(dy):
    face = make_face(eyes=((140, 130), (190, 130 + dy)))
    m = alignment_matrix(face, margin=0.1, out_size=224)
    left, right = apply(m, face.landmarks[:2])
    assert left[1] == pytest.approx(right[1], abs=1e-6)
    assert left[0] < right[0]
    # The box centre lands in the middle of the crop.
    np.testing.assert_allclose(apply(m, [[160, 155]])[0], [112, 112], atol=1e-6)


def test_alignment_ignores_implausible_tilt():
    face = make_face(eyes=((140, 130), (150, 200)))  # ~82 degrees: bad landmarks
    m = alignment_matrix(face, margin=0.0, out_size=100)
    assert m[0, 1] == pytest.approx(0.0)


def test_crop_face_shape_and_content():
    frame = np.zeros((480, 640, 3), np.uint8)
    cv2.rectangle(frame, (100, 80), (220, 230), (0, 0, 255), -1)  # red "face"
    crop = crop_face(frame, make_face(), size=64, margin=0.0, align=False)
    assert crop.shape == (64, 64, 3) and crop.dtype == np.uint8
    assert crop[32, 32].tolist() == [0, 0, 255]

    # Faces at the frame border are padded rather than failing.
    edge = crop_face(frame, make_face(x=-50, y=-50), size=64, margin=0.2)
    assert edge.shape == (64, 64, 3)


class FakeSession:
    def __init__(self, n_out, batch=None, logits=None):
        self.n_out, self.batch, self.logits, self.calls = n_out, batch, logits, []

    def get_inputs(self):
        return [type("I", (), {"name": "pixel_values", "shape": [self.batch or "batch", 3, 224, 224]})]

    def get_outputs(self):
        return [type("O", (), {"shape": ["batch", self.n_out]})]

    def run(self, _, feeds):
        x = feeds["pixel_values"]
        self.calls.append(x.shape)
        logits = self.logits if self.logits is not None else np.zeros(self.n_out, np.float32)
        return [np.tile(logits, (len(x), 1))]


def make_classifier(monkeypatch, model, **session_kwargs):
    n = len(clf.EMOTION_MODELS[model].labels)
    session = FakeSession(n, **session_kwargs)
    monkeypatch.setattr(clf.ort, "InferenceSession", lambda *a, **k: session)
    return EmotionClassifier(model, model_path="unused.onnx"), session


def test_preprocess_grayscale_and_normalisation(monkeypatch):
    c, _ = make_classifier(monkeypatch, "vit")
    crop = np.zeros((224, 224, 3), np.uint8)
    crop[..., 2] = 255  # pure red in BGR
    x = c.preprocess([crop])
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    # Grey replicated into all three channels, mapped to [-1, 1].
    np.testing.assert_allclose(x[0, 0], x[0, 1])
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[0, 0] / 255.0
    assert x[0, 0, 0, 0] == pytest.approx((gray - 0.5) / 0.5, abs=1e-6)


def test_preprocess_colour_models_get_rgb(monkeypatch):
    c, _ = make_classifier(monkeypatch, "hsemotion")
    crop = np.zeros((224, 224, 3), np.uint8)
    crop[..., 2] = 255  # red in BGR -> channel 0 in RGB
    x = c.preprocess([crop])
    assert x[0, 0, 0, 0] == pytest.approx((1 - 0.485) / 0.229, abs=1e-5)
    assert x[0, 2, 0, 0] == pytest.approx((0 - 0.406) / 0.225, abs=1e-5)


def test_predict_returns_probabilities(monkeypatch):
    c, session = make_classifier(monkeypatch, "vit", logits=np.array([0, 0, 0, 5, 0, 0, 0], np.float32))
    probs = c.predict([np.zeros((224, 224, 3), np.uint8)] * 3)
    assert probs.shape == (3, 7)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)
    assert c.labels[int(probs[0].argmax())] == "happy"
    assert session.calls == [(3, 3, 224, 224)]
    assert c.predict([]).shape == (0, 7)


def test_fixed_batch_models_run_face_by_face(monkeypatch):
    c, session = make_classifier(monkeypatch, "vit", batch=1)
    c.predict([np.zeros((224, 224, 3), np.uint8)] * 2)
    assert session.calls == [(1, 3, 224, 224)] * 2


def test_label_count_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr(clf.ort, "InferenceSession", lambda *a, **k: FakeSession(5))
    with pytest.raises(ValueError, match="5 outputs"):
        EmotionClassifier("vit", model_path="unused.onnx")


def test_softmax_is_stable():
    p = softmax(np.array([[1000.0, 0.0]]))
    np.testing.assert_allclose(p, [[1.0, 0.0]])
