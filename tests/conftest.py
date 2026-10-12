from types import SimpleNamespace

import numpy as np
import pytest

from emotion_recognition.detector import Face
from emotion_recognition.models import EMOTION_MODELS


def make_face(x=100.0, y=80.0, w=120.0, h=150.0, eyes=((140, 130), (190, 130)), score=0.95) -> Face:
    landmarks = np.array([*eyes, (165, 160), (145, 190), (185, 190)], np.float32)
    return Face(box=np.array([x, y, w, h], np.float32), landmarks=landmarks, score=score)


class FakeRecognizer:
    """Stands in for EmotionRecognizer: returns fixed faces and probabilities."""

    def __init__(self, model="hsemotion", faces=None, probs=None):
        spec = EMOTION_MODELS[model]
        self.classifier = SimpleNamespace(spec=spec)
        self.labels = spec.labels
        self.faces = faces if faces is not None else [make_face()]
        n = len(spec.labels)
        self.probs = probs if probs is not None else np.tile(np.eye(n, dtype=np.float32)[spec.labels.index("happy")], (len(self.faces), 1))

    def analyze(self, frame):
        return self.faces, self.probs


@pytest.fixture
def fake_recognizer():
    return FakeRecognizer()
