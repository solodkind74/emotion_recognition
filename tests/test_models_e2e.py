"""Runs the real ONNX models. Downloads weights, so it is opt-in:

    EMOTION_E2E=1 pytest tests/test_models_e2e.py
"""

import os

import numpy as np
import pytest

from emotion_recognition.models import EMOTION_MODELS

pytestmark = pytest.mark.skipif(not os.environ.get("EMOTION_E2E"), reason="set EMOTION_E2E=1 to run")


@pytest.mark.parametrize("model", sorted(EMOTION_MODELS))
def test_classifier_loads_and_predicts(model):
    from emotion_recognition.classifier import EmotionClassifier

    c = EmotionClassifier(model)
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (c.spec.input_size,) * 2 + (3,), dtype=np.uint8) for _ in range(2)]
    probs = c.predict(crops)
    assert probs.shape == (2, len(c.labels))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_detector_runs_on_empty_frame():
    from emotion_recognition.detector import FaceDetector

    assert FaceDetector().detect(np.zeros((720, 1280, 3), np.uint8)) == []
