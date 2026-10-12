"""Frame -> faces -> emotions, with per-stream tracking and smoothing."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .classifier import EmotionClassifier
from .detector import Face, FaceDetector
from .models import DEFAULT_EMOTION_MODEL
from .tracker import FaceTracker


@dataclass
class FaceResult:
    track_id: int
    box: tuple[int, int, int, int]  # x, y, w, h
    detection_score: float
    emotion: str
    confidence: float
    probabilities: dict[str, float]  # smoothed over time

    def to_dict(self) -> dict:
        return asdict(self)


class EmotionRecognizer:
    """Loads the models once; `analyze` is stateless and can serve many streams."""

    def __init__(
        self,
        model: str = DEFAULT_EMOTION_MODEL,
        emotion_model_path: str | Path | None = None,
        detector_path: str | Path | None = None,
        detection_threshold: float = 0.8,
        detection_max_side: int = 640,
        min_face_size: int = 32,
        align: bool = True,
        providers: list[str] | None = None,
    ):
        self.detector = FaceDetector(detector_path, detection_threshold, max_side=detection_max_side)
        self.classifier = EmotionClassifier(model, emotion_model_path, providers)
        self.min_face_size = min_face_size
        self.align = align
        # The OpenCV detector keeps per-call state, so serialise access to it.
        self._lock = threading.Lock()

    @property
    def labels(self) -> tuple[str, ...]:
        return self.classifier.labels

    def analyze(self, frame: np.ndarray) -> tuple[list[Face], np.ndarray]:
        with self._lock:
            faces = [f for f in self.detector.detect(frame) if min(f.box[2], f.box[3]) >= self.min_face_size]
            crops = [self.classifier.crop(frame, f, self.align) for f in faces]
            return faces, self.classifier.predict(crops)


class EmotionStream:
    """Stateful wrapper for one video stream: tracks faces and smooths emotions."""

    def __init__(self, recognizer: EmotionRecognizer, smoothing: float = 0.7):
        self.recognizer = recognizer
        self.tracker = FaceTracker(smoothing=smoothing)

    def process(self, frame: np.ndarray) -> list[FaceResult]:
        faces, probs = self.recognizer.analyze(frame)
        tracks = self.tracker.update([f.box for f in faces], probs)
        labels = self.recognizer.labels
        results = []
        for face, track in zip(faces, tracks):
            best = int(np.argmax(track.probs))
            results.append(
                FaceResult(
                    track_id=track.id,
                    box=tuple(int(round(v)) for v in face.box),
                    detection_score=round(face.score, 3),
                    emotion=labels[best],
                    confidence=round(float(track.probs[best]), 3),
                    probabilities={k: round(float(p), 3) for k, p in zip(labels, track.probs)},
                )
            )
        return results
