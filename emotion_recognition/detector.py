"""Face detection with YuNet (OpenCV's lightweight ONNX face detector)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .models import YUNET, ensure_model


@dataclass
class Face:
    box: np.ndarray  # x, y, w, h in frame pixels (float32)
    landmarks: np.ndarray  # 5x2: right eye, left eye, nose, right mouth, left mouth
    score: float


class FaceDetector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        score_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        max_side: int = 640,
    ):
        path = Path(model_path) if model_path else ensure_model(YUNET)
        # YuNet works best on faces up to ~300px, so frames are downscaled to
        # `max_side` before detection and boxes are mapped back afterwards.
        self.max_side = max_side
        self._input_size = (320, 320)
        self._net = cv2.FaceDetectorYN.create(
            str(path), "", self._input_size, score_threshold, nms_threshold, 5000
        )

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        img = frame if scale == 1.0 else cv2.resize(
            frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
        size = (img.shape[1], img.shape[0])
        if size != self._input_size:
            self._net.setInputSize(size)
            self._input_size = size

        _, raw = self._net.detect(img)
        if raw is None:
            return []
        coords = raw[:, :14].astype(np.float32) / np.float32(scale)
        return [
            Face(box=c[0:4], landmarks=c[4:14].reshape(5, 2), score=float(s))
            for c, s in zip(coords, raw[:, 14])
        ]
