"""Face cropping/alignment and ONNX emotion classification."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .detector import Face
from .models import EMOTION_MODELS, EmotionModelSpec, ensure_model

# Larger head tilts are more likely bad landmarks than real poses.
MAX_ALIGN_ANGLE = 45.0


def default_providers() -> list[str]:
    preferred = ("CUDAExecutionProvider", "CPUExecutionProvider")
    available = ort.get_available_providers()
    return [p for p in preferred if p in available]


def alignment_matrix(face: Face, margin: float, out_size: int, align: bool = True) -> np.ndarray:
    """Affine map from the frame to a square, upright, `out_size` face crop.

    The crop is centred on the detection box, its side is the larger box side
    grown by `margin`, and (if `align`) it is rotated so the eyes are level.
    """
    x, y, w, h = face.box
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + margin)
    angle = 0.0
    if align:
        (x0, y0), (x1, y1) = sorted(face.landmarks[:2].tolist())  # eyes, left to right in image
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        if abs(angle) > MAX_ALIGN_ANGLE:
            angle = 0.0
    m = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, out_size / side)
    m[0, 2] += out_size / 2 - cx
    m[1, 2] += out_size / 2 - cy
    return m


def crop_face(frame: np.ndarray, face: Face, size: int, margin: float = 0.0, align: bool = True) -> np.ndarray:
    # Warp at native resolution, then area-downsample: warping a large face
    # straight to `size` with bilinear sampling would alias.
    native = max(size, int(round(float(max(face.box[2], face.box[3])) * (1 + margin))))
    m = alignment_matrix(face, margin, native, align)
    crop = cv2.warpAffine(frame, m, (native, native), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if native != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class EmotionClassifier:
    def __init__(
        self,
        model: str | EmotionModelSpec = "vit",
        model_path: str | Path | None = None,
        providers: list[str] | None = None,
    ):
        self.spec = EMOTION_MODELS[model] if isinstance(model, str) else model
        path = Path(model_path) if model_path else ensure_model(self.spec.file)
        self.session = ort.InferenceSession(str(path), providers=providers or default_providers())
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Some exports have a fixed batch size of 1; run those face by face.
        self.batched = not (isinstance(inp.shape[0], int) and inp.shape[0] == 1)
        n_out = self.session.get_outputs()[0].shape[-1]
        if isinstance(n_out, int) and n_out != len(self.spec.labels):
            raise ValueError(f"{self.spec.name}: model has {n_out} outputs, expected {len(self.spec.labels)}")

    @property
    def labels(self) -> tuple[str, ...]:
        return self.spec.labels

    def crop(self, frame: np.ndarray, face: Face, align: bool = True) -> np.ndarray:
        return crop_face(frame, face, self.spec.input_size, self.spec.crop_margin, align)

    def preprocess(self, crops: list[np.ndarray]) -> np.ndarray:
        """BGR uint8 crops (already `input_size` square) -> NCHW float32."""
        batch = np.stack(crops)
        if self.spec.grayscale:
            gray = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in crops])
            batch = np.repeat(gray[..., None], 3, axis=-1)
        else:
            batch = batch[..., ::-1]  # BGR -> RGB
        x = batch.astype(np.float32) / 255.0
        x = (x - np.asarray(self.spec.mean, np.float32)) / np.asarray(self.spec.std, np.float32)
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2))

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return an (N, num_labels) array of class probabilities."""
        if not crops:
            return np.zeros((0, len(self.labels)), np.float32)
        x = self.preprocess(crops)
        if self.batched:
            logits = self.session.run(None, {self.input_name: x})[0]
        else:
            logits = np.concatenate([self.session.run(None, {self.input_name: x[i : i + 1]})[0] for i in range(len(x))])
        return softmax(logits.astype(np.float32))
