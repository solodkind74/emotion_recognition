"""Overlay rendering for annotated frames."""

from __future__ import annotations

import cv2
import numpy as np

from .pipeline import FaceResult

# BGR colours per canonical emotion label.
COLORS = {
    "angry": (40, 40, 220),
    "contempt": (150, 70, 160),
    "disgust": (40, 140, 60),
    "fear": (170, 90, 200),
    "happy": (0, 200, 255),
    "neutral": (180, 180, 180),
    "sad": (210, 120, 40),
    "surprise": (0, 140, 255),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text_box(img: np.ndarray, text: str, org: tuple[int, int], color, scale: float) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    pad = round(3 * scale / 0.55)
    cv2.rectangle(img, (x, y - th - base - 2 * pad), (x + tw + 2 * pad, y), color, -1)
    cv2.putText(img, text, (x + pad, y - base - pad), FONT, scale, (20, 20, 20), 1, cv2.LINE_AA)


def draw_results(frame: np.ndarray, results: list[FaceResult], top_k: int = 3, hud: str | None = None) -> np.ndarray:
    out = frame.copy()
    s = max(1.0, frame.shape[0] / 720)  # keep labels legible on large frames
    for r in results:
        x, y, w, h = r.box
        color = COLORS.get(r.emotion, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, round(2 * s))
        _text_box(out, f"#{r.track_id} {r.emotion} {r.confidence:.0%}", (x, max(y, round(22 * s))), color, 0.55 * s)

        # Top-k probability bars under the box.
        bar_w, bar_h = max(w, round(120 * s)), round(12 * s)
        for i, (label, p) in enumerate(sorted(r.probabilities.items(), key=lambda kv: -kv[1])[:top_k]):
            by = y + h + round(6 * s) + i * round(16 * s)
            cv2.rectangle(out, (x, by), (x + bar_w, by + bar_h), (40, 40, 40), -1)
            cv2.rectangle(out, (x, by), (x + int(bar_w * p), by + bar_h), COLORS.get(label, (255, 255, 255)), -1)
            cv2.putText(out, f"{label} {p:.0%}", (x + 3, by + bar_h - round(2 * s)), FONT, 0.38 * s,
                        (255, 255, 255), 1, cv2.LINE_AA)

    if hud:
        _text_box(out, hud, (8, round(30 * s)), (235, 235, 235), 0.6 * s)
    return out
