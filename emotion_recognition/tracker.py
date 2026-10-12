"""Minimal IoU tracker that keeps face identities and smooths emotions over time.

Per-frame emotion predictions flicker; averaging them per face (exponential
moving average) gives a much more stable read-out on a live stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two x, y, w, h boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    id: int
    box: np.ndarray
    probs: np.ndarray  # smoothed class probabilities
    age: int = 1  # frames since the track started
    missed: int = 0  # consecutive frames without a matching detection


class FaceTracker:
    def __init__(self, smoothing: float = 0.7, iou_threshold: float = 0.3, max_missed: int = 15):
        """`smoothing` is the EMA weight of the history (0 = no smoothing)."""
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes: list[np.ndarray], probs: np.ndarray) -> list[Track]:
        """Match this frame's detections to tracks; returns one track per detection, in order."""
        pairs = sorted(
            ((iou(t.box, b), ti, di) for ti, t in enumerate(self.tracks) for di, b in enumerate(boxes)),
            reverse=True,
        )
        assigned: dict[int, Track] = {}
        used_tracks: set[int] = set()
        for score, ti, di in pairs:
            if score < self.iou_threshold:
                break
            if ti in used_tracks or di in assigned:
                continue
            track = self.tracks[ti]
            track.box = boxes[di]
            track.probs = self.smoothing * track.probs + (1 - self.smoothing) * probs[di]
            track.age += 1
            track.missed = 0
            assigned[di] = track
            used_tracks.add(ti)

        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        for di, box in enumerate(boxes):
            if di not in assigned:
                track = Track(id=self._next_id, box=box, probs=np.asarray(probs[di], np.float32).copy())
                self._next_id += 1
                self.tracks.append(track)
                assigned[di] = track

        return [assigned[di] for di in range(len(boxes))]
