import numpy as np
import pytest

from emotion_recognition.tracker import FaceTracker, iou


def box(x, y, w=100, h=100):
    return np.array([x, y, w, h], np.float32)


def onehot(i, n=3):
    p = np.zeros(n, np.float32)
    p[i] = 1.0
    return p


def test_iou():
    assert iou(box(0, 0), box(0, 0)) == pytest.approx(1.0)
    assert iou(box(0, 0), box(200, 0)) == 0.0
    assert iou(box(0, 0), box(50, 0)) == pytest.approx(50 * 100 / (2 * 100 * 100 - 50 * 100))


def test_ids_persist_for_moving_faces_and_new_faces_get_new_ids():
    t = FaceTracker(smoothing=0.0)
    a, b = t.update([box(0, 0), box(300, 0)], np.stack([onehot(0), onehot(1)]))
    assert (a.id, b.id) == (1, 2)

    # Faces move a little and come back in a different order.
    b2, a2 = t.update([box(310, 5), box(10, 5)], np.stack([onehot(1), onehot(0)]))
    assert (a2.id, b2.id) == (1, 2)

    *_, c = t.update([box(10, 5), box(310, 5), box(600, 0)], np.stack([onehot(0), onehot(1), onehot(2)]))
    assert c.id == 3


def test_ema_smoothing():
    t = FaceTracker(smoothing=0.5)
    t.update([box(0, 0)], np.stack([onehot(0)]))
    (track,) = t.update([box(0, 0)], np.stack([onehot(1)]))
    np.testing.assert_allclose(track.probs, [0.5, 0.5, 0.0])
    assert track.age == 2


def test_tracks_expire_after_max_missed():
    t = FaceTracker(smoothing=0.0, max_missed=2)
    t.update([box(0, 0)], np.stack([onehot(0)]))
    for _ in range(2):
        t.update([], np.zeros((0, 3)))
    assert len(t.tracks) == 1
    t.update([], np.zeros((0, 3)))
    assert t.tracks == []
    (track,) = t.update([box(0, 0)], np.stack([onehot(0)]))
    assert track.id == 2


def test_invalid_smoothing():
    with pytest.raises(ValueError):
        FaceTracker(smoothing=1.0)
