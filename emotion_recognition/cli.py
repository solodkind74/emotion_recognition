"""Run emotion recognition on a webcam, video file or network stream.

Examples:
    python -m emotion_recognition --source 0                      # webcam
    python -m emotion_recognition --source clip.mp4 --output out.mp4 --jsonl out.jsonl --no-display
    python -m emotion_recognition --source rtsp://camera/stream --model vit-int8
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from collections import Counter
from pathlib import Path

import cv2

from .draw import draw_results
from .models import DEFAULT_EMOTION_MODEL, EMOTION_MODELS
from .pipeline import EmotionRecognizer, EmotionStream

log = logging.getLogger("emotion_recognition")


class LatestFrameReader:
    """Reads a live source on a background thread and keeps only the newest frame.

    Without this, OpenCV buffers frames while the model is busy and the
    output drifts further and further behind real time.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._frame = None
        self._ended = False
        self._cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            ok, frame = self._cap.read()
            with self._cond:
                if not ok:
                    self._ended = True
                else:
                    self._frame = frame
                self._cond.notify()
            if not ok:
                return

    def read(self):
        with self._cond:
            self._cond.wait_for(lambda: self._frame is not None or self._ended)
            frame, self._frame = self._frame, None
        return frame is not None, frame


def open_source(source: str) -> tuple[cv2.VideoCapture, bool]:
    """Returns the capture and whether it is a live source."""
    if source.isdigit():
        return cv2.VideoCapture(int(source)), True
    live = "://" in source and not source.startswith("file://")
    return cv2.VideoCapture(source), live


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="webcam index, video file, or stream URL (default: 0)")
    p.add_argument("--model", default=DEFAULT_EMOTION_MODEL, choices=sorted(EMOTION_MODELS))
    p.add_argument("--emotion-model-path", help="local ONNX file to use instead of downloading")
    p.add_argument("--detector-path", help="local YuNet ONNX file to use instead of downloading")
    p.add_argument("--smoothing", type=float, default=0.7, help="EMA weight of past frames, 0 disables (default: 0.7)")
    p.add_argument("--det-threshold", type=float, default=0.8, help="face detection confidence (default: 0.8)")
    p.add_argument("--min-face", type=int, default=32, help="ignore faces smaller than this many pixels")
    p.add_argument("--no-align", action="store_true", help="skip rotating faces upright using the eyes")
    p.add_argument("--output", help="write the annotated video here (.mp4)")
    p.add_argument("--jsonl", help="write per-frame predictions here as JSON lines")
    p.add_argument("--no-display", action="store_true", help="don't open a preview window")
    p.add_argument("--max-frames", type=int, help="stop after this many processed frames")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    recognizer = EmotionRecognizer(
        model=args.model,
        emotion_model_path=args.emotion_model_path,
        detector_path=args.detector_path,
        detection_threshold=args.det_threshold,
        min_face_size=args.min_face,
        align=not args.no_align,
    )
    stream = EmotionStream(recognizer, smoothing=args.smoothing)

    cap, live = open_source(args.source)
    if not cap.isOpened():
        log.error("Could not open video source %r", args.source)
        return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    reader = LatestFrameReader(cap) if live else cap
    log.info("Model: %s | source: %s (%s)", args.model, args.source, "live" if live else "file")

    writer = None
    jsonl = open(args.jsonl, "w") if args.jsonl else None
    display = not args.no_display
    emotion_counts: Counter[str] = Counter()
    frames, fps, started = 0, 0.0, time.perf_counter()

    try:
        while args.max_frames is None or frames < args.max_frames:
            ok, frame = reader.read()
            if not ok:
                break
            t0 = time.perf_counter()
            results = stream.process(frame)
            dt = time.perf_counter() - t0
            fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / max(dt, 1e-6)
            timestamp = time.perf_counter() - started if live or not src_fps else frames / src_fps
            frames += 1
            emotion_counts.update(r.emotion for r in results)

            if jsonl:
                record = {"frame": frames - 1, "t": round(timestamp, 3), "faces": [r.to_dict() for r in results]}
                jsonl.write(json.dumps(record) + "\n")

            if args.output or display:
                hud = f"{args.model} | {fps:4.1f} FPS | {len(results)} face(s)"
                annotated = draw_results(frame, results, hud=hud)
                if args.output:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps or 15.0, (w, h))
                    writer.write(annotated)
                if display:
                    try:
                        cv2.imshow("Emotion recognition (q to quit)", annotated)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                    except cv2.error:
                        log.warning("No GUI available (opencv-python-headless?); continuing without a window")
                        display = False
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer:
            writer.release()
        if jsonl:
            jsonl.close()
        if display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    log.info("Processed %d frames in %.1fs (%.1f FPS end-to-end)", frames, elapsed, frames / max(elapsed, 1e-6))
    if emotion_counts:
        total = sum(emotion_counts.values())
        log.info("Emotion share over all face-frames: %s",
                 ", ".join(f"{e} {n / total:.0%}" for e, n in emotion_counts.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
