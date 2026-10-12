"""Browser demo: the page streams webcam frames over a WebSocket, the server
returns faces and emotions, and the page draws the overlay.

    python -m emotion_recognition.server --model vit
    # open http://localhost:8000 (browsers only allow camera access on localhost or HTTPS)
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from .models import DEFAULT_EMOTION_MODEL, EMOTION_MODELS
from .pipeline import EmotionRecognizer, EmotionStream

STATIC = Path(__file__).parent / "static"
MAX_FRAME_BYTES = 5 * 1024 * 1024

log = logging.getLogger("emotion_recognition.server")


def decode_image(data: bytes) -> np.ndarray | None:
    if not data or len(data) > MAX_FRAME_BYTES:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def create_app(recognizer: EmotionRecognizer, smoothing: float = 0.7) -> FastAPI:
    app = FastAPI(title="Emotion recognition PoC")
    spec = recognizer.classifier.spec

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info() -> dict:
        return {"model": spec.name, "labels": list(spec.labels), "source": spec.source, "notes": spec.notes}

    @app.post("/api/analyze")
    async def analyze(request: Request) -> dict:
        """Single image (raw JPEG/PNG body) -> faces; no temporal smoothing."""
        frame = decode_image(await request.body())
        if frame is None:
            raise HTTPException(400, "body must be a JPEG or PNG image under 5 MB")
        results = await run_in_threadpool(EmotionStream(recognizer, smoothing=0.0).process, frame)
        return {"width": frame.shape[1], "height": frame.shape[0], "faces": [r.to_dict() for r in results]}

    @app.websocket("/ws")
    async def ws_stream(ws: WebSocket) -> None:
        # One tracker per connection so each viewer gets their own smoothing.
        await ws.accept()
        stream = EmotionStream(recognizer, smoothing=smoothing)
        try:
            while True:
                frame = decode_image(await ws.receive_bytes())
                if frame is None:
                    await ws.send_json({"error": "could not decode frame"})
                    continue
                t0 = time.perf_counter()
                results = await run_in_threadpool(stream.process, frame)
                await ws.send_json({
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "faces": [r.to_dict() for r in results],
                })
        except WebSocketDisconnect:
            pass

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Web demo for video-stream emotion recognition")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default=DEFAULT_EMOTION_MODEL, choices=sorted(EMOTION_MODELS))
    p.add_argument("--emotion-model-path")
    p.add_argument("--detector-path")
    p.add_argument("--smoothing", type=float, default=0.7)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    recognizer = EmotionRecognizer(
        model=args.model, emotion_model_path=args.emotion_model_path, detector_path=args.detector_path
    )
    log.info("Open http://%s:%d in your browser", "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host, args.port)
    uvicorn.run(create_app(recognizer, args.smoothing), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
