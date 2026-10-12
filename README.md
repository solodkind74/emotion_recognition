# Emotion recognition on video streams (proof of concept)

Real-time facial emotion recognition for webcams, video files and network streams (RTSP/HTTP), built only from existing open models. Everything runs locally with ONNX Runtime on the CPU, and no PyTorch is needed.

```
frame ──► YuNet face detector ──► align & crop each face ──► emotion classifier (ONNX)
                                                                   │
            annotated video / JSON lines / browser overlay ◄── IoU tracker + EMA smoothing
```

1. **Detect** faces with [YuNet](https://huggingface.co/opencv/face_detection_yunet). Frames are downscaled to 640 px first, which also suits YuNet's preferred face size.
2. **Align**: each face is rotated so the eyes are level, cropped square with a little margin, and resized to the classifier's input size.
3. **Classify** all faces in one batched ONNX call. The output is softmax probabilities over the model's emotion labels.
4. **Track and smooth**: a light IoU tracker gives each face a stable ID. Per-face probabilities are averaged over time (EMA) so the label doesn't flicker from frame to frame.

## Models

| `--model` | Weights | Classes | Notes |
|---|---|---|---|
| `vit` (default) | [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression) (`onnx/model.onnx`, 343 MB) | angry, disgust, fear, happy, neutral, sad, surprise | ViT-B/16 fine-tuned on FER2013 (71% test accuracy), the most-used FER model on the Hub. Apache-2.0. |
| `vit-int8` | [onnx-community/vit-face-expression-ONNX](https://huggingface.co/onnx-community/vit-face-expression-ONNX) (`onnx/model_int8.onnx`, 87 MB) | same | int8-quantized export of the model above. Smaller and faster on CPU. |
| `hsemotion` | [HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition) `enet_b0_8_best_afew.onnx` (16 MB, from GitHub) | + contempt | EfficientNet-B0 trained on AffectNet (in-the-wild photos). About 40x less compute than the ViT; the best choice for weak CPUs. Apache-2.0. |

Face detector: [opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet) (MIT).

Weights download on first use. Hub files go through `huggingface_hub` and its cache, so `HF_TOKEN`, `HF_ENDPOINT` and `HF_HUB_OFFLINE` all work. If the Hub can't be reached, YuNet and HSEmotion fall back to their GitHub copies in `~/.cache/emotion_recognition` (set `EMOTION_RECOGNITION_CACHE` to change it). To run fully offline, pass local files with `--emotion-model-path` / `--detector-path`.

The ViT models were trained on FER2013, which is all grayscale. Their face crops are therefore converted to grey before inference to stay in-distribution (`grayscale=True` in `emotion_recognition/models.py`).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[web]"

# Webcam with a preview window (press q to quit)
python -m emotion_recognition --source 0

# Video file -> annotated video + per-frame JSON, no window
python -m emotion_recognition --source clip.mp4 --output annotated.mp4 --jsonl emotions.jsonl --no-display

# IP camera / network stream; a faster model for CPU-only machines
python -m emotion_recognition --source rtsp://user:pass@camera/stream --model hsemotion
```

For live sources, a background thread always hands the model the **newest** frame. A slow model therefore lowers the FPS instead of building up a growing delay.

Useful flags: `--model {vit,vit-int8,hsemotion}`, `--smoothing 0.7` (0 disables it), `--det-threshold 0.8`, `--min-face 32`, `--no-align`, `--max-frames N`. See `--help`.

On a headless server, install `opencv-python-headless` in place of `opencv-python` and use `--no-display`, or use the web demo.

### Web demo (browser webcam)

```bash
python -m emotion_recognition.server --model vit      # then open http://localhost:8000
```

The page streams your webcam, or a local video file, to the server over a WebSocket and draws the results live:

- boxes with labels on the video
- probability bars for each face
- a timeline of the main face's emotion
- FPS and latency

Frames are sent one at a time, so the stream adjusts itself to the server's speed. Browsers only allow camera access on `localhost` or over HTTPS.

There is also a stateless endpoint for single images:

```bash
curl --data-binary @photo.jpg http://localhost:8000/api/analyze
```

### Output format (`--jsonl`, WebSocket, `/api/analyze`)

```json
{"frame": 42, "t": 1.68, "faces": [
  {"track_id": 1, "box": [228, 86, 79, 109], "detection_score": 0.937,
   "emotion": "happy", "confidence": 0.948,
   "probabilities": {"angry": 0.01, "disgust": 0.0, "fear": 0.0, "happy": 0.948, "neutral": 0.03, "sad": 0.01, "surprise": 0.002}}]}
```

`box` is `x, y, width, height` in frame pixels. `probabilities` are the smoothed values.

### Python API

```python
import cv2
from emotion_recognition import EmotionRecognizer, EmotionStream

recognizer = EmotionRecognizer(model="vit")      # loads the models once and can be shared
stream = EmotionStream(recognizer, smoothing=0.7) # one per video stream (tracking state)

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    for face in stream.process(frame):
        print(face.track_id, face.emotion, face.confidence)
```

## Measured so far

Results on a 4-vCPU cloud container with no GPU, using `hsemotion`:

| Scenario | Throughput |
|---|---|
| CLI, 1280×720 file, 3 faces per frame | **18 FPS** end-to-end |
| Web demo, 640 px frames, 2 faces | **13–15 FPS**, about 45 ms server latency |

On sample photos it labels a broad smile as happy (99–100%) and a grimace as angry (95%). Relaxed faces come out neutral. Track IDs stayed stable through pans, zooms and ±6° rotations.

The `vit` / `vit-int8` models have not been run in that environment yet, because its network policy blocks `huggingface.co`. Their configuration was taken from the repos' `config.json` / `preprocessor_config.json` via the Hub. Expect them to be noticeably slower than `hsemotion` on CPU (ViT-B/16 vs. EfficientNet-B0). To check them on a machine with Hub access:

```bash
EMOTION_E2E=1 pytest tests/test_models_e2e.py
```

## Tests

```bash
pip install -e ".[dev]"
pytest                                          # unit tests, no downloads needed
EMOTION_E2E=1 pytest tests/test_models_e2e.py   # downloads and runs the real models
```

## Limitations and next steps

- **What it measures.** These models classify *visible facial expressions* into fixed categories. That is not the same as how someone feels. Accuracy drops with occlusion, profile views, poor light and low resolution, and FER datasets have known demographic biases. Before any real deployment, validate on your own data. Also check local rules: the EU AI Act, for example, prohibits emotion recognition in workplaces and schools.
- **Tracker.** IoU-only matching can carry a track across a hard scene cut. For multi-camera or crowded scenes, add appearance embeddings (e.g. a face-recognition model) to the matching.
- **Speed.** Options for faster inference:
  - Install `onnxruntime-gpu` and CUDA is used automatically.
  - Classify every Nth frame and let the smoothing fill the gaps.
  - Use `vit-int8` or `hsemotion`.
- **Better models.** Swap in a stronger classifier by adding an `EmotionModelSpec` to `emotion_recognition/models.py`: point it at an ONNX file and set its labels, input size, normalization and crop margin. Candidates include larger HSEmotion variants (`enet_b2_8`) and valence/arousal models (`enet_b0_8_va_mtl`) for a continuous mood signal.
- **Multimodal.** Voice and transcript emotion models on the Hub, such as wav2vec2 speech-emotion models, could be fused with the face stream.

## Layout

```
emotion_recognition/
  models.py      model registry, HF Hub download with mirror fallback
  detector.py    YuNet face detection
  classifier.py  face alignment/cropping and ONNX emotion classification
  tracker.py     IoU tracker + EMA smoothing
  pipeline.py    EmotionRecognizer (shared models) and EmotionStream (per-stream state)
  draw.py        overlay rendering
  cli.py         webcam / file / RTSP runner
  server.py      FastAPI + WebSocket web demo (static/index.html)
tests/           unit tests (fakes, no downloads) and opt-in real-model tests
```
