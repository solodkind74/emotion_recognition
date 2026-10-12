"""Registry of the open models used by the pipeline and how to fetch them.

Weights are pulled from the Hugging Face Hub (cached by ``huggingface_hub``).
Where a model also has a public mirror, that URL is tried if the Hub is
unreachable, so the pipeline still works behind restrictive proxies.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

CACHE_DIR = Path(
    os.environ.get("EMOTION_RECOGNITION_CACHE", Path.home() / ".cache" / "emotion_recognition")
)

# Canonical label names shared by all classifiers (used for colours / logs).
EMOTIONS = ("angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelFile:
    """A single weights file, addressed on the HF Hub and/or by direct URL."""

    filename: str
    hf_repo: str | None = None
    hf_path: str | None = None
    mirrors: tuple[str, ...] = ()


@dataclass(frozen=True)
class EmotionModelSpec:
    name: str
    file: ModelFile
    labels: tuple[str, ...]
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    # FER2013-trained models only ever saw grayscale faces; feeding them grey
    # (replicated to 3 channels) keeps webcam crops in-distribution.
    grayscale: bool = False
    # Extra context around the detected face box, as a fraction of its size.
    crop_margin: float = 0.0
    source: str = ""
    license: str = ""
    notes: str = ""


YUNET = ModelFile(
    filename="face_detection_yunet_2023mar.onnx",
    hf_repo="opencv/face_detection_yunet",
    hf_path="face_detection_yunet_2023mar.onnx",
    mirrors=(
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
)

_VIT_LABELS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")

EMOTION_MODELS: dict[str, EmotionModelSpec] = {
    "vit": EmotionModelSpec(
        name="vit",
        file=ModelFile(
            filename="vit-face-expression.onnx",
            hf_repo="trpakov/vit-face-expression",
            hf_path="onnx/model.onnx",
        ),
        labels=_VIT_LABELS,
        input_size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        grayscale=True,
        crop_margin=0.1,
        source="https://huggingface.co/trpakov/vit-face-expression",
        license="Apache-2.0",
        notes="ViT-B/16 fine-tuned on FER2013 (7 classes, 71% test acc).",
    ),
    "vit-int8": EmotionModelSpec(
        name="vit-int8",
        file=ModelFile(
            filename="vit-face-expression-int8.onnx",
            hf_repo="onnx-community/vit-face-expression-ONNX",
            hf_path="onnx/model_int8.onnx",
        ),
        labels=_VIT_LABELS,
        input_size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        grayscale=True,
        crop_margin=0.1,
        source="https://huggingface.co/onnx-community/vit-face-expression-ONNX",
        license="Apache-2.0",
        notes="int8-quantized export of trpakov/vit-face-expression (~4x smaller, faster on CPU).",
    ),
    "hsemotion": EmotionModelSpec(
        name="hsemotion",
        file=ModelFile(
            filename="enet_b0_8_best_afew.onnx",
            mirrors=(
                "https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/main/"
                "models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx",
            ),
        ),
        labels=("angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
        input_size=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        grayscale=False,
        crop_margin=0.0,
        source="https://github.com/HSE-asavchenko/face-emotion-recognition",
        license="Apache-2.0",
        notes="HSEmotion EfficientNet-B0 trained on AffectNet (8 classes), fast on CPU.",
    ),
}

DEFAULT_EMOTION_MODEL = "vit"


def ensure_model(file: ModelFile) -> Path:
    """Return a local path to ``file``, downloading it if needed."""
    errors: list[str] = []

    if file.hf_repo:
        try:
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(repo_id=file.hf_repo, filename=file.hf_path or file.filename))
        except Exception as exc:  # network/proxy/auth issues: fall back to mirrors
            errors.append(f"hf://{file.hf_repo}/{file.hf_path}: {exc.__class__.__name__}: {exc}")
            log.warning("Hugging Face download failed for %s (%s)", file.hf_repo, exc.__class__.__name__)

    target = CACHE_DIR / file.filename
    if target.exists():
        return target

    for url in file.mirrors:
        try:
            _download(url, target)
            return target
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            log.warning("Download failed from %s (%s)", url, exc)

    raise RuntimeError(
        f"Could not obtain model '{file.filename}'. Tried:\n  " + "\n  ".join(errors or ["no sources"])
    )


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    tmp = tempfile.NamedTemporaryFile(dir=target.parent, delete=False)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, tmp:
            shutil.copyfileobj(resp, tmp)
        os.replace(tmp.name, target)
    except BaseException:
        Path(tmp.name).unlink(missing_ok=True)
        raise
