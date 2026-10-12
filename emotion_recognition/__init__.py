"""Real-time facial emotion recognition on video streams using open models."""

from .models import DEFAULT_EMOTION_MODEL, EMOTION_MODELS
from .pipeline import EmotionRecognizer, EmotionStream, FaceResult

__all__ = ["DEFAULT_EMOTION_MODEL", "EMOTION_MODELS", "EmotionRecognizer", "EmotionStream", "FaceResult"]
__version__ = "0.1.0"
