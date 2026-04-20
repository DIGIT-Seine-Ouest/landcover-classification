from flair_inference.model import FlairModel
from flair_inference.classes import FLAIR_CLASS_NAMES, FLAIR_CLASS_COLORS, INPUT_SIZE
from flair_inference.visualization import colorize, apply_overlay, mask_from_classes, class_stats
from flair_inference.download import download_model

__version__ = "0.1.0"
__all__ = [
    "FlairModel",
    "download_model",
    "FLAIR_CLASS_NAMES", "FLAIR_CLASS_COLORS", "INPUT_SIZE",
    "colorize", "apply_overlay", "mask_from_classes", "class_stats",
]
