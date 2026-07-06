"""Face mask detection package.

Combines an OpenCV SSD face detector with a MobileNetV2-based mask
classifier to detect whether faces in images or video streams are
wearing a mask.
"""

from mask_detection.config import DetectorConfig
from mask_detection.face_detector import FaceDetector
from mask_detection.mask_classifier import MaskClassifier
from mask_detection.pipeline import Detection, MaskDetectionPipeline

__all__ = [
    "DetectorConfig",
    "Detection",
    "FaceDetector",
    "MaskClassifier",
    "MaskDetectionPipeline",
]

__version__ = "1.0.0"
