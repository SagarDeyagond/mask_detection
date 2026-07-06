"""End-to-end mask detection pipeline: face detection + classification."""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from mask_detection.config import DetectorConfig
from mask_detection.face_detector import BoundingBox, FaceDetector
from mask_detection.mask_classifier import MaskClassifier

logger = logging.getLogger(__name__)

_COLOR_MASK = (0, 255, 0)  # green (BGR)
_COLOR_NO_MASK = (0, 0, 255)  # red (BGR)


@dataclass(frozen=True)
class Detection:
    """A single face detection with its mask classification.

    Attributes:
        box: Face bounding box ``(start_x, start_y, end_x, end_y)``.
        mask_probability: Probability the face is wearing a mask.
        no_mask_probability: Probability the face is not wearing a mask.
    """

    box: BoundingBox
    mask_probability: float
    no_mask_probability: float

    @property
    def has_mask(self) -> bool:
        """Whether the face is more likely masked than unmasked."""
        return self.mask_probability > self.no_mask_probability

    @property
    def label(self) -> str:
        """Human-readable label including the winning probability."""
        name = "Mask" if self.has_mask else "No Mask"
        confidence = max(self.mask_probability, self.no_mask_probability)
        return f"{name}: {confidence * 100:.2f}%"


class MaskDetectionPipeline:
    """Detects faces in a frame and classifies each as mask / no mask.

    Args:
        config: Paths and thresholds for both models. Defaults to the
            bundled models under ``models/``.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._config = config or DetectorConfig()
        self._config.validate()

        self._face_detector = FaceDetector(
            prototxt_path=self._config.prototxt_path,
            weights_path=self._config.face_weights_path,
            confidence_threshold=self._config.face_confidence,
            input_size=self._config.face_input_size,
            blob_mean=self._config.blob_mean,
        )
        self._mask_classifier = MaskClassifier(
            model_path=self._config.mask_model_path,
            input_size=self._config.classifier_input_size,
            batch_size=self._config.batch_size,
        )

    def process_frame(self, frame: np.ndarray) -> list[Detection]:
        """Run detection and classification on a single BGR frame.

        Args:
            frame: Image as a BGR ``numpy`` array of shape ``(H, W, 3)``.

        Returns:
            One :class:`Detection` per face found in the frame.
        """
        boxes = self._face_detector.detect(frame)
        faces = [
            frame[start_y:end_y, start_x:end_x]
            for (start_x, start_y, end_x, end_y) in boxes
        ]
        predictions = self._mask_classifier.predict(faces)

        return [
            Detection(
                box=box,
                mask_probability=float(mask),
                no_mask_probability=float(no_mask),
            )
            for box, (mask, no_mask) in zip(boxes, predictions)
        ]

    @staticmethod
    def annotate(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of ``frame``."""
        annotated = frame.copy()
        for detection in detections:
            start_x, start_y, end_x, end_y = detection.box
            color = _COLOR_MASK if detection.has_mask else _COLOR_NO_MASK
            cv2.putText(
                annotated,
                detection.label,
                (start_x, max(0, start_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )
            cv2.rectangle(annotated, (start_x, start_y), (end_x, end_y), color, 2)
        return annotated
