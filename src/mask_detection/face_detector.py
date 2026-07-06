"""Face detection using OpenCV's DNN module with an SSD Caffe model."""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BoundingBox = tuple[int, int, int, int]  # (start_x, start_y, end_x, end_y)


class FaceDetector:
    """Detects faces in a frame with a pre-trained SSD Caffe network.

    Args:
        prototxt_path: Path to the Caffe ``deploy.prototxt`` definition.
        weights_path: Path to the ``.caffemodel`` weights file.
        confidence_threshold: Minimum detection confidence to keep a face.
        input_size: Square input size the network expects (default 300).
        blob_mean: Per-channel mean subtracted when building the blob.
    """

    def __init__(
        self,
        prototxt_path: str | Path,
        weights_path: str | Path,
        confidence_threshold: float = 0.8,
        input_size: int = 300,
        blob_mean: tuple[float, float, float] = (104.0, 177.0, 123.0),
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")

        logger.info("Loading face detector model from %s", weights_path)
        self._net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(weights_path))
        self._confidence_threshold = confidence_threshold
        self._input_size = input_size
        self._blob_mean = blob_mean

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence required to keep a detection."""
        return self._confidence_threshold

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        """Detect faces in a BGR frame.

        Args:
            frame: Image as a BGR ``numpy`` array of shape ``(H, W, 3)``.

        Returns:
            Bounding boxes ``(start_x, start_y, end_x, end_y)`` clipped to
            the frame, one per detected face.
        """
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=(self._input_size, self._input_size),
            mean=self._blob_mean,
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        boxes: list[BoundingBox] = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self._confidence_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array(
                [width, height, width, height]
            )
            start_x, start_y, end_x, end_y = box.astype(int)

            # Clip the box to the frame boundaries.
            start_x, start_y = max(0, start_x), max(0, start_y)
            end_x, end_y = min(width - 1, end_x), min(height - 1, end_y)

            if end_x > start_x and end_y > start_y:
                boxes.append((start_x, start_y, end_x, end_y))

        return boxes
