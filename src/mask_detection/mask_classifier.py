"""Mask / no-mask classification of face crops with a Keras model."""

import logging
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)


class MaskClassifier:
    """Classifies face crops as "mask" or "no mask".

    Wraps the fine-tuned MobileNetV2 model produced by the training
    notebook. The model outputs two softmax probabilities in the order
    ``(mask, without_mask)``.

    Args:
        model_path: Path to the serialized Keras ``.h5`` model.
        input_size: Square input size the classifier expects (default 224).
        batch_size: Batch size used when predicting on multiple faces.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 224,
        batch_size: int = 32,
    ) -> None:
        logger.info("Loading face mask classifier from %s", model_path)
        self._model = load_model(str(model_path))
        self._input_size = input_size
        self._batch_size = batch_size

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """Convert a BGR face crop to the classifier's input format."""
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (self._input_size, self._input_size))
        return preprocess_input(face.astype("float32"))

    def predict(self, faces_bgr: list[np.ndarray]) -> np.ndarray:
        """Predict mask probabilities for a batch of BGR face crops.

        Args:
            faces_bgr: Face crops as BGR ``numpy`` arrays of any size.

        Returns:
            Array of shape ``(len(faces_bgr), 2)`` with per-face
            ``(mask, without_mask)`` probabilities. Empty input yields an
            empty ``(0, 2)`` array.
        """
        if not faces_bgr:
            return np.empty((0, 2), dtype="float32")

        batch = np.array(
            [self.preprocess(face) for face in faces_bgr], dtype="float32"
        )
        return self._model.predict(batch, batch_size=self._batch_size)
