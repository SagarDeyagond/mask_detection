"""Configuration objects for the mask detection pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

# Repository root (two levels up from this file: src/mask_detection/).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODELS_DIR = _PROJECT_ROOT / "models"


@dataclass(frozen=True)
class DetectorConfig:
    """Paths and thresholds used by the detection pipeline.

    Attributes:
        prototxt_path: Caffe network definition for the SSD face detector.
        face_weights_path: Pre-trained Caffe weights for the face detector.
        mask_model_path: Keras ``.h5`` model for the mask classifier.
        face_confidence: Minimum confidence to keep a face detection.
        face_input_size: Input size (square) expected by the face detector.
        classifier_input_size: Input size (square) expected by the classifier.
        blob_mean: Per-channel mean subtracted when building the input blob.
        batch_size: Batch size used for mask classification.
    """

    prototxt_path: Path = DEFAULT_MODELS_DIR / "deploy.prototxt"
    face_weights_path: Path = (
        DEFAULT_MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
    )
    mask_model_path: Path = DEFAULT_MODELS_DIR / "face_mask_detector.h5"
    face_confidence: float = 0.8
    face_input_size: int = 300
    classifier_input_size: int = 224
    blob_mean: tuple[float, float, float] = (104.0, 177.0, 123.0)
    batch_size: int = 32

    def validate(self) -> None:
        """Raise ``FileNotFoundError`` if any model file is missing."""
        for path in (
            self.prototxt_path,
            self.face_weights_path,
            self.mask_model_path,
        ):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Model file not found: {path}")


@dataclass(frozen=True)
class TrainingConfig:
    """Hyper-parameters for training the mask classifier.

    Mirrors the values used in ``notebooks/face_mask_training.ipynb``.
    """

    initial_learning_rate: float = 1e-4
    epochs: int = 20
    batch_size: int = 32
    test_split: float = 0.2
    input_size: int = 224
    augmentation: dict = field(
        default_factory=lambda: {
            "rotation_range": 20,
            "zoom_range": 0.15,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "shear_range": 0.15,
            "horizontal_flip": True,
            "fill_mode": "nearest",
        }
    )
