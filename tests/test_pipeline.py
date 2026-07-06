"""Tests that don't require loading the heavy model files."""

from pathlib import Path

import pytest

from mask_detection.config import DetectorConfig
from mask_detection.pipeline import Detection


class TestDetectorConfig:
    """Validation behaviour of DetectorConfig."""

    def test_default_paths_point_at_models_dir(self) -> None:
        config = DetectorConfig()
        assert config.prototxt_path.parent.name == "models"
        assert config.mask_model_path.name == "face_mask_detector.h5"

    def test_validate_passes_with_bundled_models(self) -> None:
        DetectorConfig().validate()

    def test_validate_raises_for_missing_file(self, tmp_path: Path) -> None:
        config = DetectorConfig(mask_model_path=tmp_path / "missing.h5")
        with pytest.raises(FileNotFoundError):
            config.validate()


class TestDetection:
    """Derived properties of the Detection result object."""

    def test_has_mask_true_when_mask_probability_wins(self) -> None:
        detection = Detection(
            box=(0, 0, 10, 10), mask_probability=0.9, no_mask_probability=0.1
        )
        assert detection.has_mask

    def test_has_mask_false_when_no_mask_probability_wins(self) -> None:
        detection = Detection(
            box=(0, 0, 10, 10), mask_probability=0.2, no_mask_probability=0.8
        )
        assert not detection.has_mask

    def test_label_includes_class_and_percentage(self) -> None:
        detection = Detection(
            box=(0, 0, 10, 10), mask_probability=0.95, no_mask_probability=0.05
        )
        assert detection.label == "Mask: 95.00%"

    def test_no_mask_label(self) -> None:
        detection = Detection(
            box=(0, 0, 10, 10), mask_probability=0.3, no_mask_probability=0.7
        )
        assert detection.label == "No Mask: 70.00%"
