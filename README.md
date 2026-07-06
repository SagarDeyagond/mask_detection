# Face Mask Detection

Real-time face mask detection combining two models:

1. **Face detection** — an OpenCV DNN SSD (ResNet-10 backbone, Caffe) locates faces in each frame.
2. **Mask classification** — a fine-tuned MobileNetV2 (Keras/TensorFlow) classifies each face crop as **Mask** / **No Mask**.

Detected faces are drawn with a bounding box and confidence label — green for masked, red for unmasked.

## Project structure

```
06_Mask_Detection/
├── src/mask_detection/          # Installable Python package
│   ├── config.py                # DetectorConfig / TrainingConfig dataclasses
│   ├── face_detector.py         # FaceDetector (OpenCV SSD)
│   ├── mask_classifier.py       # MaskClassifier (Keras MobileNetV2)
│   ├── pipeline.py              # MaskDetectionPipeline + Detection result
│   └── video.py                 # VideoMaskDetector (webcam / video file loop)
├── scripts/
│   └── run_video_inference.py   # CLI entry point
├── models/                      # Pre-trained model files (~21 MB total)
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── face_mask_detector.h5
├── notebooks/
│   ├── face_mask_training.ipynb     # Training workflow (Colab)
│   └── video_mask_inference.ipynb   # Notebook version of inference
├── docs/
│   ├── ARCHITECTURE.md          # Design and class responsibilities
│   └── TRAINING.md              # How the classifier was trained
├── tests/
│   └── test_pipeline.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .            # inference only
pip install -e ".[train]"   # + training/notebook dependencies
pip install -e ".[dev]"     # + pytest, ruff
```

## Usage

### Command line

```bash
# Default webcam
python scripts/run_video_inference.py

# A video file, with a lower face-detection threshold
python scripts/run_video_inference.py --source demo.mp4 --confidence 0.5
```

Press `q` in the preview window to quit.

### As a library

```python
import cv2
from mask_detection import DetectorConfig, MaskDetectionPipeline

pipeline = MaskDetectionPipeline(DetectorConfig())
frame = cv2.imread("photo.jpg")

for detection in pipeline.process_frame(frame):
    print(detection.box, detection.label)

annotated = pipeline.annotate(frame, pipeline.process_frame(frame))
cv2.imwrite("annotated.jpg", annotated)
```

## Training

The classifier was trained by transfer learning on MobileNetV2 (ImageNet weights, frozen base) with a small dense head, using augmented images of masked/unmasked faces. See [docs/TRAINING.md](docs/TRAINING.md) and [`notebooks/face_mask_training.ipynb`](notebooks/face_mask_training.ipynb). The dataset itself is not tracked in this repository.

## Notes

- Model files (~21 MB) are committed directly. If the repository grows, consider migrating them to [Git LFS](https://git-lfs.com/).
- The SSD face detector files (`deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`) are the standard OpenCV face-detection models.

## Testing & linting

```bash
pytest
ruff check .
```

## License

MIT
