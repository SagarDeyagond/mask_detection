# Architecture

The system is a two-stage pipeline: locate faces first, then classify each
face crop as mask / no mask. Each stage is encapsulated in its own class so
either model can be swapped independently.

## Data flow

```
BGR frame
   │
   ▼
FaceDetector.detect()                (OpenCV DNN, SSD ResNet-10 Caffe)
   │  list of clipped bounding boxes
   ▼
MaskClassifier.predict()             (Keras MobileNetV2, batched)
   │  (mask, without_mask) probabilities per face
   ▼
MaskDetectionPipeline.process_frame()
   │  list[Detection]
   ▼
MaskDetectionPipeline.annotate()     (boxes + labels drawn on a copy)
```

## Classes

### `DetectorConfig` (`config.py`)
Frozen dataclass holding model paths, thresholds, and input sizes.
`validate()` fails fast with `FileNotFoundError` if a model file is missing.
Defaults point at the bundled files under `models/`.

### `FaceDetector` (`face_detector.py`)
Wraps `cv2.dnn.readNetFromCaffe`. `detect(frame)` builds a 300×300 blob
(mean subtraction `(104, 177, 123)`), runs a forward pass, filters
detections below the confidence threshold, and returns bounding boxes
clipped to the frame. Degenerate (empty) boxes are dropped.

### `MaskClassifier` (`mask_classifier.py`)
Wraps the fine-tuned Keras model. `preprocess()` converts BGR → RGB,
resizes to 224×224, and applies MobileNetV2 `preprocess_input`.
`predict()` batches all faces from a frame into a single forward pass and
returns an `(N, 2)` array of `(mask, without_mask)` probabilities.

### `Detection` (`pipeline.py`)
Immutable per-face result: bounding box plus both class probabilities,
with derived `has_mask` and display `label` properties.

### `MaskDetectionPipeline` (`pipeline.py`)
Composes the two models. `process_frame()` returns `list[Detection]`;
`annotate()` is a pure drawing helper that never mutates the input frame.

### `VideoMaskDetector` (`video.py`)
Owns the `cv2.VideoCapture` loop for a webcam index or video file:
resize (aspect-ratio preserving) → process → annotate → display, until the
stream ends or `q` is pressed. Resources are released in a `finally` block.

## Design notes

- **Composition over inheritance** — the pipeline composes a detector and a
  classifier rather than subclassing either.
- **Frozen dataclasses** for config and results keep state immutable and
  hashable.
- **Batched classification** — all faces in a frame are classified in one
  `model.predict` call instead of per-face calls.
- **No global state** — the original script's module-level model loading and
  inference loop are now instance-scoped, so multiple pipelines can coexist
  (e.g. in tests).
