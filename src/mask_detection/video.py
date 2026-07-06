"""Real-time mask detection on a video stream (webcam or file)."""

import logging

import cv2

from mask_detection.config import DetectorConfig
from mask_detection.pipeline import MaskDetectionPipeline

logger = logging.getLogger(__name__)

_QUIT_KEY = ord("q")


class VideoMaskDetector:
    """Runs the mask detection pipeline over a live video source.

    Args:
        pipeline: Pipeline used for per-frame detection. A default one is
            created if omitted.
        source: OpenCV video source — a camera index (``0`` for the
            default webcam) or a path to a video file.
        display_width: Frames are resized to this width (aspect ratio
            preserved) before processing and display.
        window_name: Title of the OpenCV preview window.
    """

    def __init__(
        self,
        pipeline: MaskDetectionPipeline | None = None,
        source: int | str = 0,
        display_width: int = 400,
        window_name: str = "Face Mask Detection",
    ) -> None:
        self._pipeline = pipeline or MaskDetectionPipeline(DetectorConfig())
        self._source = source
        self._display_width = display_width
        self._window_name = window_name

    def run(self) -> None:
        """Stream frames until the source ends or ``q`` is pressed."""
        logger.info("Starting video stream from source %r", self._source)
        capture = cv2.VideoCapture(self._source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {self._source!r}")

        try:
            while True:
                grabbed, frame = capture.read()
                if not grabbed:
                    logger.info("Video source exhausted, stopping.")
                    break

                frame = self._resize(frame)
                detections = self._pipeline.process_frame(frame)
                annotated = self._pipeline.annotate(frame, detections)

                cv2.imshow(self._window_name, annotated)
                if cv2.waitKey(1) & 0xFF == _QUIT_KEY:
                    logger.info("Quit key pressed, stopping.")
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

    def _resize(self, frame):
        """Resize a frame to the display width, keeping the aspect ratio."""
        height, width = frame.shape[:2]
        scale = self._display_width / width
        return cv2.resize(frame, (self._display_width, int(height * scale)))
