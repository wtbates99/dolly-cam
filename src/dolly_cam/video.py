from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2

LOGGER = logging.getLogger(__name__)


class CameraFeed:
    """Continuously captures frames for preview and optional recording."""

    def __init__(
        self,
        device: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        frame_rate: Optional[int] = None,
    ) -> None:
        self._device = device
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = frame_rate

        self._lock = threading.Lock()
        self._writer_lock = threading.Lock()
        self._latest_frame: Optional["cv2.Mat"] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_available = threading.Event()

        self._recording_event = threading.Event()
        self._recording_writer: Optional["cv2.VideoWriter"] = None
        self._recording_size: Optional[Tuple[int, int]] = None

        self._capture_lock = threading.Lock()
        self._capture: Optional["cv2.VideoCapture"] = None

    # Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        LOGGER.debug("Starting camera feed for device %s", self._device)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="camera-feed", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        LOGGER.debug("Stopping camera feed")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        with self._capture_lock:
            if self._capture:
                self._capture.release()
                self._capture = None
        self._frame_available.clear()
        self._thread = None

    # Recording ---------------------------------------------------------

    def begin_recording(self, path: Path, frame_rate: Optional[int] = None) -> None:
        frame = self._wait_for_frame(timeout=5.0)
        if frame is None:
            raise RuntimeError("Camera not ready; unable to start recording")

        fps = frame_rate or self._requested_fps or self._get_capture_fps() or 30
        if fps <= 0:
            fps = 30

        height, width = frame.shape[:2]
        width -= width % 2
        height -= height % 2
        if width <= 0 or height <= 0:
            raise RuntimeError("Invalid frame size for recording")

        with self._writer_lock:
            if self._recording_writer:
                LOGGER.debug("Recording already active; restarting with new writer")
                self._recording_writer.release()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Failed to open video writer")
            self._recording_writer = writer
            self._recording_size = (width, height)
            self._recording_event.set()
        LOGGER.debug("Recording started at %s", path)

    def end_recording(self) -> None:
        with self._writer_lock:
            writer = self._recording_writer
            self._recording_writer = None
            self._recording_event.clear()
            self._recording_size = None
        if writer:
            writer.release()
            LOGGER.debug("Recording writer released")

    def is_recording(self) -> bool:
        return self._recording_event.is_set()

    # Preview -----------------------------------------------------------

    def get_latest_frame(self) -> Optional["cv2.Mat"]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _wait_for_frame(self, timeout: float) -> Optional["cv2.Mat"]:
        if not self._frame_available.wait(timeout=timeout):
            return None
        return self.get_latest_frame()

    # Internal ----------------------------------------------------------

    def _run(self) -> None:
        retry_delay = 1.5
        while not self._stop_event.is_set():
            if not self._ensure_capture():
                time.sleep(retry_delay)
                continue

            capture = self._capture
            if capture is None:
                time.sleep(retry_delay)
                continue

            ok, frame = capture.read()
            if not ok or frame is None:
                LOGGER.warning("Camera read failed; retrying")
                time.sleep(0.1)
                continue

            with self._lock:
                self._latest_frame = frame
            self._frame_available.set()

            if self._recording_event.is_set():
                with self._writer_lock:
                    writer = self._recording_writer
                    target_size = self._recording_size
                if writer:
                    try:
                        frame_to_write = frame
                        if target_size and (frame.shape[1], frame.shape[0]) != target_size:
                            frame_to_write = cv2.resize(frame, target_size)
                        writer.write(frame_to_write)
                    except Exception as exc:  # pragma: no cover - logging for runtime failures
                        LOGGER.exception("Failed to write frame: %s", exc)
                        self.end_recording()

        LOGGER.debug("Camera feed loop terminated")

    def _ensure_capture(self) -> bool:
        with self._capture_lock:
            if self._capture and self._capture.isOpened():
                return True
            LOGGER.debug("Opening video capture for %s", self._device)
            capture = cv2.VideoCapture(self._device)
            if self._requested_width:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._requested_width))
            if self._requested_height:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._requested_height))
            if self._requested_fps:
                capture.set(cv2.CAP_PROP_FPS, float(self._requested_fps))
            if not capture.isOpened():
                LOGGER.error("Unable to open camera device %s", self._device)
                capture.release()
                self._capture = None
                return False
            self._capture = capture
            return True

    def _get_capture_fps(self) -> Optional[int]:
        with self._capture_lock:
            if not self._capture:
                return None
            fps = self._capture.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            return int(fps)
        return None
