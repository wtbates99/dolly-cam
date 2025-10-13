from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .config import RecordingConfig, RetentionConfig
from .uploader import DriveUploader
from .video import CameraFeed

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RecorderStatus:
    running: bool
    recording_now: bool
    next_run: Optional[datetime]
    last_clip: Optional[Path]
    last_error: Optional[str]


class RecordingController:
    """Manages timed clip recording and cleanup lifecycle."""

    def __init__(
        self,
        recording_cfg: RecordingConfig,
        retention_cfg: RetentionConfig,
        uploader: DriveUploader,
    ) -> None:
        self._cfg = recording_cfg
        self._retention = retention_cfg
        self._uploader = uploader

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = threading.Event()
        self._recording_flag = threading.Event()
        self._lock = threading.Lock()

        self._next_run: Optional[datetime] = None
        self._last_clip: Optional[Path] = None
        self._last_error: Optional[str] = None

        self._camera = CameraFeed(
            recording_cfg.camera_device,
            recording_cfg.width,
            recording_cfg.height,
            recording_cfg.frame_rate,
        )
        self._camera.start()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                LOGGER.debug("Recording already running")
                return
            LOGGER.info("Starting recording scheduler")
            self._stop_event.clear()
            self._camera.start()
            self._thread = threading.Thread(target=self._run, name="recording-loop", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        thread: Optional[threading.Thread]
        with self._lock:
            thread = self._thread
            if not thread:
                return
            LOGGER.info("Stopping recording scheduler")
            self._stop_event.set()
            self._recording_flag.clear()
        if thread:
            thread.join(timeout=timeout)
        self._camera.end_recording()
        with self._lock:
            self._thread = None
            self._next_run = None
        self._is_running.clear()

    def is_running(self) -> bool:
        return self._is_running.is_set()

    def is_recording_now(self) -> bool:
        return self._recording_flag.is_set()

    def get_status(self) -> RecorderStatus:
        return RecorderStatus(
            running=self._is_running.is_set(),
            recording_now=self._recording_flag.is_set(),
            next_run=self._next_run,
            last_clip=self._last_clip,
            last_error=self._last_error,
        )

    def get_preview_frame(self):  # pragma: no cover - runtime integration
        return self._camera.get_latest_frame()

    def get_recording_directory(self) -> Path:
        return self._cfg.output_dir

    def shutdown(self) -> None:
        self.stop()
        self._uploader.shutdown()
        self._camera.end_recording()
        self._camera.stop()

    # Internal helpers -----------------------------------------------------

    def _run(self) -> None:
        self._is_running.set()
        self._next_run = datetime.now()
        LOGGER.debug("Recording loop entered")

        while not self._stop_event.is_set():
            wait_seconds = self._seconds_until_next_run()
            if wait_seconds > 0:
                LOGGER.debug("Waiting %.2f seconds for next recording", wait_seconds)
                if self._stop_event.wait(timeout=wait_seconds):
                    break

            start_time = datetime.now()
            try:
                clip_path = self._record_once(start_time)
                if clip_path:
                    self._last_clip = clip_path
                    self._uploader.enqueue(clip_path)
            except Exception as exc:  # pragma: no cover - runtime safety
                LOGGER.exception("Recording failed: %s", exc)
                self._last_error = str(exc)
            finally:
                self._schedule_next_run(start_time)
                try:
                    remove_old_local_files(self._cfg.output_dir, self._retention.days)
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("Local cleanup failed: %s", exc)

        self._is_running.clear()
        self._next_run = None
        LOGGER.debug("Recording loop exited")

    def _schedule_next_run(self, start_time: datetime) -> None:
        interval = timedelta(minutes=self._cfg.interval_minutes)
        if interval.total_seconds() <= 0:
            LOGGER.warning("Recording interval is non-positive; defaulting to immediate re-run")
            self._next_run = datetime.now()
            return
        self._next_run = start_time + interval

    def _seconds_until_next_run(self) -> float:
        if not self._next_run:
            return 0.0
        delta = (self._next_run - datetime.now()).total_seconds()
        return max(delta, 0.0)

    def _record_once(self, start_time: datetime) -> Optional[Path]:
        output_path = self._build_output_path(start_time)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Recording clip to %s", output_path)
        self._recording_flag.set()
        try:
            self._camera.begin_recording(output_path, self._cfg.frame_rate)
        except Exception as exc:
            self._last_error = str(exc)
            self._recording_flag.clear()
            raise

        try:
            end_at = start_time + timedelta(seconds=self._cfg.duration_seconds)
            while datetime.now() < end_at:
                remaining = (end_at - datetime.now()).total_seconds()
                if remaining <= 0:
                    break
                if self._stop_event.wait(timeout=min(0.5, remaining)):
                    LOGGER.info("Recording interrupted before scheduled end")
                    break
            LOGGER.info("Recording complete: %s", output_path.name)
            self._last_error = None
            return output_path
        finally:
            self._camera.end_recording()
            self._recording_flag.clear()

    def _build_output_path(self, start_time: datetime) -> Path:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self._cfg.filename_prefix}_{timestamp}.mp4"
        return self._cfg.output_dir / filename


def remove_old_local_files(directory: Path, retention_days: int) -> None:
    cutoff = datetime.now() - timedelta(days=retention_days)
    removed = 0
    for path in directory.glob("*.mp4"):
        try:
            if path.stat().st_mtime < cutoff.timestamp():
                LOGGER.info("Removing local clip %s", path.name)
                path.unlink(missing_ok=True)
                removed += 1
        except FileNotFoundError:
            continue
    if removed:
        LOGGER.debug("Removed %s old local file(s)", removed)
