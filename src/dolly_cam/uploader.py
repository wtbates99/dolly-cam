from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from .config import GoogleDriveConfig, RetentionConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class UploadJob:
    path: Path


class DriveUploader:
    """Background worker that uploads clips to Google Drive and prunes old ones."""

    def __init__(self, drive_cfg: GoogleDriveConfig, retention_cfg: RetentionConfig) -> None:
        self._cfg = drive_cfg
        self._retention = retention_cfg
        self._queue: Queue[UploadJob | None] = Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._service = None
        self._chunk_size = drive_cfg.chunk_size_mb * 1024 * 1024

        if self._cfg.enabled:
            self._init_service()
            self._worker = threading.Thread(target=self._run, name="drive-uploader", daemon=True)
            self._worker.start()

    def _init_service(self) -> None:
        if not self._cfg.credentials_file:
            raise ValueError("Google Drive credentials file must be configured")
        scopes = [
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive.metadata",
        ]
        credentials = service_account.Credentials.from_service_account_file(
            str(self._cfg.credentials_file), scopes=scopes
        )
        self._service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        LOGGER.info("Google Drive uploader initialized for folder %s", self._cfg.folder_id)

    def enqueue(self, path: Path) -> None:
        if not self._cfg.enabled:
            return
        resolved = path.resolve()
        if not resolved.exists():
            LOGGER.warning("Skipping upload; file missing: %s", resolved)
            return
        LOGGER.info("Queueing %s for Drive upload", resolved.name)
        self._queue.put(UploadJob(path=resolved))

    def request_cleanup(self) -> None:
        if not self._cfg.enabled:
            return
        LOGGER.debug("Queueing Drive cleanup job")
        self._queue.put(None)

    def shutdown(self, timeout: float = 10.0) -> None:
        if not self._cfg.enabled:
            return
        LOGGER.info("Stopping Google Drive uploader")
        self._stop_requested.set()
        # Place sentinel to unblock worker waiting on queue
        self._queue.put(None)
        if self._worker:
            self._worker.join(timeout=timeout)

    def _run(self) -> None:
        assert self._service is not None  # nosec - guarded in __init__
        while not self._stop_requested.is_set():
            try:
                job = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if job is None:
                try:
                    if self._stop_requested.is_set():
                        LOGGER.debug("Drive uploader exiting")
                        return
                    self._perform_cleanup()
                finally:
                    self._queue.task_done()
                continue

            try:
                self._upload(job.path)
                # Run retention cleanup opportunistically after uploads
                self._perform_cleanup()
            except Exception as exc:  # pragma: no cover - defensive logging for runtime issues
                LOGGER.exception("Failed to upload %s: %s", job.path, exc)
                # Brief backoff to avoid busy loop on repeated failures
                time.sleep(2)
            finally:
                self._queue.task_done()

    def _upload(self, path: Path) -> None:
        assert self._service is not None
        metadata = {"name": path.name, "parents": [self._cfg.folder_id] if self._cfg.folder_id else []}
        media = MediaFileUpload(str(path), chunksize=self._chunk_size, resumable=True)
        request = self._service.files().create(body=metadata, media_body=media, fields="id")

        LOGGER.info("Uploading %s to Google Drive", path.name)
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                LOGGER.debug("Upload progress %s: %.2f%%", path.name, status.progress() * 100)
        file_id = response.get("id") if isinstance(response, dict) else None
        LOGGER.info("Upload complete for %s (id=%s)", path.name, file_id)

    def _perform_cleanup(self) -> None:
        assert self._service is not None
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention.days)
        query = ["trashed = false"]
        if self._cfg.folder_id:
            query.append(f"'{self._cfg.folder_id}' in parents")
        query_str = " and ".join(query)

        page_token: Optional[str] = None
        removed = 0
        try:
            while True:
                response = (
                    self._service.files()
                    .list(
                        q=query_str,
                        orderBy="createdTime",
                        fields="nextPageToken, files(id, name, createdTime)",
                        pageToken=page_token,
                    )
                    .execute()
                )
                for item in response.get("files", []):
                    created = _parse_timestamp(item.get("createdTime"))
                    if created and created < cutoff:
                        LOGGER.info("Removing old Drive file %s (%s)", item.get("name"), item.get("id"))
                        self._service.files().delete(fileId=item.get("id")).execute()
                        removed += 1
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        except HttpError as exc:
            LOGGER.warning("Drive cleanup failed: %s", exc)
        if removed:
            LOGGER.info("Drive cleanup removed %s file(s)", removed)


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        # Drive timestamps are RFC3339; replace Z for fromisoformat compatibility
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        LOGGER.debug("Unable to parse timestamp %s", value)
        return None
