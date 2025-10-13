from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

LOGGER = logging.getLogger(__name__)
_DRIVE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{10,}$")


@dataclass(slots=True)
class RecordingConfig:
    camera_device: str = "/dev/video0"
    ffmpeg_path: str = "ffmpeg"
    duration_seconds: int = 300
    interval_minutes: int = 15
    width: int | None = None
    height: int | None = None
    frame_rate: int | None = None
    output_dir: Path = Path.cwd() / "videos"
    filename_prefix: str = "clip"


@dataclass(slots=True)
class GoogleDriveConfig:
    enabled: bool = False
    credentials_file: Path | None = None
    folder_id: str | None = None
    chunk_size_mb: int = 8

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.credentials_file:
            raise ValueError("google_drive.credentials_file must be set when Drive sync is enabled")
        if not self.folder_id:
            raise ValueError("google_drive.folder_id must be set when Drive sync is enabled")


@dataclass(slots=True)
class RetentionConfig:
    days: int = 3


@dataclass(slots=True)
class TouchscreenConfig:
    window_title: str = "Dolly Cam"
    full_screen: bool = True
    button_font_size: int = 28
    status_font_size: int = 18
    poll_interval_ms: int = 1000
    width: int = 800
    height: int = 480
    background_color: str = "#f3edff"
    accent_color: str = "#9a7bdc"
    accent_text_color: str = "#ffffff"
    secondary_color: str = "#d8c9ff"


@dataclass(slots=True)
class AppConfig:
    recording: RecordingConfig
    drive: GoogleDriveConfig
    retention: RetentionConfig
    touchscreen: TouchscreenConfig


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _load_table(data: dict[str, Any] | None) -> dict[str, Any]:
    return data or {}


def _make_recording_config(raw: dict[str, Any]) -> RecordingConfig:
    defaults = RecordingConfig()
    output_dir = _as_path(raw.get("output_dir")) if raw.get("output_dir") else defaults.output_dir
    cfg = RecordingConfig(
        camera_device=raw.get("camera_device", defaults.camera_device),
        ffmpeg_path=raw.get("ffmpeg_path", defaults.ffmpeg_path),
        duration_seconds=int(raw.get("duration_seconds", defaults.duration_seconds)),
        interval_minutes=int(raw.get("interval_minutes", defaults.interval_minutes)),
        width=int(raw["width"]) if raw.get("width") else None,
        height=int(raw["height"]) if raw.get("height") else None,
        frame_rate=int(raw["frame_rate"]) if raw.get("frame_rate") else None,
        output_dir=output_dir,
        filename_prefix=str(raw.get("filename_prefix", defaults.filename_prefix)),
    )
    return cfg


def _make_drive_config(raw: dict[str, Any]) -> GoogleDriveConfig:
    defaults = GoogleDriveConfig()
    cfg = GoogleDriveConfig(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        credentials_file=_as_path(raw.get("credentials_file")),
        folder_id=_normalize_drive_folder_id(raw.get("folder_id")),
        chunk_size_mb=int(raw.get("chunk_size_mb", defaults.chunk_size_mb)),
    )
    cfg.validate()
    return cfg


def _make_retention_config(raw: dict[str, Any]) -> RetentionConfig:
    defaults = RetentionConfig()
    return RetentionConfig(days=int(raw.get("days", defaults.days)))


def _make_touchscreen_config(raw: dict[str, Any]) -> TouchscreenConfig:
    defaults = TouchscreenConfig()
    return TouchscreenConfig(
        window_title=raw.get("window_title", defaults.window_title),
        full_screen=bool(raw.get("full_screen", defaults.full_screen)),
        button_font_size=int(raw.get("button_font_size", defaults.button_font_size)),
        status_font_size=int(raw.get("status_font_size", defaults.status_font_size)),
        poll_interval_ms=int(raw.get("poll_interval_ms", defaults.poll_interval_ms)),
        width=int(raw.get("width", defaults.width)),
        height=int(raw.get("height", defaults.height)),
        background_color=str(raw.get("background_color", defaults.background_color)),
        accent_color=str(raw.get("accent_color", defaults.accent_color)),
        accent_text_color=str(raw.get("accent_text_color", defaults.accent_text_color)),
        secondary_color=str(raw.get("secondary_color", defaults.secondary_color)),
    )


def load_config(path: Path) -> AppConfig:
    """Load the application configuration from a TOML file."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    LOGGER.debug("Loading configuration from %s", resolved)
    with resolved.open("rb") as f:
        data = tomllib.load(f)

    recording_cfg = _make_recording_config(_load_table(data.get("recording")))
    drive_cfg = _make_drive_config(_load_table(data.get("google_drive")))
    retention_cfg = _make_retention_config(_load_table(data.get("retention")))
    touchscreen_cfg = _make_touchscreen_config(_load_table(data.get("touchscreen")))

    # Ensure target directories exist early to avoid runtime surprises
    recording_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        recording=recording_cfg,
        drive=drive_cfg,
        retention=retention_cfg,
        touchscreen=touchscreen_cfg,
    )


def asdict(config: AppConfig) -> dict[str, Any]:
    """Expose the configuration as primitives for logging/debugging."""
    return dataclasses.asdict(config)


def _normalize_drive_folder_id(raw: str | None) -> str | None:
    """Accept either a bare folder ID or a shared URL, returning the sanitized ID."""
    if raw is None:
        return None

    value = raw.strip()
    if not value:
        return None

    candidates: list[str] = []

    if "://" in value:
        parsed = urlparse(value)
        query_id = parse_qs(parsed.query).get("id")
        if query_id:
            candidates.extend(query_id)
        path_segments = [segment for segment in parsed.path.split("/") if segment]
        candidates.extend(path_segments)
    else:
        candidates.append(value)

    expanded: list[str] = []
    for candidate in candidates:
        if "?" in candidate:
            expanded.append(candidate.split("?", 1)[0])
        if "#" in candidate:
            expanded.append(candidate.split("#", 1)[0])
        if "/" in candidate:
            expanded.extend(segment for segment in candidate.split("/") if segment)
        if "=" in candidate:
            expanded.append(candidate.split("=", 1)[-1])

    candidates.extend(expanded)

    for candidate in candidates:
        cleaned = candidate.strip().strip("/")
        if not cleaned:
            continue
        if cleaned in {"drive", "folders", "file", "u", "open", "d"}:
            continue
        if _DRIVE_ID_PATTERN.fullmatch(cleaned):
            if cleaned != value:
                LOGGER.debug("Normalized Google Drive folder ID '%s' -> '%s'", value, cleaned)
            return cleaned

    fallback = value
    if "?" in fallback:
        fallback = fallback.split("?", 1)[0]
    if "#" in fallback:
        fallback = fallback.split("#", 1)[0]
    fallback = fallback.strip().strip("/")
    if fallback != value:
        LOGGER.debug("Normalized Google Drive folder ID '%s' -> '%s'", value, fallback)
    return fallback or None
