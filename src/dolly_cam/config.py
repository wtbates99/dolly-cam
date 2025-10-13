from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

LOGGER = logging.getLogger(__name__)


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
    output_dir = _as_path(raw.get("output_dir")) if raw.get("output_dir") else Path.cwd() / "videos"
    cfg = RecordingConfig(
        camera_device=raw.get("camera_device", RecordingConfig.camera_device),
        ffmpeg_path=raw.get("ffmpeg_path", RecordingConfig.ffmpeg_path),
        duration_seconds=int(raw.get("duration_seconds", RecordingConfig.duration_seconds)),
        interval_minutes=int(raw.get("interval_minutes", RecordingConfig.interval_minutes)),
        width=int(raw["width"]) if raw.get("width") else None,
        height=int(raw["height"]) if raw.get("height") else None,
        frame_rate=int(raw["frame_rate"]) if raw.get("frame_rate") else None,
        output_dir=output_dir,
        filename_prefix=str(raw.get("filename_prefix", RecordingConfig.filename_prefix)),
    )
    return cfg


def _make_drive_config(raw: dict[str, Any]) -> GoogleDriveConfig:
    cfg = GoogleDriveConfig(
        enabled=bool(raw.get("enabled", GoogleDriveConfig.enabled)),
        credentials_file=_as_path(raw.get("credentials_file")),
        folder_id=raw.get("folder_id"),
        chunk_size_mb=int(raw.get("chunk_size_mb", GoogleDriveConfig.chunk_size_mb)),
    )
    cfg.validate()
    return cfg


def _make_retention_config(raw: dict[str, Any]) -> RetentionConfig:
    return RetentionConfig(days=int(raw.get("days", RetentionConfig.days)))


def _make_touchscreen_config(raw: dict[str, Any]) -> TouchscreenConfig:
    return TouchscreenConfig(
        window_title=raw.get("window_title", TouchscreenConfig.window_title),
        full_screen=bool(raw.get("full_screen", TouchscreenConfig.full_screen)),
        button_font_size=int(raw.get("button_font_size", TouchscreenConfig.button_font_size)),
        status_font_size=int(raw.get("status_font_size", TouchscreenConfig.status_font_size)),
        poll_interval_ms=int(raw.get("poll_interval_ms", TouchscreenConfig.poll_interval_ms)),
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
