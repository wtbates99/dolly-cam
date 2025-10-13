from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from .config import AppConfig, asdict, load_config
from .gui import DollyCamApp
from .recording import RecordingController
from .uploader import DriveUploader

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dolly Cam controller")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to configuration TOML file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def build_app(config: AppConfig) -> tuple[DollyCamApp, RecordingController]:
    uploader = DriveUploader(config.drive, config.retention, config.recording.output_dir)
    controller = RecordingController(config.recording, config.retention, uploader)
    app = DollyCamApp(controller, config.touchscreen)
    return app, controller


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    try:
        config = load_config(args.config)
    except Exception as exc:
        LOGGER.error("Failed to load config: %s", exc)
        return 1

    LOGGER.debug("Loaded configuration: %s", asdict(config))

    app, controller = build_app(config)

    def _shutdown(*_: object) -> None:
        LOGGER.info("Received stop signal")
        controller.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        app.run()
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        controller.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
