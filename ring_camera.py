#!/usr/bin/env python3
"""Raspberry Pi home camera: motion clips, live stream, and lightweight dashboard APIs."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Deque, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from storage_manager import StorageManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ring_camera")


@dataclass
class MotionConfig:
    delta_thresh: int
    min_motion_area: int
    background_alpha: float


@dataclass
class ClipConfig:
    cooldown_seconds: float
    pre_motion_seconds: float
    max_clip_seconds: float
    min_clip_seconds: float


class RingCamera:
    """Captures webcam frames, records motion clips, and serves MJPEG/dashboard data."""

    def __init__(
        self,
        storage_path: str,
        video_device: int,
        resolution: Tuple[int, int],
        fps: int,
        motion_cfg: MotionConfig,
        clip_cfg: ClipConfig,
        jpeg_quality: int,
        stream_fps: float,
        max_storage_percent: float,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.highlights_dir = self.storage_path / "highlights"
        self.highlights_dir.mkdir(parents=True, exist_ok=True)
        self.segment_index_path = self.storage_path / "segments.jsonl"
        self.highlight_index_path = self.highlights_dir / "index.jsonl"

        self.video_device = video_device
        self.resolution = resolution
        self.requested_fps = max(2, fps)
        self.motion_cfg = motion_cfg
        self.clip_cfg = clip_cfg
        self.jpeg_quality = max(10, min(95, jpeg_quality))
        self.stream_fps = max(1.0, stream_fps)

        self.storage_manager = StorageManager(str(self.storage_path), max_usage_percent=max_storage_percent)

        self.running = threading.Event()
        self.running.set()

        self.frame_lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None

        self.capture_thread: Optional[threading.Thread] = None
        self.capture_fps: float = float(self.requested_fps)

        self.prev_gray: Optional[np.ndarray] = None
        self.background_model: Optional[np.ndarray] = None
        self.pre_buffer: Deque[np.ndarray] = deque(maxlen=max(1, int(self.requested_fps * clip_cfg.pre_motion_seconds)))

        self.writer: Optional[cv2.VideoWriter] = None
        self.current_clip_path: Optional[Path] = None
        self.current_clip_start_time = 0.0
        self.last_motion_time = 0.0

        self.clip_motion_ratios: List[float] = []
        self.clip_motion_areas: List[float] = []
        self.clip_total_frames = 0
        self.last_stream_encode_ts = 0.0

        self.events_lock = threading.Lock()
        self.recent_events: Deque[Dict] = deque(maxlen=200)

    def _motion_metrics(self, frame: np.ndarray) -> Tuple[bool, float, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.background_model = gray.astype(np.float32)
            return False, 0.0, 0.0

        frame_delta = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        bg_abs = gray
        if self.background_model is not None:
            bg_abs = cv2.convertScaleAbs(self.background_model)
        bg_delta = cv2.absdiff(bg_abs, gray)

        composite_delta = cv2.max(frame_delta, bg_delta)
        _, thresh = cv2.threshold(composite_delta, self.motion_cfg.delta_thresh, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area > largest_area:
                largest_area = area

        motion_ratio = float(np.count_nonzero(thresh)) / float(thresh.size)
        detected = largest_area >= float(self.motion_cfg.min_motion_area)

        if self.background_model is not None:
            alpha = self.motion_cfg.background_alpha
            if detected:
                alpha *= 0.35
            cv2.accumulateWeighted(gray, self.background_model, alpha)

        return detected, motion_ratio, largest_area

    def _update_stream_frame(self, frame: np.ndarray, now_ts: float) -> None:
        if now_ts - self.last_stream_encode_ts < (1.0 / self.stream_fps):
            return

        ok, enc = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return

        with self.frame_lock:
            self.latest_jpeg = enc.tobytes()
        self.last_stream_encode_ts = now_ts

    def _new_clip_path(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        candidate = self.storage_path / f"motion_{ts}.mp4"
        if not candidate.exists():
            return candidate
        for idx in range(1, 1000):
            retry = self.storage_path / f"motion_{ts}_{idx:03d}.mp4"
            if not retry.exists():
                return retry
        return candidate

    def _start_clip(self, frame_shape: Tuple[int, int, int], now_ts: float) -> None:
        height, width = frame_shape[:2]
        clip_path = self._new_clip_path()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, self.capture_fps, (width, height))
        if not writer.isOpened():
            logger.error("Failed to open clip writer for %s", clip_path)
            return

        self.writer = writer
        self.current_clip_path = clip_path
        self.current_clip_start_time = now_ts
        self.last_motion_time = now_ts
        self.clip_motion_ratios = []
        self.clip_motion_areas = []
        self.clip_total_frames = 0

        for buffered in self.pre_buffer:
            self.writer.write(buffered)
            self.clip_total_frames += 1

        logger.info("Started clip: %s", clip_path.name)

    def _append_jsonl(self, path: Path, payload: Dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _build_event(self, clip_path: Path, now_ts: float, stop_reason: str) -> Dict:
        duration = max(0.0, now_ts - self.current_clip_start_time)
        motion_avg = float(np.mean(self.clip_motion_ratios)) if self.clip_motion_ratios else 0.0
        motion_peak = float(np.max(self.clip_motion_ratios)) if self.clip_motion_ratios else 0.0
        area_peak = float(np.max(self.clip_motion_areas)) if self.clip_motion_areas else 0.0

        duration_score = min(duration / 10.0, 1.0)
        avg_score = min(motion_avg / 0.02, 1.0)
        peak_score = min(motion_peak / 0.08, 1.0)
        shape_score = min(area_peak / max(1.0, float(self.motion_cfg.min_motion_area) * 2.0), 1.0)
        confidence = float((0.2 * duration_score) + (0.35 * avg_score) + (0.35 * peak_score) + (0.10 * shape_score))

        is_highlight = confidence >= 0.45 and motion_peak >= 0.01
        if confidence >= 0.75:
            event_type = "high_motion"
        elif confidence >= 0.45:
            event_type = "medium_motion"
        else:
            event_type = "low_motion"

        size_bytes = clip_path.stat().st_size if clip_path.exists() else 0

        return {
            "timestamp": datetime.now().isoformat(),
            "video_file": clip_path.name,
            "video_path": str(clip_path.resolve()),
            "event_type": event_type,
            "duration_seconds": duration,
            "frame_count": self.clip_total_frames,
            "motion_avg": motion_avg,
            "motion_peak": motion_peak,
            "motion_area_peak": area_peak,
            "highlight_score": confidence,
            "is_highlight": is_highlight,
            "stop_reason": stop_reason,
            "file_size_bytes": size_bytes,
            "noise_level": "n/a",
            "noise_spike_ratio": 0.0,
            "segment_audio_peak_rms": 0.0,
            "ambient_audio_rms": 0.0,
        }

    def _finish_clip(self, stop_reason: str, now_ts: float) -> None:
        if self.writer is None or self.current_clip_path is None:
            return

        self.writer.release()
        clip_path = self.current_clip_path

        event = self._build_event(clip_path, now_ts, stop_reason)

        with open(clip_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(event, f, indent=2)

        self._append_jsonl(self.segment_index_path, event)
        if event["is_highlight"]:
            self._append_jsonl(self.highlight_index_path, event)

        with self.events_lock:
            self.recent_events.appendleft(event)

        logger.info(
            "Finished clip: %s %.1fs score=%.3f reason=%s",
            clip_path.name,
            event["duration_seconds"],
            event["highlight_score"],
            stop_reason,
        )

        self.writer = None
        self.current_clip_path = None
        self.current_clip_start_time = 0.0
        self.last_motion_time = 0.0
        self.clip_motion_ratios = []
        self.clip_motion_areas = []
        self.clip_total_frames = 0

        self.storage_manager.check_and_cleanup()

    def _process_frame(self, frame: np.ndarray, now_ts: float) -> None:
        motion_detected, motion_ratio, area_peak = self._motion_metrics(frame)

        self.pre_buffer.append(frame.copy())
        self._update_stream_frame(frame, now_ts)

        if motion_detected and self.writer is None:
            self._start_clip(frame.shape, now_ts)

        if self.writer is None:
            return

        self.writer.write(frame)
        self.clip_total_frames += 1
        self.clip_motion_ratios.append(motion_ratio)
        self.clip_motion_areas.append(area_peak)

        if motion_detected:
            self.last_motion_time = now_ts

        elapsed_clip = now_ts - self.current_clip_start_time
        quiet_time = now_ts - self.last_motion_time

        if elapsed_clip >= self.clip_cfg.max_clip_seconds:
            self._finish_clip("max_clip_duration", now_ts)
            return

        if elapsed_clip >= self.clip_cfg.min_clip_seconds and quiet_time >= self.clip_cfg.cooldown_seconds:
            self._finish_clip("motion_stopped", now_ts)

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self.video_device)
        if not cap.isOpened():
            logger.error("Failed to open video device %s", self.video_device)
            self.running.clear()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps < 2.0:
            fps = float(self.requested_fps)
        self.capture_fps = fps
        self.pre_buffer = deque(maxlen=max(1, int(self.capture_fps * self.clip_cfg.pre_motion_seconds)))

        logger.info("Camera opened: %sx%s @ %.2f fps", width, height, self.capture_fps)

        while self.running.is_set():
            ok, frame = cap.read()
            if not ok:
                logger.warning("Failed to read camera frame; retrying")
                time.sleep(0.08)
                continue

            self._process_frame(frame, time.time())

        cap.release()

        if self.writer is not None:
            self._finish_clip("shutdown", time.time())

    def start_capture(self) -> None:
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self) -> None:
        self.running.clear()
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.frame_lock:
            return self.latest_jpeg

    def get_status(self) -> Dict:
        with self.events_lock:
            recent = list(self.recent_events)[:20]

        return {
            "status": "ok",
            "stream_ready": self.get_latest_jpeg() is not None,
            "capture_fps": self.capture_fps,
            "recording": self.writer is not None,
            "recent_events": recent,
            "time": datetime.now().isoformat(),
        }


class StreamHandler(BaseHTTPRequestHandler):
    camera: RingCamera
    access_token: Optional[str]

    def log_message(self, fmt: str, *args) -> None:
        logger.info("HTTP %s - %s", self.address_string(), fmt % args)

    def _authorized(self) -> bool:
        if not self.access_token:
            return True
        parsed = urlparse(self.path)
        token = parse_qs(parsed.query).get("token", [""])[0]
        return token == self.access_token

    def _send_json(self, payload: Dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _forbidden(self) -> None:
        self._send_json({"error": "forbidden"}, HTTPStatus.FORBIDDEN)

    def _serve_home(self) -> None:
        token_query = ""
        if self.access_token:
            token_query = f"?token={self.access_token}"

        html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Home Camera</title>
  <style>
    :root {{ --bg:#0b0f14; --panel:#121822; --line:#1f2a3b; --text:#e8edf5; --muted:#96a4b8; --good:#3ccf91; --warn:#ffad3d; }}
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: radial-gradient(circle at 20% 0%, #122037, var(--bg) 50%); color: var(--text); }}
    .wrap {{ max-width: 1100px; margin: 18px auto; padding: 0 12px; }}
    .grid {{ display:grid; grid-template-columns: 2fr 1fr; gap: 12px; }}
    .card {{ background: linear-gradient(180deg, #141d2b, var(--panel)); border:1px solid var(--line); border-radius: 14px; padding: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.26); }}
    img {{ width:100%; border-radius:12px; border:1px solid var(--line); }}
    h1, h2 {{ margin:0 0 10px; }}
    h1 {{ font-size:1.1rem; }}
    h2 {{ font-size:1rem; color: var(--muted); }}
    table {{ width:100%; border-collapse:collapse; font-size: 0.85rem; }}
    th, td {{ text-align:left; padding:6px 4px; border-bottom: 1px solid #1a2434; }}
    .badge {{ padding: 2px 6px; border-radius: 8px; font-size: 0.72rem; }}
    .high_motion {{ background: rgba(60,207,145,0.2); color:#67f3b1; }}
    .medium_motion {{ background: rgba(255,173,61,0.2); color:#ffd39b; }}
    .low_motion {{ background: rgba(130,149,172,0.2); color:#c2cedd; }}
    @media (max-width: 880px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Live Home Camera</h1>
    <div class=\"grid\">
      <div class=\"card\">
        <img src=\"/stream.mjpg{token_query}\" alt=\"Live stream\"/>
      </div>
      <div class=\"card\">
        <h2>Recent Motion Events</h2>
        <div id=\"status\" style=\"color:var(--muted);font-size:0.82rem;margin-bottom:8px;\">Loading...</div>
        <table>
          <thead>
            <tr><th>Time</th><th>Type</th><th>Score</th><th>Clip</th></tr>
          </thead>
          <tbody id=\"events\"></tbody>
        </table>
      </div>
    </div>
  </div>
<script>
async function refresh() {{
  const r = await fetch('/api/status');
  const data = await r.json();
  const status = document.getElementById('status');
  status.textContent = `stream_ready=${{data.stream_ready}} capture_fps=${{(data.capture_fps||0).toFixed(1)}} recording=${{data.recording}}`;
  const rows = (data.recent_events || []).slice(0, 12).map(e => {{
    const cls = e.event_type || 'low_motion';
    return `<tr><td>${{(e.timestamp||'').replace('T',' ').slice(0,19)}}</td><td><span class=\"badge ${{cls}}\">${{cls}}</span></td><td>${{(e.highlight_score||0).toFixed(3)}}</td><td>${{e.video_file||''}}</td></tr>`;
  }}).join('');
  document.getElementById('events').innerHTML = rows || '<tr><td colspan="4">No events yet</td></tr>';
}}
setInterval(refresh, 3000);
refresh();
</script>
</body>
</html>
"""
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self) -> None:
        if not self._authorized():
            self._forbidden()
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while self.camera.running.is_set():
                frame = self.camera.get_latest_jpeg()
                if frame is None:
                    time.sleep(0.04)
                    continue

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(max(0.01, 1.0 / max(1.0, self.camera.stream_fps)))
        except (ConnectionResetError, BrokenPipeError):
            return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._serve_home()
            return
        if parsed.path == "/health":
            self._send_json(self.camera.get_status())
            return
        if parsed.path == "/api/status":
            if not self._authorized():
                self._forbidden()
                return
            self._send_json(self.camera.get_status())
            return
        if parsed.path == "/stream.mjpg":
            self._serve_stream()
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()


def serve(camera: RingCamera, host: str, port: int, access_token: Optional[str]) -> None:
    handler_cls = type("BoundStreamHandler", (StreamHandler,), {})
    handler_cls.camera = camera
    handler_cls.access_token = access_token

    server = ThreadingHTTPServer((host, port), handler_cls)
    logger.info("Live stream ready at http://%s:%s", host, port)

    def _shutdown(*_args: object) -> None:
        logger.info("Shutting down")
        camera.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    camera.start_capture()
    server.serve_forever()


def parse_resolution(value: str) -> Tuple[int, int]:
    try:
        w, h = value.lower().split("x", maxsplit=1)
        return int(w), int(h)
    except Exception as exc:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT, e.g. 1280x720") from exc


def run_self_test() -> int:
    """Synthetic validation that motion detection creates and saves clips."""
    with TemporaryDirectory(prefix="ring-camera-test-") as tmp:
        camera = RingCamera(
            storage_path=tmp,
            video_device=0,
            resolution=(640, 360),
            fps=10,
            motion_cfg=MotionConfig(delta_thresh=18, min_motion_area=280, background_alpha=0.03),
            clip_cfg=ClipConfig(cooldown_seconds=0.8, pre_motion_seconds=0.5, max_clip_seconds=6.0, min_clip_seconds=0.5),
            jpeg_quality=70,
            stream_fps=6,
            max_storage_percent=98.0,
        )

        now_ts = time.time()
        for idx in range(90):
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            if 20 <= idx <= 58:
                x = 80 + (idx - 20) * 7
                cv2.rectangle(frame, (x, 90), (x + 120, 240), (255, 255, 255), -1)
            camera._process_frame(frame, now_ts + (idx * 0.1))

        if camera.writer is not None:
            camera._finish_clip("self_test_end", now_ts + 15.0)

        clips = sorted(Path(tmp).glob("motion_*.mp4"))
        if not clips:
            logger.error("Self-test failed: no clips were generated")
            return 1

        clip = clips[0]
        meta = clip.with_suffix(".json")
        if not meta.exists():
            logger.error("Self-test failed: metadata file missing for %s", clip.name)
            return 1

        cap = cv2.VideoCapture(str(clip))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frame_count <= 0:
            logger.error("Self-test failed: clip has no frames")
            return 1

        seg_index = Path(tmp) / "segments.jsonl"
        if not seg_index.exists() or seg_index.stat().st_size == 0:
            logger.error("Self-test failed: segments.jsonl was not written")
            return 1

        logger.info("Self-test passed: clip=%s frames=%s", clip.name, frame_count)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raspberry Pi home camera")
    parser.add_argument("--storage-path", default="./recordings", help="Directory for motion clips")
    parser.add_argument("--video-device", type=int, default=0, help="Video device index")
    parser.add_argument("--resolution", type=parse_resolution, default=parse_resolution("1280x720"), help="WIDTHxHEIGHT")
    parser.add_argument("--fps", type=int, default=15, help="Capture FPS")
    parser.add_argument("--delta-thresh", type=int, default=20, help="Pixel delta threshold")
    parser.add_argument("--min-motion-area", type=int, default=1200, help="Min contour area for motion")
    parser.add_argument("--background-alpha", type=float, default=0.02, help="Background adaptation speed")
    parser.add_argument("--cooldown-seconds", type=float, default=6.0, help="Stop clip N seconds after motion")
    parser.add_argument("--pre-motion-seconds", type=float, default=2.0, help="Include N seconds before trigger")
    parser.add_argument("--max-clip-seconds", type=float, default=45.0, help="Hard clip duration cap")
    parser.add_argument("--min-clip-seconds", type=float, default=1.0, help="Minimum clip duration")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="MJPEG quality 10-95")
    parser.add_argument("--stream-fps", type=float, default=8.0, help="Max stream frame rate")
    parser.add_argument("--max-storage-percent", type=float, default=90.0, help="Cleanup usage threshold")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument("--access-token", default=None, help="Optional stream/API token")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic clip-saving test and exit")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(run_self_test())

    camera = RingCamera(
        storage_path=args.storage_path,
        video_device=args.video_device,
        resolution=args.resolution,
        fps=args.fps,
        motion_cfg=MotionConfig(
            delta_thresh=args.delta_thresh,
            min_motion_area=args.min_motion_area,
            background_alpha=args.background_alpha,
        ),
        clip_cfg=ClipConfig(
            cooldown_seconds=args.cooldown_seconds,
            pre_motion_seconds=args.pre_motion_seconds,
            max_clip_seconds=args.max_clip_seconds,
            min_clip_seconds=args.min_clip_seconds,
        ),
        jpeg_quality=args.jpeg_quality,
        stream_fps=args.stream_fps,
        max_storage_percent=args.max_storage_percent,
    )

    serve(camera, host=args.host, port=args.port, access_token=args.access_token)


if __name__ == "__main__":
    main()
