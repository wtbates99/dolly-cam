#!/usr/bin/env python3
"""Dolly Bates memorial camera: motion clips, live stream, and lightweight dashboard APIs."""

from __future__ import annotations

import argparse
import copy
import ipaddress
import json
import logging
import secrets
import signal
import subprocess
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
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from storage_manager import StorageManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dolly_bates_camera")


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
        self.web_clips_dir = self.storage_path / "web_clips"
        self.web_clips_dir.mkdir(parents=True, exist_ok=True)
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
        self.web_clip_lock = threading.Lock()
        self.clip_feature_cache: Dict[str, Dict] = {}
        self.clip_feature_cache_lock = threading.Lock()

        self._load_recent_events_from_index()

    def _load_recent_events_from_index(self) -> None:
        if not self.segment_index_path.exists():
            return
        loaded: List[Dict] = []
        try:
            with open(self.segment_index_path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        loaded.append(payload)
        except Exception:
            logger.exception("Failed reading segment index: %s", self.segment_index_path)
            return

        loaded = loaded[-200:]
        loaded.reverse()
        with self.events_lock:
            self.recent_events.clear()
            for event in loaded:
                self.recent_events.append(event)

    def _event_clip_path(self, event: Dict) -> Optional[Path]:
        video_file = str(event.get("video_file", "")).strip()
        if video_file:
            candidate = (self.storage_path / Path(video_file).name).resolve()
            if candidate.exists():
                return candidate

        video_path = str(event.get("video_path", "")).strip()
        if not video_path:
            return None

        candidate = Path(video_path).expanduser()
        if not candidate.is_absolute():
            candidate = (self.storage_path / candidate).resolve()
        if candidate.exists():
            return candidate
        return None

    def _parse_clip_features(self, clip_path: Path) -> Dict:
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return {"parsed_ok": False}

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0.1:
            fps = self.capture_fps if self.capture_fps > 0.1 else float(self.requested_fps)

        stride = 1
        if total_frames > 0:
            stride = max(1, total_frames // 160)

        idx = 0
        sampled_frames = 0
        brightness_values: List[float] = []
        delta_values: List[float] = []
        prev_gray: Optional[np.ndarray] = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            brightness_values.append(brightness)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                delta_values.append(float(np.mean(diff)))
            prev_gray = gray
            sampled_frames += 1
            idx += 1

        cap.release()

        if sampled_frames == 0:
            return {"parsed_ok": False}

        avg_brightness = float(np.mean(brightness_values))
        brightness_std = float(np.std(brightness_values))
        mean_delta = float(np.mean(delta_values)) if delta_values else 0.0
        peak_delta = float(np.max(delta_values)) if delta_values else 0.0

        if mean_delta >= 8.0 or peak_delta >= 20.0:
            parsed_motion_type = "intense_activity"
        elif mean_delta >= 3.0 or peak_delta >= 8.0:
            parsed_motion_type = "steady_activity"
        else:
            parsed_motion_type = "subtle_activity"

        if avg_brightness < 55.0:
            lighting = "dark"
        elif avg_brightness < 140.0:
            lighting = "normal"
        else:
            lighting = "bright"

        parsed_duration = float(total_frames / fps) if total_frames > 0 and fps > 0.1 else 0.0
        return {
            "parsed_ok": True,
            "parsed_total_frames": total_frames,
            "parsed_sampled_frames": sampled_frames,
            "parsed_fps": float(fps),
            "parsed_duration_seconds": parsed_duration,
            "parsed_brightness_avg": avg_brightness,
            "parsed_brightness_std": brightness_std,
            "parsed_frame_delta_mean": mean_delta,
            "parsed_frame_delta_peak": peak_delta,
            "parsed_motion_type": parsed_motion_type,
            "parsed_lighting": lighting,
        }

    def _get_clip_features(self, clip_path: Path) -> Dict:
        try:
            stat = clip_path.stat()
        except FileNotFoundError:
            return {"parsed_ok": False}
        cache_key = f"{clip_path.resolve()}::{stat.st_mtime_ns}"
        with self.clip_feature_cache_lock:
            cached = self.clip_feature_cache.get(cache_key)
        if cached is not None:
            return cached

        parsed = self._parse_clip_features(clip_path)
        with self.clip_feature_cache_lock:
            # Keep cache bounded with simple FIFO behavior.
            if len(self.clip_feature_cache) >= 256:
                oldest_key = next(iter(self.clip_feature_cache))
                self.clip_feature_cache.pop(oldest_key, None)
            self.clip_feature_cache[cache_key] = parsed
        return parsed

    def _enrich_event(self, event: Dict) -> Dict:
        enriched = copy.deepcopy(event)
        clip_path = self._event_clip_path(enriched)
        if clip_path is None:
            enriched["parsed_ok"] = False
            enriched["filter_types"] = [str(enriched.get("event_type", "unknown"))]
            return enriched

        parsed = self._get_clip_features(clip_path)
        enriched.update(parsed)

        types = {str(enriched.get("event_type", "unknown"))}
        parsed_motion_type = str(enriched.get("parsed_motion_type", "")).strip()
        if parsed_motion_type:
            types.add(parsed_motion_type)
        stop_reason = str(enriched.get("stop_reason", "")).strip()
        if stop_reason:
            types.add(stop_reason)
        enriched["filter_types"] = sorted(types)
        return enriched

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

    def get_status(self, event_type_filter: str = "", clip_contains: str = "", limit: int = 20) -> Dict:
        with self.events_lock:
            recent = list(self.recent_events)

        all_enriched = [self._enrich_event(event) for event in recent]
        enriched_recent = all_enriched

        raw_filter = event_type_filter.strip().lower()
        if raw_filter and raw_filter != "all":
            filtered: List[Dict] = []
            for event in enriched_recent:
                event_type = str(event.get("event_type", "")).lower()
                parsed_motion_type = str(event.get("parsed_motion_type", "")).lower()
                stop_reason = str(event.get("stop_reason", "")).lower()
                if raw_filter in (event_type, parsed_motion_type, stop_reason):
                    filtered.append(event)
            enriched_recent = filtered

        clip_query = clip_contains.strip().lower()
        if clip_query:
            enriched_recent = [
                event
                for event in enriched_recent
                if clip_query in str(event.get("video_file", "")).lower()
            ]

        recent_limited = enriched_recent[: max(1, min(120, limit))]
        available_types = sorted(
            {
                str(event.get("event_type", "unknown"))
                for event in all_enriched
                if str(event.get("event_type", "")).strip()
            }
            | {
                str(event.get("parsed_motion_type", ""))
                for event in all_enriched
                if str(event.get("parsed_motion_type", "")).strip()
            }
        )
        event_type_counts: Dict[str, int] = {}
        for event in enriched_recent:
            label = str(event.get("event_type", "unknown"))
            event_type_counts[label] = event_type_counts.get(label, 0) + 1

        return {
            "status": "ok",
            "stream_ready": self.get_latest_jpeg() is not None,
            "capture_fps": self.capture_fps,
            "recording": self.writer is not None,
            "recent_events": recent_limited,
            "available_types": available_types,
            "event_type_counts": event_type_counts,
            "time": datetime.now().isoformat(),
        }

    def get_clip_features_for_name(self, clip_name: str) -> Dict:
        safe_name = Path(clip_name).name
        if safe_name != clip_name or not safe_name.endswith(".mp4"):
            return {"error": "invalid_clip_name"}
        clip_path = (self.storage_path / safe_name).resolve()
        if not clip_path.exists():
            return {"error": "clip_not_found"}
        parsed = self._get_clip_features(clip_path)
        payload = {
            "clip": safe_name,
            "video_path": str(clip_path),
        }
        payload.update(parsed)
        return payload


class StreamHandler(BaseHTTPRequestHandler):
    camera: RingCamera
    access_token: Optional[str]
    admin_token: Optional[str]
    remote_view_flag_file: Path
    trust_proxy_headers: bool

    def log_message(self, fmt: str, *args) -> None:
        logger.info("HTTP %s - %s", self.address_string(), fmt % args)

    def _send_security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")

    def _client_ip(self) -> str:
        if self.trust_proxy_headers:
            forwarded = self.headers.get("X-Forwarded-For", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return str(self.client_address[0])

    def _is_local_client(self) -> bool:
        ip_text = self._client_ip()
        try:
            addr = ipaddress.ip_address(ip_text)
            return bool(addr.is_private or addr.is_loopback)
        except ValueError:
            return False

    def _remote_view_enabled(self) -> bool:
        return self.remote_view_flag_file.exists()

    def _can_view(self) -> bool:
        if self._is_local_client():
            return True
        return self._remote_view_enabled()

    def _get_query_value(self, key: str) -> str:
        parsed = urlparse(self.path)
        return parse_qs(parsed.query).get(key, [""])[0]

    def _authorized(self) -> bool:
        # Local-only deployments can access without token.
        if self._is_local_client():
            return True
        if not self.access_token:
            return True
        token = self._get_query_value("token")
        if token and secrets.compare_digest(token, self.access_token):
            return True
        bearer = self.headers.get("Authorization", "")
        if bearer.startswith("Bearer "):
            candidate = bearer[7:].strip()
            return bool(candidate and secrets.compare_digest(candidate, self.access_token))
        return False

    def _admin_authorized(self) -> bool:
        if not self.admin_token:
            return False
        token = self._get_query_value("admin_token")
        if token and secrets.compare_digest(token, self.admin_token):
            return True
        header_token = self.headers.get("X-Admin-Token", "").strip()
        return bool(header_token and secrets.compare_digest(header_token, self.admin_token))

    def _read_json_body(self) -> Dict:
        content_len = int(self.headers.get("Content-Length", "0") or "0")
        if content_len <= 0:
            return {}
        raw = self.rfile.read(min(content_len, 8192))
        try:
            payload = json.loads(raw.decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _send_json(self, payload: Dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, private")
        self.send_header("Pragma", "no-cache")
        self._send_security_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _forbidden(self) -> None:
        self._send_json({"error": "forbidden"}, HTTPStatus.FORBIDDEN)

    def _serve_home(self) -> None:
        if not self._can_view():
            self._forbidden()
            return
        if not self._authorized():
            self._forbidden()
            return

        token_query = ""
        if self.access_token:
            token_query = f"?token={self.access_token}"

        html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Dolly Bates Camera</title>
  <style>
    :root {{ --brand:#AF69EE; --bg:#12081f; --panel:#1f1133; --line:#512b76; --text:#f5ebff; --muted:#c9b3df; --accent:#d4a9ff; --accent2:#9465c8; }}
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: radial-gradient(circle at 20% 0%, #5f2f8f, var(--bg) 52%); color: var(--text); }}
    .wrap {{ max-width: 1100px; margin: 18px auto; padding: 0 12px 24px; }}
    .grid {{ display:grid; grid-template-columns: 2fr 1fr; gap: 12px; }}
    .card {{ background: linear-gradient(180deg, #2a1646, var(--panel)); border:1px solid var(--line); border-radius: 14px; padding: 12px; box-shadow: 0 8px 30px rgba(17,5,31,0.46); }}
    img {{ width:100%; border-radius:12px; border:1px solid var(--line); }}
    video {{ width:100%; border-radius:12px; border:1px solid var(--line); margin-top:10px; background:#000; }}
    h1, h2 {{ margin:0 0 10px; }}
    h1 {{ font-size:1.1rem; }}
    h2 {{ font-size:1rem; color: var(--muted); }}
    table {{ width:100%; border-collapse:collapse; font-size: 0.85rem; }}
    th, td {{ text-align:left; padding:6px 4px; border-bottom: 1px solid #31164f; }}
    button {{ background:linear-gradient(180deg, var(--brand), #9155ce); color:#1f0935; border:1px solid #c896ff; border-radius:8px; padding:4px 8px; cursor:pointer; font-weight:600; }}
    button:hover {{ background:linear-gradient(180deg, #c58bff, #a665eb); }}
    .controls {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }}
    input, select {{ background:#190d2b; color:var(--text); border:1px solid #6f449d; border-radius:8px; padding:4px 8px; }}
    .stats {{ color:var(--muted); font-size:0.78rem; margin-bottom:8px; line-height:1.35; }}
    .badge {{ padding: 2px 6px; border-radius: 8px; font-size: 0.72rem; }}
    .high_motion {{ background: rgba(175,105,238,0.35); color:#f2dcff; }}
    .medium_motion {{ background: rgba(156,93,215,0.30); color:#edd3ff; }}
    .low_motion {{ background: rgba(120,73,170,0.30); color:#e4ccfb; }}
    .steady_activity {{ background: rgba(206,166,244,0.25); color:#f4e6ff; }}
    .intense_activity {{ background: rgba(175,105,238,0.42); color:#ffffff; }}
    .subtle_activity {{ background: rgba(102,65,143,0.40); color:#e8d6ff; }}
    @media (max-width: 880px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Dolly Bates Memorial Camera</h1>
    <div class=\"grid\">
      <div class=\"card\">
        <img src=\"/stream.mjpg{token_query}\" alt=\"Live stream\"/>
      </div>
      <div class=\"card\">
        <h2>Dolly's Recent Motion Events</h2>
        <div id=\"status\" style=\"color:var(--muted);font-size:0.82rem;margin-bottom:8px;\">Loading...</div>
        <div class=\"controls\">
          <select id=\"typeFilter\"><option value=\"all\">all types</option></select>
          <input id=\"clipFilter\" placeholder=\"clip name contains...\" />
        </div>
        <div id=\"quickStats\" class=\"stats\"></div>
        <table>
          <thead>
            <tr><th>Time</th><th>Type</th><th>Parsed</th><th>Score</th><th>Clip</th><th>Playback</th></tr>
          </thead>
          <tbody id=\"events\"></tbody>
        </table>
        <video id=\"clipPlayer\" controls preload=\"metadata\"></video>
      </div>
    </div>
  </div>
<script>
const tokenQuery = {json.dumps(token_query)};
const clipPlayer = document.getElementById('clipPlayer');
const typeFilter = document.getElementById('typeFilter');
const clipFilter = document.getElementById('clipFilter');
const eventsEl = document.getElementById('events');
const quickStats = document.getElementById('quickStats');

function escapeHtml(text) {{
  return String(text || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}}

function clipUrl(fileName) {{
  return `/clips-web/${{encodeURIComponent(fileName)}}${{tokenQuery}}`;
}}
function playClip(fileName) {{
  clipPlayer.src = clipUrl(fileName);
  clipPlayer.play().catch(() => {{}});
}}

eventsEl.addEventListener('click', (evt) => {{
  const btn = evt.target.closest('button[data-clip]');
  if (!btn) return;
  playClip(btn.getAttribute('data-clip'));
}});

let knownTypes = ['all'];
function updateTypeFilter(data) {{
  const serverTypes = (data.available_types || []).filter(Boolean);
  knownTypes = Array.from(new Set(['all', ...serverTypes])).sort();
  const current = typeFilter.value || 'all';
  typeFilter.innerHTML = knownTypes.map(t => `<option value="${{escapeHtml(t)}}">${{escapeHtml(t)}}</option>`).join('');
  typeFilter.value = knownTypes.includes(current) ? current : 'all';
}}

async function refresh() {{
  const selectedType = typeFilter.value || 'all';
  const clipContains = clipFilter.value || '';
  const params = new URLSearchParams();
  if (tokenQuery.startsWith('?token=')) {{
    params.set('token', tokenQuery.slice(7));
  }}
  if (selectedType && selectedType !== 'all') params.set('type', selectedType);
  if (clipContains) params.set('clip_contains', clipContains);

  const r = await fetch(`/api/status?${{params.toString()}}`);
  if (!r.ok) {{
    document.getElementById('status').textContent = `status fetch failed: ${{r.status}}`;
    return;
  }}
  const data = await r.json();
  updateTypeFilter(data);
  const status = document.getElementById('status');
  status.textContent = `stream_ready=${{data.stream_ready}} capture_fps=${{(data.capture_fps||0).toFixed(1)}} recording=${{data.recording}}`;
  const countBits = Object.entries(data.event_type_counts || {{}}).map(([k, v]) => `${{k}}:${{v}}`);
  quickStats.textContent = countBits.length ? `visible counts: ${{countBits.join(' | ')}}` : 'visible counts: none';
  const rows = (data.recent_events || []).slice(0, 12).map(e => {{
    const cls = e.event_type || 'low_motion';
    const parsedType = e.parsed_motion_type || 'n/a';
    const clip = e.video_file || '';
    const playButton = clip ? `<button type="button" data-clip="${{escapeHtml(clip)}}">Play</button>` : '';
    return `<tr><td>${{escapeHtml((e.timestamp||'').replace('T',' ').slice(0,19))}}</td><td><span class="badge ${{escapeHtml(cls)}}">${{escapeHtml(cls)}}</span></td><td><span class="badge ${{escapeHtml(parsedType)}}">${{escapeHtml(parsedType)}}</span></td><td>${{(e.highlight_score||0).toFixed(3)}}</td><td>${{escapeHtml(clip)}}</td><td>${{playButton}}</td></tr>`;
  }}).join('');
  eventsEl.innerHTML = rows || '<tr><td colspan="6">No events yet</td></tr>';
}}
typeFilter.addEventListener('change', refresh);
clipFilter.addEventListener('input', () => setTimeout(refresh, 0));
setInterval(refresh, 3000);
refresh();
</script>
</body>
</html>
"""
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self) -> None:
        if not self._can_view():
            self._forbidden()
            return
        if not self._authorized():
            self._forbidden()
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self._send_security_headers()
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

    def _resolve_clip_path(self, clip_name: str) -> Optional[Path]:
        decoded = unquote(clip_name)
        safe_name = Path(decoded).name
        if safe_name != decoded or not safe_name.endswith(".mp4"):
            return None
        clip_path = (self.camera.storage_path / safe_name).resolve()
        storage_root = self.camera.storage_path.resolve()
        if storage_root not in clip_path.parents or not clip_path.exists():
            return None
        return clip_path

    def _serve_video_file(self, clip_path: Path) -> None:
        if not self._can_view() or not self._authorized():
            self._forbidden()
            return

        file_size = clip_path.stat().st_size
        range_header = self.headers.get("Range", "")
        start = 0
        end = max(0, file_size - 1)
        status = HTTPStatus.OK

        if range_header.startswith("bytes="):
            try:
                start_text, end_text = range_header.replace("bytes=", "", 1).split("-", 1)
                if start_text:
                    start = int(start_text)
                if end_text:
                    end = int(end_text)
                start = max(0, start)
                end = min(end, file_size - 1)
                if start > end:
                    raise ValueError("invalid range")
                status = HTTPStatus.PARTIAL_CONTENT
            except Exception:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self._send_security_headers()
                self.end_headers()
                return

        chunk_len = (end - start) + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers()
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(chunk_len))
        self.end_headers()

        with open(clip_path, "rb") as f:
            f.seek(start)
            remaining = chunk_len
            while remaining > 0:
                block = f.read(min(64 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def _serve_clip(self, clip_name: str) -> None:
        clip_path = self._resolve_clip_path(clip_name)
        if clip_path is None:
            self._send_json({"error": "clip_not_found"}, HTTPStatus.NOT_FOUND)
            return
        self._serve_video_file(clip_path)

    def _serve_web_clip(self, clip_name: str) -> None:
        clip_path = self._resolve_clip_path(clip_name)
        if clip_path is None:
            self._send_json({"error": "clip_not_found"}, HTTPStatus.NOT_FOUND)
            return

        web_target = self.camera.web_clips_dir / f"{clip_path.stem}.web.mp4"
        needs_transcode = (not web_target.exists()) or (web_target.stat().st_mtime < clip_path.stat().st_mtime)

        if needs_transcode:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(clip_path),
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "24",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(web_target),
            ]
            with self.camera.web_clip_lock:
                needs_transcode = (not web_target.exists()) or (web_target.stat().st_mtime < clip_path.stat().st_mtime)
                if needs_transcode:
                    try:
                        subprocess.run(
                            ffmpeg_cmd,
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception:
                        # Fallback to original if transcoding is unavailable.
                        self._serve_video_file(clip_path)
                        return

        self._serve_video_file(web_target)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._serve_home()
            return
        if parsed.path == "/health":
            self._send_json(self.camera.get_status())
            return
        if parsed.path == "/api/status":
            if not self._can_view():
                self._forbidden()
                return
            if not self._authorized():
                self._forbidden()
                return
            event_type_filter = parse_qs(parsed.query).get("type", [""])[0]
            clip_contains = parse_qs(parsed.query).get("clip_contains", [""])[0]
            raw_limit = parse_qs(parsed.query).get("limit", ["20"])[0]
            try:
                limit = int(raw_limit)
            except ValueError:
                limit = 20
            self._send_json(
                self.camera.get_status(
                    event_type_filter=event_type_filter,
                    clip_contains=clip_contains,
                    limit=limit,
                )
            )
            return
        if parsed.path == "/api/clip-features":
            if not self._can_view():
                self._forbidden()
                return
            if not self._authorized():
                self._forbidden()
                return
            clip_name = parse_qs(parsed.query).get("clip", [""])[0]
            if not clip_name:
                self._send_json({"error": "missing_clip"}, HTTPStatus.BAD_REQUEST)
                return
            result = self.camera.get_clip_features_for_name(clip_name)
            if result.get("error"):
                status = HTTPStatus.BAD_REQUEST if result["error"] == "invalid_clip_name" else HTTPStatus.NOT_FOUND
                self._send_json(result, status)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/remote-view":
            if not self._authorized():
                self._forbidden()
                return
            self._send_json(
                {
                    "remote_view_enabled": self._remote_view_enabled(),
                    "client_ip": self._client_ip(),
                    "is_local_client": self._is_local_client(),
                }
            )
            return
        if parsed.path == "/stream.mjpg":
            self._serve_stream()
            return
        if parsed.path.startswith("/clips/"):
            clip_name = parsed.path[len("/clips/") :]
            self._serve_clip(clip_name)
            return
        if parsed.path.startswith("/clips-web/"):
            clip_name = parsed.path[len("/clips-web/") :]
            self._serve_web_clip(clip_name)
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/remote-view":
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return

        if not self._admin_authorized():
            self._forbidden()
            return

        action = self._get_query_value("action")
        if not action:
            body = self._read_json_body()
            action = str(body.get("action", "")).strip().lower()

        if action == "disable":
            self.remote_view_flag_file.unlink(missing_ok=True)
        elif action == "enable":
            if not self._is_local_client():
                self._send_json(
                    {"error": "enable_allowed_from_local_only"},
                    HTTPStatus.FORBIDDEN,
                )
                return
            self.remote_view_flag_file.parent.mkdir(parents=True, exist_ok=True)
            self.remote_view_flag_file.touch(exist_ok=True)
        else:
            self._send_json({"error": "action_must_be_enable_or_disable"}, HTTPStatus.BAD_REQUEST)
            return

        self._send_json(
            {
                "ok": True,
                "remote_view_enabled": self._remote_view_enabled(),
            }
        )


def serve(
    camera: RingCamera,
    host: str,
    port: int,
    access_token: Optional[str],
    admin_token: Optional[str],
    remote_view_flag_file: Path,
    trust_proxy_headers: bool,
) -> None:
    handler_cls = type("BoundStreamHandler", (StreamHandler,), {})
    handler_cls.camera = camera
    handler_cls.access_token = access_token
    handler_cls.admin_token = admin_token
    handler_cls.remote_view_flag_file = remote_view_flag_file
    handler_cls.trust_proxy_headers = trust_proxy_headers

    server = ThreadingHTTPServer((host, port), handler_cls)
    logger.info("Dolly Bates stream ready at http://%s:%s", host, port)

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
    with TemporaryDirectory(prefix="dolly-bates-camera-test-") as tmp:
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
    parser = argparse.ArgumentParser(description="Dolly Bates memorial camera")
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
    parser.add_argument("--access-token", default=None, help="Required stream/API token")
    parser.add_argument("--allow-insecure-no-token", action="store_true", help="Allow running without --access-token (not recommended)")
    parser.add_argument("--admin-token", default=None, help="Admin token for remote-view enable/disable API")
    parser.add_argument(
        "--remote-view-default",
        choices=("on", "off"),
        default="on",
        help="Initial remote-view state for non-local clients",
    )
    parser.add_argument(
        "--remote-view-flag-file",
        default=None,
        help="Path to flag file controlling remote view state (default: <storage>/remote_view.enabled)",
    )
    parser.add_argument(
        "--trust-proxy-headers",
        action="store_true",
        help="Trust X-Forwarded-For from reverse proxy for client IP detection",
    )
    parser.add_argument("--self-test", action="store_true", help="Run synthetic clip-saving test and exit")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(run_self_test())

    if not args.access_token and not args.allow_insecure_no_token:
        raise SystemExit(
            "Refusing to start without --access-token. "
            "Set a long token, or pass --allow-insecure-no-token if you explicitly accept the risk."
        )

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

    remote_view_flag_file = (
        Path(args.remote_view_flag_file).expanduser()
        if args.remote_view_flag_file
        else (Path(args.storage_path) / "remote_view.enabled")
    )
    remote_view_flag_file.parent.mkdir(parents=True, exist_ok=True)
    if args.remote_view_default == "on":
        remote_view_flag_file.touch(exist_ok=True)
    else:
        remote_view_flag_file.unlink(missing_ok=True)

    serve(
        camera,
        host=args.host,
        port=args.port,
        access_token=args.access_token,
        admin_token=args.admin_token,
        remote_view_flag_file=remote_view_flag_file,
        trust_proxy_headers=args.trust_proxy_headers,
    )


if __name__ == "__main__":
    main()
