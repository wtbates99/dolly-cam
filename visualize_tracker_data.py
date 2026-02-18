"""Create visual reports from ring camera segment metadata."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required. Install with: pip3 install matplotlib") from exc


@dataclass
class SegmentEvent:
    timestamp: datetime
    video_file: str
    video_path: str
    duration_seconds: float
    motion_avg: float
    motion_peak: float
    motion_area_peak: float
    highlight_score: float
    is_highlight: bool
    event_type: str


def parse_iso_ts(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_events(recordings_dir: Path) -> List[SegmentEvent]:
    rows = load_jsonl(recordings_dir / "segments.jsonl")
    if not rows:
        for metadata_path in sorted(recordings_dir.glob("motion_*.json")):
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    rows.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue

    events: List[SegmentEvent] = []
    for row in rows:
        ts = parse_iso_ts(str(row.get("timestamp", "")))
        if ts is None:
            continue
        events.append(
            SegmentEvent(
                timestamp=ts,
                video_file=str(row.get("video_file", "")),
                video_path=str(row.get("video_path", "")),
                duration_seconds=float(row.get("duration_seconds", 0.0)),
                motion_avg=float(row.get("motion_avg", 0.0)),
                motion_peak=float(row.get("motion_peak", 0.0)),
                motion_area_peak=float(row.get("motion_area_peak", 0.0)),
                highlight_score=float(row.get("highlight_score", 0.0)),
                is_highlight=bool(row.get("is_highlight", False)),
                event_type=str(row.get("event_type", "unknown")),
            )
        )

    events.sort(key=lambda e: e.timestamp)
    return events


def render_timeline(events: List[SegmentEvent], output_path: Path) -> None:
    x = [e.timestamp for e in events]
    y1 = [e.motion_avg for e in events]
    y2 = [e.motion_peak for e in events]
    y3 = [e.highlight_score for e in events]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Home Camera Activity Timeline")

    axes[0].plot(x, y1, color="#3a86ff")
    axes[0].set_ylabel("Motion Avg")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, y2, color="#ff006e")
    axes[1].set_ylabel("Motion Peak")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, y3, color="#8338ec")
    axes[2].set_ylabel("Highlight Score")
    axes[2].set_xlabel("Time")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_hourly_heatmap(events: List[SegmentEvent], output_path: Path) -> None:
    matrix = [[0 for _ in range(24)] for _ in range(7)]
    for e in events:
        matrix[e.timestamp.weekday()][e.timestamp.hour] += 1

    fig, ax = plt.subplots(figsize=(13, 4.2))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(h) for h in range(24)], fontsize=8)
    ax.set_title("Motion Events Heatmap (day x hour)")
    fig.colorbar(im, ax=ax, label="Events")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_event_counts(events: List[SegmentEvent], output_path: Path) -> None:
    counts = Counter(e.event_type for e in events)
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color=["#1f7a8c", "#bfdbf7", "#ff7d00"][: len(labels)] or ["#1f7a8c"])
    ax.set_title("Event Type Counts")
    ax.set_ylabel("Clips")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def top_highlights(events: List[SegmentEvent], limit: int = 30) -> List[SegmentEvent]:
    selected = [e for e in events if e.is_highlight]
    return sorted(selected, key=lambda e: e.highlight_score, reverse=True)[:limit]


def ensure_web_playable(source: Path, destination: Path) -> Path:
    web_target = destination.with_suffix(".web.mp4")
    if web_target.exists() and web_target.stat().st_mtime >= source.stat().st_mtime:
        return web_target

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
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
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return web_target
    except Exception:
        if not destination.exists():
            shutil.copy2(source, destination)
        return destination


def materialize_video(video_path_value: str, recordings_dir: Path, media_dir: Path, report_dir: Path) -> str:
    raw_path = Path(video_path_value)
    resolved = raw_path if raw_path.is_absolute() else (recordings_dir / raw_path).resolve()
    if not resolved.exists():
        fallback = (recordings_dir.parent / raw_path).resolve()
        resolved = fallback if fallback.exists() else resolved
    if not resolved.exists():
        return ""

    media_dir.mkdir(parents=True, exist_ok=True)
    target = media_dir / resolved.name
    target = ensure_web_playable(resolved, target)
    return target.relative_to(report_dir).as_posix()


def build_html_report(
    events: List[SegmentEvent],
    highlights: List[SegmentEvent],
    output_path: Path,
    recordings_dir: Path,
    timeline_png: str,
    heatmap_png: str,
    counts_png: str,
) -> None:
    media_dir = output_path.parent / "highlight_clips"
    total_duration = sum(e.duration_seconds for e in events)

    rows = []
    for e in highlights:
        video_rel = materialize_video(e.video_path, recordings_dir, media_dir, output_path.parent)
        play = f'<a href="{video_rel}" target="_blank">Play</a>' if video_rel else "Missing"
        rows.append(
            "<tr>"
            f"<td>{e.timestamp.isoformat(timespec='seconds')}</td>"
            f"<td>{e.video_file}</td>"
            f"<td>{e.event_type}</td>"
            f"<td>{e.highlight_score:.3f}</td>"
            f"<td>{e.duration_seconds:.1f}</td>"
            f"<td>{e.motion_peak:.4f}</td>"
            f"<td>{play}</td>"
            "</tr>"
        )

    featured = ""
    if highlights:
        top = highlights[0]
        src = materialize_video(top.video_path, recordings_dir, media_dir, output_path.parent)
        if src:
            featured = (
                "<h2>Featured Clip</h2>"
                f"<div><strong>{top.timestamp.isoformat(timespec='seconds')}</strong> score={top.highlight_score:.3f}</div>"
                "<video controls preload=\"metadata\" width=\"960\">"
                f"<source src=\"{src}\" type=\"video/mp4\">"
                "</video>"
            )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Home Camera Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; margin: 22px; }}
    .meta {{ margin-bottom: 16px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin-bottom: 14px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 7px; font-size: 14px; }}
    th {{ background: #f4f6f8; text-align: left; }}
  </style>
</head>
<body>
  <h1>Home Camera Activity Report</h1>
  <div class=\"meta\">
    <div><strong>Total clips:</strong> {len(events)}</div>
    <div><strong>Total highlights:</strong> {len(highlights)}</div>
    <div><strong>Total motion time:</strong> {total_duration:.1f}s</div>
  </div>

  <h2>Timeline</h2>
  <img src=\"{timeline_png}\" alt=\"timeline\" />

  <h2>Weekly Heatmap</h2>
  <img src=\"{heatmap_png}\" alt=\"heatmap\" />

  <h2>Event Counts</h2>
  <img src=\"{counts_png}\" alt=\"counts\" />

  {featured}

  <h2>Top Highlights</h2>
  <table>
    <thead>
      <tr><th>Timestamp</th><th>Clip</th><th>Type</th><th>Score</th><th>Seconds</th><th>Motion Peak</th><th>Video</th></tr>
    </thead>
    <tbody>
      {''.join(rows) if rows else '<tr><td colspan="7">No highlights yet.</td></tr>'}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate home camera report")
    parser.add_argument("--recordings-dir", type=Path, default=Path("./recordings"))
    parser.add_argument("--output-dir", type=Path, default=Path("./reports"))
    args = parser.parse_args()

    recordings_dir = args.recordings_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(recordings_dir)
    if not events:
        raise SystemExit(f"No segment data found in: {recordings_dir}")

    timeline_png = output_dir / "timeline.png"
    heatmap_png = output_dir / "hourly_heatmap.png"
    counts_png = output_dir / "event_counts.png"
    report_html = output_dir / "home_camera_report.html"

    render_timeline(events, timeline_png)
    render_hourly_heatmap(events, heatmap_png)
    render_event_counts(events, counts_png)
    highlights = top_highlights(events, limit=30)

    build_html_report(
        events=events,
        highlights=highlights,
        output_path=report_html,
        recordings_dir=recordings_dir,
        timeline_png=timeline_png.name,
        heatmap_png=heatmap_png.name,
        counts_png=counts_png.name,
    )

    print(f"Report created: {report_html}")


if __name__ == "__main__":
    main()
