"""
Create visual reports from tracker JSON/JSONL output.

Usage:
    python3 visualize_tracker_data.py --recordings-dir ./recordings --output-dir ./reports
"""

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
    raise SystemExit(
        "matplotlib is required for visualization.\n"
        "Install it with: pip3 install matplotlib"
    ) from exc


@dataclass
class SegmentEvent:
    timestamp: datetime
    video_file: str
    video_path: str
    is_highlight: bool
    highlight_score: float
    event_type: str
    noise_level: str
    noise_spike_ratio: float
    motion_avg: float
    motion_peak: float
    duration_seconds: float
    segment_audio_peak_rms: float
    ambient_audio_rms: float


def parse_iso_ts(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
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


def load_segment_events(recordings_dir: Path) -> List[SegmentEvent]:
    segments_jsonl = recordings_dir / "segments.jsonl"
    rows = load_jsonl(segments_jsonl)

    # Fallback for older runs: parse per-segment metadata JSON files.
    if not rows:
        for metadata_path in sorted(recordings_dir.glob("*.json")):
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    rows.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue

    # Fallback for highlight-only historical data.
    if not rows:
        highlight_rows = load_jsonl(recordings_dir / "highlights" / "index.jsonl")
        for item in highlight_rows:
            rows.append({
                "timestamp": item.get("timestamp"),
                "video_file": item.get("video_file", ""),
                "video_path": item.get("video_path", ""),
                "is_highlight": True,
                "highlight_score": item.get("highlight_score", 0.0),
                "event_type": item.get("event_type", "unknown"),
                "noise_level": item.get("noise_level", "unknown"),
                "noise_spike_ratio": item.get("noise_spike_ratio", 0.0),
                "motion_avg": item.get("motion_avg", 0.0),
                "motion_peak": item.get("motion_peak", 0.0),
                "duration_seconds": item.get("duration_seconds", 0.0),
                "segment_audio_peak_rms": item.get("segment_audio_peak_rms", 0.0),
                "ambient_audio_rms": item.get("ambient_audio_rms", 0.0),
            })

    events: List[SegmentEvent] = []
    for row in rows:
        ts = parse_iso_ts(row.get("timestamp"))
        if ts is None:
            continue
        events.append(
            SegmentEvent(
                timestamp=ts,
                video_file=row.get("video_file", ""),
                video_path=row.get("video_path", ""),
                is_highlight=bool(row.get("is_highlight", False)),
                highlight_score=float(row.get("highlight_score", 0.0)),
                event_type=row.get("event_type", "unknown"),
                noise_level=row.get("noise_level", "unknown"),
                noise_spike_ratio=float(row.get("noise_spike_ratio", 0.0)),
                motion_avg=float(row.get("motion_avg", 0.0)),
                motion_peak=float(row.get("motion_peak", 0.0)),
                duration_seconds=float(row.get("duration_seconds", row.get("segment_duration_seconds", 0.0))),
                segment_audio_peak_rms=float(row.get("segment_audio_peak_rms", 0.0)),
                ambient_audio_rms=float(row.get("ambient_audio_rms", 0.0)),
            )
        )

    events.sort(key=lambda e: e.timestamp)
    return events


def render_timeline(events: List[SegmentEvent], output_path: Path) -> None:
    times = [e.timestamp for e in events]
    highlight_scores = [e.highlight_score for e in events]
    motion_avg = [e.motion_avg for e in events]
    audio_peak = [e.segment_audio_peak_rms for e in events]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Dog Tracker Timeline", fontsize=14)

    axes[0].plot(times, highlight_scores, color="#cc0000")
    axes[0].set_ylabel("Highlight Score")
    axes[0].grid(alpha=0.25)

    axes[1].plot(times, motion_avg, color="#0b66c3")
    axes[1].set_ylabel("Motion Avg")
    axes[1].grid(alpha=0.25)

    axes[2].plot(times, audio_peak, color="#138a36")
    axes[2].set_ylabel("Audio Peak RMS")
    axes[2].set_xlabel("Time")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_scatter(events: List[SegmentEvent], output_path: Path) -> None:
    x = [e.motion_avg for e in events]
    y = [e.segment_audio_peak_rms for e in events]
    colors = ["#d62728" if e.is_highlight else "#7f7f7f" for e in events]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, c=colors, alpha=0.75)
    ax.set_title("Motion vs Audio (Highlight in Red)")
    ax.set_xlabel("Motion Average")
    ax.set_ylabel("Audio Peak RMS")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_event_counts(events: List[SegmentEvent], output_path: Path) -> None:
    event_counts = Counter(e.event_type for e in events)
    noise_counts = Counter(e.noise_level for e in events)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(list(event_counts.keys()), list(event_counts.values()), color="#0b66c3")
    axes[0].set_title("Event Type Count")
    axes[0].set_ylabel("Segments")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(list(noise_counts.keys()), list(noise_counts.values()), color="#138a36")
    axes[1].set_title("Noise Level Count")
    axes[1].set_ylabel("Segments")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def top_highlights(events: List[SegmentEvent], limit: int = 20) -> List[SegmentEvent]:
    selected = [e for e in events if e.is_highlight]
    return sorted(selected, key=lambda e: e.highlight_score, reverse=True)[:limit]


def build_html_report(
    events: List[SegmentEvent],
    highlights: List[SegmentEvent],
    output_path: Path,
    recordings_dir: Path,
    media_dir: Path,
    timeline_png: str,
    scatter_png: str,
    counts_png: str,
) -> None:
    if events:
        start_ts = events[0].timestamp.isoformat(timespec="seconds")
        end_ts = events[-1].timestamp.isoformat(timespec="seconds")
    else:
        start_ts, end_ts = "n/a", "n/a"

    media_dir.mkdir(parents=True, exist_ok=True)

    def ensure_web_playable(source: Path, destination: Path) -> Path:
        """
        Convert to H.264 MP4 for browser playback when possible.

        Falls back to copying original file if conversion fails.
        """
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
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(web_target),
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return web_target
        except Exception:
            if destination.is_symlink():
                destination.unlink()
            if not destination.exists():
                shutil.copy2(source, destination)
            return destination

    def materialize_video(video_path_value: str) -> str:
        raw_path = Path(video_path_value)
        if raw_path.is_absolute():
            resolved = raw_path
        else:
            # Prefer path relative to project root.
            resolved = (recordings_dir.parent / raw_path).resolve()
            if not resolved.exists():
                # Fallback: path relative to recordings directory.
                resolved = (recordings_dir / raw_path).resolve()

        if not resolved.exists():
            return ""

        target = media_dir / resolved.name
        target = ensure_web_playable(resolved, target)

        try:
            return target.relative_to(output_path.parent).as_posix()
        except ValueError:
            return target.as_posix()

    rows = []
    for e in highlights:
        video_src = materialize_video(e.video_path)
        video_cell = (
            f"<a href=\"{video_src}\" target=\"_blank\">Play</a>" if video_src else "Missing file"
        )
        rows.append(
            "<tr>"
            f"<td>{e.timestamp.isoformat(timespec='seconds')}</td>"
            f"<td>{e.video_file}</td>"
            f"<td>{e.highlight_score:.3f}</td>"
            f"<td>{e.event_type}</td>"
            f"<td>{e.noise_level}</td>"
            f"<td>{e.motion_avg:.4f}</td>"
            f"<td>{e.segment_audio_peak_rms:.4f}</td>"
            f"<td>{video_cell}</td>"
            "</tr>"
        )

    featured_video_html = ""
    if highlights:
        top = highlights[0]
        top_src = materialize_video(top.video_path)
        if top_src:
            featured_video_html = (
                "<h2>Featured Highlight</h2>"
                f"<div><strong>{top.timestamp.isoformat(timespec='seconds')}</strong> - {top.video_file}</div>"
                "<video controls preload=\"metadata\" width=\"960\">"
                f"<source src=\"{top_src}\" type=\"video/mp4\">"
                "Your browser could not play this video."
                "</video>"
            )
        else:
            featured_video_html = (
                "<h2>Featured Highlight</h2>"
                f"<div><strong>{top.timestamp.isoformat(timespec='seconds')}</strong> - {top.video_file}</div>"
                "<div>Video file missing</div>"
            )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Dog Tracker Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta {{ margin-bottom: 20px; color: #333; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; }}
    th {{ background: #f2f2f2; text-align: left; }}
  </style>
</head>
<body>
  <h1>Dog Tracker Daily Report</h1>
  <div class="meta">
    <div><strong>Total segments:</strong> {len(events)}</div>
    <div><strong>Total highlights:</strong> {len(highlights)}</div>
    <div><strong>Range:</strong> {start_ts} to {end_ts}</div>
  </div>

  <h2>Timeline</h2>
  <img src="{timeline_png}" alt="Timeline chart"/>

  <h2>Motion vs Audio</h2>
  <img src="{scatter_png}" alt="Scatter chart"/>

  <h2>Event Distribution</h2>
  <img src="{counts_png}" alt="Counts chart"/>

  {featured_video_html}

  <h2>Top Highlights</h2>
  <table>
    <thead>
      <tr>
        <th>Timestamp</th>
        <th>Clip</th>
        <th>Score</th>
        <th>Event Type</th>
        <th>Noise Level</th>
        <th>Motion Avg</th>
        <th>Audio Peak RMS</th>
        <th>Video</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows) if rows else '<tr><td colspan="8">No highlights found.</td></tr>'}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Visualize tracker JSON data")
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=script_dir / "recordings",
        help="Directory containing tracker JSON/JSONL data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "reports",
        help="Directory to write PNGs and HTML report",
    )
    args = parser.parse_args()

    recordings_dir = args.recordings_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_segment_events(recordings_dir)
    if not events:
        raise SystemExit(f"No tracker segment data found under: {recordings_dir}")

    timeline_png = output_dir / "timeline.png"
    scatter_png = output_dir / "motion_vs_audio.png"
    counts_png = output_dir / "event_counts.png"
    report_html = output_dir / "dog_tracker_report.html"
    media_dir = output_dir / "highlight_clips"

    render_timeline(events, timeline_png)
    render_scatter(events, scatter_png)
    render_event_counts(events, counts_png)
    highlights = top_highlights(events, limit=20)
    build_html_report(
        events=events,
        highlights=highlights,
        output_path=report_html,
        recordings_dir=recordings_dir,
        media_dir=media_dir,
        timeline_png=timeline_png.name,
        scatter_png=scatter_png.name,
        counts_png=counts_png.name,
    )

    print(f"Report created: {report_html}")


if __name__ == "__main__":
    main()
