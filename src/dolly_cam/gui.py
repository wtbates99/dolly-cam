from __future__ import annotations

import logging
import random
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from .config import TouchscreenConfig
from .recording import RecordingController

LOGGER = logging.getLogger(__name__)


class ClipPlayer:
    """Cycles through locally saved clips for idle playback."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._capture: Optional[cv2.VideoCapture] = None
        self._current: Optional[Path] = None

    def next_frame(self) -> Optional["cv2.Mat"]:
        for _ in range(2):
            capture = self._ensure_capture()
            if capture is None:
                return None
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
            self._close_capture()
        return None

    def reset(self) -> None:
        self._close_capture()

    def _ensure_capture(self) -> Optional[cv2.VideoCapture]:
        if self._capture and self._capture.isOpened():
            return self._capture
        clips = [p for p in self._directory.glob("*.mp4") if p.is_file()]
        if not clips:
            return None
        random.shuffle(clips)
        for candidate in clips:
            capture = cv2.VideoCapture(str(candidate))
            if capture.isOpened():
                self._current = candidate
                self._capture = capture
                return capture
            capture.release()
        return None

    def _close_capture(self) -> None:
        if self._capture:
            self._capture.release()
        self._capture = None
        self._current = None


class DollyCamApp:
    def __init__(self, controller: RecordingController, ui_cfg: TouchscreenConfig) -> None:
        self._controller = controller
        self._cfg = ui_cfg

        self._root = tk.Tk()
        self._root.title(self._cfg.window_title)
        self._root.configure(bg=self._cfg.background_color)
        self._root.geometry(f"{self._cfg.width}x{self._cfg.height}")
        self._root.minsize(self._cfg.width, self._cfg.height)
        if self._cfg.full_screen:
            self._root.attributes("-fullscreen", True)

        self._status_var = tk.StringVar()
        self._button_text = tk.StringVar()

        self._video_label: Optional[tk.Label] = None
        self._video_overlay: Optional[tk.Label] = None
        self._video_photo: Optional[ImageTk.PhotoImage] = None
        self._video_width = int(self._cfg.width * 0.7)
        self._video_height = max(int(self._cfg.height * 0.65), 240)
        self._video_poll_ms = 90

        self._idle_player = ClipPlayer(self._controller.get_recording_directory())
        self._was_recording = False

        self._build_layout()
        self._refresh_status()
        self._refresh_video()

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Main.TFrame", background=self._cfg.background_color)
        style.configure("Card.TFrame", background=self._cfg.secondary_color)
        style.configure("Title.TLabel", background=self._cfg.background_color, foreground=self._cfg.accent_color)
        style.configure(
            "Info.TLabel",
            background=self._cfg.secondary_color,
            foreground="#2d1b46",
            font=("Helvetica", self._cfg.status_font_size),
            wraplength=int(self._cfg.width * 0.26),
            justify=tk.LEFT,
        )
        style.configure(
            "Toggle.TButton",
            padding=(18, 14),
            font=("Helvetica", self._cfg.button_font_size, "bold"),
            background=self._cfg.accent_color,
            foreground=self._cfg.accent_text_color,
        )
        style.map(
            "Toggle.TButton",
            background=[("active", self._cfg.accent_color), ("disabled", "#c9c0de")],
            foreground=[("disabled", "#6d6488")],
        )

        container = ttk.Frame(self._root, style="Main.TFrame")
        container.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

        title = ttk.Label(
            container,
            text="Dolly Cam",
            style="Title.TLabel",
            font=("Helvetica", 32, "bold"),
        )
        title.pack(anchor=tk.W, pady=(0, 12))

        main = ttk.Frame(container, style="Main.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        video_card = tk.Frame(
            main,
            bg=self._cfg.secondary_color,
            highlightbackground=self._cfg.accent_color,
            highlightthickness=3,
            bd=0,
            relief=tk.FLAT,
        )
        video_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))

        self._video_label = tk.Label(video_card, bg="#1b1033")
        self._video_label.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        self._video_overlay = tk.Label(
            video_card,
            text="Warming up Dolly's spotlight...",
            fg=self._cfg.accent_text_color,
            bg=self._cfg.accent_color,
            font=("Helvetica", 18, "bold"),
            padx=16,
            pady=10,
        )
        self._video_overlay.place(relx=0.5, rely=0.5, anchor="center")

        info_card = tk.Frame(
            main,
            bg=self._cfg.secondary_color,
            highlightbackground=self._cfg.accent_color,
            highlightthickness=3,
            bd=0,
            relief=tk.FLAT,
        )
        info_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        status_label = ttk.Label(info_card, textvariable=self._status_var, style="Info.TLabel")
        status_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=(18, 12))

        toggle_btn = ttk.Button(
            info_card,
            textvariable=self._button_text,
            command=self._toggle_recording,
            style="Toggle.TButton",
        )
        toggle_btn.pack(fill=tk.X, padx=20, pady=(0, 20))

    def _toggle_recording(self) -> None:
        if self._controller.is_running():
            LOGGER.info("Recording toggle: stop")
            self._controller.stop()
        else:
            LOGGER.info("Recording toggle: start")
            self._controller.start()
        self._refresh_status()

    def _refresh_status(self) -> None:
        status = self._controller.get_status()
        status_lines: list[str] = []

        if status.running:
            self._button_text.set("Pause Dolly Cam")
        else:
            self._button_text.set("Start Dolly Cam")

        if status.recording_now:
            status_lines.append("🎥 Dolly is on camera right now!")
        elif status.running:
            status_lines.append("🐾 Waiting for the next Dolly spotlight clip")
        else:
            status_lines.append("Tap start to begin capturing Dolly moments")

        if status.next_run and not status.recording_now:
            formatted = _format_time(status.next_run)
            status_lines.append(f"Next clip scheduled at {formatted}")

        if status.last_clip:
            status_lines.append(f"Last clip saved: {status.last_clip.name}")

        if status.last_error:
            status_lines.append(f"⚠️ Last error: {status.last_error}")

        self._status_var.set("\n\n".join(status_lines))

        is_recording = status.recording_now
        if is_recording and not self._was_recording:
            self._idle_player.reset()
            if self._video_overlay:
                self._video_overlay.configure(text="Recording Dolly in real time!")
        elif not is_recording and self._was_recording:
            if self._video_overlay:
                self._video_overlay.configure(text="Playing Dolly highlights ✨")
        self._was_recording = is_recording

        self._root.after(self._cfg.poll_interval_ms, self._refresh_status)

    def _refresh_video(self) -> None:
        status = self._controller.get_status()
        frame = None
        if status.recording_now:
            frame = self._controller.get_preview_frame()
        else:
            frame = self._idle_player.next_frame()

        if frame is not None:
            self._show_frame(frame)
        else:
            self._show_placeholder(status)

        self._root.after(self._video_poll_ms, self._refresh_video)

    def _show_frame(self, frame) -> None:  # pragma: no cover - GUI rendering
        if self._video_label is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = image.resize((self._video_width, self._video_height), Image.LANCZOS)
        self._video_photo = ImageTk.PhotoImage(image)
        self._video_label.configure(image=self._video_photo)
        if self._video_overlay:
            self._video_overlay.place_forget()

    def _show_placeholder(self, status) -> None:
        if not self._video_label or not self._video_overlay:
            return
        self._video_label.configure(image="")
        self._video_photo = None
        if status.recording_now:
            message = "Capturing Dolly..."
        elif status.last_clip:
            message = "Finding Dolly highlights"
        else:
            message = "Waiting for Dolly to appear"
        self._video_overlay.configure(text=message)
        self._video_overlay.place(relx=0.5, rely=0.5, anchor="center")

    def run(self) -> None:
        LOGGER.info("Touchscreen UI ready")
        self._root.mainloop()

    def _on_close(self) -> None:
        LOGGER.info("Shutting down application")
        self._idle_player.reset()
        self._controller.shutdown()
        self._root.destroy()


def _format_time(value: datetime | None) -> str:
    if not value:
        return ""
    return value.strftime("%I:%M %p").lstrip("0")
