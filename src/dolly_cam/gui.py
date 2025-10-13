from __future__ import annotations

import logging
import tkinter as tk
from datetime import datetime
from tkinter import ttk

from .config import TouchscreenConfig
from .recording import RecordingController

LOGGER = logging.getLogger(__name__)


class DollyCamApp:
    def __init__(self, controller: RecordingController, ui_cfg: TouchscreenConfig) -> None:
        self._controller = controller
        self._cfg = ui_cfg

        self._root = tk.Tk()
        self._root.title(self._cfg.window_title)
        if self._cfg.full_screen:
            self._root.attributes("-fullscreen", True)
        self._root.configure(bg="#111111")

        self._status_var = tk.StringVar()
        self._button_text = tk.StringVar()

        self._build_layout()
        self._refresh_status()

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        container = ttk.Frame(self._root, padding=30)
        container.pack(fill=tk.BOTH, expand=True)

        status_label = ttk.Label(
            container,
            textvariable=self._status_var,
            anchor=tk.CENTER,
            font=("Helvetica", self._cfg.status_font_size),
        )
        status_label.pack(fill=tk.X, pady=20)

        toggle_btn = ttk.Button(
            container,
            textvariable=self._button_text,
            command=self._toggle_recording,
        )
        toggle_btn.pack(fill=tk.X, pady=20)
        toggle_btn.configure(width=20)
        toggle_btn.configure(style="Toggle.TButton")

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Toggle.TButton",
            font=("Helvetica", self._cfg.button_font_size, "bold"),
            padding=20,
        )

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
        if status.running:
            button_text = "Stop Recording"
            status_text = "Recording schedule active"
            if status.next_run:
                formatted = _format_time(status.next_run)
                status_text = f"Recording active\nNext clip at {formatted}"
        else:
            button_text = "Start Recording"
            status_text = "Recording is idle"
            if status.next_run:
                formatted = _format_time(status.next_run)
                status_text += f"\nNext scheduled clip at {formatted}"
        if status.last_clip:
            status_text += f"\nLast clip: {status.last_clip.name}"
        if status.last_error:
            status_text += f"\nLast error: {status.last_error}"

        self._button_text.set(button_text)
        self._status_var.set(status_text)

        self._root.after(self._cfg.poll_interval_ms, self._refresh_status)

    def run(self) -> None:
        LOGGER.info("Touchscreen UI ready")
        self._root.mainloop()

    def _on_close(self) -> None:
        LOGGER.info("Shutting down application")
        self._controller.shutdown()
        self._root.destroy()


def _format_time(value: datetime | None) -> str:
    if not value:
        return ""
    return value.strftime("%H:%M:%S")
