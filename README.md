# Dolly Cam

Touchscreen-friendly Raspberry Pi app that schedules Logitech webcam recordings, uploads clips to Google Drive, and prunes anything older than three days.

## Features
- One-tap start/stop button designed for a Pi-mounted touchscreen.
- Records five minute clips (`duration_seconds`) on a fifteen minute cadence (`interval_minutes`).
- Uses `ffmpeg` to capture from a USB/V4L2 webcam (tested with Logitech devices).
- Background worker uploads finished clips to a Google Drive folder via a service-account.
- Automatic retention policy deletes clips older than three days both locally and in Drive.

## Hardware Assumptions
- Raspberry Pi 4 (or newer) running Raspberry Pi OS (Bullseye or later).
- Official Raspberry Pi touchscreen (or any display that can run a Tkinter GUI).
- Logitech USB webcam that exposes a V4L2 device (typically `/dev/video0`).
- Reliable network connection for Drive uploads.

## Software Prerequisites
1. Install system packages:
   ```bash
   sudo apt update
   sudo apt install python3 python3-venv python3-pip python3-tk ffmpeg v4l-utils
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install Python dependencies:
   ```bash
   pip install --upgrade pip
   pip install .
   ```

## Google Drive Setup
1. Create or reuse a Google Cloud project and enable the **Google Drive API**.
2. Create a **service account**, download the JSON key file, and copy it to the Pi (e.g. `/home/pi/dolly_cam/service-account.json`).
3. In Google Drive, create a folder for clips, copy the folder ID from the URL, and **share the folder with the service account email** so it has write permission.
4. Update the configuration file with the service account path and folder ID.

> Tip: You can disable Drive uploads by setting `enabled = false` in the `[google_drive]` block. The recorder will keep working locally.

## Configuration
1. Copy the example config and edit it:
   ```bash
   cp config.example.toml config.toml
   nano config.toml
   ```
2. Adjust at least the following:
   - `camera_device` if your webcam is not `/dev/video0` (`v4l2-ctl --list-devices` helps).
   - `ffmpeg_path` if `ffmpeg` lives somewhere else (`which ffmpeg`).
   - `output_dir` where clips should be stored locally.
   - `credentials_file` and `folder_id` once the Drive service account is ready.

Key options in `config.toml`:
- `[recording]` – cadence and capture parameters.
- `[google_drive]` – Drive credentials, folder, and chunk size.
- `[retention]` – number of days to keep clips (3 by default).
- `[touchscreen]` – UI text, font sizes, and refresh interval.

The loader ensures the output directory exists before recording.

## Running the App
From the project root (with your virtual environment active if you made one):
```bash
python3 main.py --config config.toml
```

You can still launch via `python -m dolly_cam.app` if you prefer the module entry point.

The Tkinter window will fill the touchscreen (set `full_screen = false` to keep it windowed). Tap **Start Recording** to activate the scheduler: a five minute clip is recorded immediately and then every fifteen minutes. Tap **Stop Recording** to pause the schedule. Status text shows the next clip window, last clip filename, and any recent error.

### Service Mode (optional)
To start the recorder automatically on boot, create a `systemd` unit such as `/etc/systemd/system/dolly-cam.service`:
```ini
[Unit]
Description=Dolly Cam Recorder
After=multi-user.target network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/dolly-cam
ExecStart=/home/pi/dolly-cam/.venv/bin/python -m dolly_cam.app --config /home/pi/dolly-cam/config.toml
Restart=on-failure
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/pi/.Xauthority

[Install]
WantedBy=graphical.target
```
Then enable it with:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now dolly-cam.service
```

Adjust paths/usernames to match your installation.

## Maintenance & Troubleshooting
- **Check camera availability**: `v4l2-ctl --list-devices` or `ffmpeg -f v4l2 -list_formats all -i /dev/video0`.
- **Logs**: the app logs to stdout; for long-running installs, redirect output to a file or configure `systemd` to journal the logs.
- **Retention**: local files and Drive items older than `retention.days` are deleted after each upload cycle.
- **Network dropouts**: uploads retry on the next scheduled clip; the source file remains locally until Drive confirms the upload.

## Development
- Launch via `python -m dolly_cam.app --config config.toml --log-level DEBUG` for verbose logs.
- Run `pip install -e .[dev]` if you add linting/formatting tools.

Enjoy keeping an eye on your dog!
