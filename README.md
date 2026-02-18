# Raspberry Pi Ring-Style Home Camera

This setup is optimized for Raspberry Pi 4 + USB webcam and gives you:

- Motion-triggered MP4 clip recording
- Live stream (`/stream.mjpg`) and live dashboard (`/`)
- Event scoring + highlight indexing (`segments.jsonl`, `highlights/index.jsonl`)
- Offline analytics dashboard report (`reports/home_camera_report.html`)
- Storage auto-cleanup when disk usage crosses a threshold

## Install on Pi

```bash
cd /home/pi/dolly-cam
sudo apt update
sudo apt install -y python3 python3-pip python3-opencv ffmpeg
pip3 install numpy matplotlib
```

## Verify clip saving works (synthetic test)

This test does not require a real webcam.

```bash
python3 ring_camera.py --self-test
```

Expected: `Self-test passed` and exit code `0`.

## Run camera

```bash
python3 ring_camera.py \
  --storage-path ./recordings \
  --video-device 0 \
  --resolution 1280x720 \
  --fps 15 \
  --stream-fps 8 \
  --host 0.0.0.0 \
  --port 8080 \
  --access-token YOUR_LONG_RANDOM_TOKEN
```

Open on LAN:

- `http://<PI_LOCAL_IP>:8080/` (dashboard + live stream)
- `http://<PI_LOCAL_IP>:8080/health`
- `http://<PI_LOCAL_IP>:8080/stream.mjpg?token=YOUR_LONG_RANDOM_TOKEN`

## Tune motion detection

- `--min-motion-area 1200`: increase to reduce false triggers
- `--delta-thresh 20`: increase to reduce sensitivity
- `--background-alpha 0.02`: lower for slower background adaptation
- `--cooldown-seconds 6`: how long to keep recording after motion drops
- `--pre-motion-seconds 2`: pre-roll buffer at clip start
- `--min-clip-seconds 1`: avoid tiny clips
- `--max-clip-seconds 45`: hard clip cap

Example (conservative):

```bash
python3 ring_camera.py --min-motion-area 2400 --delta-thresh 24 --background-alpha 0.015
```

## Generate analytics dashboard report

After clips are recorded:

```bash
python3 visualize_tracker_data.py --recordings-dir ./recordings --output-dir ./reports
```

Open `reports/home_camera_report.html`.

## Run on boot (systemd)

```bash
sudo cp deploy/ring-camera.service /etc/systemd/system/ring-camera.service
sudo nano /etc/systemd/system/ring-camera.service
sudo systemctl daemon-reload
sudo systemctl enable --now ring-camera.service
sudo systemctl status ring-camera.service
```

Set these values in the service file:

- `User`
- `WorkingDirectory`
- `ExecStart` path
- `--access-token`

## Synology router remote access

1. Reserve static DHCP lease for Pi in SRM.
2. Port-forward WAN TCP `8080` to `<PI_LOCAL_IP>:8080`.
3. Configure DDNS hostname in SRM.
4. Prefer HTTPS reverse-proxy on Synology and restrict source access.
5. Use a strong `--access-token`.

## Files

- `ring_camera.py`: motion engine, clip saving, stream server, API, dashboard
- `storage_manager.py`: cleanup policy
- `visualize_tracker_data.py`: offline analytics + HTML report
- `deploy/ring-camera.service`: service template
