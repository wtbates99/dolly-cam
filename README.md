# Dog Tracking Application

A Raspberry Pi application for tracking dogs using webcam video and audio recording. Features intelligent storage management and metadata extraction for ML model training.

## Features

- **Video Recording**: Records video from USB webcam with configurable resolution and FPS
- **Audio Recording**: Simultaneous audio capture with the video
- **Smart Storage Management**: Automatically deletes old footage when storage exceeds 90% capacity
- **Highlight Capture**: Scores segments by motion + audio and writes a fast review index at `recordings/highlights/index.jsonl`
- **Metadata Extraction**: Extracts comprehensive metadata including:
  - Video properties (resolution, FPS, duration, codec)
  - Audio features (MFCC, spectral centroid, chroma, tempo, RMS energy)
  - File information (size, timestamps)
- **Segment-based Recording**: Records in configurable time segments (default: 60 seconds)
- **ML-Ready Data**: All metadata saved as JSON files alongside video files

## Requirements

- Raspberry Pi (tested on Raspberry Pi 4)
- USB Webcam
- Python 3.8+
- Sufficient storage space for recordings

## Installation

1. Clone or download this repository:
```bash
cd /path/to/dolly-cam
```

2. Install system dependencies (required for audio support):
```bash
# On Ubuntu/Debian/Raspberry Pi OS:
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio

# On other Linux distributions, install PortAudio development libraries
# For example on Fedora: sudo dnf install portaudio-devel
```

3. Install Python dependencies:
```bash
pip3 install -r requirements.txt
```

**Note**: PortAudio is a system library required for audio recording. If you see "PortAudio library not found" error, install it using the system package manager (not pip/uv).

For audio support, you can use either:
- `sounddevice` (recommended, requires PortAudio system library)
- `pyaudio` (alternative, also requires PortAudio)

If you encounter issues with `librosa`, it's optional - the application will work without it but with limited audio features.

## Usage

### Basic Usage

Start recording with default settings:
```bash
python3 main.py
```

### Command Line Options

```bash
python3 main.py [OPTIONS]

Options:
  --storage-path PATH       Path to store recordings (default: ./recordings)
  --video-device INDEX      Video device index (default: 0)
  --audio-device INDEX      Audio device index (default: auto-detect)
  --segment-duration SEC    Duration of each segment in seconds (default: 60)
  --fps FPS                 Frames per second (default: 30)
  --resolution WIDTHxHEIGHT Video resolution (default: 1280x720)
  --motion-sample-stride N  Analyze motion every Nth frame (default: 2)
  --min-audio-rms FLOAT     Audio threshold for highlights (default: 0.03)
  --min-motion-avg FLOAT    Avg motion threshold for highlights (default: 0.015)
  --min-motion-peak FLOAT   Peak motion threshold for highlights (default: 0.05)
  --storage-info            Show storage information and exit
```

### Examples

Record with custom settings:
```bash
python3 main.py --storage-path /mnt/usb/recordings --segment-duration 120 --fps 15 --resolution 1920x1080
```

Check storage usage:
```bash
python3 main.py --storage-info
```

### Stopping Recording

Press `Ctrl+C` to stop recording gracefully. The current segment will be finalized and metadata will be saved.

## Storage Management

The application automatically monitors disk usage and deletes footage when storage exceeds 90% capacity. Cleanup prioritizes deleting non-highlight clips first; highlights are only pruned as a last resort and a minimum set is retained.

Storage information includes:
- Total/used/free disk space
- Number of video files
- Oldest files list
- Current usage percentage

## Output Files

### Video Files
- Format: `dog_tracking_YYYYMMDD_HHMMSS_seg####.mp4`
- Codec: MP4V (H.264 compatible)
- Location: `storage_path/` directory

### Metadata Files
- Format: `dog_tracking_YYYYMMDD_HHMMSS_seg####.json`
- Contains:
  - Video metadata (resolution, FPS, duration, codec)
  - Audio features (if available)
  - Timestamps
  - File sizes
  - ML-ready features (MFCC, spectral features, etc.)
  - Highlight fields (`is_highlight`, `highlight_score`, `highlight_reasons`)

### Highlight Review Index
- Location: `storage_path/highlights/index.jsonl`
- One JSON event per line with score, reason, and clip path for fast "what happened?" review

## Metadata for ML Models

The extracted metadata includes features commonly used in ML models:

### Audio Features
- **MFCC (Mel-frequency cepstral coefficients)**: 13 coefficients with mean/std
- **Spectral Centroid**: Mean and standard deviation
- **Chroma**: 12-dimensional pitch class profile
- **Tempo**: Estimated beats per minute
- **RMS Energy**: Root mean square energy
- **Zero Crossing Rate**: Indicator of noise/speech

### Video Features
- Resolution and aspect ratio
- Frame count and duration
- FPS and codec information

All features are normalized and ready for ML model training.

## Troubleshooting

### Camera Not Found
- Check camera is connected: `lsusb`
- List video devices: `v4l2-ctl --list-devices`
- Try different `--video-device` index

### Audio Issues

**PortAudio library not found:**
```bash
# Install PortAudio system library:
sudo apt-get install portaudio19-dev

# Then reinstall sounddevice if needed:
pip3 install --force-reinstall sounddevice
```

**Other audio troubleshooting:**
- List audio devices: `python3 -c "import sounddevice as sd; print(sd.query_devices())"`
- Try different `--audio-device` index
- Check microphone permissions
- If sounddevice doesn't work, try pyaudio: `sudo apt-get install python3-pyaudio`

### Storage Issues
- Ensure sufficient disk space
- Check write permissions on storage path
- Monitor storage: `python3 main.py --storage-info`

### Performance on Raspberry Pi
- Lower resolution: `--resolution 640x480`
- Lower FPS: `--fps 15`
- Longer segments: `--segment-duration 120` (reduces file overhead)

## Running as a Service

To run continuously on Raspberry Pi, you can create a systemd service:

```bash
sudo nano /etc/systemd/system/dog-tracker.service
```

Add:
```ini
[Unit]
Description=Dog Tracking Application
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/path/to/dolly-cam
ExecStart=/usr/bin/python3 /path/to/dolly-cam/main.py --storage-path /mnt/usb/recordings
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable dog-tracker.service
sudo systemctl start dog-tracker.service
```

## License

MIT License - feel free to modify and use as needed.
