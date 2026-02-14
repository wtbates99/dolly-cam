"""
Dog Tracking Application
Records video and audio from webcam, manages storage, and extracts metadata for ML.
"""
import cv2
import numpy as np
import threading
import time
import json
import importlib.util
import wave
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import logging

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    PYAUDIO_AVAILABLE = False
except (ImportError, OSError) as e:
    SOUNDDEVICE_AVAILABLE = False
    if isinstance(e, OSError) and 'PortAudio' in str(e):
        logging.warning("PortAudio library not found. Install with: sudo apt-get install portaudio19-dev")
    PYAUDIO_AVAILABLE = importlib.util.find_spec("pyaudio") is not None
    if not PYAUDIO_AVAILABLE:
        logging.warning("No audio library available - audio recording will be disabled")

from storage_manager import StorageManager
from metadata_extractor import MetadataExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DogTracker:
    """Main application for tracking dogs with video and audio recording."""
    
    def __init__(self, storage_path: str = "./recordings", 
                 video_device: int = 0,
                 audio_device: Optional[int] = None,
                 audio_channels: int = 1,
                 segment_duration: int = 60,
                 fps: int = 30,
                 resolution: tuple = (1280, 720),
                 motion_sample_stride: int = 2,
                 min_audio_rms: float = 0.03,
                 min_motion_avg: float = 0.015,
                 min_motion_peak: float = 0.05):
        """
        Initialize dog tracker.
        
        Args:
            storage_path: Directory to save recordings
            video_device: Camera device index (default: 0)
            audio_device: Audio device index (None for default)
            segment_duration: Duration of each video segment in seconds (default: 60)
            fps: Frames per second (default: 30)
            resolution: Video resolution (width, height) (default: 1280x720)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.highlights_path = self.storage_path / "highlights"
        self.highlights_path.mkdir(parents=True, exist_ok=True)
        self.highlight_index_path = self.highlights_path / "index.jsonl"
        self.segment_index_path = self.storage_path / "segments.jsonl"
        self.video_device = video_device
        self.audio_device = audio_device
        self.requested_audio_channels = max(1, int(audio_channels))
        self.segment_duration = segment_duration
        self.fps = fps
        self.resolution = resolution
        self.motion_sample_stride = max(1, motion_sample_stride)
        self.min_audio_rms = min_audio_rms
        self.min_motion_avg = min_motion_avg
        self.min_motion_peak = min_motion_peak
        
        # Initialize components
        self.storage_manager = StorageManager(storage_path, max_usage_percent=90.0)
        self.metadata_extractor = MetadataExtractor()
        
        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.audio_frames = []
        self.audio_sample_rate = 44100
        self.audio_channels = self.requested_audio_channels
        self.current_segment_path: Optional[Path] = None
        self.current_segment_frame_count = 0
        self.current_segment_motion_scores: List[float] = []
        self.current_segment_audio_peak_rms = 0.0
        self.current_segment_audio_rms_samples: List[float] = []
        self.previous_gray_frame: Optional[np.ndarray] = None
        self.frame_index = 0
        
        # Threading
        self.video_thread = None
        self.audio_thread = None
        self.recording_lock = threading.Lock()
        
        # Check storage before starting
        self.storage_manager.check_and_cleanup()

    def _reset_segment_trackers(self) -> None:
        """Reset all per-segment accumulators."""
        self.current_segment_frame_count = 0
        self.current_segment_motion_scores = []
        self.current_segment_audio_peak_rms = 0.0
        self.current_segment_audio_rms_samples = []
        with self.recording_lock:
            self.audio_frames = []

    def _ingest_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """Store chunk and update audio analytics."""
        if audio_chunk.size == 0:
            return

        chunk_rms = float(np.sqrt(np.mean(np.square(audio_chunk))))
        with self.recording_lock:
            self.audio_frames.append(audio_chunk.copy())
            self.current_segment_audio_peak_rms = max(self.current_segment_audio_peak_rms, chunk_rms)
            # Keep lightweight signal profile for noise/ambient analysis.
            self.current_segment_audio_rms_samples.append(chunk_rms)
    
    def setup_audio(self):
        """Setup audio recording parameters."""
        if SOUNDDEVICE_AVAILABLE:
            devices = sd.query_devices()
            logger.info("Available audio devices:")
            for i, device in enumerate(devices):
                logger.info(f"  {i}: {device['name']} ({device['max_input_channels']} input channels)")
            
            if self.audio_device is None:
                self.audio_device = sd.default.device[0]
            
            device_info = sd.query_devices(self.audio_device)
            self.audio_sample_rate = int(device_info['default_samplerate'])
            max_input = int(device_info['max_input_channels'])
            self.audio_channels = max(1, min(self.requested_audio_channels, max_input, 2))
            logger.info(f"Using audio device {self.audio_device}: {device_info['name']}")
            logger.info(
                "Sample rate: %s Hz, Channels: %s (requested=%s, device_max=%s)",
                self.audio_sample_rate,
                self.audio_channels,
                self.requested_audio_channels,
                max_input,
            )
        
        elif PYAUDIO_AVAILABLE:
            pyaudio = importlib.import_module("pyaudio")
            p = pyaudio.PyAudio()
            if self.audio_device is None:
                self.audio_device = p.get_default_input_device_info()['index']
            
            device_info = p.get_device_info_by_index(self.audio_device)
            self.audio_sample_rate = int(device_info['defaultSampleRate'])
            max_input = int(device_info['maxInputChannels'])
            self.audio_channels = max(1, min(self.requested_audio_channels, max_input, 2))
            logger.info(f"Using audio device {self.audio_device}: {device_info['name']}")
            logger.info(
                "Sample rate: %s Hz, Channels: %s (requested=%s, device_max=%s)",
                self.audio_sample_rate,
                self.audio_channels,
                self.requested_audio_channels,
                max_input,
            )
            p.terminate()
        else:
            logger.warning("No audio library available - audio recording disabled")
    
    def record_audio_sounddevice(self):
        """Record audio using sounddevice library."""
        try:
            with sd.InputStream(device=self.audio_device,
                              channels=self.audio_channels,
                              samplerate=self.audio_sample_rate,
                              dtype='float32',
                              blocksize=int(self.audio_sample_rate * 0.1)) as stream:
                logger.info("Audio recording started (sounddevice)")
                while self.is_recording:
                    audio_chunk, overflowed = stream.read(int(self.audio_sample_rate * 0.1))
                    if overflowed:
                        logger.warning("Audio buffer overflow")
                    self._ingest_audio_chunk(audio_chunk)
        except Exception as e:
            logger.error(f"Error in audio recording: {e}")
    
    def record_audio_pyaudio(self):
        """Record audio using pyaudio library."""
        try:
            pyaudio = importlib.import_module("pyaudio")
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                          channels=self.audio_channels,
                          rate=self.audio_sample_rate,
                          input=True,
                          input_device_index=self.audio_device,
                          frames_per_buffer=int(self.audio_sample_rate * 0.1))
            
            logger.info("Audio recording started (pyaudio)")
            while self.is_recording:
                audio_chunk = stream.read(int(self.audio_sample_rate * 0.1), exception_on_overflow=False)
                audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
                audio_array = audio_array.reshape(-1, self.audio_channels)
                self._ingest_audio_chunk(audio_array)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            logger.error(f"Error in audio recording: {e}")

    def compute_motion_score(self, frame: np.ndarray) -> float:
        """Compute normalized motion score from consecutive frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.previous_gray_frame is None:
            self.previous_gray_frame = gray
            return 0.0

        frame_delta = cv2.absdiff(self.previous_gray_frame, gray)
        self.previous_gray_frame = gray

        # Use thresholded motion area for better robustness to low-light noise.
        _, thresh = cv2.threshold(frame_delta, 20, 255, cv2.THRESH_BINARY)
        motion_pixels = float(np.count_nonzero(thresh))
        total_pixels = float(thresh.size) if thresh.size else 1.0
        motion_score = motion_pixels / total_pixels
        return motion_score

    def classify_noise_profile(self) -> Tuple[float, float, float, str]:
        """
        Return (ambient_rms, peak_rms, spike_ratio, noise_level).

        noise_level is a human-friendly summary for quick timeline reviews.
        """
        if not self.current_segment_audio_rms_samples:
            return 0.0, 0.0, 0.0, "unknown"

        arr = np.array(self.current_segment_audio_rms_samples, dtype=np.float32)
        ambient_rms = float(np.percentile(arr, 25))
        peak_rms = float(np.max(arr))
        spike_ratio = float(peak_rms / max(ambient_rms, 1e-6))

        if ambient_rms < 0.01 and peak_rms < 0.02:
            noise_level = "quiet"
        elif spike_ratio >= 4.0:
            noise_level = "bursty"
        elif ambient_rms >= 0.04:
            noise_level = "loud"
        else:
            noise_level = "normal"

        return ambient_rms, peak_rms, spike_ratio, noise_level
    
    def record_video(self):
        """Record video frames."""
        cap = cv2.VideoCapture(self.video_device)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Get actual properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video: {actual_width}x{actual_height} @ {actual_fps} fps")
        self.previous_gray_frame = None
        self.frame_index = 0
        
        segment_start_time = time.time()
        segment_number = 0
        
        try:
            while self.is_recording:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Check if we need to start a new segment
                elapsed = time.time() - segment_start_time
                if elapsed >= self.segment_duration and self.video_writer is not None:
                    self.finish_segment()
                    segment_number += 1
                    segment_start_time = time.time()
                
                # Start new segment if needed
                if self.video_writer is None:
                    self.start_new_segment(segment_number, actual_width, actual_height, actual_fps)
                
                # Write frame
                if self.video_writer is not None:
                    self.video_writer.write(frame)
                    self.current_segment_frame_count += 1

                if self.frame_index % self.motion_sample_stride == 0:
                    motion_score = self.compute_motion_score(frame)
                    self.current_segment_motion_scores.append(motion_score)
                self.frame_index += 1
                
                # Optional: display frame (comment out for headless operation)
                # cv2.imshow("Dog Tracker", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     self.is_recording = False
                #     break
        
        finally:
            cap.release()
            if self.video_writer is not None:
                self.finish_segment()
    
    def start_new_segment(self, segment_number: int, width: int, height: int, fps: float):
        """Start a new video segment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dog_tracking_{timestamp}_seg{segment_number:04d}.mp4"
        video_path = self.storage_path / filename
        
        # Use H.264 codec (widely supported)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            fps,
            (width, height)
        )
        
        if not self.video_writer.isOpened():
            logger.error(f"Failed to open video writer for {video_path}")
            self.video_writer = None
            return
        
        logger.info(f"Started new segment: {filename}")
        self.current_segment_path = video_path
        self._reset_segment_trackers()
    
    def finish_segment(self):
        """Finish current segment and save metadata."""
        if self.video_writer is None or self.current_segment_path is None:
            return

        video_path = self.current_segment_path
        self.video_writer.release()
        self.video_writer = None
        self.current_segment_path = None
        
        logger.info(f"Finished segment: {video_path.name}")
        
        # Get audio data
        audio_data = None
        with self.recording_lock:
            if self.audio_frames:
                audio_data = np.concatenate(self.audio_frames, axis=0)
                self.audio_frames = []

        video_path = self.embed_audio_into_video(video_path, audio_data)

        motion_avg = float(np.mean(self.current_segment_motion_scores)) if self.current_segment_motion_scores else 0.0
        motion_peak = float(np.max(self.current_segment_motion_scores)) if self.current_segment_motion_scores else 0.0
        ambient_rms, peak_rms, spike_ratio, noise_level = self.classify_noise_profile()

        # Extract metadata + highlight decision
        self.extract_and_save_metadata(
            video_path,
            audio_data,
            frame_count_hint=self.current_segment_frame_count,
            motion_avg=motion_avg,
            motion_peak=motion_peak,
            audio_peak_rms=self.current_segment_audio_peak_rms,
            ambient_rms=ambient_rms,
            peak_rms=peak_rms,
            spike_ratio=spike_ratio,
            noise_level=noise_level,
        )
        
        # Check storage and cleanup if needed
        self.storage_manager.check_and_cleanup()

    def write_audio_wav(self, video_path: Path, audio_data: np.ndarray) -> Optional[Path]:
        """Write segment audio as a WAV sidecar file for ML and muxing."""
        if audio_data is None or len(audio_data) == 0:
            return None

        audio_path = video_path.with_suffix(".wav")
        channels = 1 if audio_data.ndim == 1 else int(audio_data.shape[1])
        if channels > 2:
            # Keep muxing/browser compatibility by downmixing excessive channels.
            audio_data = np.mean(audio_data, axis=1)
            channels = 1

        pcm = np.clip(audio_data, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)

        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.audio_sample_rate)
            wav_file.writeframes(pcm.tobytes())

        return audio_path

    def embed_audio_into_video(self, video_path: Path, audio_data: Optional[np.ndarray]) -> Path:
        """Mux recorded WAV audio into the segment MP4 in place."""
        if audio_data is None or len(audio_data) == 0:
            return video_path

        audio_path = self.write_audio_wav(video_path, audio_data)
        if audio_path is None:
            return video_path

        muxed_path = video_path.with_name(f"{video_path.stem}.mux.mp4")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(muxed_path),
        ]

        try:
            subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            muxed_path.replace(video_path)
            logger.info("Muxed audio into video: %s", video_path.name)
        except Exception as e:
            if muxed_path.exists():
                muxed_path.unlink(missing_ok=True)
            stderr_tail = ""
            if hasattr(e, "stderr") and getattr(e, "stderr"):
                stderr_text = str(getattr(e, "stderr"))
                stderr_tail = stderr_text[-400:]
            logger.warning("Failed to mux audio into %s: %s %s", video_path.name, e, stderr_tail)

        return video_path

    def append_highlight_event(self, metadata: Dict) -> None:
        """Append highlight event metadata to a JSONL index for fast review."""
        self.highlights_path.mkdir(parents=True, exist_ok=True)
        event = {
            'timestamp': metadata.get('timestamp'),
            'video_file': metadata.get('video_file'),
            'video_path': metadata.get('video_path'),
            'highlight_score': metadata.get('highlight_score'),
            'highlight_reasons': metadata.get('highlight_reasons', []),
            'event_type': metadata.get('event_type', 'unknown'),
            'motion_avg': metadata.get('motion_avg', 0.0),
            'motion_peak': metadata.get('motion_peak', 0.0),
            'segment_audio_peak_rms': metadata.get('segment_audio_peak_rms', 0.0),
            'ambient_audio_rms': metadata.get('ambient_audio_rms', 0.0),
            'noise_spike_ratio': metadata.get('noise_spike_ratio', 0.0),
            'noise_level': metadata.get('noise_level', 'unknown'),
            'duration_seconds': metadata.get('video_metadata', {}).get('duration_seconds', 0.0),
        }
        with open(self.highlight_index_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + "\n")

    def append_segment_event(self, metadata: Dict) -> None:
        """Append every segment event for complete day timeline analysis."""
        event = {
            'timestamp': metadata.get('timestamp'),
            'video_file': metadata.get('video_file'),
            'video_path': metadata.get('video_path'),
            'is_highlight': metadata.get('is_highlight', False),
            'highlight_score': metadata.get('highlight_score', 0.0),
            'event_type': metadata.get('event_type', 'unknown'),
            'noise_level': metadata.get('noise_level', 'unknown'),
            'noise_spike_ratio': metadata.get('noise_spike_ratio', 0.0),
            'motion_avg': metadata.get('motion_avg', 0.0),
            'motion_peak': metadata.get('motion_peak', 0.0),
            'duration_seconds': metadata.get('video_metadata', {}).get('duration_seconds', 0.0),
        }
        with open(self.segment_index_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + "\n")
    
    def extract_and_save_metadata(
        self,
        video_path: Path,
        audio_data: Optional[np.ndarray],
        frame_count_hint: int,
        motion_avg: float,
        motion_peak: float,
        audio_peak_rms: float,
        ambient_rms: float,
        peak_rms: float,
        spike_ratio: float,
        noise_level: str,
    ):
        """Extract and save metadata for a video segment."""
        try:
            # Get video metadata
            cap = cv2.VideoCapture(str(video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                frame_count = frame_count_hint
            cap.release()
            
            video_metadata = self.metadata_extractor.extract_video_metadata(
                width, height, fps, frame_count, 'mp4v'
            )
            
            # Extract audio features if available
            audio_features = None
            if audio_data is not None and len(audio_data) > 0:
                audio_mono = self.metadata_extractor._to_mono_float32(audio_data)
                audio_features = self.metadata_extractor.extract_audio_features(
                    audio_mono, self.audio_sample_rate
                )

            # Build highlight score from motion and audio.
            highlight = self.metadata_extractor.compute_highlight_score(
                audio_features=audio_features,
                motion_avg=motion_avg,
                motion_peak=motion_peak,
                audio_peak_rms=audio_peak_rms,
                min_audio_rms=self.min_audio_rms,
                min_motion_avg=self.min_motion_avg,
                min_motion_peak=self.min_motion_peak,
            )
            
            # Create complete metadata
            metadata = self.metadata_extractor.create_metadata(
                video_path,
                audio_features=audio_features,
                video_metadata=video_metadata,
                additional_info={
                    'application': 'dog_tracker',
                    'segment_duration_seconds': self.segment_duration,
                    'motion_avg': motion_avg,
                    'motion_peak': motion_peak,
                    'segment_audio_peak_rms': audio_peak_rms,
                    'ambient_audio_rms': ambient_rms,
                    'noise_peak_rms': peak_rms,
                    'noise_spike_ratio': spike_ratio,
                    'noise_level': noise_level,
                    'is_highlight': highlight['is_highlight'],
                    'highlight_score': highlight['score'],
                    'highlight_reasons': highlight['reasons'],
                    'highlight_thresholds': highlight['thresholds'],
                    'event_type': highlight['event_type'],
                }
            )
            
            # Save metadata
            metadata_path = video_path.with_suffix('.json')
            self.metadata_extractor.save_metadata(metadata, metadata_path)
            self.append_segment_event(metadata)

            # Keep lightweight index of important moments for quick review.
            if highlight['is_highlight']:
                self.append_highlight_event(metadata)
                logger.info(
                    "Saved highlight: %s (score=%.3f, reasons=%s)",
                    video_path.name,
                    highlight['score'],
                    ",".join(highlight['reasons']) or "n/a",
                )
            
        except Exception as e:
            logger.error(f"Error extracting metadata for {video_path}: {e}")
    
    def start_recording(self):
        """Start recording video and audio."""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return
        
        logger.info("Starting dog tracking recording...")
        self.is_recording = True
        
        # Setup audio
        self.setup_audio()
        
        # Start audio thread
        if SOUNDDEVICE_AVAILABLE:
            self.audio_thread = threading.Thread(target=self.record_audio_sounddevice, daemon=True)
        elif PYAUDIO_AVAILABLE:
            self.audio_thread = threading.Thread(target=self.record_audio_pyaudio, daemon=True)
        
        if self.audio_thread:
            self.audio_thread.start()
        
        # Start video recording (runs in main thread)
        self.record_video()
    
    def stop_recording(self):
        """Stop recording."""
        logger.info("Stopping recording...")
        self.is_recording = False
        
        # Wait for audio thread to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        # Finish current segment
        if self.video_writer is not None:
            self.finish_segment()
        
        logger.info("Recording stopped")
    
    def get_storage_info(self):
        """Get storage information."""
        return self.storage_manager.get_storage_info()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dog Tracking Application')
    parser.add_argument('--storage-path', type=str, default='./recordings',
                       help='Path to store recordings (default: ./recordings)')
    parser.add_argument('--video-device', type=int, default=0,
                       help='Video device index (default: 0)')
    parser.add_argument('--audio-device', type=int, default=None,
                       help='Audio device index (default: auto-detect)')
    parser.add_argument('--audio-channels', type=int, default=1,
                       help='Audio channels to capture (1=mono, 2=stereo, default: 1)')
    parser.add_argument('--segment-duration', type=int, default=60,
                       help='Duration of each segment in seconds (default: 60)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--resolution', type=str, default='1280x720',
                       help='Video resolution WIDTHxHEIGHT (default: 1280x720)')
    parser.add_argument('--motion-sample-stride', type=int, default=2,
                       help='Analyze motion every Nth frame (default: 2)')
    parser.add_argument('--min-audio-rms', type=float, default=0.03,
                       help='Audio RMS threshold for highlight scoring (default: 0.03)')
    parser.add_argument('--min-motion-avg', type=float, default=0.015,
                       help='Average motion threshold for highlight scoring (default: 0.015)')
    parser.add_argument('--min-motion-peak', type=float, default=0.05,
                       help='Peak motion threshold for highlight scoring (default: 0.05)')
    parser.add_argument('--storage-info', action='store_true',
                       help='Show storage information and exit')
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Create tracker
    tracker = DogTracker(
        storage_path=args.storage_path,
        video_device=args.video_device,
        audio_device=args.audio_device,
        audio_channels=args.audio_channels,
        segment_duration=args.segment_duration,
        fps=args.fps,
        resolution=(width, height),
        motion_sample_stride=args.motion_sample_stride,
        min_audio_rms=args.min_audio_rms,
        min_motion_avg=args.min_motion_avg,
        min_motion_peak=args.min_motion_peak,
    )
    
    # Show storage info if requested
    if args.storage_info:
        info = tracker.get_storage_info()
        import json
        print(json.dumps(info, indent=2))
        return
    
    # Start recording
    try:
        tracker.start_recording()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        tracker.stop_recording()


if __name__ == "__main__":
    main()
