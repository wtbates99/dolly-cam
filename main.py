"""
Dog Tracking Application
Records video and audio from webcam, manages storage, and extracts metadata for ML.
"""
import cv2
import numpy as np
import threading
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import logging

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    PYAUDIO_AVAILABLE = False
except (ImportError, OSError) as e:
    SOUNDDEVICE_AVAILABLE = False
    if isinstance(e, OSError) and 'PortAudio' in str(e):
        logging.warning("PortAudio library not found. Install with: sudo apt-get install portaudio19-dev")
    try:
        import pyaudio
        PYAUDIO_AVAILABLE = True
    except (ImportError, OSError):
        PYAUDIO_AVAILABLE = False
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
        self.video_device = video_device
        self.audio_device = audio_device
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
        self.audio_channels = 1
        self.current_segment_path: Optional[Path] = None
        self.current_segment_frame_count = 0
        self.current_segment_motion_scores: List[float] = []
        self.current_segment_audio_peak_rms = 0.0
        self.previous_gray_frame: Optional[np.ndarray] = None
        self.frame_index = 0
        
        # Threading
        self.video_thread = None
        self.audio_thread = None
        self.recording_lock = threading.Lock()
        
        # Check storage before starting
        self.storage_manager.check_and_cleanup()
    
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
            self.audio_channels = device_info['max_input_channels']
            logger.info(f"Using audio device {self.audio_device}: {device_info['name']}")
            logger.info(f"Sample rate: {self.audio_sample_rate} Hz, Channels: {self.audio_channels}")
        
        elif PYAUDIO_AVAILABLE:
            p = pyaudio.PyAudio()
            if self.audio_device is None:
                self.audio_device = p.get_default_input_device_info()['index']
            
            device_info = p.get_device_info_by_index(self.audio_device)
            self.audio_sample_rate = int(device_info['defaultSampleRate'])
            self.audio_channels = device_info['maxInputChannels']
            logger.info(f"Using audio device {self.audio_device}: {device_info['name']}")
            logger.info(f"Sample rate: {self.audio_sample_rate} Hz, Channels: {self.audio_channels}")
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
                    chunk_rms = float(np.sqrt(np.mean(np.square(audio_chunk)))) if audio_chunk.size else 0.0
                    with self.recording_lock:
                        self.audio_frames.append(audio_chunk.copy())
                        self.current_segment_audio_peak_rms = max(self.current_segment_audio_peak_rms, chunk_rms)
        except Exception as e:
            logger.error(f"Error in audio recording: {e}")
    
    def record_audio_pyaudio(self):
        """Record audio using pyaudio library."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                          channels=self.audio_channels,
                          rate=self.audio_sample_rate,
                          input=True,
                          input_device_index=self.audio_device,
                          frames_per_buffer=int(self.audio_sample_rate * 0.1))
            
            logger.info("Audio recording started (pyaudio)")
            while self.is_recording:
                audio_chunk = stream.read(int(self.audio_sample_rate * 0.1))
                audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
                audio_array = audio_array.reshape(-1, self.audio_channels)
                chunk_rms = float(np.sqrt(np.mean(np.square(audio_array)))) if audio_array.size else 0.0
                with self.recording_lock:
                    self.audio_frames.append(audio_array.copy())
                    self.current_segment_audio_peak_rms = max(self.current_segment_audio_peak_rms, chunk_rms)
            
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
        self.current_segment_frame_count = 0
        self.current_segment_motion_scores = []
        self.current_segment_audio_peak_rms = 0.0
        
        # Clear audio frames for new segment
        with self.recording_lock:
            self.audio_frames = []
    
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

        motion_avg = float(np.mean(self.current_segment_motion_scores)) if self.current_segment_motion_scores else 0.0
        motion_peak = float(np.max(self.current_segment_motion_scores)) if self.current_segment_motion_scores else 0.0

        # Extract metadata + highlight decision
        self.extract_and_save_metadata(
            video_path,
            audio_data,
            frame_count_hint=self.current_segment_frame_count,
            motion_avg=motion_avg,
            motion_peak=motion_peak,
            audio_peak_rms=self.current_segment_audio_peak_rms,
        )
        
        # Check storage and cleanup if needed
        self.storage_manager.check_and_cleanup()

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
            'duration_seconds': metadata.get('video_metadata', {}).get('duration_seconds', 0.0),
        }
        with open(self.highlight_index_path, 'a') as f:
            f.write(json.dumps(event) + "\n")
    
    def extract_and_save_metadata(
        self,
        video_path: Path,
        audio_data: Optional[np.ndarray],
        frame_count_hint: int,
        motion_avg: float,
        motion_peak: float,
        audio_peak_rms: float,
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
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_mono = np.mean(audio_data, axis=1)
                else:
                    audio_mono = audio_data
                
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
