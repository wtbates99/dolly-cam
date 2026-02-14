"""
Metadata extraction module for ML model training.
Extracts audio features, video properties, and other metadata from recordings.
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import logging

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available - audio features will be limited")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts metadata from video and audio for ML model training."""
    
    def __init__(self):
        """Initialize metadata extractor."""
        self.librosa_available = LIBROSA_AVAILABLE
    
    def extract_audio_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """
        Extract audio features from audio data.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary of audio features
        """
        features = {
            'sample_rate': sample_rate,
            'duration_seconds': len(audio_data) / sample_rate if len(audio_data) > 0 else 0,
            'num_samples': len(audio_data),
        }
        
        if len(audio_data) == 0:
            return features
        
        # Basic statistics
        features['audio_mean'] = float(np.mean(audio_data))
        features['audio_std'] = float(np.std(audio_data))
        features['audio_max'] = float(np.max(np.abs(audio_data)))
        features['audio_min'] = float(np.min(audio_data))
        
        # RMS energy
        features['rms_energy'] = float(np.sqrt(np.mean(audio_data**2)))
        
        # Zero crossing rate (indicator of noise/speech)
        if len(audio_data) > 1:
            zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
            features['zero_crossing_rate'] = float(zero_crossings / len(audio_data))
        
        # Advanced features with librosa if available
        if self.librosa_available:
            try:
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_mono = np.mean(audio_data, axis=1)
                else:
                    audio_mono = audio_data
                
                # Ensure float32 format
                if audio_mono.dtype != np.float32:
                    audio_mono = audio_mono.astype(np.float32)
                
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=audio_mono, sr=sample_rate)[0]
                features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
                features['spectral_centroid_std'] = float(np.std(spectral_centroids))
                
                # MFCC features (commonly used in audio ML)
                mfccs = librosa.feature.mfcc(y=audio_mono, sr=sample_rate, n_mfcc=13)
                features['mfcc_mean'] = [float(x) for x in np.mean(mfccs, axis=1)]
                features['mfcc_std'] = [float(x) for x in np.std(mfccs, axis=1)]
                
                # Chroma features (musical pitch information)
                chroma = librosa.feature.chroma_stft(y=audio_mono, sr=sample_rate)
                features['chroma_mean'] = [float(x) for x in np.mean(chroma, axis=1)]
                
                # Tempo estimation
                tempo, _ = librosa.beat.beat_track(y=audio_mono, sr=sample_rate)
                features['tempo'] = float(tempo)
                
            except Exception as e:
                logger.warning(f"Error extracting advanced audio features: {e}")
        
        return features
    
    def extract_video_metadata(self, width: int, height: int, fps: float, 
                               frame_count: int, codec: str) -> Dict:
        """
        Extract video metadata.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            frame_count: Total number of frames
            codec: Video codec used
            
        Returns:
            Dictionary of video metadata
        """
        duration = frame_count / fps if fps > 0 else 0
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration_seconds': duration,
            'codec': codec,
            'resolution': f"{width}x{height}",
            'aspect_ratio': width / height if height > 0 else 0,
            'total_pixels': width * height,
        }
    
    def create_metadata(self, video_path: Path, audio_features: Optional[Dict] = None,
                       video_metadata: Optional[Dict] = None, 
                       additional_info: Optional[Dict] = None) -> Dict:
        """
        Create comprehensive metadata dictionary.
        
        Args:
            video_path: Path to video file
            audio_features: Audio features dictionary
            video_metadata: Video metadata dictionary
            additional_info: Additional information dictionary
            
        Returns:
            Complete metadata dictionary
        """
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'video_file': str(video_path.name),
            'video_path': str(video_path),
        }
        
        if audio_features:
            metadata['audio_features'] = audio_features
        
        if video_metadata:
            metadata['video_metadata'] = video_metadata
        
        if additional_info:
            metadata.update(additional_info)
        
        # File size
        if video_path.exists():
            metadata['file_size_bytes'] = video_path.stat().st_size
            metadata['file_size_gb'] = video_path.stat().st_size / (1024 ** 3)
        
        return metadata
    
    def save_metadata(self, metadata: Dict, output_path: Path):
        """
        Save metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {output_path}")
    
    def load_metadata(self, metadata_path: Path) -> Dict:
        """
        Load metadata from JSON file.
        
        Args:
            metadata_path: Path to metadata JSON file
            
        Returns:
            Metadata dictionary
        """
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def compute_highlight_score(
        self,
        audio_features: Optional[Dict],
        motion_avg: float,
        motion_peak: float,
        audio_peak_rms: float = 0.0,
        min_audio_rms: float = 0.03,
        min_motion_avg: float = 0.015,
        min_motion_peak: float = 0.05,
    ) -> Dict:
        """
        Compute a simple highlight score using audio and motion signals.

        Returns:
            Dictionary with score, flags, and explanation.
        """
        audio_rms = 0.0
        if audio_features:
            audio_rms = float(audio_features.get('rms_energy', 0.0))
        effective_audio = max(audio_rms, float(audio_peak_rms))

        # Normalize each signal into [0, 1] for a weighted score.
        audio_score = min(effective_audio / max(min_audio_rms, 1e-6), 1.0)
        motion_avg_score = min(motion_avg / max(min_motion_avg, 1e-6), 1.0)
        motion_peak_score = min(motion_peak / max(min_motion_peak, 1e-6), 1.0)

        combined_score = (0.4 * audio_score) + (0.3 * motion_avg_score) + (0.3 * motion_peak_score)

        reasons = []
        if effective_audio >= min_audio_rms:
            reasons.append('audio_peak')
        if motion_avg >= min_motion_avg:
            reasons.append('sustained_motion')
        if motion_peak >= min_motion_peak:
            reasons.append('motion_spike')

        is_highlight = len(reasons) > 0 and combined_score >= 0.45
        if ('audio_peak' in reasons) and ('motion_spike' in reasons or 'sustained_motion' in reasons):
            event_type = 'audio_and_motion'
        elif 'audio_peak' in reasons:
            event_type = 'audio_only'
        elif reasons:
            event_type = 'motion_only'
        else:
            event_type = 'quiet'

        return {
            'is_highlight': is_highlight,
            'score': float(combined_score),
            'audio_rms': float(audio_rms),
            'audio_peak_rms': float(audio_peak_rms),
            'effective_audio_rms': float(effective_audio),
            'motion_avg': float(motion_avg),
            'motion_peak': float(motion_peak),
            'event_type': event_type,
            'reasons': reasons,
            'thresholds': {
                'min_audio_rms': float(min_audio_rms),
                'min_motion_avg': float(min_motion_avg),
                'min_motion_peak': float(min_motion_peak),
                'min_combined_score': 0.45,
            },
        }
