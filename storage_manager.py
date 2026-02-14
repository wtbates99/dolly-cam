"""
Storage management module for monitoring disk usage and cleaning up old footage.
Ensures storage never exceeds 90% capacity.
"""
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageManager:
    """Manages storage space by monitoring usage and deleting old footage."""

    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov')

    def __init__(
        self,
        storage_path: str,
        max_usage_percent: float = 90.0,
        min_highlight_files_to_keep: int = 25,
    ):
        """
        Initialize storage manager.
        
        Args:
            storage_path: Path to directory where footage is stored
            max_usage_percent: Maximum disk usage percentage before cleanup (default: 90%)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_usage_percent = max_usage_percent
        self.min_highlight_files_to_keep = max(min_highlight_files_to_keep, 0)
        
    def get_disk_usage(self) -> Tuple[float, float]:
        """
        Get current disk usage statistics.
        
        Returns:
            Tuple of (used_percent, free_gb)
        """
        stat = shutil.disk_usage(self.storage_path)
        total_gb = stat.total / (1024 ** 3)
        used_gb = stat.used / (1024 ** 3)
        free_gb = stat.free / (1024 ** 3)
        used_percent = (stat.used / stat.total) * 100
        
        logger.info(f"Disk usage: {used_percent:.1f}% used, {free_gb:.2f} GB free")
        return used_percent, free_gb
    
    def get_oldest_files(self, num_files: int = 10) -> List[Path]:
        """
        Get oldest files in storage directory.
        
        Args:
            num_files: Number of oldest files to return
            
        Returns:
            List of Path objects sorted by modification time (oldest first)
        """
        normal_files = self._collect_video_files(only_highlights=False)
        normal_files.sort(key=lambda p: p.stat().st_mtime)
        return normal_files[:num_files]

    def _collect_video_files(self, only_highlights: bool) -> List[Path]:
        """Collect video files recursively filtered by highlight flag."""
        video_files: List[Path] = []
        for ext in self.VIDEO_EXTENSIONS:
            video_files.extend(self.storage_path.rglob(f'*{ext}'))

        return [p for p in video_files if self._is_highlight_file(p) == only_highlights]

    def _is_highlight_file(self, video_path: Path) -> bool:
        """Determine highlight status from sidecar metadata file."""
        metadata_file = video_path.with_suffix('.json')
        if not metadata_file.exists():
            return False

        try:
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return bool(metadata.get('is_highlight', False))
        except Exception:
            # If metadata cannot be parsed, treat as normal footage.
            return False
    
    def cleanup_old_footage(self, target_free_gb: float = None) -> int:
        """
        Delete oldest footage until storage usage is below threshold.
        
        Args:
            target_free_gb: Target free space in GB (optional, uses max_usage_percent if not provided)
            
        Returns:
            Number of files deleted
        """
        used_percent, free_gb = self.get_disk_usage()
        
        if used_percent < self.max_usage_percent:
            logger.info(f"Storage usage ({used_percent:.1f}%) is below threshold ({self.max_usage_percent}%)")
            return 0
        
        # Calculate how much space we need to free
        if target_free_gb is None:
            stat = shutil.disk_usage(self.storage_path)
            total_gb = stat.total / (1024 ** 3)
            target_free_gb = total_gb * (1 - self.max_usage_percent / 100)
        
        current_free_gb = free_gb
        files_deleted = 0
        
        logger.info(f"Starting cleanup: {used_percent:.1f}% used, need {target_free_gb:.2f} GB free")
        
        while current_free_gb < target_free_gb:
            normal_files = self._collect_video_files(only_highlights=False)
            normal_files.sort(key=lambda p: p.stat().st_mtime)

            highlight_files = self._collect_video_files(only_highlights=True)
            highlight_files.sort(key=lambda p: p.stat().st_mtime)

            if normal_files:
                file_to_delete = normal_files[0]
                delete_reason = "normal footage"
            elif len(highlight_files) > self.min_highlight_files_to_keep:
                file_to_delete = highlight_files[0]
                delete_reason = "highlight footage (fallback)"
            else:
                logger.warning("No more files to delete")
                break

            file_size_gb = file_to_delete.stat().st_size / (1024 ** 3)
            
            try:
                # Also delete associated metadata file if it exists
                metadata_file = file_to_delete.with_suffix('.json')
                if metadata_file.exists():
                    metadata_file.unlink()
                
                file_to_delete.unlink()
                files_deleted += 1
                current_free_gb += file_size_gb
                
                logger.info(f"Deleted {delete_reason}: {file_to_delete.name} ({file_size_gb:.2f} GB)")
                
            except Exception as e:
                logger.error(f"Error deleting {file_to_delete}: {e}")
                break
        
        used_percent, free_gb = self.get_disk_usage()
        logger.info(f"Cleanup complete: {files_deleted} files deleted, {used_percent:.1f}% used, {free_gb:.2f} GB free")
        
        return files_deleted
    
    def check_and_cleanup(self) -> int:
        """
        Check disk usage and cleanup if necessary.
        
        Returns:
            Number of files deleted
        """
        used_percent, _ = self.get_disk_usage()
        
        if used_percent >= self.max_usage_percent:
            return self.cleanup_old_footage()
        
        return 0
    
    def get_storage_info(self) -> dict:
        """
        Get comprehensive storage information.
        
        Returns:
            Dictionary with storage statistics
        """
        stat = shutil.disk_usage(self.storage_path)
        used_percent, free_gb = self.get_disk_usage()
        
        video_files = []
        total_size_gb = 0
        for file in self._collect_video_files(only_highlights=False) + self._collect_video_files(only_highlights=True):
            size = file.stat().st_size / (1024 ** 3)
            video_files.append({
                'name': file.name,
                'size_gb': size,
                'modified': datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                'is_highlight': self._is_highlight_file(file),
            })
            total_size_gb += size
        
        return {
            'total_gb': stat.total / (1024 ** 3),
            'used_gb': stat.used / (1024 ** 3),
            'free_gb': free_gb,
            'used_percent': used_percent,
            'max_usage_percent': self.max_usage_percent,
            'video_files_count': len(video_files),
            'highlight_files_count': sum(1 for f in video_files if f['is_highlight']),
            'total_video_size_gb': total_size_gb,
            'oldest_files': [f['name'] for f in sorted(video_files, key=lambda x: x['modified'])[:5]]
        }
