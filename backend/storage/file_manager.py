 # File upload/management
import os
import aiofiles
from typing import Optional, Dict, Any
import hashlib
from datetime import datetime
import logging
from pathlib import Path
from core.config import settings

logger = logging.getLogger(__name__)

class FileManager:
    """Handles file upload, storage, and management operations"""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIRECTORY)
        self.upload_dir.mkdir(exist_ok=True)
        logger.info(f"File manager initialized with upload directory: {self.upload_dir}")
    
    def _generate_safe_filename(self, original_filename: str) -> str:
        """Generate a safe, unique filename"""
        # Extract name and extension
        name_part = Path(original_filename).stem
        extension = Path(original_filename).suffix.lower()
        
        # Clean the filename
        safe_name = "".join(c for c in name_part if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name[:50]  # Limit length
        
        # Add timestamp and hash for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(original_filename.encode()).hexdigest()[:8]
        
        return f"{timestamp}_{name_hash}_{safe_name}{extension}"
    
    def _validate_file(self, filename: str, file_size: int) -> None:
        """Validate uploaded file"""
        # Check file size
        if file_size > settings.MAX_FILE_SIZE:
            raise ValueError(f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB")
        
        # Check file extension
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        if extension not in settings.ALLOWED_FILE_TYPES:
            raise ValueError(f"File type not supported. Allowed types: {', '.join(settings.ALLOWED_FILE_TYPES)}")
        
        # Check filename
        if not filename or len(filename) > 255:
            raise ValueError("Invalid filename")
    
    async def save_file(self, file_content: bytes, original_filename: str) -> Dict[str, Any]:
        """
        Save uploaded file to disk
        
        Returns:
            Dict with file information
        """
        try:
            # Validate file
            self._validate_file(original_filename, len(file_content))
            
            # Generate safe filename
            safe_filename = self._generate_safe_filename(original_filename)
            file_path = self.upload_dir / safe_filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            # Create file info
            file_info = {
                'original_filename': original_filename,
                'saved_filename': safe_filename,
                'file_path': str(file_path),
                'file_size': len(file_content),
                'upload_time': datetime.now().isoformat(),
                'file_type': original_filename.split('.')[-1].lower() if '.' in original_filename else 'unknown'
            }
            
            logger.info(f"File saved successfully: {safe_filename}")
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to save file '{original_filename}': {e}")
            raise Exception(f"File save error: {str(e)}")
    
    async def get_file(self, filename: str) -> Optional[bytes]:
        """Retrieve file content"""
        try:
            file_path = self.upload_dir / filename
            if not file_path.exists():
                return None
            
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            
            logger.debug(f"Retrieved file: {filename}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to retrieve file '{filename}': {e}")
            return None
    
    def get_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get file information without reading content"""
        try:
            file_path = self.upload_dir / filename
            if not file_path.exists():
                return None
            
            stat = file_path.stat()
            return {
                'filename': filename,
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'exists': True
            }
        except Exception as e:
            logger.error(f"Failed to get file info for '{filename}': {e}")
            return None
    
    def delete_file(self, filename: str) -> bool:
        """Delete a file"""
        try:
            file_path = self.upload_dir / filename
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {filename}")
                return True
            else:
                logger.warning(f"File not found for deletion: {filename}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete file '{filename}': {e}")
            return False
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Clean up files older than specified hours"""
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            deleted_count = 0
            
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_ctime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete old file {file_path}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"File cleanup failed: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            total_size = 0
            file_count = 0
            
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                'total_files': file_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'upload_directory': str(self.upload_dir)
            }
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {
                'total_files': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0,
                'error': str(e)
            }

# Global file manager instance
file_manager = FileManager()