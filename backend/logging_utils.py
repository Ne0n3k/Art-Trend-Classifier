# Shared logging utilities for Art Classifier
import logging
import time
import json
from typing import Dict, Any, Optional

def setup_logging(level: int = logging.INFO, is_lambda: bool = False) -> logging.Logger:
    """Setup structured logging for backend or lambda"""
    if is_lambda:
        # Lambda uses CloudWatch - JSON format
        logger = logging.getLogger()
        logger.setLevel(level)
        return logger
    else:
        # Backend uses structured format
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

class AnalysisLogger:
    """Context manager for analysis logging"""
    
    def __init__(self, logger: logging.Logger, filename: str, file_size: int, is_lambda: bool = False):
        self.logger = logger
        self.filename = filename
        self.file_size = file_size
        self.is_lambda = is_lambda
        self.start_time = time.time()
        
    def __enter__(self):
        self._log_start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._log_success()
        else:
            self._log_error(str(exc_val))
    
    def _log_start(self):
        """Log analysis start"""
        log_data = {
            "event": "analysis_started",
            "filename": self.filename,
            "file_size_bytes": self.file_size,
            "timestamp": self.start_time
        }
        self._log(log_data, "info")
    
    def log_success(self, predicted_class: str = "", confidence: float = 0.0, 
                    inference_time_ms: float = 0.0, image_dimensions: str = ""):
        """Log successful analysis"""
        total_time = time.time() - self.start_time
        log_data = {
            "event": "analysis_completed",
            "filename": self.filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "inference_time_ms": round(inference_time_ms, 2),
            "total_time_ms": round(total_time * 1000, 2),
            "file_size_bytes": self.file_size,
            "image_dimensions": image_dimensions,
            "success": True
        }
        self._log(log_data, "info")
    
    def log_error(self, error: str):
        """Log analysis error"""
        total_time = time.time() - self.start_time
        log_data = {
            "event": "analysis_failed",
            "filename": self.filename,
            "error": error,
            "file_size_bytes": self.file_size,
            "total_time_ms": round(total_time * 1000, 2),
            "success": False
        }
        self._log(log_data, "error")
    
    def _log(self, data: Dict[str, Any], level: str):
        """Log with appropriate format"""
        if self.is_lambda:
            # Lambda: JSON format for CloudWatch
            getattr(self.logger, level)(json.dumps(data))
        else:
            # Backend: structured format
            getattr(self.logger, level)(f"Analysis event: {data['event']}", extra=data)
