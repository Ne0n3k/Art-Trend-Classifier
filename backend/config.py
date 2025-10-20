# Configuration constants for Art Classifier
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ImageLimits:
    """Image dimension and size limits"""
    MIN_DIMENSION: int = 32
    MAX_DIMENSION: int = 8192
    MIN_FILE_SIZE: int = 1024  # 1KB
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB

@dataclass
class SecurityConfig:
    """Security headers configuration"""
    HEADERS: Dict[str, str] = None
    
    def __post_init__(self):
        if self.HEADERS is None:
            self.HEADERS = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
            }

@dataclass
class APIConfig:
    """API configuration"""
    ALLOWED_MIME_TYPES: List[str] = None
    CORS_ORIGINS: List[str] = None
    RATE_LIMITS: Dict[str, str] = None
    
    def __post_init__(self):
        if self.ALLOWED_MIME_TYPES is None:
            self.ALLOWED_MIME_TYPES = [
                'image/jpeg', 'image/jpg', 'image/png'  # WebP removed - not supported by PIL
            ]
        
        if self.CORS_ORIGINS is None:
            self.CORS_ORIGINS = [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "https://d3hxkd3bumy7al.cloudfront.net"
            ]
        
        if self.RATE_LIMITS is None:
            self.RATE_LIMITS = {
                "default": "200 per day, 50 per hour",
                "root": "30/minute",
                "analyze": "100/minute"  # Increased for testing
            }

# Global configuration instances
IMAGE_LIMITS = ImageLimits()
SECURITY_CONFIG = SecurityConfig()
API_CONFIG = APIConfig()
