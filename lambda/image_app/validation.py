# Shared validation utilities for Art Classifier
from fastapi import HTTPException, UploadFile
from PIL import Image
from typing import Tuple
import io
from config import IMAGE_LIMITS, API_CONFIG

class ValidationError(HTTPException):
    """Custom validation error"""
    pass

async def validate_file_input(file: UploadFile) -> Tuple[bytes, int]:
    """Validate file input and return content with size"""
    if not file or not file.filename:
        raise ValidationError(status_code=400, detail="No file provided")

    if not file.content_type or not file.content_type.startswith('image/'):
        raise ValidationError(status_code=400, detail="File must be an image")

    # Read file content once
    content = await file.read()
    file_size = len(content)
    
    if file_size > IMAGE_LIMITS.MAX_FILE_SIZE:
        raise ValidationError(
            status_code=413, 
            detail=f"File too large (max {IMAGE_LIMITS.MAX_FILE_SIZE // (1024*1024)}MB)"
        )

    if file_size < IMAGE_LIMITS.MIN_FILE_SIZE:
        raise ValidationError(
            status_code=400, 
            detail=f"File too small (min {IMAGE_LIMITS.MIN_FILE_SIZE // 1024}KB)"
        )

    # Validate MIME type
    if file.content_type.lower() not in API_CONFIG.ALLOWED_MIME_TYPES:
        raise ValidationError(
            status_code=400, 
            detail=f"Unsupported image format. Allowed: {', '.join(API_CONFIG.ALLOWED_MIME_TYPES)}"
        )

    return content, file_size

def validate_image_dimensions(image: Image.Image) -> None:
    """Validate image dimensions"""
    width, height = image.size
    
    if width > IMAGE_LIMITS.MAX_DIMENSION or height > IMAGE_LIMITS.MAX_DIMENSION:
        raise ValidationError(
            status_code=400, 
            detail=f"Image too large. Maximum dimension: {IMAGE_LIMITS.MAX_DIMENSION}px. Got: {width}x{height}"
        )
    
    if width < IMAGE_LIMITS.MIN_DIMENSION or height < IMAGE_LIMITS.MIN_DIMENSION:
        raise ValidationError(
            status_code=400,
            detail=f"Image too small. Minimum dimension: {IMAGE_LIMITS.MIN_DIMENSION}px. Got: {width}x{height}"
        )

def process_image_from_content(content: bytes) -> Image.Image:
    """Process image from bytes content"""
    try:
        image = Image.open(io.BytesIO(content)).convert('RGB')
        validate_image_dimensions(image)
        return image
    except Exception as e:
        raise ValidationError(status_code=400, detail=f"Invalid image: {str(e)}")
