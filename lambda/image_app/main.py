# AWS Lambda FastAPI Application for Art Classification
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from typing import Any, Dict, List
import boto3
import botocore
import io
import json
import os
import tempfile
from PIL import Image
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import time

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Torch imports are optional; container includes them. If import fails, we fall back to mock.
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms as T
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    models = None  # type: ignore
    T = None  # type: ignore


app = FastAPI(title="Art Classifier")

# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: RESTRICTED to specific origins only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://d3hxkd3bumy7al.cloudfront.net",  # Production domain
    ],
    allow_credentials=False,  # Disabled for security
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# -------- Model loading (lazy, S3-backed) --------
_model = None
_class_names: List[str] = []
_model_loaded_error: str | None = None

def _device():
    if torch is None:
        return "cpu"
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

def _get_s3_client():
    return boto3.client("s3", region_name=os.getenv("AWS_REGION"))

def _download_weights_s3(local_path: str) -> None:
    bucket = os.getenv("MODEL_S3_BUCKET")
    key = os.getenv("MODEL_S3_KEY")
    if not bucket or not key:
        raise RuntimeError("MODEL_S3_BUCKET and MODEL_S3_KEY must be set for real inference")
    s3 = _get_s3_client()
    s3.download_file(bucket, key, local_path)

def _build_model(num_classes: int):
    assert models is not None and nn is not None
    backbone = models.resnet50(weights=None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(in_features, num_classes)
    )
    return backbone

def _load_model_if_needed() -> None:
    """Lazy load model from S3 on first request"""
    global _model, _class_names, _model_loaded_error
    if _model is not None or _model_loaded_error is not None:
        return

    # If torch missing, keep mock mode
    if torch is None:
        _model_loaded_error = "torch not available in runtime"
        return

    # Download checkpoint to /tmp
    ckpt_path = os.path.join(tempfile.gettempdir(), "model.ckpt")
    if not os.path.exists(ckpt_path):
        try:
            _download_weights_s3(ckpt_path)
        except botocore.exceptions.ClientError as e:
            _model_loaded_error = f"S3 error: {e}"
            return
        except Exception as e:  # noqa: S110
            _model_loaded_error = f"Download error: {e}"
            return

    try:
        checkpoint = torch.load(ckpt_path, map_location=_device())
        _class_names = checkpoint.get("class_names") or []
        num_classes = checkpoint.get("num_classes") or (len(_class_names) or 0)
        if not num_classes:
            raise RuntimeError("Checkpoint missing class metadata")
        model = _build_model(num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(_device())
        model.eval()
        _model = model
    except Exception as e:  # noqa: S110
        _model_loaded_error = f"Load error: {e}"


def _validate_image_dimensions(image: Image.Image) -> None:
    """Validate image dimensions to prevent DoS attacks"""
    width, height = image.size
    max_dimension = 8192  # 8K limit
    
    if width > max_dimension or height > max_dimension:
        raise HTTPException(
            status_code=400, 
            detail=f"Image too large. Maximum dimension: {max_dimension}px. Got: {width}x{height}"
        )
    
    if width < 32 or height < 32:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small. Minimum dimension: 32px. Got: {width}x{height}"
        )


# -------- Routes --------
@app.get("/")
@limiter.limit("30/minute")
async def root():
    """Health check endpoint"""
    return {"message": "ok", "model_ready": _model is not None, "classes": len(_class_names)}


@app.get("/health")
async def health():
    """Detailed health check with model and dependency status"""
    # Try to load model on health check to surface errors early
    _load_model_if_needed()
    numpy_info: Dict[str, Any] = {"available": False}
    try:
        import numpy as np  # type: ignore
        numpy_info = {"available": True, "version": getattr(np, "__version__", None)}
    except Exception as e:  # noqa: S110
        numpy_info = {"available": False, "error": str(e)}

    torch_info: Dict[str, Any] = {"available": torch is not None}
    if torch is not None:
        try:
            torch_info["version"] = torch.__version__
        except Exception:
            pass

    return {
        "status": "healthy",
        "model_error": _model_loaded_error,
        "numpy": numpy_info,
        "torch": torch_info,
    }


def _mock_response(file_name: str) -> Dict[str, Any]:
    """Mock response for fallback (not used in production)"""
    return {
        "predicted_style": "Unknown",
        "confidence": 0.0,
        "review": f"Model unavailable for {file_name}",
        "top_predictions": [],
        "all_predictions": []
    }


@app.post("/analyze")
@limiter.limit("10/minute", methods=["POST"])
async def analyze(file: UploadFile = File(...)):
    """Analyze uploaded image and return art style prediction"""

    # Structured logging: start analysis
    start_time = time.time()

    # Comprehensive input validation
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Single file read - validate size and get content
    content = await file.read()
    file_size = len(content)
    
    if file_size > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    if file_size < 1024:  # min 1KB
        raise HTTPException(status_code=400, detail="File too small")

    # Validate image format more strictly
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    if file.content_type.lower() not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported image format. Allowed: {', '.join(allowed_types)}")

    # Log analysis start
    logger.info(json.dumps({
        "event": "lambda_analysis_started",
        "filename": file.filename,
        "file_size_bytes": file_size,
        "content_type": file.content_type,
        "timestamp": start_time
    }))

    # Ensure model is available; if not, return an error (no mock fallback)
    _load_model_if_needed()

    if torch is None:
        logger.error(json.dumps({
            "event": "lambda_analysis_failed",
            "filename": file.filename,
            "error": "Torch not available in runtime",
            "file_size_bytes": file_size,
            "total_time_ms": round((time.time() - start_time) * 1000, 2),
            "success": False
        }))
        raise HTTPException(status_code=500, detail="Torch not available in runtime")
    
    if _model is None:
        detail = _model_loaded_error or "Model not loaded"
        logger.error(json.dumps({
            "event": "lambda_analysis_failed",
            "filename": file.filename,
            "error": detail,
            "file_size_bytes": file_size,
            "total_time_ms": round((time.time() - start_time) * 1000, 2),
            "success": False
        }))
        raise HTTPException(status_code=503, detail=detail)

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Validate image dimensions
        _validate_image_dimensions(image)
        
        image_tensor = _preprocess(image)
        
        # Inference timing
        inference_start = time.time()
        with torch.no_grad():
            outputs = _model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        inference_time = time.time() - inference_start
        total_time = time.time() - start_time

        predicted_class = _class_names[predicted.item()] if _class_names else "Unknown"
        confidence_score = float(confidence.item())

        # top-k
        k = min(3, len(_class_names) or 3)
        top_probs, top_indices = torch.topk(probabilities, k=k)
        top_predictions = [
            {"style": _class_names[idx.item()] if _class_names else str(idx.item()),
             "confidence": float(prob.item())}
            for prob, idx in zip(top_probs[0], top_indices[0])
        ]

        all_k = len(_class_names) or probabilities.shape[1]
        all_probs, all_indices = torch.topk(probabilities, k=all_k)
        all_predictions = [
            {"style": _class_names[idx.item()] if _class_names else str(idx.item()),
             "confidence": float(prob.item())}
            for prob, idx in zip(all_probs[0], all_indices[0])
        ]

        review = f"Analysis for {file.filename} completed"

        # Structured logging: successful analysis
        logger.info(json.dumps({
            "event": "lambda_analysis_completed",
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence_score,
            "inference_time_ms": round(inference_time * 1000, 2),
            "total_time_ms": round(total_time * 1000, 2),
            "file_size_bytes": file_size,
            "image_dimensions": f"{image.width}x{image.height}",
            "success": True,
            "model_loaded": _model is not None
        }))

        return {
            "predicted_style": predicted_class,
            "confidence": confidence_score,
            "review": review,
            "top_predictions": top_predictions,
            "all_predictions": all_predictions
        }
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:  # noqa: S110
        # Structured logging: analysis failed
        logger.error(json.dumps({
            "event": "lambda_analysis_failed",
            "filename": file.filename,
            "error": str(e),
            "file_size_bytes": file_size,
            "total_time_ms": round((time.time() - start_time) * 1000, 2),
            "success": False,
            "model_loaded": _model is not None
        }))
        # Surface real errors to caller to aid debugging when using real model
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


handler = Mangum(app)


