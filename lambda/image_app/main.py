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

# Import shared modules (copy to lambda directory)
import sys
sys.path.append('/opt/python')
from config import IMAGE_LIMITS, API_CONFIG
from validation import validate_file_input, process_image_from_content, ValidationError
from reviews import generate_review
from logging_utils import setup_logging, AnalysisLogger

# Configure logging for Lambda
logger = setup_logging(is_lambda=True)

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
limiter = Limiter(key_func=get_remote_address, default_limits=[API_CONFIG.RATE_LIMITS["default"]])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: RESTRICTED to specific origins only
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG.CORS_ORIGINS,
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


# Validation functions moved to validation.py module


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
@limiter.limit(API_CONFIG.RATE_LIMITS["analyze"], methods=["POST"])
async def analyze(file: UploadFile = File(...)):
    """Analyze uploaded image and return art style prediction"""

    try:
        # Validate input and get content
        content, file_size = await validate_file_input(file)
        
        # Ensure model is available
        _load_model_if_needed()

        if torch is None:
            raise HTTPException(status_code=500, detail="Torch not available in runtime")
        
        if _model is None:
            detail = _model_loaded_error or "Model not loaded"
            raise HTTPException(status_code=503, detail=detail)

        # Process image
        image = process_image_from_content(content)
        image_tensor = _preprocess(image)
        
        # Inference timing
        inference_start = time.time()
        with torch.no_grad():
            outputs = _model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        inference_time = time.time() - inference_start

        predicted_class = _class_names[predicted.item()] if _class_names else "Unknown"
        confidence_score = float(confidence.item())

        # Generate predictions
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

        # Generate AI review
        review = generate_review(predicted_class, confidence_score)

        # Log success
        with AnalysisLogger(logger, file.filename, file_size, is_lambda=True) as analysis_logger:
            analysis_logger.log_success(
                predicted_class, confidence_score, 
                inference_time * 1000, f"{image.width}x{image.height}"
            )

        return {
            "predicted_style": predicted_class,
            "confidence": confidence_score,
            "review": review,
            "top_predictions": top_predictions,
            "all_predictions": all_predictions
        }
        
    except ValidationError:
        # Re-raise validation errors
        raise
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:  # noqa: S110
        # Log and handle other errors
        with AnalysisLogger(logger, file.filename, file_size, is_lambda=True) as analysis_logger:
            analysis_logger.log_error(str(e))
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


handler = Mangum(app)


