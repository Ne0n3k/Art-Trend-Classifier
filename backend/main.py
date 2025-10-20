# Backend API for Art Trend Classifier
import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import time

from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import our refactored modules
from config import IMAGE_LIMITS, SECURITY_CONFIG, API_CONFIG
from validation import validate_file_input, process_image_from_content, ValidationError
from reviews import generate_review
from logging_utils import setup_logging, AnalysisLogger

# Setup logging
logger = setup_logging(is_lambda=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    try:
        load_model()
        print(f"Model loaded in lifespan: {model is not None}")
        print(f"Class names loaded in lifespan: {len(class_names)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    yield
    print("Shutting down...")

app = FastAPI(title="Art Trend Classifier", version="1.0.0", lifespan=lifespan)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    for header, value in SECURITY_CONFIG.HEADERS.items():
        response.headers[header] = value
    return response

# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address, default_limits=[API_CONFIG.RATE_LIMITS["default"]])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS for frontend - RESTRICTED origins only
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG.CORS_ORIGINS,
    allow_credentials=False,  # Disabled for security
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Global model state
model = None
class_names = []
device = torch.device('cuda' if torch.cuda.is_available() else (
    'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))

def load_model():
    """Load ResNet50 model with custom classifier head"""
    global model, class_names
    
    # Try different paths for different execution contexts
    possible_paths = [
        "ml_model/model/model_best_82_73.pth",  # Project root
        "../ml_model/model/model_best_82_73.pth"  # Backend directory
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError(f"Model file not found in any of these locations: {possible_paths}")
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']
    
    # Build ResNet50 with custom classifier
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(in_features, num_classes)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded successfully! Classes: {len(class_names)}")

# Pre-compiled transform for better performance
TRANSFORM = A.Compose([
    A.Resize(height=352, width=352),
    A.CenterCrop(height=320, width=320),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Review generation moved to reviews.py module


@app.get("/")
@limiter.limit("30/minute")
async def root(request: Request):
    """Health check endpoint"""
    return {"message": "Art Trend Classifier API is running!", "classes": len(class_names)}

@app.options("/analyze")
async def analyze_options():
    return {"message": "OK"}

@app.post("/analyze")
@limiter.limit(API_CONFIG.RATE_LIMITS["analyze"], methods=["POST"])
async def analyze_artwork(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    """Analyze uploaded artwork image and return style prediction"""
    
    try:
        print(f"Model available: {model is not None}")
        print(f"Class names available: {len(class_names)}")
        
        # Validate input and get content
        content, file_size = await validate_file_input(file)
        print(f"File validated, size: {file_size}")
        
        # Process image
        image = process_image_from_content(content)
        image_np = np.array(image)
        print(f"Image processed, shape: {image_np.shape}")
        
        # Apply transforms and predict
        transformed = TRANSFORM(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        print(f"Tensor created, shape: {image_tensor.shape}, device: {image_tensor.device}")
        
        # Inference timing
        inference_start = time.time()
        print(f"Starting inference with model: {model is not None}")
        with torch.no_grad():
            outputs = model(image_tensor)
            print(f"Inference completed, output shape: {outputs.shape}")
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = class_names[predicted.item()]
            confidence_score = confidence.item()
        
        inference_time = time.time() - inference_start
        
        # Generate predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(3, len(class_names)))
        top_predictions = [
            {"style": class_names[idx.item()], "confidence": float(prob.item())}
            for prob, idx in zip(top_probs[0], top_indices[0])
        ]
        
        all_probs, all_indices = torch.topk(probabilities, k=len(class_names))
        all_predictions = [
            {"style": class_names[idx.item()], "confidence": float(prob.item())}
            for prob, idx in zip(all_probs[0], all_indices[0])
        ]
        
        # Generate AI review
        review = generate_review(predicted_class, confidence_score)
        
        # Log success
        try:
            with AnalysisLogger(logger, file.filename, file_size) as analysis_logger:
                analysis_logger.log_success(
                    predicted_class, confidence_score, 
                    inference_time * 1000, f"{image.width}x{image.height}"
                )
        except Exception as log_error:
            print(f"Logging error: {log_error}")
            # Continue without logging
        
        return {
            "predicted_style": predicted_class,
            "confidence": confidence_score,
            "review": review,
            "top_predictions": top_predictions,
            "all_predictions": all_predictions,
            "filename": file.filename
        }
        
    except ValidationError:
        # Re-raise validation errors
        raise
    except Exception as e:
        # Log and handle other errors
        file_size = getattr(file, 'size', 0) if hasattr(file, 'size') else 0
        with AnalysisLogger(logger, file.filename if file else "unknown", file_size) as analysis_logger:
            analysis_logger.log_error(str(e))
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
