# Backend API for Art Trend Classifier
import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    yield
    print("Shutting down...")

app = FastAPI(title="Art Trend Classifier", version="1.0.0", lifespan=lifespan)

# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS for frontend - RESTRICTED origins only
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

def get_transform():
    """Image preprocessing pipeline"""
    return A.Compose([
        A.Resize(height=352, width=352),
        A.CenterCrop(height=320, width=320),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def generate_review(style: str, confidence: float) -> str:
    """Generate AI review based on predicted style and confidence"""
    reviews = {
        "Impressionism": [
            "Soft light and blurred contours.",
            "Bright colors and fleeting impressions.",
            "Romantic play of light and shadow."
        ],
        "Post_Impressionism": [
            "Stronger contrasts and expression.",
            "Saturated colors and symbolism.",
            "Evolution from impressionism."
        ],
        "Expressionism": [
            "Intense colors and emotions.",
            "Distorted forms and drama.",
            "Artist's inner experiences."
        ],
        "Cubism": [
            "Geometric forms and fragmentation.",
            "Multi-perspective view of subject.",
            "Analytical breakdown of reality."
        ],
        "Abstract_Expressionism": [
            "Pure abstraction and spontaneity.",
            "Expressive brush gesture.",
            "Emotions through abstract forms."
        ],
        "Fauvism": [
            "Wildness and color intensity.",
            "Unnatural color combinations.",
            "Expressive use of color."
        ],
        "Pop_Art": [
            "Bright colors and contrasts.",
            "Popular culture in art.",
            "Commercial aesthetics transformed."
        ],
        "Minimalism": [
            "Reduction to essence.",
            "Simplicity and purity of form.",
            "Less is more."
        ],
        "Color_Field_Painting": [
            "Large color planes.",
            "Meditative composition.",
            "Peace through uniformity."
        ],
        "Art_Nouveau_Modern": [
            "Organic flowing forms.",
            "Decorative elegance.",
            "Nature-inspired designs."
        ],
        "Symbolism": [
            "Hidden meanings and symbols.",
            "Mysterious atmosphere.",
            "Expression of spiritual ideas."
        ],
        "Romanticism": [
            "Emotion over rationality.",
            "Melancholy and nature.",
            "Cult of feeling."
        ],
        "Baroque": [
            "Theatricality and opulence.",
            "Dynamic composition.",
            "Rich details."
        ],
        "Rococo": [
            "Delicacy and grace.",
            "Pastel colors.",
            "Aristocratic elegance."
        ],
        "Northern_Renaissance": [
            "Precision and realism.",
            "Attention to detail.",
            "Religious symbolism."
        ],
        "High_Renaissance": [
            "Classical harmony.",
            "Technical perfection.",
            "Idealization of form."
        ],
        "Naive_Art_Primitivism": [
            "Naive spontaneity.",
            "Direct expression.",
            "Authenticity of art."
        ],
        "Ukiyo_e": [
            "Japanese woodblock print.",
            "Flat colors.",
            "Fleeting beauty."
        ]
    }
    
    # Select review based on confidence
    style_reviews = reviews.get(style, ["Interesting work with unique artistic character."])
    
    if confidence > 0.8:
        review = style_reviews[0]
    elif confidence > 0.6:
        review = style_reviews[1] if len(style_reviews) > 1 else style_reviews[0]
    else:
        review = style_reviews[-1] if len(style_reviews) > 2 else style_reviews[0]
    
    # Add confidence-based modifier
    if confidence > 0.9:
        confidence_text = "Analysis with very high confidence indicates "
    elif confidence > 0.7:
        confidence_text = "With high confidence we can state that "
    elif confidence > 0.5:
        confidence_text = "We probably have "
    else:
        confidence_text = "The work may represent "
    
    return f"{confidence_text}{style.replace('_', ' ').lower()}. {review}"


@app.get("/")
@limiter.limit("30/minute")
async def root():
    """Health check endpoint"""
    return {"message": "Art Trend Classifier API is running!", "classes": len(class_names)}

@app.options("/analyze")
async def analyze_options():
    return {"message": "OK"}

@app.post("/analyze")
@limiter.limit("10/minute", methods=["POST"])
async def analyze_artwork(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Analyze uploaded artwork image and return style prediction"""

    # Comprehensive input validation
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Validate file size (max 10MB)
    file_size = 0
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

    # Reset file pointer for further processing
    import io
    file.file = io.BytesIO(content)
    
    try:
        # Preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        # Apply transforms and predict
        transform = get_transform()
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = class_names[predicted.item()]
            confidence_score = confidence.item()
        
        # Generate AI review
        review = generate_review(predicted_class, confidence_score)
        
        # Top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(3, len(class_names)))
        top_predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            top_predictions.append({
                "style": class_names[idx.item()],
                "confidence": float(prob.item())
            })
        
        # All predictions sorted by confidence
        all_probs, all_indices = torch.topk(probabilities, k=len(class_names))
        all_predictions = []
        for prob, idx in zip(all_probs[0], all_indices[0]):
            all_predictions.append({
                "style": class_names[idx.item()],
                "confidence": float(prob.item())
            })
        
        return {
            "predicted_style": predicted_class,
            "confidence": confidence_score,
            "review": review,
            "top_predictions": top_predictions,
            "all_predictions": all_predictions,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
