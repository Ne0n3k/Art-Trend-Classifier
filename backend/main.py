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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(title="Art Trend Classifier", version="1.0.0", lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
class_names = []
device = torch.device('cuda' if torch.cuda.is_available() else (
    'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))

def load_model():
    # Load the trained model
    global model, class_names
    
    # Try different paths for different execution contexts
    possible_paths = [
        "ml_model/model/model_best_82_73.pth",  # When run from project root
        "../ml_model/model/model_best_82_73.pth"  # When run from backend/ directory
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
    
    # Create model architecture
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
    # Get image preprocessing transform
    return A.Compose([
        A.Resize(height=352, width=352),
        A.CenterCrop(height=320, width=320),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def generate_review(style: str, confidence: float) -> str:
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
async def root():
    # Health check endpoint
    return {"message": "Art Trend Classifier API is running!", "classes": len(class_names)}

@app.options("/analyze")
async def analyze_options():
    return {"message": "OK"}

@app.post("/analyze")
async def analyze_artwork(file: UploadFile = File(...)) -> Dict[str, Any]:
    # Analyze uploaded artwork image
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Convert PIL to numpy for albumentations
        image_np = np.array(image)
        
        # Apply transforms
        transform = get_transform()
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = class_names[predicted.item()]
            confidence_score = confidence.item()
        
        # Generate review
        review = generate_review(predicted_class, confidence_score)
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(3, len(class_names)))
        top_predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            top_predictions.append({
                "style": class_names[idx.item()],
                "confidence": float(prob.item())
            })
        
        # Get all predictions (sorted by confidence)
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
