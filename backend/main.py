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

app = FastAPI(title="AI Art Critic", version="1.0.0")

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
    
    model_path = "ml_model/model/model_best_82_73.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
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
            "Miękkie światło i rozmyte kontury.",
            "Jasne kolory i ulotne wrażenia.",
            "Romantyczna gra światłocienia."
        ],
        "Post_Impressionism": [
            "Silniejsze kontrasty i ekspresja.",
            "Nasycone kolory i symbolizm.",
            "Ewolucja od impresjonizmu."
        ],
        "Expressionism": [
            "Intensywne kolory i emocje.",
            "Zniekształcone formy i dramat.",
            "Wewnętrzne przeżycia artysty."
        ],
        "Cubism": [
            "Geometryczne formy i fragmentacja.",
            "Wieloperspektywiczne ujęcie tematu.",
            "Analityczne rozbicie rzeczywistości."
        ],
        "Abstract_Expressionism": [
            "Czysta abstrakcja i spontaniczność.",
            "Ekspresyjny gest malarski.",
            "Emocje przez abstrakcyjne formy."
        ],
        "Fauvism": [
            "Dzikość i intensywność kolorów.",
            "Nienaturalne zestawienia barw.",
            "Ekspresyjne użycie koloru."
        ],
        "Pop_Art": [
            "Jasne kolory i kontrasty.",
            "Kultura popularna w sztuce.",
            "Komercyjna estetyka przetworzona."
        ],
        "Minimalism": [
            "Redukcja do istoty.",
            "Prostota i czystość formy.",
            "Mniej znaczy więcej."
        ],
        "Color_Field_Painting": [
            "Duże płaszczyzny koloru.",
            "Medytacyjna kompozycja.",
            "Spokój przez jednolitość."
        ],
        "Art_Nouveau_Modern": [
            "Organiczne płynne formy.",
            "Dekoracyjność i elegancja.",
            "Inspiracje naturą."
        ],
        "Symbolism": [
            "Ukryte znaczenia i symbole.",
            "Tajemnicza atmosfera.",
            "Wyrażenie idei duchowych."
        ],
        "Romanticism": [
            "Emocjonalność nad racjonalnością.",
            "Melancholia i natura.",
            "Kult uczucia."
        ],
        "Baroque": [
            "Teatralność i przepych.",
            "Dynamiczna kompozycja.",
            "Bogate detale."
        ],
        "Rococo": [
            "Delikatność i gracja.",
            "Pastelowe kolory.",
            "Arystokratyczna elegancja."
        ],
        "Northern_Renaissance": [
            "Precyzja i realizm.",
            "Dbałość o detale.",
            "Symbolika religijna."
        ],
        "High_Renaissance": [
            "Klasyczna harmonia.",
            "Perfekcja techniczna.",
            "Idealizacja formy."
        ],
        "Naive_Art_Primitivism": [
            "Naiwna spontaniczność.",
            "Bezpośredniość wyrazu.",
            "Autentyczność sztuki."
        ],
        "Ukiyo_e": [
            "Japoński drzeworyt.",
            "Płaskie kolory.",
            "Ulotna piękność."
        ]
    }
    
    # Select review based on confidence
    style_reviews = reviews.get(style, ["Interesujące dzieło o unikalnym charakterze artystycznym."])
    
    if confidence > 0.8:
        review = style_reviews[0]
    elif confidence > 0.6:
        review = style_reviews[1] if len(style_reviews) > 1 else style_reviews[0]
    else:
        review = style_reviews[-1] if len(style_reviews) > 2 else style_reviews[0]
    
    # Add confidence-based modifier
    if confidence > 0.9:
        confidence_text = "Analiza z bardzo wysoką pewnością wskazuje na "
    elif confidence > 0.7:
        confidence_text = "Z dużą pewnością można stwierdzić, że "
    elif confidence > 0.5:
        confidence_text = "Prawdopodobnie mamy do czynienia z "
    else:
        confidence_text = "Dzieło może reprezentować "
    
    return f"{confidence_text}{style.replace('_', ' ').lower()}. {review}"

@app.on_event("startup")
async def startup_event():
    # Load model on startup
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    # Health check endpoint
    return {"message": "AI Art Critic API is running!", "classes": len(class_names)}

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
        
        return {
            "predicted_style": predicted_class,
            "confidence": confidence_score,
            "review": review,
            "top_predictions": top_predictions,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
