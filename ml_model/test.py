import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_model(model_path="ml_model/model/model_best_82_73.pth"):
    # Load the trained ResNet50 model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']
    
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(in_features, num_classes)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded successfully! Classes: {len(class_names)}")
    return model, class_names


def get_transform():
    # Get image preprocessing transform
    return A.Compose([
        A.Resize(height=352, width=352),
        A.CenterCrop(height=320, width=320),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def predict_image(image_path, model, class_names, device='cpu'):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    transform = get_transform()
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
    
    top_probs, top_indices = torch.topk(probabilities, k=min(3, len(class_names)))
    top_predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        top_predictions.append({
            "style": class_names[idx.item()],
            "confidence": float(prob.item())
        })
    
    return predicted_class, confidence_score, top_predictions


def test_on_sample_images():
    # Test the model on sample images
    device = torch.device('cuda' if torch.cuda.is_available() else (
        'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))
    
    model, class_names = load_model()
    model = model.to(device)
    
    print(f"Using device: {device}")
    print(f"Available classes: {class_names}")
    print("=" * 60)
    
    sample_dir = "data/wikiart"
    if os.path.exists(sample_dir):
        print("Testing on sample images from dataset...")
        
        for class_name in class_names[:5]:
            class_dir = os.path.join(sample_dir, class_name)
            if os.path.exists(class_dir):
                images = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    test_image = os.path.join(class_dir, images[0])
                    try:
                        predicted_style, confidence, top_predictions = predict_image(
                            test_image, model, class_names, device
                        )
                        
                        print(f"Image: {os.path.basename(test_image)}")
                        print(f"True class: {class_name}")
                        print(f"Predicted: {predicted_style} ({confidence:.3f})")
                        top3_str = [(p['style'], f"{p['confidence']:.3f}") for p in top_predictions]
                        print(f"Top 3: {top3_str}")
                        print("-" * 40)
                        
                    except Exception as e:
                        print(f"Error processing {test_image}: {e}")
    else:
        print("No sample images found. Please provide an image path to test.")


if __name__ == "__main__":
    test_on_sample_images()

