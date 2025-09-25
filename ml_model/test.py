import torch
from torchvision import models, transforms
from PIL import Image
import os

def load_model(model_path="ml_model/model.pth"):
    """Load trained model from checkpoint"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']
    
    # Create model architecture
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, class_names

def predict_image(image_path, model, class_names):
    """Predict art style for a single image"""
    
    # Image preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    predicted_style = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_style, confidence_score, probabilities

def generate_review(style, confidence):
    """Generate simple AI review based on style and confidence"""
    
    reviews = {
        'Impressionism': "This artwork displays the characteristic soft brushstrokes and light effects of Impressionism, capturing fleeting moments with vibrant colors.",
        'Cubism': "The geometric forms and fragmented perspectives are hallmarks of Cubism, showing multiple viewpoints simultaneously.",
        'Baroque': "Rich in dramatic contrasts and ornate details, this piece embodies the grandeur and emotional intensity of Baroque art.",
        'Expressionism': "Bold colors and distorted forms convey deep emotions, typical of Expressionist movement's focus on inner feelings.",
        'Surrealism': "Dreamlike imagery and unexpected juxtapositions create the fantastical quality characteristic of Surrealism.",
        'Abstract_Expressionism': "The spontaneous, gestural marks and emphasis on pure emotion reflect Abstract Expressionist principles.",
        'Romanticism': "Dramatic scenes and emotional intensity capture the Romantic movement's emphasis on feeling over reason.",
        'Realism': "Faithful representation of everyday subjects shows the Realist commitment to depicting life as it truly appears.",
        'Post_Impressionism': "Building on Impressionist techniques while adding symbolic content and structural elements.",
        'Art_Nouveau_Modern': "Flowing, organic forms and decorative elements showcase Art Nouveau's integration of art and design."
    }
    
    # Get base review or default
    base_review = reviews.get(style, f"This artwork appears to be in the {style.replace('_', ' ')} style.")
    
    # Add confidence qualifier
    if confidence > 0.8:
        confidence_text = "I'm quite confident that "
    elif confidence > 0.6:
        confidence_text = "This appears to be "
    else:
        confidence_text = "This might be "
    
    return f"{confidence_text}{base_review.lower()}"

def test_model():
    """Test the trained model with example images"""
    
    print("Loading trained model...")
    try:
        model, class_names = load_model()
        print(f"Model loaded successfully! Found {len(class_names)} art styles.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run training first: python ml_model/train.py")
        return
    
    # Test with example images
    examples_dir = "docs/examples"
    if not os.path.exists(examples_dir):
        print(f"Examples directory not found: {examples_dir}")
        return
    
    test_images = [f for f in os.listdir(examples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_images:
        print("No test images found in docs/examples/")
        return
    
    print(f"\nTesting with {len(test_images)} example images:")
    print("=" * 60)
    
    for img_file in test_images:
        img_path = os.path.join(examples_dir, img_file)
        
        try:
            style, confidence, probabilities = predict_image(img_path, model, class_names)
            review = generate_review(style, confidence)
            
            print(f"\nImage: {img_file}")
            print(f"Predicted Style: {style}")
            print(f"Confidence: {confidence:.1%}")
            print(f"Review: {review}")
            
            # Show top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, min(3, len(class_names)))
            print("Top predictions:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                print(f"  {i+1}. {class_names[idx]}: {prob:.1%}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    test_model()