# Art Trend Classifier

AI-powered web application for recognizing art styles and generating detailed artistic reviews. Upload any artwork image and get instant analysis with style classification and AI-generated reviews.

## 🌟 Features

### Core Functionality
- **18 Art Styles Recognition**: From Impressionism to Pop Art
- **Instant Analysis**: Get results in seconds
- **AI-Generated Reviews**: Detailed artistic analysis based on confidence levels
- **High Accuracy**: 82% training, 73% validation accuracy - will be updated in future
- **Free to Use**: No registration required

### User Interface
- **Drag & Drop Upload**: Easy image upload with preview
- **Real-time Results**: Live confidence bars and predictions
- **Expandable Predictions**: View all 18 style predictions
- **Modern Design**: Clean, intuitive interface
- **Landing Page**: Professional presentation with animations

## 🏗️ Project Structure
- `backend/` – FastAPI REST API (Python)
- `ml_model/` – Deep learning model (PyTorch)
- `frontend/` – Web application (HTML/CSS/JS)
  - `landing.html` – Professional landing page
  - `index.html` – Main application interface
- `docs/` – Documentation and notes

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip/conda package manager

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/art-trend-classifier.git
cd art-trend-classifier
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
1. Start the backend server:
```bash
cd backend
python main.py
```

2. Open the frontend:
- **Landing Page**: Open `frontend/landing.html` in your browser
- **Main App**: Open `frontend/index.html` in your browser

3. Upload an artwork image and get instant analysis!

## 🤖 ML Model Details

### Architecture
- **Backbone**: ResNet50 (transfer learning)
- **Classes**: 18 art styles (Impressionism, Cubism, Abstract Expressionism, etc.)
- **Performance**: 82% training accuracy, 73% validation accuracy
- **Model Size**: ~100MB (ResNet50 + custom head)

### Training Features
- **Data Augmentation**: Albumentations (ColorJitter, GaussianBlur, Rotate, Brightness/Contrast, CoarseDropout)
- **Advanced Techniques**: Mixup/CutMix, EMA (Exponential Moving Average), TTA (Test Time Augmentation)
- **Optimization**: AdamW optimizer, CosineAnnealingWarmRestarts scheduler
- **Regularization**: Label smoothing (0.05), Weight decay (0.02), Gradient clipping, Early stopping

### Dataset
- **Source**: WikiArt dataset
- **Split**: 80% train, 10% validation, 10% test (stratified)
- **Selected Styles**: 18 most representative art movements
- **Total Images**: ~60k training samples

### Model Files
- `ml_model/train.py` – Training script with advanced techniques
- `ml_model/test.py` – Model testing and inference
- `ml_model/dataset.py` – Data loading and preprocessing
- `ml_model/model/` – Saved model checkpoints (not in git due to size)

## 🌐 API Documentation

### Endpoints
- `GET /` – Health check
- `POST /analyze` – Image analysis endpoint

### Request Format
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@artwork.jpg"
```

### Response Format
```json
{
  "predicted_style": "Impressionism",
  "confidence": 0.85,
  "review": "With high confidence we can state that impressionism. Soft light and blurred contours.",
  "top_predictions": [
    {"style": "Impressionism", "confidence": 0.85},
    {"style": "Post_Impressionism", "confidence": 0.12},
    {"style": "Expressionism", "confidence": 0.03}
  ],
  "all_predictions": [...],
  "filename": "artwork.jpg"
}
```

## 🎨 Supported Art Styles

1. **Impressionism** - Soft light and blurred contours
2. **Post-Impressionism** - Stronger contrasts and expression
3. **Expressionism** - Intense colors and emotions
4. **Cubism** - Geometric forms and fragmentation
5. **Abstract Expressionism** - Pure abstraction and spontaneity
6. **Fauvism** - Wildness and color intensity
7. **Pop Art** - Bright colors and contrasts
8. **Minimalism** - Reduction to essence
9. **Color Field Painting** - Large color planes
10. **Art Nouveau Modern** - Organic flowing forms
11. **Symbolism** - Hidden meanings and symbols
12. **Romanticism** - Emotion over rationality
13. **Baroque** - Theatricality and opulence
14. **Rococo** - Delicacy and grace
15. **Northern Renaissance** - Precision and realism
16. **High Renaissance** - Classical harmony
17. **Naive Art Primitivism** - Naive spontaneity
18. **Ukiyo-e** - Japanese woodblock print

## 🛠️ Technologies Used

### Backend
- **FastAPI** - High-performance API framework
- **PyTorch** - Deep learning framework
- **Albumentations** - Image preprocessing
- **PIL/Pillow** - Image handling

### Frontend
- **HTML5/CSS3** - Modern web interface
- **JavaScript** - Interactive functionality
- **Responsive Design** - Mobile-first approach

### ML/AI
- **ResNet50** - Neural network architecture
- **Transfer Learning** - Pre-trained model fine-tuning
- **Data Augmentation** - Enhanced training data

## 📱 Features Overview

### Landing Page (`landing.html`)
- **Professional Design** - Modern gradient backgrounds
- **Animated Sections** - Scroll-triggered animations
- **Responsive Layout** - Works on all devices
- **Call-to-Action** - Direct links to main application

### Main Application (`index.html`)
- **Drag & Drop Upload** - Easy image selection
- **Real-time Preview** - Instant image preview
- **Confidence Visualization** - Visual confidence bars
- **Expandable Results** - View all predictions
- **AI Reviews** - Contextual artistic analysis

## 🔧 Development

### Training New Models
```bash
cd ml_model
python train.py
```

### Testing Models
```bash
cd ml_model
python test.py
```

### Backend Development
```bash
cd backend
python main.py
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
