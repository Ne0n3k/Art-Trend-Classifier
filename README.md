# Art-Trend-Classifier

Project of a web application for recognizing art styles and generating short image reviews.

## Structure
- `backend/` – FastAPI (Python)
- `ml_model/` – model ML (PyTorch)
- `frontend/` – aplikacja web (HTML/JS)
- `docs/` – dokumentacja, notatki

## Final ML Model Details

### Architecture
- **Backbone**: ResNet50 (transfer learning)
- **Classes**: 18 art styles (Impressionism, Cubism, Abstract Expressionism, etc.)
- **Performance**: 82% train accuracy, 73% validation accuracy

### Training Features
- **Data Augmentation**: Albumentations (ColorJitter, GaussianBlur, Rotate, Brightness/Contrast, CoarseDropout)
- **Advanced Techniques**: Mixup/CutMix, EMA (Exponential Moving Average), TTA (Test Time Augmentation)
- **Optimization**: AdamW optimizer, CosineAnnealingWarmRestarts scheduler
- **Regularization**: Label smoothing (0.02), Weight decay (0.05), Gradient clipping, Early stopping

### Dataset
- **Source**: WikiArt dataset
- **Split**: 80% train, 10% validation, 10% test (stratified)
- **Selected Styles**: 18 most representative art movements
- **Total Images**: ~60k training samples

### Files
- `train.py` – training script with advanced techniques
- `test.py` – model testing and inference
- `dataset.py` – data loading and preprocessing
- `model/` – saved model checkpoints (not in git due to size)
