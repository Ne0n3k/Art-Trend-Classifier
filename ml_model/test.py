import os

import torch
from torchvision import models, transforms
from PIL import Image


def load_model(model_path="ml_model/model_best.pth"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    chkpt = torch.load(model_path, map_location='cpu')
    class_names = chkpt['class_names']
    num_classes = chkpt['num_classes']

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(chkpt['model_state_dict'])
    model.eval()
    return model, class_names


def predict_image(image_path, model, class_names):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_t = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(img_t)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        conf, pred = torch.max(probs, 0)
    return class_names[pred.item()], conf.item(), probs


if __name__ == "__main__":
    model, classes = load_model()
    print(f"Loaded model. Classes: {len(classes)}")

