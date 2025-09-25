import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataset import create_dataloaders


def train():
    base_dir = "data/wikiart"
    batch_size = 16
    num_workers = 0
    image_size = 224
    num_epochs = 30
    lr_head = 5e-4
    lr_backbone = 1e-4
    weight_decay = 0.01
    label_smoothing = 0.2

    device = torch.device('cuda' if torch.cuda.is_available() else (
        'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    selection = [
        "Impressionism",
        "Post_Impressionism",
        "Expressionism",
        "Cubism",
        "Abstract_Expressionism",
        "Fauvism",
        "Pop_Art",
        "Minimalism",
        "Color_Field_Painting",
        "Art_Nouveau_Modern",
        "Symbolism",
        "Romanticism",
        "Baroque",
        "Rococo",
        "Northern_Renaissance",
        "High_Renaissance",
        "Naive_Art_Primitivism",
        "Ukiyo_e",
    ]

    train_loader, val_loader, _, classes = create_dataloaders(
        base_dir=base_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        selected_classes=selection,
    )
    print(f"Datasets → train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}")
    num_classes = len(classes)

    model = models.resnet18(weights='DEFAULT')
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)
    
    optimizer = optim.AdamW([
        { 'params': model.fc.parameters(), 'lr': lr_head },
        { 'params': [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad], 'lr': lr_backbone }
    ], weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_acc = 0.0
    best_path = "ml_model/model_best.pth"

    print(f"Starting training for {num_epochs} epochs...")
    print("=" * 60)
    unfroze = False
    for epoch in range(num_epochs):
        if not unfroze and epoch == 3:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=lr_backbone, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            unfroze = True
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)
        scheduler.step(epoch)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss = val_loss_sum / max(1, len(val_loader))
        val_acc = 100.0 * val_correct / max(1, val_total)

        print(f"Epoch [{epoch+1}/{num_epochs}]  Train: loss={train_loss:.4f} acc={train_acc:.2f}%  |  Val: loss={val_loss:.4f} acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': classes,
                'num_classes': num_classes,
                'model_type': 'resnet18',
                'epoch': epoch + 1,
                'val_acc': val_acc
            }, best_path)
            print(f"✓ Best model updated: {best_val_acc:.2f}% -> {best_path}")

    # final save
    final_path = "ml_model/model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': classes,
        'num_classes': num_classes,
        'model_type': 'resnet18'
    }, final_path)
    print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    train()


