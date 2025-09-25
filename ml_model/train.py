import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import get_loaders
import time
import os

def train_model():
    """Improved training to maximize training accuracy while remaining stable."""
    
    # Hyperparameters (tune here)
    epochs = 6
    batch_size = 24
    base_lr = 3e-4
    weight_decay = 0.0  # maximize train accuracy
    max_grad_norm = 1.0
    
    # Device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else (
            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
    )
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    # Disable augmentations to maximize fit on train (higher training accuracy)
    train_loader, val_loader, class_names = get_loaders(batch_size=batch_size, augment_level="none")
    num_classes = len(class_names)
    
    print(f"Found {num_classes} art styles")
    print(f"Training batches: {len(train_loader)} (batch_size={batch_size})")
    
    # Model
    print("Loading pretrained ResNet18...")
    model = models.resnet18(weights='DEFAULT')
    
    # Unfreeze deeper layers for higher capacity
    for name, param in model.named_parameters():
        if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2']):
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Replace classifier head with higher-capacity MLP (no dropout for max train acc)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 768),
        nn.ReLU(inplace=True),
        nn.Linear(768, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)
    
    # Loss & Optimizer (different LR for pretrained vs head)
    criterion = nn.CrossEntropyLoss()
    pretrained_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (head_params if name.startswith('fc') else pretrained_params).append(param)
    
    optimizer = optim.Adam([
        {'params': pretrained_params, 'lr': base_lr * 0.1, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': base_lr, 'weight_decay': weight_decay},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"Starting training for {epochs} epochs...")
    print("=" * 60)
    
    best_train_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0:
                current_acc = 100.0 * correct / total if total else 0.0
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {current_acc:.2f}% LR: {current_lr:.6f}")
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        train_loss = running_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)
        best_train_acc = max(best_train_acc, train_acc)
        
        print(f"\nEpoch [{epoch+1}/{epochs}] Results:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Acc: {train_acc:.2f}% (best {best_train_acc:.2f}%)")
        
        # Optional quick validation (first 100 batches) just to monitor; skip if slow
        if val_loader and len(val_loader) > 0:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for v_idx, (v_images, v_labels) in enumerate(val_loader):
                    if v_idx >= 100:
                        break
                    v_images = v_images.to(device)
                    v_labels = v_labels.to(device)
                    v_out = model(v_images)
                    _, v_pred = torch.max(v_out, 1)
                    val_total += v_labels.size(0)
                    val_correct += (v_pred == v_labels).sum().item()
            if val_total > 0:
                val_acc = 100.0 * val_correct / val_total
                print(f"  Val Acc (sample): {val_acc:.2f}%")
        
        print("-" * 60)
        
        # Save checkpoint each epoch
        ckpt_path = f"ml_model/model_epoch_{epoch+1}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': class_names,
            'num_classes': num_classes,
            'model_type': 'resnet18_improved',
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc
        }, ckpt_path)
        print(f"Model saved: {ckpt_path}")
    
    # Final model save
    final_model_path = "ml_model/model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes,
        'model_type': 'resnet18_improved',
        'best_train_acc': best_train_acc
    }, final_model_path)
    
    if os.path.exists(final_model_path):
        file_size = os.path.getsize(final_model_path) / (1024*1024)
        print(f"\nFinal model saved: {final_model_path} ({file_size:.1f} MB)")
    else:
        print("\nWarning: model file not found after save!")
    
    return model, class_names

if __name__ == "__main__":
    train_model()
