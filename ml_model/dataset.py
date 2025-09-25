import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import random

def _build_stratified_indices(full_dataset, train_split: float = 0.8, seed: int = 42):
    class_to_indices = {}
    for idx, (_, class_idx) in enumerate(full_dataset.samples):
        class_to_indices.setdefault(class_idx, []).append(idx)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []
    for class_idx, indices in class_to_indices.items():
        rng.shuffle(indices)
        split_point = int(len(indices) * train_split)
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    # Keep a consistent overall order for reproducibility
    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices

def get_loaders(data_dir="data/wikiart", batch_size=32, train_split=0.8, augment_level="strong", balance: bool = False, seed: int = 42):
    if augment_level == "none":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir)
    
    train_indices, val_indices = _build_stratified_indices(full_dataset, train_split=train_split, seed=seed)
    
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    if balance:
        temp_dataset = datasets.ImageFolder(root=data_dir)
        class_counts = [0] * len(temp_dataset.classes)
        for idx in train_indices:
            _, class_idx = temp_dataset.samples[idx]
            class_counts[class_idx] += 1
        weights = []
        for idx in train_indices:
            _, class_idx = temp_dataset.samples[idx]
            count = max(1, class_counts[class_idx])
            weights.append(1.0 / count)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    temp_dataset = datasets.ImageFolder(root=data_dir)
    class_names = temp_dataset.classes
    
    return train_loader, val_loader, class_names

def get_class_names(data_dir="data/wikiart"):
    dataset = datasets.ImageFolder(root=data_dir)
    return dataset.classes
