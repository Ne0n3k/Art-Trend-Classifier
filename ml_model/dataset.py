import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(data_dir="data/wikiart", batch_size=32, train_split=0.8, augment_level="strong"):
    """
    Create train and validation data loaders from WikiArt dataset
    augment_level: "strong" | "none"
    """
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

    # Load datasets with different transforms
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    # Split indices
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    train_size = int(train_split * dataset_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create separate datasets with different transforms
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Get class names from the original dataset
    temp_dataset = datasets.ImageFolder(root=data_dir)
    class_names = temp_dataset.classes
    
    return train_loader, val_loader, class_names

def get_class_names(data_dir="data/wikiart"):
    """Get class names from dataset"""
    dataset = datasets.ImageFolder(root=data_dir)
    return dataset.classes
