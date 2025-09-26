import os
from typing import Tuple, List, Optional, Dict

import cv2
cv2.setNumThreads(0)
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ArtDataset(Dataset):
    def __init__(self, data_dir: str, classes: List[str], transform=None):
        self.data_dir = data_dir
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        self.transform = transform
        self.image_paths: List[str] = []
        self.labels: List[int] = []

        for class_name in classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            aug = self.transform(image=image)
            image = aug["image"]

        return image, label


def get_transforms(image_size: int = 224) -> Tuple[A.Compose, A.Compose]:
    train_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Cutout(num_holes=1, max_h_size=32, max_w_size=32, fill_value=0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_test_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_test_transform


def _count_images_per_class(base_dir: str, class_names: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name in class_names:
        class_dir = os.path.join(base_dir, name)
        count = 0
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    count += 1
        counts[name] = count
    return counts


def create_dataloaders(base_dir: str, batch_size: int = 16, num_workers: int = 2, image_size: int = 224, top_k: Optional[int] = None, selected_classes: Optional[List[str]] = None):
    classes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    if selected_classes is not None and len(selected_classes) > 0:
        selected_set = set(selected_classes)
        classes = [c for c in classes if c in selected_set]
    elif top_k is not None and top_k > 0 and top_k < len(classes):
        counts = _count_images_per_class(base_dir, classes)
        classes = sorted(classes, key=lambda c: counts.get(c, 0), reverse=True)[:top_k]
    train_tf, val_tf = get_transforms(image_size=image_size)

    full_set = ArtDataset(base_dir, classes, transform=None)

    class_to_indices: List[List[int]] = [[] for _ in classes]
    for idx, label in enumerate(full_set.labels):
        class_to_indices[label].append(idx)

    train_indices, val_indices, test_indices = [], [], []
    for inds in class_to_indices:
        inds = sorted(inds)
        n = len(inds)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_indices.extend(inds[:n_train])
        val_indices.extend(inds[n_train:n_train + n_val])
        test_indices.extend(inds[n_train + n_val:])

    train_base = ArtDataset(base_dir, classes, transform=train_tf)
    val_base = ArtDataset(base_dir, classes, transform=val_tf)
    test_base = ArtDataset(base_dir, classes, transform=val_tf)

    train_set = Subset(train_base, train_indices)
    val_set = Subset(val_base, val_indices)
    test_set = Subset(test_base, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, test_loader, classes


