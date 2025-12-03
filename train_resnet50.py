"""
One-click training script for dog breed classification (ResNet50 + tqdm progress bar)

Folder structure (same directory as this script):
    - labels.csv
    - train/
    - test/          (optional)
    - train_resnet50_dog.py

Run:
    python train_resnet50.py
"""

import os
from typing import Tuple

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

from tqdm import tqdm   # ★ NEW


# ==============================
# 0. CONFIG
# ==============================
LABELS_CSV   = "labels.csv"
TRAIN_DIR    = "train"
OUTPUT_DIR   = "checkpoints"

EPOCHS       = 15
BATCH_SIZE   = 128
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAL_RATIO    = 0.2
NUM_WORKERS  = 4
USE_CPU_ONLY = False


# ==============================
# 1. Dataset
# ==============================
class DogBreedDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, breed2idx: dict, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.breed2idx = breed2idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = str(row["id"])
        breed = row["breed"]

        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidate = os.path.join(self.img_dir, img_id + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            raise FileNotFoundError(f"Image not found for {img_id}")

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.breed2idx[breed]


# ==============================
# 2. Build Model
# ==============================
def build_model(num_classes: int) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ==============================
# 3. Train / Eval Loops (tqdm)
# ==============================
def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())

    return running_loss / len(loader.dataset)


def eval_one_epoch(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Validating", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item())

    val_loss = running_loss / len(loader.dataset)
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc


# ==============================
# 4. Main Training Logic
# ==============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() and not USE_CPU_ONLY else "cpu")
    print(f"Using device: {device}")

    # Read labels.csv
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError("labels.csv not found")

    df = pd.read_csv(LABELS_CSV)
    breeds = sorted(df["breed"].unique())
    breed2idx = {b: i for i, b in enumerate(breeds)}
    idx2breed = {i: b for b, i in breed2idx.items()}
    num_classes = len(breeds)

    print(f"{len(df)} images, {num_classes} breeds")

    # Transforms
    train_tf = T.Compose([
        T.Resize((256, 256)),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    # Dataset + split
    full_dataset = DogBreedDataset(LABELS_CSV, TRAIN_DIR, breed2idx, transform=train_tf)
    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_tf

    print(f"Train: {train_size}, Val: {val_size}")

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Save directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_ckpt = os.path.join(OUTPUT_DIR, "resnet50_dog_best.pth")
    best_val_acc = 0.0

    # Epoch loop
    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch [{epoch}/{EPOCHS}] =====")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "breed2idx": breed2idx,
                "idx2breed": idx2breed,
                "val_acc": best_val_acc,
                "epoch": epoch,
            }, best_ckpt)
            print(f"🔥 New best model saved to {best_ckpt} (acc={best_val_acc:.4f})")

    print("\nTraining completed!")
    print(f"Best accuracy: {best_val_acc:.4f}")
    print(f"Best model saved at: {best_ckpt}")


# ==============================
# 5. Entry
# ==============================
if __name__ == "__main__":
    main()