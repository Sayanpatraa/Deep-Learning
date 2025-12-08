"""
One-click training script for dog breed classification (ResNet50 + tqdm progress bar + full metrics)

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

from tqdm import tqdm   # progress bar

# NEW: metrics
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ==============================
# 0. CONFIG
# ==============================
LABELS_CSV   = "labels.csv"
TRAIN_DIR    = "train"
OUTPUT_DIR   = "checkpoints"

EPOCHS       = 15
BATCH_SIZE   = 32
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
    """
    Light-weight eval used during training (loss + accuracy only).
    Full metrics are computed after training with a separate function.
    """
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
# 3b. Full Metrics Evaluation
# ==============================
import json

def evaluate_full_metrics(model, loader, device, idx2breed: dict, output_dir="checkpoints"):
    """
    Run once at the end on the validation set using the best checkpoint.
    Prints and saves:
    - accuracy
    - macro/micro/weighted precision/recall/F1
    - detailed classification report
    - confusion matrix
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating best model", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # ----------- Compute metrics -----------
    acc = accuracy_score(all_labels, all_preds)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="micro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    # Per-class metrics
    num_classes = len(idx2breed)
    target_names = [idx2breed[i] for i in range(num_classes)]

    cls_report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        zero_division=0,
        output_dict=True,  # ← IMPORTANT for JSON
    )

    conf_mat = confusion_matrix(all_labels, all_preds).tolist()  # JSON-friendly

    # ============ PRINT METRICS =============
    print("\n========== Final Evaluation (Validation Set) ==========")
    print(f"Accuracy: {acc:.4f}\n")

    print("Macro-averaged metrics:")
    print(f"  Precision (macro):  {p_macro:.4f}")
    print(f"  Recall (macro):     {r_macro:.4f}")
    print(f"  F1-score (macro):   {f1_macro:.4f}\n")

    print("Micro-averaged metrics:")
    print(f"  Precision (micro):  {p_micro:.4f}")
    print(f"  Recall (micro):     {r_micro:.4f}")
    print(f"  F1-score (micro):   {f1_micro:.4f}\n")

    print("Weighted-averaged metrics:")
    print(f"  Precision (weighted): {p_weighted:.4f}")
    print(f"  Recall (weighted):    {r_weighted:.4f}")
    print(f"  F1-score (weighted):  {f1_weighted:.4f}\n")

    print("----- Classification Report (per breed) -----")
    print(classification_report(
        all_labels, all_preds, target_names=target_names, zero_division=0
    ))

    print("Confusion matrix shape:", len(conf_mat), "x", len(conf_mat))

    # ============ SAVE METRICS TO JSON =============
    metrics_dict = {
        "accuracy": acc,
        "macro": {
            "precision": p_macro,
            "recall": r_macro,
            "f1": f1_macro,
        },
        "micro": {
            "precision": p_micro,
            "recall": r_micro,
            "f1": f1_micro,
        },
        "weighted": {
            "precision": p_weighted,
            "recall": r_weighted,
            "f1": f1_weighted,
        },
        "classification_report": cls_report,  # dict version
        "confusion_matrix": conf_mat,
        "num_classes": num_classes,
        "class_names": target_names,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "metrics.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"\n📁 All metrics saved to: {json_path}\n")



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
    # use val transforms for validation subset
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
    print(f"Best accuracy (Val Acc): {best_val_acc:.4f}")

    # ===== Final full-metrics evaluation on the best checkpoint =====
    if os.path.exists(best_ckpt):
        print("\nLoading best checkpoint and evaluating full metrics on validation set...")
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        # Prefer idx2breed from checkpoint if available
        idx2breed_best = ckpt.get("idx2breed", idx2breed)

        evaluate_full_metrics(model, val_loader, device, idx2breed_best, OUTPUT_DIR)
    else:
        print("Best checkpoint not found, skipping full metrics evaluation.")


# ==============================
# 5. Entry
# ==============================
if __name__ == "__main__":
    main()
