import os
import math
import random
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report




# ============================================================
# 0. CONFIG
# ============================================================

class Config:
    # Paths (adapt these to your Kaggle layout)
    DATA_DIR = "dog-breed-identification"
    TRAIN_CSV = os.path.join(DATA_DIR, "labels.csv")  # Kaggle: 'labels.csv'
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")   # contains {id}.jpg
    CKPT_DIR = "checkpoints"

    # Training
    NUM_CLASSES = 120
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS_STAGE1 = 5      # feature extraction
    EPOCHS_STAGE2 = 10     # partial fine-tuning
    LR_HEAD = 1e-3
    LR_FINE_TUNE = 1e-4
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Stable Diffusion
    SD_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    SD_GUIDANCE_SCALE = 7.5
    SD_NUM_INFERENCE_STEPS = 40
    SD_IMG_SIZE = 512

    # Misc
    TOP_K = 5


def set_seed(seed: int = 42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(Config.SEED)


# ============================================================
# 1. DATASET
# ============================================================

class DogBreedDataset(Dataset):
    """
    Expects Kaggle 'labels.csv' with columns: id, breed
    And images in 'train/{id}.jpg'
    """

    def __init__(self,
                 df: pd.DataFrame,
                 img_dir: str,
                 label2idx: dict,
                 transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = row["id"]
        breed = row["breed"]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.label2idx[breed]
        return image, label


def get_transforms(img_size: int = 224):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])

    valid_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_tfms, valid_tfms


def create_dataloaders():
    df = pd.read_csv(Config.TRAIN_CSV)

    # train/val split (ps-style: could predefine indices; here simple split)
    df = df.sample(frac=1.0, random_state=Config.SEED).reset_index(drop=True)
    n_train = int(0.8 * len(df))
    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:]

    breeds = sorted(df["breed"].unique())
    label2idx = {b: i for i, b in enumerate(breeds)}
    idx2label = {i: b for b, i in label2idx.items()}

    train_tfms, valid_tfms = get_transforms(Config.IMG_SIZE)

    train_ds = DogBreedDataset(df_train, Config.TRAIN_IMG_DIR, label2idx, transform=train_tfms)
    val_ds = DogBreedDataset(df_val, Config.TRAIN_IMG_DIR, label2idx, transform=valid_tfms)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, label2idx, idx2label


# ============================================================
# 2. MODEL — RESNET CLASSIFIER
# ============================================================

def create_resnet_classifier(num_classes: int = 120, pretrained: bool = True):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_all_but_head(model: nn.Module):
    for name, param in model.named_parameters():
        if "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_last_blocks(model: nn.Module, num_blocks: int = 1):
    """
    Unfreeze last num_blocks of layer4 (deepest stage) plus head.
    """
    # unfreeze fc
    for name, param in model.named_parameters():
        if "fc" in name:
            param.requires_grad = True

    # unfreeze last residual blocks in layer4
    for name, module in model.named_modules():
        if name.startswith("layer4"):
            for param in module.parameters():
                param.requires_grad = True


# ============================================================
# 3. TRAINING & EVAL
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, acc, f1


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, acc, f1, all_labels, all_preds


def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}")


def train_resnet_pipeline():
    train_loader, val_loader, label2idx, idx2label = create_dataloaders()

    model = create_resnet_classifier(num_classes=len(label2idx), pretrained=True)
    model.to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()

    # -------- Stage 1: feature extraction (only head) ----------
    print("\n=== Stage 1: Feature Extraction (freeze backbone) ===")
    freeze_all_but_head(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LR_HEAD)

    best_f1 = 0.0
    for epoch in range(1, Config.EPOCHS_STAGE1 + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS_STAGE1}")
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_acc, val_f1, y_true, y_pred = evaluate(model, val_loader, criterion, Config.DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, os.path.join(Config.CKPT_DIR, "resnet_stage1_best.pth"))

    # -------- Stage 2: partial fine-tuning ----------
    print("\n=== Stage 2: Partial Fine-Tuning (unfreeze last blocks) ===")
    unfreeze_last_blocks(model, num_blocks=1)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LR_FINE_TUNE)

    for epoch in range(1, Config.EPOCHS_STAGE2 + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS_STAGE2}")
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_acc, val_f1, y_true, y_pred = evaluate(model, val_loader, criterion, Config.DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, os.path.join(Config.CKPT_DIR, "resnet_stage2_best.pth"))

    print("\n=== Final Evaluation (classification report) ===")
    _, _, _, y_true, y_pred = evaluate(model, val_loader, criterion, Config.DEVICE)
    print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(len(idx2label))]))

    # store label mapping
    mapping_path = os.path.join(Config.CKPT_DIR, "label_mapping.csv")
    pd.DataFrame({
        "idx": list(idx2label.keys()),
        "breed": [idx2label[i] for i in range(len(idx2label))]
    }).to_csv(mapping_path, index=False)
    print(f"Saved label mapping to {mapping_path}")


# ============================================================
# 4. INFERENCE: IMAGE to  BREED
# ============================================================

class DogBreedPredictor:
    def __init__(self, ckpt_path: str, mapping_csv: str):
        self.device = Config.DEVICE
        df_map = pd.read_csv(mapping_csv)
        self.idx2label = dict(zip(df_map["idx"], df_map["breed"]))

        self.model = create_resnet_classifier(num_classes=len(self.idx2label), pretrained=False)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        _, self.val_tfms = get_transforms(Config.IMG_SIZE)

    def predict_proba(self, img_path: str):
        image = Image.open(img_path).convert("RGB")
        x = self.val_tfms(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    def top_k(self, img_path: str, k: int = Config.TOP_K):
        probs = self.predict_proba(img_path)
        idxs = np.argsort(probs)[::-1][:k]
        breeds = [self.idx2label[i] for i in idxs]
        scores = probs[idxs]
        return list(zip(breeds, scores))


# ============================================================
# 5. STABLE DIFFUSION GENERATOR
# ============================================================

class DogImageGenerator:
    def __init__(self, model_name: str = Config.SD_MODEL_NAME):
        self.device = Config.DEVICE
        print(f"Loading Stable Diffusion model: {model_name}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def generate(
        self,
        breed_name: str,
        extra_prompt: Optional[str] = None,
        num_images: int = 1,
        guidance_scale: float = Config.SD_GUIDANCE_SCALE,
        num_inference_steps: int = Config.SD_NUM_INFERENCE_STEPS
    ) -> List[Image.Image]:
        if extra_prompt is None:
            prompt = f"A high quality, realistic photograph of a {breed_name} dog, 4k, studio lighting"
        else:
            prompt = extra_prompt.replace("{breed}", breed_name)

        images = []
        for _ in range(num_images):
            out = self.pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            images.append(out.images[0])
        return images


# ============================================================
# 6. CANISNET PIPELINE: IMAGE → PREDICT → GENERATE
# ============================================================

class CanisNet:
    """
    High-level wrapper that combines classifier + generator.
    """

    def __init__(self,
                 ckpt_path: str,
                 mapping_csv: str,
                 sd_model_name: str = Config.SD_MODEL_NAME):
        self.predictor = DogBreedPredictor(ckpt_path, mapping_csv)
        self.generator = DogImageGenerator(sd_model_name)

    def process_image(
        self,
        img_path: str,
        extra_prompt_template: Optional[str] = None,
        num_images: int = 1
    ):
        """
        1. Predict top-k breeds for the uploaded dog image.
        2. Generate images conditioned on top-1 (or more).
        """
        topk = self.predictor.top_k(img_path, k=Config.TOP_K)
        top1_breed, top1_score = topk[0]

        print("Top-k predictions:")
        for b, s in topk:
            print(f"{b:25s} | {s:.4f}")

        print(f"\nUsing top-1 breed for generation: {top1_breed} (p={top1_score:.3f})")

        images = self.generator.generate(
            breed_name=top1_breed,
            extra_prompt=extra_prompt_template,
            num_images=num_images
        )
        return topk, images


# ============================================================
# 7. STREAMLIT FRONTEND (WORKING ON)
# ============================================================

"""
Run with:
    streamlit run canisnet.py

This will call main_streamlit() defined below.
"""

import streamlit as st

def main_streamlit():
    st.set_page_config(page_title="CanisNet: Dog Breed Recognition & Image Generation", layout="wide")
    st.title("🐶 CanisNet: Dog Breed Recognition & Image Generation")

    ckpt_path = os.path.join(Config.CKPT_DIR, "resnet_stage2_best.pth")
    mapping_csv = os.path.join(Config.CKPT_DIR, "label_mapping.csv")

    if not (os.path.exists(ckpt_path) and os.path.exists(mapping_csv)):
        st.error("Trained model checkpoint or label mapping not found. Train the classifier first.")
        return

    st.sidebar.header("Generation Settings")
    num_images = st.sidebar.slider("Number of generated images", 1, 4, 1)
    custom_prompt = st.sidebar.text_area(
        "Custom prompt (use {breed} placeholder, optional)",
        value="A realistic photo of a {breed} dog running in a park, 4k, natural lighting"
    )

    uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        temp_path = "temp_upload.png"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        st.image(temp_path, caption="Uploaded image", use_column_width=True)

        if st.button("Predict breed & generate images"):
            with st.spinner("Loading models & generating..."):
                canisnet = CanisNet(ckpt_path, mapping_csv, sd_model_name=Config.SD_MODEL_NAME)
                topk, gen_images = canisnet.process_image(
                    temp_path,
                    extra_prompt_template=custom_prompt,
                    num_images=num_images
                )

            st.subheader("Predicted Breeds")
            for breed, score in topk:
                st.write(f"- **{breed}** (p={score:.3f})")

            st.subheader("Generated Images")
            cols = st.columns(num_images)
            for img, col in zip(gen_images, cols):
                with col:
                    st.image(img, use_column_width=True)


# ============================================================
# 8. CLI ENTRYPOINTS
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CanisNet: Dog Breed Recognition & Image Generation")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "demo", "infer"],
                        help="train: train classifier; demo: run Streamlit; infer: CLI infer+generate")
    parser.add_argument("--image", type=str, default=None,
                        help="Image path for infer mode")
    args = parser.parse_args()

    if args.mode == "train":
        train_resnet_pipeline()

    elif args.mode == "demo":
        # This will NOT start streamlit automatically – you still run `streamlit run canisnet.py`
        print("Use:  streamlit run canisnet.py  to launch the UI.")

    elif args.mode == "infer":
        if args.image is None:
            raise ValueError("Please provide --image path for infer mode.")

        ckpt_path = os.path.join(Config.CKPT_DIR, "resnet_stage2_best.pth")
        mapping_csv = os.path.join(Config.CKPT_DIR, "label_mapping.csv")

        canisnet = CanisNet(ckpt_path, mapping_csv, sd_model_name=Config.SD_MODEL_NAME)
        topk, gen_images = canisnet.process_image(args.image, num_images=2)

        # save generated images
        os.makedirs("generated_samples", exist_ok=True)
        for i, img in enumerate(gen_images):
            out_path = os.path.join("generated_samples", f"generated_{i}.png")
            img.save(out_path)
            print(f"Saved generated image to {out_path}")


