"""
Dog Breed Classifier using fine-tuned ResNet-50
"""

import json
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
from PIL import Image
from huggingface_hub import hf_hub_download
import torchvision.transforms as T
from torchvision.models import resnet50
import numpy as np

from config import (
    CLASSIFIER_HF_REPO,
    CLASSIFIER_WEIGHT_FILE,
    CLASSIFIER_LABEL_FILE,
)


class DogClassifier(nn.Module):
    """
    Dog breed classifier for inference.

    - Architecture: ResNet50 (same as training)
    - Weights loaded from Hugging Face checkpoint (model_state)
    - Preprocessing strictly matches validation transform
    """

    def __init__(self, num_classes: int, id2breed: dict):
        super().__init__()

        # Build the same backbone as in training
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.eval()

        # Preprocessing matching training validation transform
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.device: str = "cpu"
        self.id2breed: dict = id2breed

    @classmethod
    def from_hf(cls, device: Optional[str] = None) -> "DogClassifier":
        """
        Load id2breed.json and model weights from Hugging Face Hub.
        Returns a ready-to-use classifier.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load label mapping
        json_path = hf_hub_download(
            repo_id=CLASSIFIER_HF_REPO,
            filename=CLASSIFIER_LABEL_FILE,
        )
        with open(json_path, "r", encoding="utf-8") as f:
            id2breed = json.load(f)

        num_classes = len(id2breed)

        # Build classifier
        clf = cls(num_classes=num_classes, id2breed=id2breed)

        # Load checkpoint
        ckpt_path = hf_hub_download(
            repo_id=CLASSIFIER_HF_REPO,
            filename=CLASSIFIER_WEIGHT_FILE,
        )
        ckpt = torch.load(ckpt_path, map_location=device)

        # Extract state_dict
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt

        clf.model.load_state_dict(state_dict, strict=True)
        clf.model.to(device)
        clf.device = device

        return clf

    @torch.no_grad()
    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> Tuple[str, float]:
        """
        Predict dog breed from image.

        Args:
            image: PIL.Image or numpy array in HWC format.

        Returns:
            (breed_name, confidence_score)
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        conf, idx = probs.max(dim=1)
        idx_int = idx.item()
        breed_name = self.id2breed.get(str(idx_int), f"class_{idx_int}")

        return breed_name, float(conf.item())

    @torch.no_grad()
    def predict_top_k(
        self,
        image: Union[Image.Image, np.ndarray],
        k: int = 5,
    ) -> list[Tuple[str, float]]:
        """
        Predict top-k dog breeds from image.

        Args:
            image: PIL.Image or numpy array in HWC format.
            k: Number of top predictions to return.

        Returns:
            List of (breed_name, confidence_score) tuples
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        top_probs, top_indices = probs.topk(k, dim=1)

        results = []
        for i in range(k):
            idx_int = top_indices[0, i].item()
            conf = top_probs[0, i].item()
            breed_name = self.id2breed.get(str(idx_int), f"class_{idx_int}")
            results.append((breed_name, float(conf)))

        return results

    def get_breed_list(self) -> list[str]:
        """Return list of all known breeds."""
        return list(self.id2breed.values())
