import json
from typing import Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from huggingface_hub import hf_hub_download
import torchvision.transforms as T
from torchvision.models import resnet50


# Hugging Face repo & filenames
HF_REPO_ID = "djhua0103/dog-breed-resnet50"
WEIGHT_FILE = "resnet50_dog_best.pth"
LABEL_FILE = "id2breed.json"


class DogClassifier(nn.Module):
    """
    Dog breed classifier for inference.

    - Architecture: ResNet50 (same as training)
    - Weights loaded from Hugging Face checkpoint (model_state)
    - Preprocessing strictly matches validation transform (val_tf)
    """

    def __init__(self, num_classes: int, id2breed: dict):
        super().__init__()

        # Build the same backbone as in training (weights will be overwritten)
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.eval()

        # Use EXACTLY the same preprocessing as val_tf in train_resnet50.py:
        #   Resize((224, 224)) + CenterCrop(224) + ToTensor + Normalize
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
        # id2breed: string index -> breed name
        # e.g. {"0": "beagle", "1": "golden_retriever", ...}
        self.id2breed: dict = id2breed

    # ---------- construction from Hugging Face ----------
    @classmethod
    def from_hf(cls, device: str | None = None) -> "DogClassifier":
        """
        Load id2breed.json and model_state weights from Hugging Face Hub,
        and return a ready-to-use classifier.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1) load label mapping
        json_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=LABEL_FILE,
        )
        with open(json_path, "r", encoding="utf-8") as f:
            id2breed = json.load(f)

        num_classes = len(id2breed)

        # 2) build classifier
        clf = cls(num_classes=num_classes, id2breed=id2breed)

        # 3) load checkpoint
        ckpt_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=WEIGHT_FILE,
        )
        ckpt = torch.load(ckpt_path, map_location=device)

        # 4) extract state_dict
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt

        # 5) strict=True，
        clf.model.load_state_dict(state_dict, strict=True)

        clf.model.to(device)
        clf.device = device

        return clf

    # ---------- single-image inference ----------
    @torch.no_grad()
    def predict(
        self,
        image: Union[Image.Image, "np.ndarray"],
    ) -> Tuple[str, float]:
        """
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
