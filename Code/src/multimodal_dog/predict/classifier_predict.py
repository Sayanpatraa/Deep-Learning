from PIL import Image
from multimodal_dog.models.classifier import DogClassifier

# 1. load model from HF
model = DogClassifier.from_hf()
print("Model loaded.")

# 2. load local image
img = Image.open("border_collie.jpg").convert("RGB")
breed, conf = model.predict(img)

print("Predicted breed:", breed)
print("Confidence:", conf)
