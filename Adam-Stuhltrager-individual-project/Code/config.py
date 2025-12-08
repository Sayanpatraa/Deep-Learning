"""
Configuration for Dog Breed Classifier and Generator Streamlit App
"""

# ============================================================
# CLASSIFIER SETTINGS
# ============================================================
CLASSIFIER_HF_REPO = "djhua0103/dog-breed-resnet50"
CLASSIFIER_WEIGHT_FILE = "resnet50_dog_best.pth"
CLASSIFIER_LABEL_FILE = "id2breed.json"

# ============================================================
# GENERATOR SETTINGS
# ============================================================
GENERATOR_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
GENERATOR_REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"

# ============================================================
# STYLE CONFIGURATIONS
# ============================================================
LORA_STYLES = {
    "Realistic": {
        "repo": None,
        "weight_name": None,
        "scale": None,
        "description": "Photorealistic dog images with accurate breed anatomy",
        "use_realistic_pipeline": True,
    },
    "Manga (LineAni)": {
        "repo": "artificialguybr/LineAniRedmond-LinearMangaSDXL-V2",
        "weight_name": None,
        "scale": 0.9,
        "description": "Black and white manga style with speed lines",
    },
    "Anime": {
        "repo": "ntc-ai/SDXL-LoRA-slider.anime",
        "weight_name": "anime.safetensors",
        "scale": 1.5,
        "description": "Anime style slider effect",
    },
    "Pastel Anime": {
        "repo": "Linaqruf/pastel-anime-xl-lora",
        "weight_name": "pastel-anime-xl.safetensors",
        "scale": 0.8,
        "description": "Soft pastel anime illustration style",
    },
    "Pixel Art": {
        "repo": "nerijs/pixel-art-xl",
        "weight_name": None,
        "scale": 0.9,
        "description": "Retro pixel art style",
    },
}

DEFAULT_LORA_STYLE = "Realistic"

# ============================================================
# PROMPT TEMPLATES (for stylized generation)
# ============================================================
PROMPT_TEMPLATES = {
    "Realistic": None,  # Uses custom prompt in app.py
    "Manga (LineAni)": (
        "Black and white side view of a {breed} dog, accurate canine anatomy, "
        "single visible tail, one tail only, proper proportions, full body in frame, "
        "natural limb spacing, dynamic pose, consistent perspective, "
        "shonen jump manga style, screentone shading, inked lineart, high contrast, "
        "speed lines, impact frame, dramatic action"
    ),
    "Anime": (
        "Beautiful illustration of a {breed} dog, anime style, "
        "vibrant colors, detailed fur, expressive eyes, "
        "soft lighting, peaceful scene, high quality"
    ),
    "Pastel Anime": (
        "Cute {breed} dog, pastel colors, soft anime style, "
        "gentle lighting, dreamy atmosphere, detailed fur, "
        "masterpiece, best quality"
    ),
    "Pixel Art": (
        "Pixel art of a {breed} dog, 16-bit style, "
        "retro gaming aesthetic, clean pixels, vibrant colors, "
        "side view sprite, detailed shading"
    ),
}

# ============================================================
# NEGATIVE PROMPTS
# ============================================================
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, extra limbs, extra legs, extra tails, "
    "duplicate tail, extra heads, wrong dog anatomy, elongated body, "
    "tiny head, giant head, short limbs, missing legs, mutated paws, "
    "fused anatomy, deformed pose, unnatural posture, "
    "jpeg artifacts, bad anatomy, text, logo, watermark"
)

NEGATIVE_PROMPTS = {
    "Realistic": None,  # Uses NEG from realistic_generator.py
    "Manga (LineAni)": (
        "extra tail, duplicate tail, second tail, tail duplication, tail artifact, "
        "blurry, low quality, distorted, extra limbs, extra legs, extra tails, "
        "duplicate tail, extra heads, wrong dog anatomy, elongated body, "
        "tiny head, giant head, short limbs, missing legs, mutated paws, "
        "fused anatomy, deformed pose, unnatural posture, "
        "color, colorful, pastel colors, 3d, cgi, photorealistic, painting, "
        "edge highlight, jpeg artifacts, bad anatomy, text, logo, watermark"
    ),
    "Anime": DEFAULT_NEGATIVE_PROMPT,
    "Pastel Anime": DEFAULT_NEGATIVE_PROMPT,
    "Pixel Art": DEFAULT_NEGATIVE_PROMPT + ", anti-aliasing, smooth gradients",
}

# ============================================================
# GENERATION DEFAULTS
# ============================================================
# Realistic defaults
DEFAULT_REALISTIC_HEIGHT = 1024
DEFAULT_REALISTIC_WIDTH = 1024
DEFAULT_STEPS_BASE = 28
DEFAULT_STEPS_REFINER = 20
DEFAULT_REALISTIC_GUIDANCE = 5.5

# Stylized defaults
DEFAULT_HEIGHT = 720
DEFAULT_WIDTH = 1080
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 5.0

DEFAULT_SEED = None  # Random

# ============================================================
# DOG BREEDS (from classifier model id2breed.json)
# ============================================================
DOG_BREEDS = [
    "Affenpinscher",
    "Afghan Hound",
    "African Hunting Dog",
    "Airedale",
    "American Staffordshire Terrier",
    "Appenzeller",
    "Australian Terrier",
    "Basenji",
    "Basset",
    "Beagle",
    "Bedlington Terrier",
    "Bernese Mountain Dog",
    "Black-and-tan Coonhound",
    "Blenheim Spaniel",
    "Bloodhound",
    "Bluetick",
    "Border Collie",
    "Border Terrier",
    "Borzoi",
    "Boston Bull",
    "Bouvier Des Flandres",
    "Boxer",
    "Brabancon Griffon",
    "Briard",
    "Brittany Spaniel",
    "Bull Mastiff",
    "Cairn",
    "Cardigan",
    "Chesapeake Bay Retriever",
    "Chihuahua",
    "Chow",
    "Clumber",
    "Cocker Spaniel",
    "Collie",
    "Curly-coated Retriever",
    "Dandie Dinmont",
    "Dhole",
    "Dingo",
    "Doberman",
    "English Foxhound",
    "English Setter",
    "English Springer",
    "Entlebucher",
    "Eskimo Dog",
    "Flat-coated Retriever",
    "French Bulldog",
    "German Shepherd",
    "German Short-haired Pointer",
    "Giant Schnauzer",
    "Golden Retriever",
    "Gordon Setter",
    "Great Dane",
    "Great Pyrenees",
    "Greater Swiss Mountain Dog",
    "Groenendael",
    "Ibizan Hound",
    "Irish Setter",
    "Irish Terrier",
    "Irish Water Spaniel",
    "Irish Wolfhound",
    "Italian Greyhound",
    "Japanese Spaniel",
    "Keeshond",
    "Kelpie",
    "Kerry Blue Terrier",
    "Komondor",
    "Kuvasz",
    "Labrador Retriever",
    "Lakeland Terrier",
    "Leonberg",
    "Lhasa",
    "Malamute",
    "Malinois",
    "Maltese Dog",
    "Mexican Hairless",
    "Miniature Pinscher",
    "Miniature Poodle",
    "Miniature Schnauzer",
    "Newfoundland",
    "Norfolk Terrier",
    "Norwegian Elkhound",
    "Norwich Terrier",
    "Old English Sheepdog",
    "Otterhound",
    "Papillon",
    "Pekinese",
    "Pembroke",
    "Pomeranian",
    "Pug",
    "Redbone",
    "Rhodesian Ridgeback",
    "Rottweiler",
    "Saint Bernard",
    "Saluki",
    "Samoyed",
    "Schipperke",
    "Scotch Terrier",
    "Scottish Deerhound",
    "Sealyham Terrier",
    "Shetland Sheepdog",
    "Shih-tzu",
    "Siberian Husky",
    "Silky Terrier",
    "Soft-coated Wheaten Terrier",
    "Staffordshire Bullterrier",
    "Standard Poodle",
    "Standard Schnauzer",
    "Sussex Spaniel",
    "Tibetan Mastiff",
    "Tibetan Terrier",
    "Toy Poodle",
    "Toy Terrier",
    "Vizsla",
    "Walker Hound",
    "Weimaraner",
    "Welsh Springer Spaniel",
    "West Highland White Terrier",
    "Whippet",
    "Wire-haired Fox Terrier",
    "Yorkshire Terrier",
]
