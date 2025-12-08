# 🐕 Dog Breed AI

A Streamlit application for dog breed classification and AI image generation.

## Features

### 🔍 Classify Tab
- Upload an image of a dog
- Get top-5 breed predictions with confidence scores
- Uses a fine-tuned ResNet-50 model hosted on Hugging Face

### 🎨 Generate Tab
- Generate dog images in multiple styles
- Select from 120 dog breeds or enter custom breed names
- Adjustable generation parameters

#### Available Styles

| Style | Description | Pipeline |
|-------|-------------|----------|
| **Realistic** | Photorealistic images with accurate anatomy | SDXL Base + Refiner |
| **Manga (LineAni)** | Black and white manga with speed lines | SDXL + LoRA |
| **Anime** | Colorful anime style | SDXL + LoRA |
| **Pastel Anime** | Soft pastel anime illustrations | SDXL + LoRA |
| **Pixel Art** | Retro 16-bit pixel art | SDXL + LoRA |

## Project Structure

```
streamlit_app/
├── app.py                  # Main Streamlit application
├── config.py               # Configuration and settings
├── classifier.py           # Dog breed classifier (ResNet-50)
├── generator.py            # Stylized generator (SDXL + LoRA)
├── realistic_generator.py  # Photorealistic generator (SDXL + Refiner)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended: 16GB+ VRAM for SDXL)
- ~20GB disk space for model weights

## Installation

### 1. Clone/Copy Files

```bash
git clone <your-repo-url>
cd streamlit_app
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For CUDA support, ensure you have the correct PyTorch version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Models

### Classifier
- **Architecture**: ResNet-50
- **Dataset**: 120 dog breeds
- **Hosted at**: [huggingface.co/djhua0103/dog-breed-resnet50](https://huggingface.co/djhua0103/dog-breed-resnet50)
- **Auto-downloaded** on first use

### Generators
- **Base Model**: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- **Refiner Model**: [stabilityai/stable-diffusion-xl-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
- **LoRA Styles**: Various from Hugging Face (see config.py)

Models are downloaded automatically on first use (~15GB total).

## Configuration

Edit `config.py` to customize:

### Add New LoRA Styles

```python
LORA_STYLES = {
    # ... existing styles ...
    "New Style": {
        "repo": "huggingface/repo-name",
        "weight_name": "weights.safetensors",  # or None for auto-detect
        "scale": 0.8,
        "description": "Description of the style",
    },
}

# Add corresponding prompt template
PROMPT_TEMPLATES["New Style"] = "Your prompt with {breed} placeholder"
```

### Adjust Default Parameters

```python
# Realistic generation
DEFAULT_STEPS_BASE = 28      # Base model steps
DEFAULT_STEPS_REFINER = 20   # Refiner steps

# Stylized generation
DEFAULT_STEPS = 50           # Total inference steps
DEFAULT_GUIDANCE = 5.0       # CFG scale
```

## Usage Tips

### Classification
1. Upload a clear image of a dog
2. Click "Classify Breed"
3. View top-5 predictions
4. Click "Use in Generator →" in sidebar to generate art of that breed

### Generation
1. Select a breed from dropdown or enter custom name
2. Choose an art style
3. Expand "Advanced Settings" to adjust:
   - **Realistic**: Base steps, Refiner steps, dimensions
   - **Stylized**: Inference steps, dimensions, guidance scale
4. Optionally set a fixed seed for reproducibility
5. Click "Generate Image"
6. Download the result

### Performance Notes
- First generation takes longer (model loading)
- Subsequent generations are faster (models cached)
- Realistic mode uses more VRAM (two models)
- Reduce dimensions if encountering OOM errors

## AWS Deployment

### Recommended Instance
- **Type**: g4dn.xlarge or g5.xlarge
- **GPU**: NVIDIA T4 (16GB) or A10G (24GB)
- **Storage**: 50GB+ SSD

### Setup
```bash
# Install NVIDIA drivers and CUDA
# Install Python 3.10+

# Clone and setup
git clone <your-repo>
cd streamlit_app
pip install -r requirements.txt

# Run with external access
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Security Group
- Open port 8501 for Streamlit access
- Consider adding HTTPS via reverse proxy (nginx)

## Troubleshooting

### Out of Memory (OOM)
- Reduce image dimensions (512x512 for testing)
- Reduce inference steps
- Use only one generator at a time (restart app between modes)

### Slow Generation
- First run downloads models (~15GB)
- Ensure CUDA is available: check sidebar shows "Device: cuda"
- Reduce steps for faster iteration

### LoRA Loading Errors
- Verify the Hugging Face repo exists and is public
- Check `weight_name` matches actual file in repo
- Some LoRAs may be incompatible with SDXL base

### Import Errors
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version: 3.10+ required

## License

[Your License Here]

## Acknowledgments

- [Stable Diffusion XL](https://stability.ai/) by Stability AI
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- LoRA model creators (see config.py for credits)
