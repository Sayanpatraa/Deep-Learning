# Dog Breed AI - Streamlit App

A Streamlit application for dog breed classification and stylized image generation.

## Features

### 🔍 Classify Tab
- Upload an image of a dog
- Get top-5 breed predictions with confidence scores
- One-click transfer to Generate tab

### 🎨 Generate Tab
- Select from common breeds or enter custom breed
- Choose from multiple art styles:
  - **Manga (LineAni)** - Black and white manga style
  - **Anime** - Colorful anime illustration
  - **Pixel Art** - Retro 16-bit style
  - **Watercolor** - Soft watercolor painting
- Adjustable generation parameters
- Download generated images

## Setup

### 1. Clone/Copy Files

Ensure you have all files:
```
streamlit_app/
├── app.py
├── config.py
├── classifier.py
├── generator.py
├── requirements.txt
└── README.md
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt

# For CUDA support, ensure you have the correct PyTorch version:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Models

### Classifier
- **Model**: ResNet-50 fine-tuned for dog breed classification
- **Hosted at**: `huggingface.co/djhua0103/dog-breed-resnet50`
- **Auto-downloaded** on first use

### Generator
- **Base Model**: Stable Diffusion XL 1.0
- **LoRA Styles**: Multiple styles from Hugging Face
- **Auto-downloaded** on first use (may take several minutes)

## Configuration

Edit `config.py` to:
- Add/remove LoRA styles
- Modify prompt templates
- Adjust default generation parameters
- Update breed list

### Adding a New LoRA Style

```python
# In config.py, add to LORA_STYLES:
LORA_STYLES = {
    # ... existing styles ...
    "New Style": {
        "repo": "huggingface/repo-name",
        "weight_name": None,  # or specific .safetensors file
        "scale": 0.8,
        "description": "Description of the style"
    },
}

# Add corresponding prompt template:
PROMPT_TEMPLATES = {
    # ... existing templates ...
    "New Style": "Your prompt template with {breed} placeholder",
}
```

## AWS Deployment Notes

### Recommended Instance
- **Type**: g4dn.xlarge or better (for SDXL)
- **GPU Memory**: 16GB+ recommended
- **Storage**: 50GB+ (for model caching)

### Environment Setup
```bash
# Install CUDA drivers if not present
# Install Python 3.10+

# Clone your repo
git clone <your-github-repo>
cd streamlit_app

# Install dependencies
pip install -r requirements.txt

# Run with specific port
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Security Group
- Open port 8501 for Streamlit access

## Troubleshooting

### Out of Memory
- Reduce image dimensions in Advanced Settings
- Reduce inference steps
- Ensure only one model is loaded at a time

### Slow First Generation
- First run downloads models (~10GB)
- Subsequent runs use cached models

### LoRA Not Loading
- Check Hugging Face repo accessibility
- Verify weight_name if specified

## License

[Your License Here]
