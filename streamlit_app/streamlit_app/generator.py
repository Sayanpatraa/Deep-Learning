"""
Dog Image Generator using Stable Diffusion XL with LoRA styles
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from config import (
    GENERATOR_BASE_MODEL,
    LORA_STYLES,
    PROMPT_TEMPLATES,
    NEGATIVE_PROMPTS,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE,
)


class DogGenerator:
    """
    Dog image generator using Stable Diffusion XL with swappable LoRA styles.
    """

    def __init__(
        self,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.pipe: Optional[StableDiffusionXLPipeline] = None
        self.current_lora: Optional[str] = None
        self.base_model = GENERATOR_BASE_MODEL

    def load_base_model(self) -> None:
        """Load the base SDXL model."""
        if self.pipe is not None:
            return

        print(f"[INFO] Loading base model: {self.base_model}")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            variant="fp16" if self.device == "cuda" else None,
            use_safetensors=True,
        ).to(self.device)
        print("[INFO] Base model loaded")

    def load_lora(self, style_name: str) -> None:
        """
        Load a LoRA style. Unloads previous LoRA if different.

        Args:
            style_name: Key from LORA_STYLES dict
        """
        if style_name not in LORA_STYLES:
            raise ValueError(f"Unknown style: {style_name}. Available: {list(LORA_STYLES.keys())}")

        # Ensure base model is loaded
        self.load_base_model()

        # Skip if same LoRA already loaded
        if self.current_lora == style_name:
            print(f"[INFO] LoRA '{style_name}' already loaded")
            return

        # Unload previous LoRA if any
        if self.current_lora is not None:
            print(f"[INFO] Unloading previous LoRA: {self.current_lora}")
            self.pipe.unload_lora_weights()

        # Load new LoRA
        lora_config = LORA_STYLES[style_name]
        print(f"[INFO] Loading LoRA: {style_name}")
        print(f"       repo: {lora_config['repo']}")

        if lora_config["weight_name"] is not None:
            self.pipe.load_lora_weights(
                lora_config["repo"],
                weight_name=lora_config["weight_name"],
            )
        else:
            self.pipe.load_lora_weights(lora_config["repo"])

        self.current_lora = style_name
        print(f"[INFO] LoRA '{style_name}' loaded")

    def generate(
        self,
        breed: str,
        style_name: str,
        custom_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate a dog image.

        Args:
            breed: Dog breed to generate
            style_name: LoRA style to use
            custom_prompt: Optional custom prompt (overrides template)
            negative_prompt: Optional custom negative prompt
            height: Image height
            width: Image width
            num_inference_steps: Number of diffusion steps
            guidance_scale: CFG scale
            seed: Random seed for reproducibility

        Returns:
            Generated PIL Image
        """
        # Load LoRA if needed
        self.load_lora(style_name)

        # Build prompt
        if custom_prompt:
            prompt = custom_prompt.format(breed=breed)
        else:
            template = PROMPT_TEMPLATES.get(style_name, PROMPT_TEMPLATES["Manga (LineAni)"])
            prompt = template.format(breed=breed)

        # Get negative prompt
        if negative_prompt is None:
            negative_prompt = NEGATIVE_PROMPTS.get(style_name, DEFAULT_NEGATIVE_PROMPT)

        # Set up generator for seed
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        # LoRA scale
        lora_scale = LORA_STYLES[style_name].get("scale", 0.8)
        extra_kwargs = {"cross_attention_kwargs": {"scale": lora_scale}}

        print(f"[INFO] Generating image...")
        print(f"       Breed: {breed}")
        print(f"       Style: {style_name}")
        print(f"       Seed: {seed if seed else 'random'}")

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **extra_kwargs,
        )

        print("[INFO] Generation complete")
        return out.images[0]

    @staticmethod
    def get_available_styles() -> dict:
        """Return available LoRA styles with descriptions."""
        return {
            name: config["description"]
            for name, config in LORA_STYLES.items()
        }

    def is_loaded(self) -> bool:
        """Check if base model is loaded."""
        return self.pipe is not None
