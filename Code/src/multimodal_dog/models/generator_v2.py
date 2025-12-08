import os
from typing import Optional, Dict, Any

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image


class DogGenerator:
    """
    SDXL-based image generator that keeps a single base pipeline in memory
    and switches LoRA styles via adapters.

    This design is VRAM-friendly and works well on a 12GB GPU:
      - Only one SDXL UNet / VAE / Text Encoder is loaded.
      - LoRA weights are small adapters applied on top of the base model.
    """

    def __init__(
        self,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        enable_cpu_offload: bool = False,
    ):
        """
        Initialize the base SDXL pipeline and preload LoRA adapters.

        Args:
            base_model: Hugging Face repo id of the SDXL base model.
            device: "cuda", "cpu", or None. If None, automatically choose.
            dtype: Torch dtype used for SDXL weights (fp16 recommended for 12GB).
            enable_cpu_offload: If True, use model CPU offload to reduce VRAM.
                                On a 12GB GPU you usually can leave this False.
        """

        # ---- decide device ----
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.base_model = base_model
        self.dtype = dtype
        self.enable_cpu_offload = enable_cpu_offload

        print(f"[INFO] Using device: {self.device}")
        print(f"[INFO] Loading base SDXL model: {base_model}")

        # ---- load single SDXL base pipeline ----
        self.pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype if self.device == "cuda" else torch.float32,
            variant="fp16" if (self.device == "cuda" and dtype == torch.float16) else None,
            use_safetensors=True,
        )

        # Move pipeline to device or enable CPU offload
        if self.device == "cuda":
            if enable_cpu_offload:
                # Slower, but more memory efficient on very small GPUs
                print("[INFO] Enabling model CPU offload (accelerate).")
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(self.device)
        else:
            self.pipe.to("cpu")

        # ---- memory optimizations for ~12GB VRAM ----
        # VAE tiling + slicing reduce peak memory usage.
        self.pipe.enable_vae_tiling()
        self.pipe.enable_vae_slicing()

        # Try xFormers for memory-efficient attention.
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("[INFO] xFormers memory-efficient attention enabled.")
        except Exception as e:
            print(f"[WARN] xFormers not enabled: {e}")
            print("[INFO] Falling back to attention slicing.")
            self.pipe.enable_attention_slicing()

        # Track which adapter (LoRA style) is active
        self.current_adapter: Optional[str] = None

        # Mapping from style name to adapter name
        # You can extend this dict with more styles if needed.
        self.style_to_adapter: Dict[str, str] = {
            "manga": "manga",
            "realistic": "realistic",
        }

        # ---- preload LoRA adapters ----
        print("[INFO] Loading LoRA adapters (manga, realistic) from Hugging Face...")

        # Manga LoRA -> adapter "manga"
        self.pipe.load_lora_weights(
            "artificialguybr/LineAniRedmond-LinearMangaSDXL-V2",
            adapter_name="manga",
        )

        # Realistic LoRA -> adapter "realistic"
        self.pipe.load_lora_weights(
            "djhua0103/DogDiffusion",
            adapter_name="realistic",
        )

        print("[INFO] LoRA adapters loaded: manga, realistic.")

    def set_style(self, style: Optional[str], scale: float = 1.0) -> None:
        """
        Activate a specific LoRA style via adapters.

        Args:
            style: "manga", "realistic", or None to disable all LoRA.
            scale: Relative strength of the LoRA adapter (if supported by diffusers version).
        """

        # Disable all LoRA -> back to pure base SDXL
        if style is None or style == "" or style not in self.style_to_adapter:
            print("[INFO] Disabling all LoRA adapters, using base SDXL only.")
            self.pipe.set_adapters([])  # clear all adapters
            self.current_adapter = None
            return

        adapter_name = self.style_to_adapter[style]

        # Only switch if the style actually changed
        if adapter_name != self.current_adapter:
            print(f"[INFO] Switching LoRA style to: {style} (adapter: {adapter_name})")
            try:
                # Newer diffusers versions accept a single string
                self.pipe.set_adapters(adapter_name)
            except TypeError:
                # Fallback: older versions expect a list
                self.pipe.set_adapters([adapter_name])

            self.current_adapter = adapter_name

        # Try to set adapter weights (scales) if diffusers version supports it
        try:
            self.pipe.set_adapters([adapter_name], adapter_weights=[scale])
            print(f"[INFO] Setting LoRA scale for '{adapter_name}' to: {scale}")
        except Exception:
            # Safe to ignore if not supported; the adapter will still be active.
            print("[WARN] Adapter weights (scale) not supported by this diffusers version.")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "extra tail, duplicate tail, second tail, tail duplication, tail artifact,"
                               "blurry, low quality, distorted, extra limbs, extra legs, extra tails, duplicate tail, extra heads, "
                               "wrong dog anatomy, elongated body, tiny head, giant head, short limbs, missing legs, mutated paws, "
                               "fused anatomy, deformed pose, unnatural posture, "
                               "color, colorful, pastel colors, 3d, cgi, photorealistic, painting, edge highlight, "
                               "jpeg artifacts, bad anatomy, text, logo, watermark",
        height: int = 720,
        width: int = 1080,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        """
        Generate a single image from the current style (adapter) and prompts.

        Args:
            prompt: Positive prompt (description of the desired image).
            negative_prompt: Negative prompt to avoid artifacts / unwanted content.
            height: Output image height in pixels.
            width: Output image width in pixels.
            num_inference_steps: Number of diffusion steps (higher = better but slower).
            guidance_scale: Classifier-free guidance scale.
            seed: Optional random seed for reproducibility.
            extra_kwargs: Extra keyword arguments passed to the pipeline call.

        Returns:
            A PIL.Image.Image object.
        """
        if extra_kwargs is None:
            extra_kwargs = {}

        # Set up a deterministic generator if seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)
            extra_kwargs["generator"] = generator

        print("[INFO] Generating image...")
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **extra_kwargs,
        )
        return out.images[0]


if __name__ == "__main__":
    import datetime
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--breed",
        type=str,
        required=True,
        help="Dog breed to be inserted into the prompt",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="manga",
        choices=["manga", "realistic"],
        help="LoRA style to use (manga or realistic)",
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(__file__)
    print(f"[INFO] Current dir: {current_dir}")

    # Create generator with a single base SDXL pipeline
    generator = DogGenerator(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        device=None,              # auto-select cuda / cpu
        dtype=torch.float16,      # good for 12GB VRAM
        enable_cpu_offload=False, # usually not needed on 12GB
    )

    # Switch LoRA style via adapter
    if args.style == "manga":
        generator.set_style("manga", scale=0.9)
    else:
        generator.set_style("realistic", scale=1.0)

    # Prompt (you can later sync this with your Streamlit prompts)
    prompt = (
        f"Black and white side view of a {args.breed} dog sprinting, accurate canine anatomy, single visible tail, "
        "one tail only, proper proportions, full body in frame, natural limb spacing, dynamic running pose, "
        "consistent perspective, shonen jump manga style, screentone shading, inked lineart, high contrast, "
        "speed lines, impact frame, dramatic action"
    )

    img = generator.generate(
        prompt=prompt,
        height=720,
        width=1080,
        num_inference_steps=50,
        guidance_scale=5.0,
        seed=39,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"dogdiffusionxl_{timestamp}.png"

    output_dir = os.path.join(current_dir, "..", "output")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, file_name)
    img.save(save_path)

    print(f"[INFO] Saved to: {save_path}")
