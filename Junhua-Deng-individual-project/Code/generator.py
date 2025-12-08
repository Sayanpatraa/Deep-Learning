import os
from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image


class DogDiffusionXLGenerator:
    def __init__(
        self,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_repo: str = "artificialguybr/LineAniRedmond-LinearMangaSDXL-V2",
        lora_weight_name: Optional[str] = None,
        device: Optional[str] = None,
        lora_scale: float = 0.9,
    ):

        self.lora_repo = lora_repo
        self.lora_weight_name = lora_weight_name
        self.has_lora = True
        self.lora_scale = lora_scale

        print(f"[INFO] LoRA will be loaded from Hugging Face:")
        print(f"       repo: {self.lora_repo}")
        print(
            f"       weight_name: "
            f"{self.lora_weight_name if self.lora_weight_name is not None else '<auto-detect>'}"
        )

        # ---- device ----
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # ---- base model ----
        print(f"[INFO] Loading base model: {base_model}")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
            use_safetensors=True,
        ).to(device)
        '''
        if device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("[INFO] xFormers enabled")
            except Exception as e:
                print(f"[WARN] xFormers not enabled: {e}")
        '''

        # ---- LoRA (from Hugging Face) ----
        print("[INFO] Loading LoRA weights from Hugging Face...")
        if self.lora_weight_name is not None:
            # Assigned safetensors
            self.pipe.load_lora_weights(
                self.lora_repo, weight_name=self.lora_weight_name
            )
        else:
            # Neglect weight_name if only one LoRA exist
            self.pipe.load_lora_weights(self.lora_repo)
        print("[INFO] LoRA loaded")

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
    ) -> Image.Image:
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        extra_kwargs = {}
        if getattr(self, "has_lora", False) and self.lora_scale is not None:
            extra_kwargs["cross_attention_kwargs"] = {"scale": self.lora_scale}

        print(f"[INFO] Generating image...")
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **extra_kwargs,
        )
        return out.images[0]


if __name__ == "__main__":
    import datetime
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--breed", type=str, required=True,
                        help="Dog breed to be inserted into the prompt")
    args = parser.parse_args()

    current_dir = os.path.dirname(__file__)
    print(f"[INFO] Current dir: {current_dir}")

    generator = DogDiffusionXLGenerator(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        lora_repo="artificialguybr/LineAniRedmond-LinearMangaSDXL-V2",
        lora_weight_name=None,
        lora_scale=0.9,
    )

    # prompt
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

