import streamlit as st
import torch
from PIL import Image

# Adjust imports according to your project structure.
# If classifier.py and generator.py are in src/multimodal_dog/models/:
from src.multimodal_dog.models.classifier import DogClassifier
from src.multimodal_dog.models.generator_v2 import DogGenerator
# If they are in the same folder as this script, use instead:
# from classifier import DogClassifier
# from generator import DogDiffusionXLGenerator


# ---------- helpers ----------

@st.cache_resource
def load_classifier():
    """
    Load DogClassifier model only once and cache it.
    Hugging Face weights will be downloaded only the first time.
    """
    clf = DogClassifier.from_hf()
    return clf


@st.cache_resource
def load_generator():
    """
    Load a single SDXL base pipeline and keep it cached.

    LoRA styles (manga / realistic) are switched later via adapters
    using generator.set_style(style_name, scale).
    This design is friendly for a 12GB VRAM GPU.
    """
    gen = DogGenerator(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        device=None,              # auto-select cuda / cpu
        dtype=torch.float16,      # good for 12GB GPUs
        enable_cpu_offload=False  # usually not needed on 12GB
    )
    return gen


def approx_token_count(text: str) -> int:
    """
    Rough approximation of token count using whitespace-separated words.
    CLIP actually tokenizes differently, but this is good enough
    as a safety margin under the 77 token limit.
    """
    return len(text.strip().split()) if text.strip() else 0


# ---------- main app ----------

def main():
    st.title("🐶 Dog Breed Classifier + DogDiffusionXL (Single Base + LoRA Switch)")

    st.write(
        "Workflow:\n"
        "1. Upload a dog image and classify its breed (ResNet50).\n"
        "2. Choose to use the predicted breed or enter a breed manually.\n"
        "3. Choose LoRA style (Manga / Realistic), prompts auto-adjust.\n"
        "4. Generate a new dog image with DogDiffusionXL."
    )

    # --- sidebar: generation settings ---
    with st.sidebar:
        st.header("Generation Settings")

        height = st.number_input("Height", min_value=512, max_value=1536, value=720, step=64)
        width = st.number_input("Width", min_value=512, max_value=1536, value=1080, step=64)

        steps = st.slider("Num inference steps", min_value=10, max_value=80, value=50, step=5)
        guidance_scale = st.slider("Guidance scale", min_value=1.0, max_value=12.0, value=5.0, step=0.5)

        seed = st.number_input("Seed (0 = random)", min_value=0, max_value=999999, value=39, step=1)

    # --- Step 0: image upload ---
    uploaded_file = st.file_uploader(
        "Upload a dog image (jpg/jpeg/png):",
        type=["jpg", "jpeg", "png"],
    )

    # When a new file is uploaded, store it in session_state
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state["uploaded_img"] = img

        st.image(img, caption="Uploaded Image", width='stretch')

        if st.button("Predict Breed"):
            with st.spinner("Loading classifier & predicting breed..."):
                clf = load_classifier()
                breed, conf = clf.predict(img)

            st.session_state["predicted_breed"] = breed
            st.session_state["pred_conf"] = float(conf)

    # --- Step 1: show prediction & breed selection if available ---
    if "predicted_breed" in st.session_state:
        predicted_breed = st.session_state["predicted_breed"]
        conf = st.session_state.get("pred_conf", 0.0)

        st.success("Classification complete ✅")
        st.markdown(f"### 🐾 Predicted breed: **{predicted_breed}**")
        st.markdown(f"### 🔢 Confidence: **{conf:.4f}**")

        st.markdown("### 1️⃣ Choose breed for generation")

        # Let user choose to use predicted breed or manual input
        breed_mode = st.radio(
            "Breed source:",
            ("Use predicted breed", "Enter breed manually"),
        )

        if breed_mode == "Use predicted breed":
            final_breed = predicted_breed
        else:
            final_breed = st.text_input("Enter a breed manually:")

        # --- Step 2: LoRA style selection ---
        st.markdown("### 2️⃣ Choose LoRA style")

        lora_choice = st.selectbox(
            "LoRA / Style:",
            (
                "Manga (black & white, Jump-style)",
                "Realistic (photo, DogDiffusion)",
            ),
        )

        if lora_choice.startswith("Manga"):
            style_name = "manga"
            lora_scale = 0.9
        else:
            style_name = "realistic"
            lora_scale = 1.0

        # --- Step 3: build style-specific prompts ---
        st.markdown("### 3️⃣ Prompts")

        # If user did not provide breed, fall back to generic word "dog"
        subject = final_breed.strip() if final_breed and final_breed.strip() else "dog"

        if style_name == "manga":
            # Manga-style positive prompt (kept well under 77 tokens)
            positive_prompt_template = (
                "black and white side view of a {subject} dog standing, accurate anatomy, "
                "single visible tail, all four legs visible, all four paws visible, full body, consistent perspective, "
                "shonen jump manga style, screentone shading, inked lineart, high contrast, "
                "speed lines, impact frame, dramatic action"
            )

            default_negative_prompt = (
                "missing legs, missing paws, cropped legs, legs out of frame, body cropped, only two legs, three legs, hidden legs, occluded legs, "
                "extra limbs, extra legs, extra tails, duplicate tail, extra heads, "
                "wrong dog anatomy, deformed limbs, fused anatomy, elongated body, tiny head, giant head, short limbs, unnatural posture, "
                "blurry, low quality, distorted, "
                "color, colorful, pastel colors, 3d, cgi, painting, sketch, "
                "jpeg artifacts, text, logo, watermark"
            )
        else:
            # Realistic-style positive prompt (also under 77 tokens)
            positive_prompt_template = (
                "high quality photo of a {subject} dog, full body, single tail, correct body proportions, "
                "structurally accurate canine body,natural lighting, detailed fur, sharp focus, "
                "outdoor background, shallow depth of field, realistic colors, 4k, ultra detailed"
            )

            default_negative_prompt = (
                "cartoon, anime, manga, illustration, drawing, painting, sketch, 3d render, cgi, "
                "low quality, blurry, distorted, extra limbs, extra tails, duplicate tail, extra heads, "
                "wrong anatomy, jpeg artifacts, watermark, logo, text"
            )

        positive_prompt = positive_prompt_template.format(subject=subject)

        st.markdown("**Positive prompt (auto-generated by style + breed):**")
        st.text(positive_prompt)

        pos_tokens = approx_token_count(positive_prompt)
        if pos_tokens > 77:
            st.error(f"Approximate token count: {pos_tokens} (⚠️ above 77, please shorten the template).")
        else:
            st.caption(f"Approximate token count: **{pos_tokens}** (under 77 is recommended).")

        negative_prompt_user = st.text_area(
            "Negative prompt (optional, leave empty to use style-specific default):",
            value="",
            placeholder=default_negative_prompt,
        )

        # --- Step 4: generate button ---
        if st.button("🚀 Generate Image"):
            if "uploaded_img" not in st.session_state:
                st.warning("Please upload an image first.")
                st.stop()

            if breed_mode == "Enter breed manually" and not final_breed.strip():
                st.warning("Please enter a breed name before generating.")
                st.stop()

            with st.spinner("Loading generator and creating an image..."):
                # 1) Load the single base generator (cached)
                generator = load_generator()

                # 2) Switch LoRA style via adapter
                generator.set_style(style_name, scale=lora_scale)

                # 3) Decide which negative prompt to use
                negative_prompt_to_use = negative_prompt_user.strip() or default_negative_prompt

                # 4) Seed: 0 means random
                final_seed = None if seed == 0 else int(seed)

                # 5) Run generation
                image = generator.generate(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt_to_use,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    seed=final_seed,
                )

            st.success("Image generation complete 🐕🖼️")
            st.image(
                image,
                caption=f"Style: {style_name} | Breed: {subject}",
                width='stretch',
            )


if __name__ == "__main__":
    main()
