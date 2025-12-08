import streamlit as st
from PIL import Image

# Import classifier & generator from your project
from src.multimodal_dog.models.classifier import DogClassifier
from src.multimodal_dog.models.generator import DogDiffusionXLGenerator


@st.cache_resource
def load_classifier():
    """
    Load DogClassifier model only once and cache it.
    HuggingFace weights will be downloaded only the first time.
    """
    model = DogClassifier.from_hf()
    return model


@st.cache_resource
def load_generator():
    """
    Load DogDiffusionXLGenerator (SDXL + LoRA) once and cache it.
    This is heavy (several GB), so we definitely want caching.
    """
    gen = DogDiffusionXLGenerator(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        lora_repo="artificialguybr/LineAniRedmond-LinearMangaSDXL-V2",
        lora_weight_name=None,
        lora_scale=0.9,
    )
    return gen


def main():
    st.title("🐶 Dog Breed Classifier + DogDiffusionXL")

    st.write(
        "Upload a dog image, the app will:\n"
        "1) Classify the breed using a fine-tuned ResNet50.\n"
        "2) Let you choose the breed (predicted or manual).\n"
        "3) Generate a new image using DogDiffusionXL with optional negative prompts."
    )

    # --- Image upload ---
    uploaded_file = st.file_uploader(
        "Upload an image (jpg/jpeg/png):",
        type=["jpg", "jpeg", "png"],
    )

    # If user uploads a new file, save it in session_state
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state["uploaded_img"] = img

        # Show preview
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # --- Step 1: Predict breed only ---
        if st.button("Predict Breed"):
            with st.spinner("Loading classifier & predicting breed..."):
                clf = load_classifier()
                breed, conf = clf.predict(img)

            # Save results into session_state so they persist across reruns
            st.session_state["predicted_breed"] = breed
            st.session_state["pred_conf"] = float(conf)

    # --- Show prediction and generation UI if we already have a predicted breed ---
    if "predicted_breed" in st.session_state:
        predicted_breed = st.session_state["predicted_breed"]
        conf = st.session_state.get("pred_conf", 0.0)

        st.success("Classification complete ✅")
        st.markdown(f"### 🐾 Predicted breed: **{predicted_breed}**")
        st.markdown(f"### 🔢 Confidence: **{conf:.4f}**")

        st.markdown("### 🔧 Choose the breed for generation")

        # Select whether to use predicted breed or manual input
        option = st.radio(
            "Select breed input mode:",
            ("Use predicted breed", "Enter breed manually"),
        )

        if option == "Use predicted breed":
            final_breed = predicted_breed
        else:
            final_breed = st.text_input("Enter a breed manually:")

        # Negative prompt input
        negative_prompt = st.text_area(
            "Negative prompt (optional):",
            placeholder=(
                "extra tail, duplicate tail, second tail, tail duplication, tail artifact,"
                "blurry, low quality, distorted, extra limbs, extra legs, extra tails, duplicate tail, extra heads, "
                "wrong dog anatomy, elongated body, tiny head, giant head, short limbs, missing legs, mutated paws, "
                "fused anatomy, deformed pose, unnatural posture, "
                "color, colorful, pastel colors, 3d, cgi, photorealistic, painting, edge highlight, "
                "jpeg artifacts, bad anatomy, text, logo, watermark",
            ),
        )

        # --- Step 2: Generate image ---
        if st.button("Generate Image"):
            # Basic checks
            if "uploaded_img" not in st.session_state:
                st.warning("Please upload an image first.")
                return

            if option == "Enter breed manually" and not final_breed.strip():
                st.warning("Please enter a breed name before generating.")
                return

            with st.spinner(
                "Loading generator & creating a new image... this may take a while"
            ):
                generator = load_generator()

                # Build positive prompt
                # You can change the wording below to any style you prefer
                positive_prompt = (
                    f"Black and white side view of a {final_breed} dog sprinting, accurate canine anatomy, single visible tail, "
                     "one tail only, proper proportions, full body in frame, natural limb spacing, dynamic running pose, "
                     "consistent perspective, shonen jump manga style, screentone shading, inked lineart, high contrast, "
                     "speed lines, impact frame, dramatic action"
                )

                # If user provided a negative prompt, override; otherwise,
                # do NOT pass it and let the generator use its own default.
                if negative_prompt.strip():
                    gen_img = generator.generate(
                        prompt=positive_prompt,
                        negative_prompt=negative_prompt.strip(),
                        height=720,
                        width=1080,
                        num_inference_steps=50,
                        guidance_scale=5.0,
                        seed=39,
                    )
                else:
                    # No custom negative_prompt: rely on the default defined in generator.py
                    gen_img = generator.generate(
                        prompt=positive_prompt,
                        height=720,
                        width=1080,
                        num_inference_steps=50,
                        guidance_scale=5.0,
                        seed=39,
                    )

            st.success("Image generation complete 🖼️")
            st.markdown(f"### 🎨 Generated image ({final_breed})")
            st.image(gen_img, caption=f"DogDiffusionXL - {final_breed}", use_container_width=True)


if __name__ == "__main__":
    main()
