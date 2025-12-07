"""
Dog Breed Classifier & Generator - Streamlit App
"""

import streamlit as st
from PIL import Image
import torch
import io

from classifier import DogClassifier
from generator import DogGenerator
from realistic_generator import RealLifeDogSDXL
from config import (
    DOG_BREEDS,
    LORA_STYLES,
    DEFAULT_LORA_STYLE,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE,
)


# Page config
st.set_page_config(
    page_title="Dog Breed AI",
    page_icon="🐕",
    layout="wide",
)

# Custom CSS for clean UI
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 24px;
        font-size: 18px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .breed-name {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    .confidence {
        font-size: 18px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# ============ Initialize Session State ============
if "classification_results" not in st.session_state:
    st.session_state.classification_results = None
if "classified_breed" not in st.session_state:
    st.session_state.classified_breed = None
if "use_classified_breed" not in st.session_state:
    st.session_state.use_classified_breed = False


# ============ Model Loading (Cached) ============

@st.cache_resource
def load_classifier():
    """Load classifier model (cached)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return DogClassifier.from_hf(device=device)


@st.cache_resource
def load_stylized_generator():
    """Load stylized generator for LoRA-based styles (cached)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return DogGenerator(device=device)


@st.cache_resource
def load_realistic_generator():
    """Load realistic generator with refiner (cached)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return RealLifeDogSDXL(device=device)


def is_realistic_style(style_name: str) -> bool:
    """Check if the selected style uses the realistic pipeline."""
    style_config = LORA_STYLES.get(style_name, {})
    return style_config.get("use_realistic_pipeline", False)


# ============ Main App ============

def main():
    st.title("🐕 Dog Breed AI")
    st.markdown("Classify dog breeds and generate stylized dog artwork")

    # Device info in sidebar
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.markdown(f"**Device:** `{device}`")
    if device == "cuda":
        st.sidebar.markdown(f"**GPU:** `{torch.cuda.get_device_name(0)}`")

    # Show classified breed info in sidebar if available
    if st.session_state.classified_breed:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Last Classified:**")
        st.sidebar.markdown(f"🐕 {st.session_state.classified_breed.replace('_', ' ').title()}")
        if st.sidebar.button("Use in Generator →"):
            st.session_state.use_classified_breed = True
            st.rerun()

    # Tabs
    tab_classify, tab_generate = st.tabs(["🔍 Classify", "🎨 Generate"])

    # ============ CLASSIFY TAB ============
    with tab_classify:
        st.header("Dog Breed Classification")
        st.markdown("Upload an image of a dog to identify its breed.")

        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png", "webp"],
                key="classifier_upload",
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)

                if st.button("🔍 Classify Breed", type="primary", use_container_width=True):
                    with st.spinner("Loading model..."):
                        classifier = load_classifier()

                    with st.spinner("Analyzing image..."):
                        results = classifier.predict_top_k(image, k=5)

                    # Store results in session state
                    st.session_state.classification_results = results
                    st.session_state.classified_breed = results[0][0]

        with col2:
            # Display results from session state (persists across reruns)
            if st.session_state.classification_results is not None:
                results = st.session_state.classification_results

                st.markdown("### Results")

                # Top prediction
                top_breed, top_conf = results[0]
                st.markdown(f"""
                <div class="result-box">
                    <div class="breed-name">{top_breed.replace('_', ' ').title()}</div>
                    <div class="confidence">Confidence: {top_conf:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

                # Other predictions
                if len(results) > 1:
                    st.markdown("**Other possibilities:**")
                    for breed, conf in results[1:]:
                        st.markdown(f"- {breed.replace('_', ' ').title()}: {conf:.1%}")

                # Quick action - this now persists!
                st.markdown("---")
                st.info("💡 To generate art of this breed, click **'Use in Generator →'** in the sidebar, then switch to the **Generate** tab.")

            else:
                # Placeholder
                st.markdown("""
                <div style="
                    border: 2px dashed #ccc;
                    border-radius: 10px;
                    padding: 100px 20px;
                    text-align: center;
                    color: #888;
                ">
                    <p style="font-size: 48px; margin: 0;">📊</p>
                    <p>Classification results will appear here</p>
                </div>
                """, unsafe_allow_html=True)

    # ============ GENERATE TAB ============
    with tab_generate:
        st.header("Dog Image Generator")
        st.markdown("Generate stylized artwork of any dog breed.")

        col1, col2 = st.columns([1, 1])

        with col1:
            # Breed selection
            st.subheader("Breed")

            # Check if user clicked "Use in Generator" from sidebar
            if st.session_state.use_classified_breed and st.session_state.classified_breed:
                breed_from_classifier = st.session_state.classified_breed
                st.success(f"✓ Using classified breed: **{breed_from_classifier.replace('_', ' ').title()}**")
                # Reset flag
                st.session_state.use_classified_breed = False
            else:
                breed_from_classifier = None

            breed_option = st.radio(
                "Select breed method",
                ["Choose from list", "Custom input"],
                horizontal=True,
            )

            if breed_option == "Choose from list":
                # Try to find classified breed in list
                default_idx = 0
                if breed_from_classifier:
                    formatted = breed_from_classifier.replace('_', ' ').title()
                    if formatted in DOG_BREEDS:
                        default_idx = DOG_BREEDS.index(formatted)

                selected_breed = st.selectbox(
                    "Dog breed",
                    DOG_BREEDS,
                    index=default_idx,
                )
            else:
                default_val = ""
                if breed_from_classifier:
                    default_val = breed_from_classifier.replace('_', ' ').title()

                selected_breed = st.text_input(
                    "Enter breed name",
                    value=default_val,
                    placeholder="e.g., Golden Retriever",
                )

            # Style selection
            st.subheader("Style")
            style_options = list(LORA_STYLES.keys())
            selected_style = st.selectbox(
                "Art style",
                style_options,
                index=style_options.index(DEFAULT_LORA_STYLE),
            )
            st.caption(LORA_STYLES[selected_style]["description"])

            # Check if realistic style
            use_realistic = is_realistic_style(selected_style)

            # Advanced settings - different options for realistic vs stylized
            with st.expander("⚙️ Advanced Settings"):
                if use_realistic:
                    # Realistic generator settings
                    st.markdown("**Realistic Generation Settings**")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        height = st.number_input("Height", value=1024, step=64, min_value=512, max_value=1024)
                        steps_base = st.slider("Base Steps", min_value=15, max_value=50, value=28)
                    with col_b:
                        width = st.number_input("Width", value=1024, step=64, min_value=512, max_value=1280)
                        steps_refiner = st.slider("Refiner Steps", min_value=10, max_value=40, value=20)
                    
                    guidance = st.slider("Guidance Scale", min_value=1.0, max_value=10.0, value=5.5, step=0.5)
                    
                    use_seed = st.checkbox("Use fixed seed")
                    seed = None
                    if use_seed:
                        seed = st.number_input("Seed", value=1234, min_value=0, max_value=2**32-1)
                else:
                    # Stylized generator settings
                    steps_base = None
                    steps_refiner = None
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        height = st.number_input("Height", value=DEFAULT_HEIGHT, step=64, min_value=512, max_value=1024)
                        steps = st.slider("Inference Steps", min_value=20, max_value=100, value=DEFAULT_STEPS)
                    with col_b:
                        width = st.number_input("Width", value=DEFAULT_WIDTH, step=64, min_value=512, max_value=1280)
                        guidance = st.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=DEFAULT_GUIDANCE, step=0.5)

                    use_seed = st.checkbox("Use fixed seed")
                    seed = None
                    if use_seed:
                        seed = st.number_input("Seed", value=42, min_value=0, max_value=2**32-1)

            # Generate button
            generate_btn = st.button(
                "🎨 Generate Image",
                type="primary",
                use_container_width=True,
                disabled=not selected_breed,
            )

        with col2:
            if generate_btn and selected_breed:
                # Generate
                progress_text = st.empty()
                progress_text.markdown(f"**Generating {selected_breed} in {selected_style} style...**")

                with st.spinner("Loading model (this may take a minute on first run)..."):
                    if use_realistic:
                        generator = load_realistic_generator()
                    else:
                        generator = load_stylized_generator()

                with st.spinner("Generating..."):
                    try:
                        if use_realistic:
                            # Use realistic generator
                            generated_image = generator.generate(
                                breed=selected_breed,
                                seed=seed,
                                steps_base=steps_base,
                                steps_refiner=steps_refiner,
                                scale=guidance,
                                width=width,
                                height=height,
                            )
                        else:
                            # Use stylized generator
                            generated_image = generator.generate(
                                breed=selected_breed,
                                style_name=selected_style,
                                height=height,
                                width=width,
                                num_inference_steps=steps,
                                guidance_scale=guidance,
                                seed=seed,
                            )

                        progress_text.empty()
                        st.image(generated_image, caption=f"{selected_breed} - {selected_style}", use_container_width=True)

                        # Download button
                        buf = io.BytesIO()
                        generated_image.save(buf, format="PNG")
                        st.download_button(
                            label="💾 Download Image",
                            data=buf.getvalue(),
                            file_name=f"{selected_breed.lower().replace(' ', '_')}_{selected_style.lower().replace(' ', '_')}.png",
                            mime="image/png",
                        )

                    except Exception as e:
                        progress_text.empty()
                        st.error(f"Generation failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            elif not selected_breed and generate_btn:
                st.warning("Please enter a breed name.")

            else:
                # Placeholder when no generation yet
                st.markdown("""
                <div style="
                    border: 2px dashed #ccc;
                    border-radius: 10px;
                    padding: 100px 20px;
                    text-align: center;
                    color: #888;
                ">
                    <p style="font-size: 48px; margin: 0;">🖼️</p>
                    <p>Generated image will appear here</p>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
