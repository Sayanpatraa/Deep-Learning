"""
Dog Breed Classifier & Generator - Streamlit App

Two tabs:
1. Classify - Upload an image to identify dog breed
2. Generate - Create dog images in various styles (realistic or stylized)
"""

import streamlit as st
from PIL import Image
import torch
import io

from classifier import DogClassifier
from generator import DogGenerator
from realistic_generator import RealLifeDogSDXL, NEG, tone, microfur, real_sensor_noise, snap
from config import (
    DOG_BREEDS,
    LORA_STYLES,
    DEFAULT_LORA_STYLE,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE,
)


# ============ Page Configuration ============
st.set_page_config(
    page_title="Dog Breed AI",
    page_icon="🐕",
    layout="wide",
)

# Clean CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; padding: 10px 24px; font-size: 18px; }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .breed-name { font-size: 28px; font-weight: bold; color: #1f77b4; }
    .confidence { font-size: 18px; color: #666; }
</style>
""", unsafe_allow_html=True)


# ============ Session State ============
if "classification_results" not in st.session_state:
    st.session_state.classification_results = None
if "classified_breed" not in st.session_state:
    st.session_state.classified_breed = None
if "use_classified_breed" not in st.session_state:
    st.session_state.use_classified_breed = False


# ============ Cached Model Loaders ============
@st.cache_resource
def load_classifier():
    """Load classifier model (cached)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return DogClassifier.from_hf(device=device)


@st.cache_resource
def load_stylized_generator():
    """Load stylized LoRA generator (cached)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return DogGenerator(device=device)


@st.cache_resource
def load_realistic_generator():
    """Load realistic base+refiner generator (cached)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return RealLifeDogSDXL(device=device)


# ============ Helper Functions ============
def is_realistic_style(style_name: str) -> bool:
    """Check if style uses the realistic pipeline."""
    return LORA_STYLES.get(style_name, {}).get("use_realistic_pipeline", False)


def generate_realistic(
    generator: RealLifeDogSDXL,
    breed: str,
    seed: int,
    steps_base: int,
    steps_refiner: int,
    scale: float,
    width: int,
    height: int,
) -> Image.Image:
    """
    Generate realistic dog image.
    Uses prompt strings directly to avoid encode_prompt compatibility issues.
    """
    width, height = snap(width), snap(height)
    prompt = generator.build_prompt(breed)
    
    print(f"[PROMPT] {prompt[:100]}...")
    
    # Handle seed
    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
    print(f"[SEED] {seed}")
    
    g = torch.Generator(generator.device).manual_seed(seed)
    
    # Use high_noise_frac for base/refiner split (standard approach)
    high_noise_frac = 0.8
    
    # Base pass - use prompt string directly
    print("[INFO] Running base model...")
    base_out = generator.base(
        prompt=prompt,
        negative_prompt=NEG,
        height=height,
        width=width,
        num_inference_steps=steps_base,
        guidance_scale=scale,
        generator=g,
        denoising_end=high_noise_frac,
        output_type="latent",
    )
    
    latents = base_out.images
    
    # Refiner pass
    print("[INFO] Running refiner...")
    g_refiner = torch.Generator(generator.device).manual_seed(seed)
    
    refined = generator.refiner(
        prompt=prompt,
        negative_prompt=NEG,
        image=latents,
        num_inference_steps=steps_refiner,
        guidance_scale=scale,
        generator=g_refiner,
        denoising_start=high_noise_frac,
    )
    
    img = refined.images[0]
    
    # Post-processing for realism
    print("[INFO] Applying realism filters...")
    img = tone(img)
    img = microfur(img)
    img = real_sensor_noise(img)
    
    print("[INFO] Generation complete!")
    return img


# ============ Main App ============
def main():
    st.title("🐕 Dog Breed AI")
    st.markdown("Classify dog breeds and generate stylized dog artwork")
    
    # Sidebar - device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.markdown(f"**Device:** `{device}`")
    if device == "cuda":
        st.sidebar.markdown(f"**GPU:** `{torch.cuda.get_device_name(0)}`")
    
    # Sidebar - show last classified breed
    if st.session_state.classified_breed:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Last Classified:**")
        breed_display = st.session_state.classified_breed.replace('_', ' ').title()
        st.sidebar.markdown(f"🐕 {breed_display}")
        if st.sidebar.button("Use in Generator →"):
            st.session_state.use_classified_breed = True
            st.rerun()
    
    # Main tabs
    tab_classify, tab_generate = st.tabs(["🔍 Classify", "🎨 Generate"])
    
    # ==================== CLASSIFY TAB ====================
    with tab_classify:
        st.header("Dog Breed Classification")
        st.markdown("Upload an image of a dog to identify its breed.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png", "webp"],
                key="classifier_upload",
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🔍 Classify Breed", type="primary", use_container_width=True):
                    with st.spinner("Loading classifier..."):
                        classifier = load_classifier()
                    
                    with st.spinner("Analyzing..."):
                        results = classifier.predict_top_k(image, k=5)
                    
                    st.session_state.classification_results = results
                    st.session_state.classified_breed = results[0][0]
        
        with col2:
            if st.session_state.classification_results:
                results = st.session_state.classification_results
                
                st.markdown("### Results")
                
                top_breed, top_conf = results[0]
                st.markdown(f"""
                <div class="result-box">
                    <div class="breed-name">{top_breed.replace('_', ' ').title()}</div>
                    <div class="confidence">Confidence: {top_conf:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if len(results) > 1:
                    st.markdown("**Other possibilities:**")
                    for breed, conf in results[1:]:
                        st.markdown(f"- {breed.replace('_', ' ').title()}: {conf:.1%}")
                
                st.markdown("---")
                st.info("💡 Click **'Use in Generator →'** in the sidebar to generate art of this breed.")
            else:
                st.markdown("""
                <div style="border: 2px dashed #ccc; border-radius: 10px; padding: 80px 20px; text-align: center; color: #888;">
                    <p style="font-size: 48px; margin: 0;">📊</p>
                    <p>Results will appear here</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ==================== GENERATE TAB ====================
    with tab_generate:
        st.header("Dog Image Generator")
        st.markdown("Generate stylized artwork of any dog breed.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Breed selection
            st.subheader("Breed")
            
            # Check if coming from classifier
            if st.session_state.use_classified_breed and st.session_state.classified_breed:
                breed_from_classifier = st.session_state.classified_breed
                st.success(f"✓ Using: **{breed_from_classifier.replace('_', ' ').title()}**")
                st.session_state.use_classified_breed = False
            else:
                breed_from_classifier = None
            
            breed_method = st.radio("Select breed", ["From list", "Custom"], horizontal=True)
            
            if breed_method == "From list":
                default_idx = 0
                if breed_from_classifier:
                    formatted = breed_from_classifier.replace('_', ' ').title()
                    if formatted in DOG_BREEDS:
                        default_idx = DOG_BREEDS.index(formatted)
                selected_breed = st.selectbox("Dog breed", DOG_BREEDS, index=default_idx)
            else:
                default_val = breed_from_classifier.replace('_', ' ').title() if breed_from_classifier else ""
                selected_breed = st.text_input("Enter breed", value=default_val, placeholder="e.g., Golden Retriever")
            
            # Style selection
            st.subheader("Style")
            style_options = list(LORA_STYLES.keys())
            selected_style = st.selectbox(
                "Art style",
                style_options,
                index=style_options.index(DEFAULT_LORA_STYLE),
            )
            st.caption(LORA_STYLES[selected_style]["description"])
            
            use_realistic = is_realistic_style(selected_style)
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings"):
                if use_realistic:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        height = st.number_input("Height", value=1024, step=64, min_value=512, max_value=1024)
                        steps_base = st.slider("Base Steps", 15, 50, 28)
                    with col_b:
                        width = st.number_input("Width", value=1024, step=64, min_value=512, max_value=1280)
                        steps_refiner = st.slider("Refiner Steps", 10, 40, 20)
                    guidance = st.slider("Guidance Scale", 1.0, 10.0, 5.5, 0.5)
                else:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        height = st.number_input("Height", value=DEFAULT_HEIGHT, step=64, min_value=512, max_value=1024)
                        steps = st.slider("Inference Steps", 20, 100, DEFAULT_STEPS)
                    with col_b:
                        width = st.number_input("Width", value=DEFAULT_WIDTH, step=64, min_value=512, max_value=1280)
                        guidance = st.slider("Guidance Scale", 1.0, 15.0, DEFAULT_GUIDANCE, 0.5)
                    steps_base = steps_refiner = None
                
                use_seed = st.checkbox("Use fixed seed")
                seed = st.number_input("Seed", value=1234, min_value=0) if use_seed else None
            
            # Generate button
            generate_btn = st.button(
                "🎨 Generate Image",
                type="primary",
                use_container_width=True,
                disabled=not selected_breed,
            )
        
        with col2:
            if generate_btn and selected_breed:
                progress = st.empty()
                progress.markdown(f"**Generating {selected_breed} in {selected_style} style...**")
                
                try:
                    with st.spinner("Loading model..."):
                        if use_realistic:
                            generator = load_realistic_generator()
                        else:
                            generator = load_stylized_generator()
                    
                    with st.spinner("Generating image..."):
                        if use_realistic:
                            image = generate_realistic(
                                generator=generator,
                                breed=selected_breed,
                                seed=seed,
                                steps_base=steps_base,
                                steps_refiner=steps_refiner,
                                scale=guidance,
                                width=width,
                                height=height,
                            )
                        else:
                            image = generator.generate(
                                breed=selected_breed,
                                style_name=selected_style,
                                height=height,
                                width=width,
                                num_inference_steps=steps,
                                guidance_scale=guidance,
                                seed=seed,
                            )
                    
                    progress.empty()
                    st.image(image, caption=f"{selected_breed} - {selected_style}", use_container_width=True)
                    
                    # Download button
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    filename = f"{selected_breed.lower().replace(' ', '_')}_{selected_style.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
                    st.download_button("💾 Download", buf.getvalue(), filename, "image/png")
                
                except Exception as e:
                    progress.empty()
                    st.error(f"Generation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            
            elif not generate_btn:
                st.markdown("""
                <div style="border: 2px dashed #ccc; border-radius: 10px; padding: 80px 20px; text-align: center; color: #888;">
                    <p style="font-size: 48px; margin: 0;">🖼️</p>
                    <p>Generated image will appear here</p>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
