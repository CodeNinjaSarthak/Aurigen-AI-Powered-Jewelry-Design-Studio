import logging
import time
import warnings

import streamlit as st
import torch  # kept: used in sidebar for pipe.dtype and torch.cuda.is_available()
from models import load_pipeline
from utils import preprocess_image

warnings.filterwarnings("ignore")

#######################
# Page configuration
st.set_page_config(
    page_title="Aurigen - Jewelry Design Studio",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup logging
logging.basicConfig(level=logging.INFO)

#######################
# CSS Styling — Art Deco "Cartier Atelier" theme
# Lines 26–188 (single injection, no logic changed)
st.markdown(
    """
    <style>
    /* ── Fonts ─────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400;1,600&family=Jost:wght@200;300;400;500&display=swap');

    /* ── Design Tokens ──────────────────────────────────────────────────── */
    :root {
        --gold:          #C9A84C;
        --gold-light:    #E8C97A;
        --gold-dim:      rgba(201, 168, 76, 0.40);
        --obsidian:      #0A0A0A;
        --surface:       #111111;
        --surface2:      #181818;
        --border:        rgba(201, 168, 76, 0.18);
        --border-bright: rgba(201, 168, 76, 0.45);
        --text:          #F0EDE8;
        --text-dim:      #8A8680;
    }

    /* ── Global ─────────────────────────────────────────────────────────── */
    body, .stApp {
        background: var(--obsidian) !important;
        color: var(--text) !important;
        font-family: 'Jost', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--obsidian) !important;
    }

    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="block-container"] {
        background: var(--obsidian) !important;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border-bright) !important;
        position: relative;
    }

    /* Thin gold gradient line on right edge */
    [data-testid="stSidebar"]::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 1px;
        height: 100%;
        background: linear-gradient(to bottom,
            transparent 0%,
            var(--gold) 40%,
            var(--gold) 60%,
            transparent 100%);
        pointer-events: none;
    }

    /* ── Brand Mark ─────────────────────────────────────────────────────── */
    .brand {
        text-align: center;
        padding: 28px 0 32px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 20px;
    }
    .brand-icon {
        color: var(--gold);
        font-size: 28px;
        line-height: 1;
        margin-bottom: 8px;
    }
    .brand-name {
        font-family: 'Cormorant Garamond', serif;
        font-weight: 600;
        font-size: 22px;
        letter-spacing: 0.35em;
        color: var(--text);
        margin-bottom: 4px;
    }
    .brand-sub {
        font-family: 'Jost', sans-serif;
        font-weight: 200;
        font-size: 10px;
        letter-spacing: 0.4em;
        color: var(--gold);
        text-transform: uppercase;
    }

    /* ── Sidebar Labels ──────────────────────────────────────────────────── */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stSlider   label,
    [data-testid="stSidebar"] .stCheckbox label {
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        font-size: 10px !important;
        color: var(--text-dim) !important;
    }

    /* ── Sidebar section header bold text ────────────────────────────────── */
    [data-testid="stSidebar"] strong {
        font-family: 'Jost', sans-serif;
        font-weight: 300;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-size: 10px;
        color: var(--text-dim);
    }

    /* ── Text Areas ─────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] .stTextArea textarea {
        background: #161616 !important;
        border: 1px solid var(--border) !important;
        border-radius: 2px !important;
        color: var(--text) !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
    }
    [data-testid="stSidebar"] .stTextArea textarea:focus {
        border-color: var(--gold) !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* ── Sliders ────────────────────────────────────────────────────────── */
    /* Thumb */
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: var(--gold) !important;
        border: none !important;
        box-shadow: 0 0 6px var(--gold-dim) !important;
    }
    /* Active / filled track */
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderTrackFill"],
    [data-testid="stSidebar"] .stSlider [class*="TrackFill"] {
        background: var(--gold) !important;
    }

    /* ── Checkbox ────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stCheckbox"] input:checked + span,
    [data-testid="stSidebar"] [data-testid="stCheckbox"] [aria-checked="true"] {
        background: var(--gold) !important;
        border-color: var(--gold) !important;
    }

    /* ── Section Dividers ────────────────────────────────────────────────── */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 20px 0 !important;
    }

    /* ── Main Heading ────────────────────────────────────────────────────── */
    h1, h2 {
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 300 !important;
        font-size: 48px !important;
        letter-spacing: 0.05em !important;
        color: var(--text) !important;
        border-bottom: 1px solid var(--border-bright) !important;
        padding-bottom: 16px !important;
        margin-bottom: 4px !important;
    }

    /* Italic subtitle injected via CSS */
    h2::after {
        content: 'by Aurigen Studio';
        display: block;
        font-family: 'Cormorant Garamond', serif;
        font-style: italic;
        font-weight: 300;
        font-size: 16px;
        color: var(--gold);
        letter-spacing: 0.05em;
        border: none;
        margin-top: 6px;
    }

    /* ── Generate Button (primary kind) ─────────────────────────────────── */
    [data-testid="stButton"] > button[kind="primary"],
    [data-testid="stBaseButton-primary"] {
        background: transparent !important;
        border: 1px solid var(--gold) !important;
        color: var(--gold) !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        letter-spacing: 0.2em !important;
        text-transform: uppercase !important;
        font-size: 11px !important;
        padding: 12px 36px !important;
        border-radius: 0 !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background: var(--gold) !important;
        color: var(--obsidian) !important;
    }

    /* ── Image Containers (thin gold frame) ──────────────────────────────── */
    [data-testid="stImage"] {
        border: 1px solid var(--border) !important;
        background: var(--surface2) !important;
        padding: 3px !important;
    }

    /* ── Image Captions ──────────────────────────────────────────────────── */
    [data-testid="stImage"] figcaption,
    [data-testid="stImageCaption"],
    .stImage caption {
        font-family: 'Jost', sans-serif !important;
        font-weight: 200 !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        font-size: 10px !important;
        color: var(--text-dim) !important;
        text-align: center !important;
    }

    /* ── Expanders (Studio Guide + Enhance) ──────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 2px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] + span {
        font-family: 'Cormorant Garamond', serif !important;
        font-style: italic !important;
        font-size: 18px !important;
        color: var(--text) !important;
    }

    /* ── System Status Card ───────────────────────────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-family: 'Jost', sans-serif;
        font-weight: 300;
        font-size: 11px;
        color: var(--text);
        letter-spacing: 0.05em;
        margin: 4px 0;
    }

    /* ── Scrollbar ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar {
        width: 4px;
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: var(--gold-dim);
        border-radius: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#######################
# Model Setup
@st.cache_resource
def get_pipeline():
    return load_pipeline()

pipe = get_pipeline()

#######################
# Sidebar Controls
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-icon">◆</div>
          <div class="brand-name">AURIGEN</div>
          <div class="brand-sub">Design Studio</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Generation Parameters
    prompt = st.text_area(
        "Design Description",
        "A luxurious diamond necklace with intricate gold filigree, studio lighting, 8k resolution",
    )
    negative_prompt = st.text_area(
        "Exclusions",
        "blurry, low quality, plastic, unrealistic proportions, poor lighting",
    )

    # ControlNet Settings
    st.markdown("---")
    st.markdown("**Design Guidance**")
    reference_image = st.file_uploader(
        "Upload Sketch/Reference", type=["png", "jpg", "jpeg"]
    )
    conditioning_scale = st.slider("Guidance Strength", 0.0, 2.0, 1.2)
    apply_edges = st.checkbox("Auto-detect Edges", True)

    # Generation Settings
    st.markdown("---")
    num_images = st.slider("Number of Designs", 1, 4, 2)
    num_inference_steps = st.slider("Refinement Steps", 20, 100, 45)
    guidance_scale = st.slider("Prompt Adherence", 1.0, 20.0, 7.5)

#######################
# Main Interface
col = st.columns((1, 6, 1))
with col[1]:
    st.markdown("## Craft Your Jewelry Masterpiece")

    # Generation Controls
    col1, col2 = st.columns([1, 3])
    with col1:
        generate_btn = st.button("✨ Generate Designs", type="primary")
    with col2:
        if "generated_images" in st.session_state:
            selected_refinement = st.selectbox(
                "Select Design to Enhance",
                options=list(range(len(st.session_state.generated_images))),
                format_func=lambda x: f"Design {x+1}",
            )

    # Image Display and Refinement
    if "generated_images" in st.session_state:
        st.markdown("---")
        cols = st.columns(2)
        for idx, img in enumerate(st.session_state.generated_images):
            with cols[idx % 2]:
                st.image(img, use_container_width=True, caption=f"Design {idx+1}")

                # Refinement Options
                with st.expander(f"Enhance Design {idx+1}"):
                    refine_prompt = st.text_input(
                        "Modification Request",
                        key=f"refine_{idx}",
                        placeholder="Make the gems more emerald-colored",
                    )
                    if st.button("Apply Changes", key=f"btn_refine_{idx}"):
                        with st.spinner("Refining design..."):
                            try:
                                # Process existing image for ControlNet
                                control_image = preprocess_image(img, apply_edges)

                                # Run refinement
                                refined = pipe(
                                    prompt=refine_prompt or prompt,
                                    negative_prompt=negative_prompt,
                                    image=control_image,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    controlnet_conditioning_scale=conditioning_scale,
                                ).images[0]

                                # Update the design
                                st.session_state.generated_images[idx] = refined
                                st.rerun()
                            except Exception as e:
                                st.error(f"Refinement failed: {str(e)}")

    # Initial Generation
    if generate_btn:
        start_time = time.time()
        with st.spinner("Crafting your designs..."):
            try:
                control_image = preprocess_image(reference_image, apply_edges)

                images = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=control_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images,
                    controlnet_conditioning_scale=conditioning_scale,
                ).images

                st.session_state.generated_images = images
                st.success(
                    f"Created {len(images)} designs in {time.time()-start_time:.1f}s"
                )
                st.rerun()

            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

    # Instructions Section
    with st.expander("🛠️ Studio Guide", expanded=True):
        st.markdown(
            """
        **Design Studio Features:**
        - 🖼️ **Reference Upload**: Add sketches or inspiration images (defaults to white background if none uploaded)
        - 🎛️ **Precision Control**: Adjust guidance strength and edge detection
        - ✨ **Iterative Refinement**: Modify specific designs after generation
        - 💎 **Material Focus**: Use terms like "24k gold", "flawless diamonds", "vintage engraving"
        
        **Pro Tips:**
        1. Start with broad concepts, then refine details
        2. Use "enhance" for color/material adjustments
        3. Combine reference images with textual descriptions
        4. Experiment with guidance strength (0.8-1.5 recommended)
        """
        )

#######################
# System Information
st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")
st.sidebar.write(f"Device: {'GPU 🔥' if torch.cuda.is_available() else 'CPU ⚙️'}")
st.sidebar.write(f"Model: SDXL 1.0 + ControlNet")
st.sidebar.write(f"Precision: {pipe.dtype}")