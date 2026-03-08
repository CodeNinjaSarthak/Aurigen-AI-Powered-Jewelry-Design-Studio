# Aurigen — Architecture Overview

## System Summary

Aurigen is an AI-powered jewelry design studio built on Stable Diffusion XL (SDXL) with ControlNet guidance. A Streamlit web UI allows users to upload reference sketches and generate high-quality jewelry renders using fine-tuned diffusion weights.

---

## Pipeline Diagram

```
User Input (text prompt / sketch)
       ↓
Streamlit UI (app/controlnet_app.py)
       ↓
preprocess_image() → utils/image_utils.py
[Canny edge detection if sketch uploaded]
       ↓
load_pipeline() → models/sdxl_pipeline.py
[SDXL 1.0 + ControlNet Canny + fine-tuned UNet weights]
       ↓
Generated Jewelry Image
```

---

## Design Decisions

### Why ControlNet?

ControlNet extends diffusion models with spatial conditioning signals — in Aurigen's case, Canny edge maps extracted from user-uploaded sketches. This lets the model preserve the structural intent of a rough sketch (proportions, topology, silhouette) while still applying full diffusion-model texture and material quality. Without ControlNet, text-only generation ignores sketch geometry entirely; with it, users get structure-preserving generation that respects their design input.

### Why SDXL over SD 1.5?

SDXL 1.0 operates natively at 1024×1024, versus SD 1.5's 512×512. For jewelry — a category where fine surface detail, gem facets, and metalwork texture are critical — the higher base resolution produces substantially sharper outputs without post-processing upscaling. SDXL also uses a dual-encoder text architecture (CLIP ViT-L + OpenCLIP ViT-bigG) that yields stronger prompt adherence for material descriptors like "18k white gold" or "pavé diamond setting", which tend to be poorly encoded by SD 1.5's single encoder.

### Fine-Tuning Approach

The UNet component of SDXL was fine-tuned on the custom 6,157-image jewelry dataset using a DreamBooth/LoRA-style regime that targets domain adaptation: the goal is to shift the model's prior toward photorealistic jewelry renders without catastrophic forgetting of general prompt-following ability. Training was ControlNet-conditioned throughout, so the fine-tuned weights preserve spatial controllability. The resulting checkpoint (`unet_epoch_3.pth`) is loaded on top of the base SDXL pipeline at inference time, with the model falling back to base weights gracefully if the file is absent.

### Dataset Composition — Why Earrings Dominate

The dataset contains 6,157 images across four categories: Earrings (3,298), Necklaces (1,738), Bracelets (888), and Rings (233). Earrings are the most widely scraped jewelry category online — they are small, flat objects that photograph well against plain backgrounds, making them dominant in e-commerce imagery and therefore in any web-scraped dataset. The class imbalance is acknowledged; future training runs may benefit from oversampling Rings and Bracelets or using class-conditional loss weighting to improve generation quality uniformly across categories.

---

## Hardware Requirements

| Tier | Specification |
|------|--------------|
| Minimum | 8 GB VRAM GPU, Python 3.10+ |
| Tested | CUDA 11.8, NVIDIA RTX 3080 / RTX 4090 |
| CPU mode | Supported but very slow (~10 min/image) |

---

## Key Technologies

| Component | Library / Model |
|-----------|----------------|
| UI | Streamlit |
| Diffusion backbone | `stabilityai/stable-diffusion-xl-base-1.0` |
| ControlNet | `diffusers/controlnet-canny-sdxl-1.0` |
| Edge detection | OpenCV Canny |
| Fine-tuned weights | Custom UNet checkpoint (`unet_epoch_3.pth`, epoch 3) |
| Inference device | CUDA (falls back to CPU) |

---

## Data Flow

1. User provides a text prompt and optional sketch/reference image.
2. `preprocess_image()` applies Canny edge detection (if enabled) and resizes to 1024×1024.
3. The SDXL + ControlNet pipeline runs inference guided by both the text prompt and the edge map.
4. Generated images are displayed in the UI; users can select any for iterative refinement.
5. Refinement re-runs inference using the selected image as the new ControlNet conditioning input.

---

## Running the App

```bash
# From the project root:
streamlit run app/controlnet_app.py
```

Environment variables (see `.env.example`):
- `HUGGINGFACE_TOKEN` — required to download gated models from HuggingFace Hub
