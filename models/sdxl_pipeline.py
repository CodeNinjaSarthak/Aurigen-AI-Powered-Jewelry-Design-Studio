"""
models/sdxl_pipeline.py
Loads the SDXL 1.0 + ControlNet Canny pipeline with optional fine-tuned UNet weights.
Run the app from the project root: streamlit run app/controlnet_app.py
"""
import logging
from pathlib import Path

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent          # project root
WEIGHTS_DIR = ROOT / "fine-tuned-weights"
UNET_CHECKPOINT = WEIGHTS_DIR / "unet_epoch_3.pth"


def load_pipeline():
    """Load SDXL + ControlNet pipeline with fine-tuned UNet weights.

    Returns:
        StableDiffusionXLControlNetPipeline ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to(device)

    try:
        if UNET_CHECKPOINT.exists():
            pipe.unet.load_state_dict(
                torch.load(UNET_CHECKPOINT, map_location=device, weights_only=True)
            )
            logger.info("Fine-tuned UNet weights loaded from %s", UNET_CHECKPOINT)
        else:
            logger.warning(
                "UNet checkpoint not found at %s — using base SDXL weights.", UNET_CHECKPOINT
            )
    except Exception as exc:
        logger.warning("Failed to load UNet weights: %s — using base weights.", exc)

    return pipe
