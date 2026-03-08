"""
models/sdxl_pipeline.py
Loads the SDXL 1.0 + ControlNet Canny pipeline with optional fine-tuned UNet weights.
Run the app from the project root: streamlit run app/controlnet_app.py
"""

import logging
from pathlib import Path

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    DPMSolverMultistepScheduler,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = ROOT / "fine-tuned-weights"
UNET_CHECKPOINT = WEIGHTS_DIR / "unet_epoch_3.pth"


def load_pipeline():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        # Faster scheduler — good quality at 20 steps instead of 50
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )

        # Key fix: offloads unused layers to CPU during generation
        # Reduces active VRAM from ~15GB → ~8GB
        pipe.enable_model_cpu_offload()

        # Saves ~2GB VRAM with no quality loss
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

        # Faster attention (xformers) if available
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers enabled")
        except Exception:
            pipe.enable_attention_slicing(1)
            logger.info("xformers not available, using attention slicing")

        # Load fine-tuned weights
        try:
            if UNET_CHECKPOINT.exists():
                pipe.unet.load_state_dict(
                    torch.load(UNET_CHECKPOINT, map_location=device, weights_only=True)
                )
                logger.info("Fine-tuned UNet weights loaded from %s", UNET_CHECKPOINT)
            else:
                logger.warning(
                    "UNet checkpoint not found at %s — using base SDXL weights.",
                    UNET_CHECKPOINT,
                )
        except Exception as exc:
            logger.warning("Failed to load UNet weights: %s — using base weights.", exc)

        return pipe
    except Exception as e:
        logger.error("load_pipeline FAILED: %s", e, exc_info=True)
        raise
