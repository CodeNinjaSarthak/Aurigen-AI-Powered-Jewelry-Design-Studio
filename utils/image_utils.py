"""
utils/image_utils.py
Image preprocessing utilities for the Aurigen ControlNet pipeline.
"""
import cv2
import numpy as np
import streamlit as st
from PIL import Image


def detect_edges(image: Image.Image) -> Image.Image:
    """Apply Canny edge detection to a PIL Image.

    Args:
        image: RGB PIL Image.

    Returns:
        PIL Image with 3-channel Canny edge map (same spatial size as input).
    """
    image_np = np.array(image)
    canny = cv2.Canny(image_np, 100, 200)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    return Image.fromarray(canny)


@st.cache_data
def preprocess_image(image_input, apply_edges: bool = True) -> Image.Image:
    """Load, optionally edge-detect, and resize an image for ControlNet input.

    Accepts None (white canvas), a file-like object (from st.file_uploader),
    or an existing PIL Image (from refinement loop).

    Args:
        image_input: None | file-like object | PIL.Image.Image
        apply_edges: If True, apply Canny edge detection before resizing.

    Returns:
        PIL Image resized to 1024×1024.
    """
    if image_input is None:
        image = Image.new("RGB", (1024, 1024), (255, 255, 255))
    elif hasattr(image_input, "read"):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        # Unknown input type — fall back to white canvas (same as None case)
        image = Image.new("RGB", (1024, 1024), (255, 255, 255))

    image = image.resize((1024, 1024))

    if apply_edges:
        image = detect_edges(image)

    return image
