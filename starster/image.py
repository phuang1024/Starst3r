__all__ = (
    "process_image",
    "load_image",
)

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from PIL.ImageOps import exif_transpose

_to_tensor = T.ToTensor()
_normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


def process_image(img: np.ndarray) -> torch.Tensor:
    """
    Process image to model requirements.

    Crops (around center) HW to a multiple of 8.

    img: Shape (H,W,C), dtype uint8
    """
    cx = img.shape[1] // 2
    cy = img.shape[0] // 2

    # Compute width and height as mult of 8
    width_half = (cx // 8) * 8
    height_half = (cy // 8) * 8

    # Crop around center
    img = img[cy - height_half : cy + height_half, cx - width_half : cx + width_half]

    # Torchvision transforms
    img = _to_tensor(img)
    img = _normalize(img)

    return img


def load_image(path: str | Path) -> torch.Tensor:
    img = Image.open(path)
    img = exif_transpose(img)
    img = img.convert("RGB")
    img = np.array(img)

    img = process_image(img)
    return img
