__all__ = (
    "make_pair_indices",
    "process_image",
    "resize_image",
    "load_image",
    "prepare_images_for_mast3r",
)

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from PIL.ImageOps import exif_transpose

_to_tensor = T.ToTensor()
_normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


def make_pair_indices(n: int, symmetric: bool = True) -> list[tuple[int, int]]:
    """
    Generate all pairs of indices for `n` elements.

    Fully connected graph; i.e. (i, j) exists for i, j < n and:
    Symmetric: i != j;
    Not symmetric: i < j.
    """
    pairs = []
    for i in range(n):
        for j in range(i):
            pairs.append((i, j))
    if symmetric:
        for i, j in pairs:
            pairs.append((j, i))
    return pairs


def process_image(img: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Process image to model requirements.

    Crops (around center) HW to a multiple of 8.

    img: Shape (C,H,W), dtype uint8
    return: Tensor, shape (C,H,W), dtype float32
    """
    cx = img.shape[2] // 2
    cy = img.shape[1] // 2

    # Compute width and height as mult of 8
    width_half = (cx // 8) * 8
    height_half = (cy // 8) * 8

    # Crop around center
    img = img[..., cy - height_half : cy + height_half, cx - width_half : cx + width_half]

    # Torchvision transforms
    img = _normalize(img)

    return img


def resize_image(img: Image.Image, size: int) -> Image.Image:
    """
    Resize longest edge of image to `size`.
    """
    if max(img.size) > size:
        interp = Image.LANCZOS
    else:
        interp = Image.BICUBIC
    new_size = [int(x * size / max(img.size)) for x in img.size]
    return img.resize(new_size, interp)


def load_image(path: str | Path, size: int = 512) -> torch.Tensor:
    img = Image.open(path)
    img = resize_image(img, size)
    img = exif_transpose(img)
    img = img.convert("RGB")
    img = np.array(img)
    img = _to_tensor(img)

    img = process_image(img)
    return img


def prepare_images_for_mast3r(imgs: list[torch.Tensor]):
    """
    Returns image format as supported by mast3r legacy code.

    Each image is
    {
        img: Tensor image, shape (1, 3, H, W)
        true_shape: [[H, W]]
        idx: int
        instance: str(idx)
    }

    TODO legacy code should be replaced and this function should be removed.
    """
    ret = []
    for i in range(len(imgs)):
        img = imgs[i]
        ret.append(
            dict(
                img=img[None],
                true_shape=np.int32([img.shape[-2:]]),
                idx=i,
                instance=str(i),
            )
        )

    return ret
