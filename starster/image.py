__all__ = (
    "make_pair_indices",
    "process_image",
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
        for i in range(len(pairs)):
            pairs.append((pairs[i][1], pairs[i][0]))
    return pairs


def process_image(img: np.ndarray | torch.Tensor, size: int) -> torch.Tensor:
    """
    Process image to model requirements.

    Resize longest edge of image to `size`.
    Crops (around center) HW to a multiple of 8.

    img: Shape (C,H,W), dtype uint8
    return: Tensor, shape (C,H,W), dtype float32
    """
    new_size = [int(x * size / max(img.shape[1:])) for x in img.shape[1:]]
    img = T.functional.resize(img, new_size, T.InterpolationMode.BICUBIC)

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


def load_image(path: str | Path, size: int = 512) -> torch.Tensor:
    img = Image.open(path)
    img = exif_transpose(img)
    img = img.convert("RGB")
    img = _to_tensor(np.array(img))
    img = process_image(img, size)
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
