"""
Reconstructed point cloud wrapper class.
"""

__all__ = (
    "Scene",
)

import tempfile
from typing import Optional

import torch


class Scene:
    """Starst3r scene. Contains Mast3r and 3DGS reconstructions, and helper methods.
    """

    raw_imgs: list[torch.Tensor]
    """GT images, unscaled. Each img is shape (H, W, 3)."""
    imgs: list[torch.Tensor]
    """GT images after scaling via Mast3r code."""

    dense_pts: list[torch.Tensor]
    """Dense point 3D coords from Mast3r reconstruction. From each camera view."""
    dense_cols: list[torch.Tensor]
    """Dense point colors from Mast3r reconstruction. From each camera view."""

    c2w: torch.Tensor
    """Camera-to-world matrix, shape (C, 4, 4)."""
    intrinsics: torch.Tensor
    """Camera intrinsic matrices, shape (C, 3, 3)."""

    cache_dir: str

    def __init__(self, imgs: Optional[list[torch.Tensor]] = None, cache_dir: Optional[str] = None):
        """Initialize a new scene.

        Parameters
        ----------

        imgs:
            Starting GT images. If None, then no images are added now.

        cache_dir:
            Tmp dir. If None, a new one is created via tempfile.
        """
        self.raw_imgs = []
        self.imgs = []
        self.dense_pts = []
        self.dense_cols = []
        self.c2w = None
        self.intrinsics = None

        self.cache_dir = cache_dir
        if cache_dir is None:
            self.cache_dir = tempfile.mkdtemp()

        if imgs is not None:
            self.add_images(imgs)

    @property
    def dense_pts_flat(self):
        """Dense points concatenated from all cameras."""
        return torch.cat(self.dense_pts, dim=0)

    @property
    def dense_cols_flat(self):
        """Dense colors concatenated from all cameras."""
        return torch.cat(self.dense_cols, dim=0)

    @property
    def w2c(self) -> torch.Tensor:
        """World-to-camera transformation matrix (inverse of ``c2w``)."""
        return torch.inverse(self.c2w)

    def add_images(self, imgs: list[torch.Tensor]):
        """Add GT images to the scene. Solve camera pose and update dense points with Mast3r.
        """
