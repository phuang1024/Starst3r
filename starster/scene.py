"""
Reconstructed point cloud wrapper class.
"""

__all__ = (
    "Scene",
)

import tempfile
from typing import Optional

import torch

from .reconstruct import reconstruct_scene


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

    def add_images(
            self,
            model,
            imgs: list[torch.Tensor],
            device,
            conf_thres=1.5,
        ):
        """Add GT images to the scene. Solve camera pose and update dense points with Mast3r.

        Parameters
        ----------

        model:
            Model instance. :class:`starst3r.Mast3rModel`.

        imgs:
            New GT images to add. Each img is shape (H, W, 3).

        device:
            Torch device to use.

        conf_thres:
            Confidence threshold for Mast3r dense points.
        """
        # Generate fake filelist.
        filelist = [f"{i + len(self.imgs)}.png" for i in range(len(imgs))]

        scene = reconstruct_scene(model, imgs, filelist, device, tmpdir=self.cache_dir)

        self.raw_imgs.extend(imgs)
        self.imgs.extend(scene.imgs)

        if self.c2w is None:
            self.c2w = scene.cam2w
            self.intrinsics = scene.intrinsics
        else:
            self.c2w = torch.stack(self.c2w, scene.cam2w)
            self.intrinsics = torch.stack(self.intrinsics, scene.intrinsics)

        pts, _, confs = scene.get_dense_pts3d(clean_depth=True)
        for i in range(len(imgs)):
            mask = (confs[i] > conf_thres).reshape(-1).cpu()
            colors = torch.tensor(scene.imgs[i]).reshape(-1, 3)
            self.dense_pts.append(pts[i][mask])
            self.dense_cols.append(colors[mask])
