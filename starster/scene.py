"""
Reconstructed point cloud wrapper class.
"""

__all__ = (
    "Scene",
)

import tempfile
from typing import Any, Optional

import torch

from .gs import *
from .reconstruct import reconstruct_scene


class Scene:
    """Starst3r scene. Contains Mast3r and 3DGS reconstructions, and helper methods.
    """

    device: str
    cache_dir: str

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

    optim_params: dict[str, Any]
    """SLAM parameters passed to Mast3r scene optimization."""

    gs_params: dict[str, torch.Tensor]
    gs_optims: dict[str, Any]
    gs_strategy: Any
    gs_state: Any

    def __init__(self, cache_dir: Optional[str] = None, device="cuda"):
        """Initialize a new scene.

        Parameters
        ----------

        cache_dir:
            Tmp dir. If None, a new one is created via tempfile.

        device:
            Torch device to use.
        """
        self.device = device
        self.cache_dir = cache_dir
        if cache_dir is None:
            self.cache_dir = tempfile.mkdtemp()

        self.raw_imgs = []
        self.imgs = []

        self.dense_pts = []
        self.dense_cols = []
        self.c2w = None
        self.intrinsics = None

        self.optim_params = None

        self.gs_params = None
        self.gs_optims = None
        self.gs_strategy = None
        self.gs_state = None

    @property
    def dense_pts_flat(self):
        """Dense points concatenated from all cameras."""
        assert self.dense_pts, "No dense points available."
        return torch.cat(self.dense_pts, dim=0)

    @property
    def dense_cols_flat(self):
        """Dense colors concatenated from all cameras."""
        assert self.dense_cols, "No dense colors available."
        return torch.cat(self.dense_cols, dim=0)

    @property
    def w2c(self) -> torch.Tensor:
        """World-to-camera transformation matrix (inverse of ``c2w``)."""
        assert self.c2w is not None, "No c2w matrix available."
        return torch.inverse(self.c2w)

    def add_images(
            self,
            model,
            imgs: list[torch.Tensor],
            conf_thres=1.5,
        ):
        """Add GT images to the scene. Solve camera pose and update dense points with Mast3r.

        Parameters
        ----------

        model:
            Model instance. :class:`starst3r.Mast3rModel`.

        imgs:
            New GT images to add. Each img is shape (H, W, 3).

        conf_thres:
            Confidence threshold for Mast3r dense points.
        """
        self.raw_imgs.extend(imgs)

        # Generate fake filelist.
        filelist = [f"{i}.png" for i in range(len(self.raw_imgs))]

        scene, optim_params = reconstruct_scene(
            model,
            self.raw_imgs,
            filelist,
            self.device,
            optim_params=self.optim_params,
            tmpdir=self.cache_dir,
        )
        self.optim_params = optim_params

        curr_len = len(self.imgs)
        self.imgs.extend(scene.imgs[curr_len:])

        # TODO: New images should not completely replace previous points.
        # A coordinate system shift is needed as new Mast3r instance could be different.

        self.c2w = scene.cam2w
        self.intrinsics = scene.intrinsics
        """
        if self.c2w is None:
            pass
        else:
            self.c2w = torch.cat((self.c2w, scene.cam2w[curr_len:]), dim=0)
            self.intrinsics = torch.cat((self.intrinsics, scene.intrinsics[curr_len:]), dim=0)
        """

        pts, _, confs = scene.get_dense_pts3d(clean_depth=True)
        self.dense_pts = []
        self.dense_cols = []
        for i in range(len(scene.imgs)):
            mask = (confs[i] > conf_thres).reshape(-1).cpu()
            colors = torch.tensor(scene.imgs[i]).reshape(-1, 3)
            self.dense_pts.append(pts[i][mask])
            self.dense_cols.append(colors[mask])

    def init_3dgs(self, init_scale=3e-3, lr=1e-3):
        init_3dgs(self, init_scale, lr)

    def render_3dgs(self, w2c, intrinsics, width, height):
        return render_3dgs(self, w2c, intrinsics, width, height)

    def render_3dgs_original(self, width, height):
        return render_3dgs_original(self, width, height)

    def run_3dgs_optim(
            self,
            iters: int,
            enable_pruning: bool = False,
            loss_ssim_fac=0.2,
            loss_opacity_fac=0.01,
            loss_scale_fac=0.01,
            verbose: bool = False,
        ) -> list[float]:
        return run_3dgs_optim(
            self,
            iters,
            enable_pruning,
            loss_ssim_fac,
            loss_opacity_fac,
            loss_scale_fac,
            verbose,
        )
