"""
Reconstructed point cloud wrapper class.
"""

__all__ = (
    "Scene",
)

import tempfile
from typing import Any, Optional

import gsplat
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from tqdm import trange

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

        if self.c2w is None:
            self.c2w = scene.cam2w
            self.intrinsics = scene.intrinsics
        """
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

    def init_3dgs(self, init_scale=1e-3, lr=1e-3):
        """Initialize 3DGS splats and optims from Mast3r dense points.
        """

        pts = self.dense_pts_flat
        colors = self.dense_cols_flat
        self.gaussians = {
            "means": pts,
            "scales": torch.full_like(pts, init_scale),
            "quats": torch.zeros(pts.shape[0], 4),
            "opacities": torch.ones(pts.shape[0]),
            "sh0": torch.zeros(pts.shape[0], 1, 3),
            "shN": torch.zeros(pts.shape[0], 24, 3),
        }
        self.gaussians["quats"][:, 0] = 1.0
        self.gaussians["sh0"][:, 0] = 1 - colors
        for i in range(24):
            self.gaussians["shN"][:, i] = 1 - colors

        for k, v in self.gaussians.items():
            self.gaussians[k] = torch.nn.Parameter(v.to(self.device))

        # Create optimizers
        self.optimizers = {k: torch.optim.Adam([v], lr=lr) for k, v in self.gaussians.items()}

        self.ssim = SSIM(data_range=1).to(self.device)

        # Create pruning strategy
        # Using MC strategy bc DefaultStrategy has a bug.
        self.strategy = gsplat.MCMCStrategy()
        self.strategy.check_sanity(self.gaussians, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

    def render_3dgs(self, w2c: torch.Tensor, intrinsics: torch.Tensor, width: int, height: int):
        """Render the splats from a set of camera views.

        TODO currently width and height can only be the original image size.

        Parameters
        ----------

        w2c:
            World-to-camera matrices. Shape (N, 4, 4).

        intrinsics:
            Camera intrinsics. Shape (N, 3, 3).

        width:
            Image width.

        height:
            Image height.

        Returns
        -------

        Output of ``gsplat.rasterization``.

        Tuple of ``(render_img, render_alpha, info)``.

        - render_img: Color image. Shape (N, H, W, 3).
        """
        render = gsplat.rasterization(
            means=self.gaussians["means"],
            quats=self.gaussians["quats"],
            scales=self.gaussians["scales"],
            opacities=self.gaussians["opacities"],
            colors=self.gaussians["shN"],
            viewmats=w2c,
            Ks=intrinsics,
            width=width,
            height=height,
            sh_degree=1,
        )
        return render

    def render_3dgs_original(self, width: int, height: int):
        """Render from camera views of original scene (``self.scene``).

        See ``render_views``.
        """
        return self.render_3dgs(self.w2c, self.intrinsics, width, height)

    def run_3dgs_optim(
            self,
            iters: int,
            enable_pruning: bool = False,
            loss_ssim_fac=0.2,
            loss_opacity_fac=0.01,
            loss_scale_fac=0.01,
            verbose: bool = False,
        ) -> list[float]:
        """Run 3DGS optimization and pruning (optional) for a number of iterations.

        Parameters
        ----------

        iters:
            Number of iterations.

        enable_pruning:
            Enable pruning and densification via the gsplat pruning strategy.

        verbose:
            Enable tqdm progress bar.

        Returns
        -------

        List of losses at each iteration.
        """

        def compute_loss(truth_img, render_img, render_alpha):
            l1 = torch.nn.functional.l1_loss(truth_img, render_img)

            ssim = 1 - self.ssim(truth_img.permute(2, 0, 1).unsqueeze(0), render_img.permute(2, 0, 1).unsqueeze(0))
            loss = l1 * (1 - loss_ssim_fac) + ssim * loss_ssim_fac

            loss += loss_opacity_fac * torch.abs(torch.sigmoid(self.gaussians["opacities"])).mean()

            loss += loss_scale_fac * torch.abs(torch.exp(self.gaussians["scales"])).mean()

            return loss

        height, width = self.imgs[0].shape[:2]

        losses = []

        pbar = trange(iters, disable=not verbose)
        for step in pbar:
            render_img, render_alpha, info = self.render_3dgs_original(width, height)

            if enable_pruning:
                self.strategy.step_pre_backward(self.gaussians, self.optimizers, self.strategy_state, step, info)

            loss = 0
            for i in range(len(self.imgs)):
                img = torch.tensor(self.imgs[i], device=self.device)
                loss += compute_loss(img, render_img[i], render_alpha[i])
            loss.backward()

            desc = f"Gsplat optimization: loss={loss.item()}"
            pbar.set_description(desc)
            losses.append(loss.item())

            for optim in self.optimizers.values():
                optim.step()
                optim.zero_grad(set_to_none=True)

            if enable_pruning:
                self.strategy.step_post_backward(self.gaussians, self.optimizers, self.strategy_state, step, info, 1e-3)

        return losses
