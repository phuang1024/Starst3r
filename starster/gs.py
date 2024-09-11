"""
3D Gaussian Splatting scene refinement.
"""

__all__ = (
    "GSTrainer",
)

import torch
import gsplat
from tqdm import trange

from .scene import PointCloudScene


class GSTrainer:
    """3DGS training and refinement.

    Uses gsplat for rendering and optimization.

    Uses the Mast3r reconstruction as a starting point.
    """

    def __init__(self, scene: PointCloudScene, init_scale=3e-3, lr=1e-3, device: str = "cuda"):
        """Initialize the splats and optimizers from the Mast3r reconstruction.

        Parameters
        ----------

        scene:
            Reconstructed scene from Mast3r.
            Return value of :func:`starster.reconstruct_scene`.

        init_scale:
            Initial scale of the splats.

        lr:
            Learning rate for Adam optimizers.

        device:
            Device to use.
        """
        self.scene = scene
        self.device = device

        # Create splats
        pts, colors = self.scene.pts_dense_flat()
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

        # Create pruning strategy
        # Using MC strategy bc DefaultStrategy has a bug.
        self.strategy = gsplat.MCMCStrategy()
        self.strategy.check_sanity(self.gaussians, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

    def render_views(self, w2c: torch.Tensor, intrinsics: torch.Tensor, width: int, height: int):
        """Render the splats from a set of camera views.

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

    def render_views_original(self, width: int, height: int):
        """Render from camera views of original scene (``self.scene``).

        See ``render_views``.
        """
        return self.render_views(self.scene.w2c(), self.scene.intrinsics(), width, height)

    def run_optimization(self, iters: int, enable_pruning: bool = False, verbose: bool = False) -> list[float]:
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
        height, width = self.scene.imgs[0].shape[:2]

        losses = []

        pbar = trange(iters, disable=not verbose)
        for step in pbar:
            render_img, render_alpha, info = self.render_views_original(width, height)

            if enable_pruning:
                self.strategy.step_pre_backward(self.gaussians, self.optimizers, self.strategy_state, step, info)

            loss = 0
            for i in range(self.scene.num_cams):
                img = torch.tensor(self.scene.imgs[i], device=self.device)
                loss += torch.nn.functional.mse_loss(render_img[i], img)
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
