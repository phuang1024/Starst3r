"""
Gaussian splatting integration with gsplat.
"""

__all__ = (
    "GSTrainer",
)

import torch
import gsplat
from tqdm import trange

from .scene import PointCloud


class GSTrainer:
    def __init__(self, scene: PointCloud, device: str = "cuda"):
        self.scene = scene
        self.device = device

        self.gaussians = {}
        self.optimizers = {}
        self.strategy = None
        self.strategy_state = None

    def init_gaussians(self, init_scale=0.003):
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

    def render_views(self, w2c: torch.Tensor, intrinsics: torch.Tensor, width: int, height: int):
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

    def render_scene_views(self, width: int, height: int):
        """
        Render from camera views of original scene (i.e. ``self.scene``).
        """
        return self.render_views(self.scene.w2c(), self.scene.intrinsics(), width, height)

    def init_optimizer(self):
        self.optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in self.gaussians.items()}

        self.strategy = gsplat.DefaultStrategy()
        self.strategy.check_sanity(self.gaussians, self.optimizers)

        self.strategy_state = self.strategy.initialize_state()

    def run_optimization(self, iters: int):
        height, width = self.scene.imgs[0].shape[:2]

        for step in trange(iters):
            render_img, render_alpha, info = self.render_scene_views(width, height)
            #self.strategy.step_pre_backward(self.gaussians, self.optimizers, self.strategy_state, step, info)

            loss = 0
            for i in range(self.scene.num_cams):
                img = torch.tensor(self.scene.imgs[i], device=self.device)
                loss += torch.nn.functional.mse_loss(render_img[i], img)
            loss.backward()
            print(loss)

            for optim in self.optimizers.values():
                optim.step()
                optim.zero_grad(set_to_none=True)

            #self.strategy.step_post_backward(self.gaussians, self.optimizers, self.strategy_state, step, info)
