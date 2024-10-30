__all__ = (
    "init_3dgs",
    "render_3dgs",
    "render_3dgs_original",
    "run_3dgs_optim",
)

import gsplat
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from tqdm import trange


def init_3dgs(scene, init_scale=3e-3, lr=1e-3):
    """Initialize 3DGS splats and optims from Mast3r dense points.
    """

    pts = scene.dense_pts_flat
    colors = scene.dense_cols_flat
    scene.gaussians = {
        "means": pts,
        "scales": torch.full_like(pts, init_scale),
        "quats": torch.zeros(pts.shape[0], 4),
        "opacities": torch.ones(pts.shape[0]),
        "sh0": torch.zeros(pts.shape[0], 1, 3),
        "shN": torch.zeros(pts.shape[0], 24, 3),
    }
    scene.gaussians["quats"][:, 0] = 1.0
    scene.gaussians["sh0"][:, 0] = 1 - colors
    for i in range(24):
        scene.gaussians["shN"][:, i] = 1 - colors

    for k, v in scene.gaussians.items():
        scene.gaussians[k] = torch.nn.Parameter(v.to(scene.device))

    # Create optimizers
    scene.optimizers = {k: torch.optim.Adam([v], lr=lr) for k, v in scene.gaussians.items()}

    scene.ssim = SSIM(data_range=1).to(scene.device)

    # Create pruning strategy
    # Using MC strategy bc DefaultStrategy has a bug.
    scene.strategy = gsplat.MCMCStrategy()
    scene.strategy.check_sanity(scene.gaussians, scene.optimizers)
    scene.strategy_state = scene.strategy.initialize_state()

def render_3dgs(scene, w2c: torch.Tensor, intrinsics: torch.Tensor, width: int, height: int):
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
        means=scene.gaussians["means"],
        quats=scene.gaussians["quats"],
        scales=scene.gaussians["scales"],
        opacities=scene.gaussians["opacities"],
        colors=scene.gaussians["shN"],
        viewmats=w2c,
        Ks=intrinsics,
        width=width,
        height=height,
        sh_degree=1,
    )
    return render

def render_3dgs_original(scene, width: int, height: int):
    """Render from camera views of original scene (``scene.scene``).

    See ``render_views``.
    """
    return scene.render_3dgs(scene.w2c, scene.intrinsics, width, height)

def run_3dgs_optim(
        scene,
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

        ssim = 1 - scene.ssim(truth_img.permute(2, 0, 1).unsqueeze(0), render_img.permute(2, 0, 1).unsqueeze(0))
        loss = l1 * (1 - loss_ssim_fac) + ssim * loss_ssim_fac

        loss += loss_opacity_fac * torch.abs(torch.sigmoid(scene.gaussians["opacities"])).mean()

        loss += loss_scale_fac * torch.abs(torch.exp(scene.gaussians["scales"])).mean()

        return loss

    height, width = scene.imgs[0].shape[:2]

    losses = []

    pbar = trange(iters, disable=not verbose)
    for step in pbar:
        render_img, render_alpha, info = scene.render_3dgs_original(width, height)

        if enable_pruning:
            scene.strategy.step_pre_backward(scene.gaussians, scene.optimizers, scene.strategy_state, step, info)

        loss = 0
        for i in range(len(scene.imgs)):
            img = torch.tensor(scene.imgs[i], device=scene.device)
            loss += compute_loss(img, render_img[i], render_alpha[i])
        loss.backward()

        desc = f"Gsplat optimization: loss={loss.item()}"
        pbar.set_description(desc)
        losses.append(loss.item())

        for optim in scene.optimizers.values():
            optim.step()
            optim.zero_grad(set_to_none=True)

        if enable_pruning:
            scene.strategy.step_post_backward(scene.gaussians, scene.optimizers, scene.strategy_state, step, info, 1e-3)

    return losses
