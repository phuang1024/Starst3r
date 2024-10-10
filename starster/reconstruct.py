"""
Functions for running the Mast3r inference pipeline.
"""

__all__ = (
    "reconstruct_scene",
)

import tempfile

from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment, extract_correspondences

from .image import prepare_images_for_mast3r


def reconstruct_scene(model, imgs, filelist, device, tmpdir=None):
    """Run Mast3r reconstruction pipeline.

    Mast3r model inference and global adjustment.

    Parameters
    ----------

    model:
        Model instance :class:`starster.Mast3rModel`.

    imgs:
        List of images from :func:`starster.load_image`.
        Tensor shape (C,H,W), dtype float32.

    filelist:
        List of image paths corresponding to each image.
        Due to Mast3r legacy code, this is required.

    device:
        Torch device to run on.

    tmpdir:
        Temp directory. If None, a new one is created tempfile.

    Returns
    -------

    Reconstructed scene as Mast3r SparseGA instance.
    """
    imgs = prepare_images_for_mast3r(imgs)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    scene = sparse_global_alignment(
        filelist,
        pairs,
        tmpdir,
        model,
        lr1=0.07,
        niter1=500,
        lr2=0.014,
        niter2=200,
        device=device,
        opt_depth=False,
        matching_conf_thr=5,
        shared_intrinsics=False,
    )

    return scene
