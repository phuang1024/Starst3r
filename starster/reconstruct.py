"""
Functions for running the Mast3r inference pipeline.
"""

__all__ = (
    "reconstruct_scene",
)

import tempfile

import torch

from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import (
    convert_dust3r_pairs_naming,
    forward_mast3r,
    prepare_canonical_data,
    compute_min_spanning_tree,
    condense_data,
    sparse_scene_optimizer,
    SparseGA,
)

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
    scene = run_sparse_ga(
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


def run_sparse_ga(
        imgs,
        pairs_in,
        cache_path,
        model,
        subsample=8,
        desc_conf='desc_conf',
        device='cuda',
        dtype=torch.float32,
        shared_intrinsics=False,
        **kw
    ):
    """Copied and modified from sparse_ga.py:sparse_global_alignment
    """
    pairs_in = convert_dust3r_pairs_naming(imgs, pairs_in)

    pairs, cache_path = forward_mast3r(pairs_in, model,
                                       cache_path=cache_path, subsample=subsample,
                                       desc_conf=desc_conf, device=device)

    tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21 = \
        prepare_canonical_data(imgs, pairs, subsample, cache_path=cache_path, mode='avg-angle', device=device)

    mst = compute_min_spanning_tree(pairwise_scores)

    imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21 = \
        condense_data(imgs, tmp_pairs, canonical_views, preds_21, dtype)

    imgs, res_coarse, res_fine = sparse_scene_optimizer(
        imgs, subsample, imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21, canonical_paths, mst,
        shared_intrinsics=shared_intrinsics, cache_path=cache_path, device=device, dtype=dtype, **kw)

    return SparseGA(imgs, pairs_in, res_fine or res_coarse, anchors, canonical_paths)
