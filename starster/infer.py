"""
Functions for running the inference pipeline.
"""

__all__ = (
    #"symmetric_inference",
    #"pairs_inference",
    "reconstruct_scene",
)

import tempfile

import numpy as np
import torch
from tqdm import tqdm

from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment, extract_correspondences

from .image import prepare_images_for_mast3r


def symmetric_inference(model, img1, img2) -> dict[str, list[torch.Tensor]]:
    """
    Run inference on a pair of images w.r.t. both images.

    Returns x11, x21, x22, x12.

    x11, x21 are obtained w.r.t. img1.
        x11 is img1 pointmap in img1 coords; x21 is img2 pointmap in img1 coords.
    x22, x12 w.r.t. img2, similarly.

    Put model and images on the same device externally.

    img1, img2: Tensor, shape (3, H, W).
    return: {"pts3d": [...,], "conf": [...,], "desc": [...,], "desc_conf": [...,]}
        i.e. flattens keys of model output.
        Each value is a list of 4, corresponding to x11, x21, x22, x12.
    """
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    shape1 = torch.tensor(img1.shape[-2:]).unsqueeze(0)
    shape2 = torch.tensor(img2.shape[-2:]).unsqueeze(0)

    # compute encoder only once
    feat1, feat2, pos1, pos2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        with torch.cuda.amp.autocast(enabled=False):
            res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
        return res1, res2

    # decoder 1-2
    res11, res21 = decoder(feat1, feat2, pos1, pos2, shape1, shape2)
    # decoder 2-1
    res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1)

    # Flatten keys
    results = [res11, res21, res22, res12]
    keys = ("pts3d", "conf", "desc", "desc_conf")
    ret = {}
    for key in keys:
        ret[key] = [r[key][0] for r in results]
    return ret


@torch.no_grad()
def pairs_inference(model, imgs, pair_indices, verbose=False):
    """
    Run symmetric inference on all pairs of images.

    Returns key-value map of pair indices to results.

    return:
    {
        (i1, i2): {
            "pts3d": ...,
            "conf": ...,
            "matching_score": ...,
            "corres": ...,
        },
    }
    See original function (mast3r/cloud_opt/sparse_ga.py:forward_mast3r) for details.
    """
    ret = {}

    for i1, i2 in tqdm(pair_indices, disable=not verbose, desc="Pairs inference"):
        output = symmetric_inference(model, imgs[i1], imgs[i2])
        xs = output["pts3d"]
        confs = output["conf"]
        descs = output["desc"]
        qonfs = output["desc_conf"]

        corres = extract_correspondences(descs, qonfs, device=imgs[0].device, subsample=8)
        # Geometric mean of confidences
        conf_score = np.prod([c.mean().item() for c in confs]) ** (1 / 4)
        matching_score = (float(conf_score), float(corres[2].sum()), len(corres[2]))

        ret[(i1, i2)] = {
            "pts3d": xs,
            "conf": confs,
            "matching_score": matching_score,
            "corres": corres,
        }

    return ret


def reconstruct_scene(model, imgs, filelist, device):
    """
    model: Model instance.
    imgs: List of images from load_image.
        Tensor shape (C,H,W), dtype float32.
    filelist: List of image paths corresponding to each image.
        Due to Mast3r legacy code, this is required.
    device: Device to run on.
    """
    imgs = prepare_images_for_mast3r(imgs)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)

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
