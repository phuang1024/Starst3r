"""
Functions for running the inference pipeline.
"""

__all__ = (
    "symmetric_inference",
    "pairs_inference",
    "reconstruct_scene",
)

import numpy as np
import torch
from tqdm import tqdm

from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment


def symmetric_inference(model, img1, img2):
    """
    Run inference on a pair of images w.r.t. both images.

    Returns x11, x21, x22, x12.

    x11, x21 are obtained w.r.t. img1.
    x22, x12 w.r.t. img2.

    Put model and images on the same device externally.

    img1, img2: Tensor, shape (3, H, W).
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

    return (res11, res21, res22, res12)


@torch.no_grad()
def pairs_inference(model, imgs, pair_indices):
    """
    Run inference on all pairs of images.
    """
    for i1, i2 in tqdm(pair_indices):
        res = symmetric_inference(model, imgs[i1], imgs[i2])
        X11, X21, X22, X12 = [r['pts3d'][0] for r in res]
        C11, C21, C22, C12 = [r['conf'][0] for r in res]
        descs = [r['desc'][0] for r in res]
        #qonfs = [r[desc_conf][0] for r in res]

        # save
        #torch.save(to_cpu((X11, C11, X21, C21)), mkdir_for(path1))
        #torch.save(to_cpu((X22, C22, X12, C12)), mkdir_for(path2))

        # perform reciprocal matching
        #corres = extract_correspondences(descs, qonfs, device=device, subsample=subsample)

        #conf_score = (C11.mean() * C12.mean() * C21.mean() * C22.mean()).sqrt().sqrt()
        #matching_score = (float(conf_score), float(corres[2].sum()), len(corres[2]))
        #if cache_path is not None:
        #    torch.save((matching_score, corres), mkdir_for(path_corres))

    return


def reconstruct_scene(model, imgs, filelist, device):
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)

    scene = sparse_global_alignment(
        filelist,
        pairs,
        "/tmp",
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
