"""
Functions for running the inference pipeline.
"""

__all__ = (
    "get_reconstructed_scene",
)

import copy
import tempfile

from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.demo import SparseGAState


def get_reconstructed_scene(model, image_size, filelist, device):
    imgs = load_images(filelist, size=image_size)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

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

    #outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)
    #scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)

    return scene
