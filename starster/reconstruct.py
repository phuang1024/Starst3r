"""
Functions for running the Mast3r inference pipeline.
"""

__all__ = (
    "reconstruct_scene",
)

import tempfile

import torch

from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import *

from .image import prepare_images_for_mast3r


def reconstruct_scene(model, imgs, filelist, device, optim_params=None, tmpdir=None):
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

    optim_params: TODO

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
        optim_params=optim_params,
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
        optim_params=None,
        **kw
    ):
    """Copied and modified from sparse_ga.py:sparse_global_alignment

    return:
        (Optimized scene, optimization parameters)
        Pass optimization parameters to this function again to resume from previous optimization results.
            See :func:`sparse_scene_optimizer_slam`.
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

    imgs, res_coarse, res_fine, optim_params = sparse_scene_optimizer_slam(
        imgs, subsample, imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21, canonical_paths, mst,
        shared_intrinsics=shared_intrinsics, cache_path=cache_path, device=device, dtype=dtype, prev_params=optim_params, **kw)

    return SparseGA(imgs, pairs_in, res_fine or res_coarse, anchors, canonical_paths), optim_params


def sparse_scene_optimizer_slam(imgs, subsample, imsizes, pps, base_focals, core_depth, anchors, corres, corres2d,
                           preds_21, canonical_paths, mst, cache_path,
                           lr1=0.2, niter1=500, loss1=gamma_loss(1.1),
                           lr2=0.02, niter2=500, loss2=gamma_loss(0.4),
                           lossd=gamma_loss(1.1),
                           opt_pp=True, opt_depth=True,
                           schedule=cosine_schedule, depth_mode='add', exp_depth=False,
                           lora_depth=False,  # dict(k=96, gamma=15, min_norm=.5),
                           shared_intrinsics=False,
                           init={}, device='cuda', dtype=torch.float32,
                           matching_conf_thr=5., loss_dust3r_w=0.01,
                           verbose=True, dbg=(), prev_params=None):
    """
    Global adjustment optimizer function, copied and modified from sparse_ga.py:158

    Optimized parameters: ``params = pps + log_focals + quats + trans + log_sizes + core_depth``

    Changes to the function
    -----------------------

    Maintain previous parameters: Use previous optimization results as a starting point if applicable.
    If the parameter ``prev_params`` is not None, optimized params are set to the values in ``prev_params``.

    prev_params: dict
        Key is optimized parameter name (e.g. "pps").
        Value is previous optimization result for the parameter.
        If more cameras were added, only the first N params are set to prev_params.
            N = len(prev_params[...])

    return:
        Add the new optimized parameters to the return value.
    """
    init = copy.deepcopy(init)
    # extrinsic parameters
    vec0001 = torch.tensor((0, 0, 0, 1), dtype=dtype, device=device)
    quats = [nn.Parameter(vec0001.clone()) for _ in range(len(imgs))]
    trans = [nn.Parameter(torch.zeros(3, device=device, dtype=dtype)) for _ in range(len(imgs))]

    # intialize
    ones = torch.ones((len(imgs), 1), device=device, dtype=dtype)
    median_depths = torch.ones(len(imgs), device=device, dtype=dtype)
    for img in imgs:
        idx = imgs.index(img)
        init_values = init.setdefault(img, {})
        if verbose and init_values:
            print(f' >> initializing img=...{img[-25:]} [{idx}] for {set(init_values)}')

        K = init_values.get('intrinsics')
        if K is not None:
            K = K.detach()
            focal = K[:2, :2].diag().mean()
            pp = K[:2, 2]
            base_focals[idx] = focal
            pps[idx] = pp
        pps[idx] /= imsizes[idx]  # default principal_point would be (0.5, 0.5)

        depth = init_values.get('depthmap')
        if depth is not None:
            core_depth[idx] = depth.detach()

        median_depths[idx] = med_depth = core_depth[idx].median()
        core_depth[idx] /= med_depth

        cam2w = init_values.get('cam2w')
        if cam2w is not None:
            rot = cam2w[:3, :3].detach()
            cam_center = cam2w[:3, 3].detach()
            quats[idx].data[:] = roma.rotmat_to_unitquat(rot)
            trans_offset = med_depth * torch.cat((imsizes[idx] / base_focals[idx] * (0.5 - pps[idx]), ones[:1, 0]))
            trans[idx].data[:] = cam_center + rot @ trans_offset
            del rot
            assert False, 'inverse kinematic chain not yet implemented'

    # intrinsics parameters
    if shared_intrinsics:
        # Optimize a single set of intrinsics for all cameras. Use averages as init.
        confs = torch.stack([torch.load(pth)[0][2].mean() for pth in canonical_paths]).to(pps)
        weighting = confs / confs.sum()
        pp = nn.Parameter((weighting @ pps).to(dtype))
        pps = [pp for _ in range(len(imgs))]
        focal_m = weighting @ base_focals
        log_focal = nn.Parameter(focal_m.view(1).log().to(dtype))
        log_focals = [log_focal for _ in range(len(imgs))]
    else:
        pps = [nn.Parameter(pp.to(dtype)) for pp in pps]
        log_focals = [nn.Parameter(f.view(1).log().to(dtype)) for f in base_focals]

    diags = imsizes.float().norm(dim=1)
    min_focals = 0.25 * diags  # diag = 1.2~1.4*max(W,H) => beta >= 1/(2*1.2*tan(fov/2)) ~= 0.26
    max_focals = 10 * diags

    assert len(mst[1]) == len(pps) - 1

    def make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth):
        # make intrinsics
        focals = torch.cat(log_focals).exp().clip(min=min_focals, max=max_focals)
        pps = torch.stack(pps)
        K = torch.eye(3, dtype=dtype, device=device)[None].expand(len(imgs), 3, 3).clone()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, 0:2, 2] = pps * imsizes
        if trans is None:
            return K

        # security! optimization is always trying to crush the scale down
        sizes = torch.cat(log_sizes).exp()
        global_scaling = 1 / sizes.min()

        # compute distance of camera to focal plane
        # tan(fov) = W/2 / focal
        z_cameras = sizes * median_depths * focals / base_focals

        # make extrinsic
        rel_cam2cam = torch.eye(4, dtype=dtype, device=device)[None].expand(len(imgs), 4, 4).clone()
        rel_cam2cam[:, :3, :3] = roma.unitquat_to_rotmat(F.normalize(torch.stack(quats), dim=1))
        rel_cam2cam[:, :3, 3] = torch.stack(trans)

        # camera are defined as a kinematic chain
        tmp_cam2w = [None] * len(K)
        tmp_cam2w[mst[0]] = rel_cam2cam[mst[0]]
        for i, j in mst[1]:
            # i is the cam_i_to_world reference, j is the relative pose = cam_j_to_cam_i
            tmp_cam2w[j] = tmp_cam2w[i] @ rel_cam2cam[j]
        tmp_cam2w = torch.stack(tmp_cam2w)

        # smart reparameterizaton of cameras
        trans_offset = z_cameras.unsqueeze(1) * torch.cat((imsizes / focals.unsqueeze(1) * (0.5 - pps), ones), dim=-1)
        new_trans = global_scaling * (tmp_cam2w[:, :3, 3:4] - tmp_cam2w[:, :3, :3] @ trans_offset.unsqueeze(-1))
        cam2w = torch.cat((torch.cat((tmp_cam2w[:, :3, :3], new_trans), dim=2),
                          vec0001.view(1, 1, 4).expand(len(K), 1, 4)), dim=1)

        depthmaps = []
        for i in range(len(imgs)):
            core_depth_img = core_depth[i]
            if exp_depth:
                core_depth_img = core_depth_img.exp()
            if lora_depth:  # compute core_depth as a low-rank decomposition of 3d points
                core_depth_img = lora_depth_proj[i] @ core_depth_img
            if depth_mode == 'add':
                core_depth_img = z_cameras[i] + (core_depth_img - 1) * (median_depths[i] * sizes[i])
            elif depth_mode == 'mul':
                core_depth_img = z_cameras[i] * core_depth_img
            else:
                raise ValueError(f'Bad {depth_mode=}')
            depthmaps.append(global_scaling * core_depth_img)

        return K, (inv(cam2w), cam2w), depthmaps

    K = make_K_cam_depth(log_focals, pps, None, None, None, None)

    if shared_intrinsics:
        print('init focal (shared) = ', to_numpy(K[0, 0, 0]).round(2))
    else:
        print('init focals =', to_numpy(K[:, 0, 0]))

    # spectral low-rank projection of depthmaps
    if lora_depth:
        core_depth, lora_depth_proj = spectral_projection_of_depthmaps(
            imgs, K, core_depth, subsample, cache_path=cache_path, **lora_depth)
    if exp_depth:
        core_depth = [d.clip(min=1e-4).log() for d in core_depth]
    core_depth = [nn.Parameter(d.ravel().to(dtype)) for d in core_depth]
    log_sizes = [nn.Parameter(torch.zeros(1, dtype=dtype, device=device)) for _ in range(len(imgs))]

    # Fetch img slices
    _, confs_sum, imgs_slices = corres

    # Define which pairs are fine to use with matching
    def matching_check(x): return x.max() > matching_conf_thr
    is_matching_ok = {}
    for s in imgs_slices:
        is_matching_ok[s.img1, s.img2] = matching_check(s.confs)

    # Prepare slices and corres for losses
    dust3r_slices = [s for s in imgs_slices if not is_matching_ok[s.img1, s.img2]]
    loss3d_slices = [s for s in imgs_slices if is_matching_ok[s.img1, s.img2]]
    cleaned_corres2d = []
    for cci, (img1, pix1, confs, confsum, imgs_slices) in enumerate(corres2d):
        cf_sum = 0
        pix1_filtered = []
        confs_filtered = []
        curstep = 0
        cleaned_slices = []
        for img2, slice2 in imgs_slices:
            if is_matching_ok[img1, img2]:
                tslice = slice(curstep, curstep + slice2.stop - slice2.start, slice2.step)
                pix1_filtered.append(pix1[tslice])
                confs_filtered.append(confs[tslice])
                cleaned_slices.append((img2, slice2))
            curstep += slice2.stop - slice2.start
        if pix1_filtered != []:
            pix1_filtered = torch.cat(pix1_filtered)
            confs_filtered = torch.cat(confs_filtered)
            cf_sum = confs_filtered.sum()
        cleaned_corres2d.append((img1, pix1_filtered, confs_filtered, cf_sum, cleaned_slices))

    def loss_dust3r(cam2w, pts3d, pix_loss):
        # In the case no correspondence could be established, fallback to DUSt3R GA regression loss formulation (sparsified)
        loss = 0.
        cf_sum = 0.
        for s in dust3r_slices:
            if init[imgs[s.img1]].get('freeze') and init[imgs[s.img2]].get('freeze'):
                continue
            # fallback to dust3r regression
            tgt_pts, tgt_confs = preds_21[imgs[s.img2]][imgs[s.img1]]
            tgt_pts = geotrf(cam2w[s.img2], tgt_pts)
            cf_sum += tgt_confs.sum()
            loss += tgt_confs @ pix_loss(pts3d[s.img1], tgt_pts)
        return loss / cf_sum if cf_sum != 0. else 0.

    def loss_3d(K, w2cam, pts3d, pix_loss):
        # For each correspondence, we have two 3D points (one for each image of the pair).
        # For each 3D point, we have 2 reproj errors
        if any(v.get('freeze') for v in init.values()):
            pts3d_1 = []
            pts3d_2 = []
            confs = []
            for s in loss3d_slices:
                if init[imgs[s.img1]].get('freeze') and init[imgs[s.img2]].get('freeze'):
                    continue
                pts3d_1.append(pts3d[s.img1][s.slice1])
                pts3d_2.append(pts3d[s.img2][s.slice2])
                confs.append(s.confs)
        else:
            pts3d_1 = [pts3d[s.img1][s.slice1] for s in loss3d_slices]
            pts3d_2 = [pts3d[s.img2][s.slice2] for s in loss3d_slices]
            confs = [s.confs for s in loss3d_slices]

        if pts3d_1 != []:
            confs = torch.cat(confs)
            pts3d_1 = torch.cat(pts3d_1)
            pts3d_2 = torch.cat(pts3d_2)
            loss = confs @ pix_loss(pts3d_1, pts3d_2)
            cf_sum = confs.sum()
        else:
            loss = 0.
            cf_sum = 1.

        return loss / cf_sum

    def loss_2d(K, w2cam, pts3d, pix_loss):
        # For each correspondence, we have two 3D points (one for each image of the pair).
        # For each 3D point, we have 2 reproj errors
        proj_matrix = K @ w2cam[:, :3]
        loss = npix = 0
        for img1, pix1_filtered, confs_filtered, cf_sum, cleaned_slices in cleaned_corres2d:
            if init[imgs[img1]].get('freeze', 0) >= 1:
                continue  # no need
            pts3d_in_img1 = [pts3d[img2][slice2] for img2, slice2 in cleaned_slices]
            if pts3d_in_img1 != []:
                pts3d_in_img1 = torch.cat(pts3d_in_img1)
                loss += confs_filtered @ pix_loss(pix1_filtered, reproj2d(proj_matrix[img1], pts3d_in_img1))
                npix += confs_filtered.sum()

        return loss / npix if npix != 0 else 0.

    def optimize_loop(loss_func, lr_base, niter, pix_loss, lr_end=0):
        # create optimizer
        params = pps + log_focals + quats + trans + log_sizes + core_depth
        optimizer = torch.optim.Adam(params, lr=1, weight_decay=0, betas=(0.9, 0.9))
        ploss = pix_loss if 'meta' in repr(pix_loss) else (lambda a: pix_loss)

        with tqdm(total=niter) as bar:
            for iter in range(niter or 1):
                K, (w2cam, cam2w), depthmaps = make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth)
                pts3d = make_pts3d(anchors, K, cam2w, depthmaps, base_focals=base_focals)
                if niter == 0:
                    break

                alpha = (iter / niter)
                lr = schedule(alpha, lr_base, lr_end)
                adjust_learning_rate_by_lr(optimizer, lr)
                pix_loss = ploss(1 - alpha)
                optimizer.zero_grad()
                loss = loss_func(K, w2cam, pts3d, pix_loss) + loss_dust3r_w * loss_dust3r(cam2w, pts3d, lossd)
                loss.backward()
                optimizer.step()

                # make sure the pose remains well optimizable
                for i in range(len(imgs)):
                    quats[i].data[:] /= quats[i].data.norm()

                loss = float(loss)
                if loss != loss:
                    break  # NaN loss
                bar.set_postfix_str(f'{lr=:.4f}, {loss=:.3f}')
                bar.update(1)

        if niter:
            print(f'>> final loss = {loss}')
        return dict(intrinsics=K.detach(), cam2w=cam2w.detach(),
                    depthmaps=[d.detach() for d in depthmaps], pts3d=[p.detach() for p in pts3d])

    # MODIFICATION: Set params to prev_params if given.
    if prev_params is not None:
        pps[:len(prev_params['pps'])] = prev_params['pps']
        log_focals[:len(prev_params['log_focals'])] = prev_params['log_focals']
        quats[:len(prev_params['quats'])] = prev_params['quats']
        trans[:len(prev_params['trans'])] = prev_params['trans']
        log_sizes[:len(prev_params['log_sizes'])] = prev_params['log_sizes']
        core_depth[:len(prev_params['core_depth'])] = prev_params['core_depth']

    # at start, don't optimize 3d points
    for i, img in enumerate(imgs):
        trainable = not (init[img].get('freeze'))
        pps[i].requires_grad_(False)
        log_focals[i].requires_grad_(False)
        quats[i].requires_grad_(trainable)
        trans[i].requires_grad_(trainable)
        log_sizes[i].requires_grad_(trainable)
        core_depth[i].requires_grad_(False)

    res_coarse = optimize_loop(loss_3d, lr_base=lr1, niter=niter1, pix_loss=loss1)

    res_fine = None
    if niter2:
        # now we can optimize 3d points
        for i, img in enumerate(imgs):
            if init[img].get('freeze', 0) >= 1:
                continue
            pps[i].requires_grad_(bool(opt_pp))
            log_focals[i].requires_grad_(True)
            core_depth[i].requires_grad_(opt_depth)

        # refinement with 2d reproj
        res_fine = optimize_loop(loss_2d, lr_base=lr2, niter=niter2, pix_loss=loss2)

    K = make_K_cam_depth(log_focals, pps, None, None, None, None)
    if shared_intrinsics:
        print('Final focal (shared) = ', to_numpy(K[0, 0, 0]).round(2))
    else:
        print('Final focals =', to_numpy(K[:, 0, 0]))

    params_ret = {
        'pps': pps,
        'log_focals': log_focals,
        'quats': quats,
        'trans': trans,
        'log_sizes': log_sizes,
        'core_depth': core_depth,
    }

    return imgs, res_coarse, res_fine, params_ret
