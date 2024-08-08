"""
Functions for running inference.
"""


def symmetric_inference(model, img1, img2, device):
    """
    Run inference on a pair of images.
    Return results w.r.t both images.
    Returns r11, r21, r22, r12

    img1, img2: dict (as in dust3r/utils/image.py:load_images)
    """
    shape1 = torch.from_numpy(img1['true_shape']).to(device, non_blocking=True)
    shape2 = torch.from_numpy(img2['true_shape']).to(device, non_blocking=True)
    img1 = img1['img'].to(device, non_blocking=True)
    img2 = img2['img'].to(device, non_blocking=True)

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
