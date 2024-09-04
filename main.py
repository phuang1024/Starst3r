import sys
sys.path.append("mast3r")
sys.path.append("mast3r/dust3r")
sys.path.append("mast3r/dust3r/croco")

from pathlib import Path

import starster
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


files = []
dir = Path("../data/room/").absolute()
for file in dir.iterdir():
    if file.suffix.lower() == ".jpg":
        files.append(str(file))


"""
# Test load image
img = starster.load_image(files[0])
print(type(img))
print(img.shape, img.dtype)
exit()
"""


imgs = []
for file in files:
    imgs.append(starster.load_image(file, 224))

"""
imgs_mast3r = starster.prepare_images_for_mast3r(imgs)
for i in range(len(imgs)):
    imgs[i] = imgs[i].to(DEVICE)
pairs = starster.make_pair_indices(len(imgs), symmetric=True)
"""

model = starster.Mast3rModel.from_pretrained("../models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth").to(DEVICE)

scene = starster.reconstruct_scene(model, imgs, files, DEVICE, "/tmp/starster_main_test")
#starster.pairs_inference(model, imgs, pairs)

#pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))

"""
import numpy as np
for i, pts in enumerate(scene.pts3d):
    print(pts.shape)
    np.savetxt(f"pts{i}.txt", pts)

print(len(scene.pts3d), scene.pts3d[0].shape)
"""
