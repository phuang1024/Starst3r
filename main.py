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

print(scene.imgs[0].shape)

#pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))

"""
import numpy as np
for i, pts in enumerate(scene.pts3d):
    print(pts.shape)
    np.savetxt(f"pts{i}.txt", pts)

print(len(scene.pts3d), scene.pts3d[0].shape)
"""

import numpy as np
import cv2

gs = starster.GSTrainer(scene)
gs.init_gaussians()

gs.run_optimization(1000, enable_pruning=True, verbose=True)
gs.run_optimization(5000, enable_pruning=False, verbose=True)

render = gs.render_views_original(224, 224)
for i, img in enumerate(render[0]):
    cv2.imwrite(f"{i}.png", (img.detach().cpu().numpy()[..., ::-1] * 255).astype(np.uint8))
