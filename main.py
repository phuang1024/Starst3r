"""
Testing script
"""

import sys
sys.path.append("mast3r")
sys.path.append("mast3r/dust3r")
sys.path.append("mast3r/dust3r/croco")

from pathlib import Path

import starster
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RES = 224


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
    imgs.append(starster.load_image(file, RES))

"""
imgs_mast3r = starster.prepare_images_for_mast3r(imgs)
for i in range(len(imgs)):
    imgs[i] = imgs[i].to(DEVICE)
pairs = starster.make_pair_indices(len(imgs), symmetric=True)
"""

model = starster.Mast3rModel.from_pretrained("../models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth").to(DEVICE)

scene = starster.Scene()
scene.add_images(model, imgs, files, DEVICE)

print(scene.imgs[0].shape)


"""
import numpy as np
for i, pts in enumerate(scene.pts3d):
    print(pts.shape)
    np.savetxt(f"pts{i}.txt", pts)

print(len(scene.pts3d), scene.pts3d[0].shape)
"""

"""
import numpy as np
import cv2

gs = starster.GSTrainer(scene)

# Show progress.
for i in range(50):
    imgs, alpha, info = gs.render_views_original(RES, RES)
    imgs = torch.clip(imgs.detach().cpu(), 0, 1)
    imgs = (imgs.numpy()[..., ::-1] * 255).astype(np.uint8)
    cv2.imwrite(f"imgs/{i}.jpg", imgs[0])

    gs.run_optimization(10, enable_pruning=True, verbose=True)

"""
"""
gs.run_optimization(400, enable_pruning=True, verbose=True)
gs.run_optimization(100, enable_pruning=False, verbose=True)

imgs, alpha, info = gs.render_views_original(RES, RES)
print(imgs.shape)
imgs = torch.clip(imgs.detach().cpu(), 0, 1)
imgs = (imgs.numpy()[..., ::-1] * 255).astype(np.uint8)
for i, img in enumerate(imgs):
    cv2.imwrite(f"{i}.png", img)
"""
