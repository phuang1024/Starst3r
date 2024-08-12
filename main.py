import sys
sys.path.append("mast3r")
sys.path.append("mast3r/dust3r")
sys.path.append("mast3r/dust3r/croco")

from pathlib import Path

import starster
from mast3r.model import AsymmetricMASt3R

"""
# Test load image
img = starster.load_image("/tmp/test.jpg")
print(type(img))
print(img.shape, img.dtype)
stop
"""

files = []
dir = Path("../data/room/").absolute()
for file in dir.iterdir():
    if file.suffix.lower() == ".jpg":
        files.append(str(file))

imgs = []
for file in files:
    imgs.append(starster.load_image(file).to("cuda"))

imgs_mast3r = starster.prepare_images_for_mast3r(imgs)

pairs = starster.make_pair_indices(len(imgs), symmetric=True)

model = AsymmetricMASt3R.from_pretrained("../models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth").to("cuda")

#starster.reconstruct_scene(model, imgs, files, "cuda")
starster.pairs_inference(model, imgs, pairs)
