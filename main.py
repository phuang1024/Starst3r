import sys
sys.path.append("mast3r")
sys.path.append("mast3r/dust3r")
sys.path.append("mast3r/dust3r/croco")

from pathlib import Path

import starster
from mast3r.model import AsymmetricMASt3R

"""
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

images = []
for file in files:
    images.append(starster.load_image(file))

images = starster.prepare_images_for_mast3r(images)

model = AsymmetricMASt3R.from_pretrained("../models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth").to("cuda")
starster.get_reconstructed_scene(model, images, files, "cuda")
