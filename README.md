# Starst3r

![](docs/images/Poster.jpg)

![](docs/images/demo.gif)

Fast 3D reconstruction framework using Mast3r.

Docs: https://starst3r.readthedocs.io/en/latest/

```py
import starster

device = "cuda"

files = (
  "/path/to/img1.jpg",
  ...
)

imgs = starster.load_images(files, size=224)
model = starster.Mast3rModel.from_pretrained("/path/to/model.pth").to(device)
scene = starster.reconstruct_scene(model, imgs, files, device)
```
