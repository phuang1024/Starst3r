# MasterSplat

Mast3r reconstruction and localization with 3DGS view rendering.

## User experience

```py
images = [...]

scene = reconstruct(images)
for cam in scene.cameras:
    pose = cam.c2w  # or something

point_cloud = scene.point_cloud
```
