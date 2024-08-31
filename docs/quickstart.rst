Quickstart
==========

Download model
--------------

Download the pretrained Mast3r model.

https://github.com/naver/mast3r/?tab=readme-ov-file#checkpoints

Starst3r uses the same Mast3r model internally.

Reconstruct scene
-----------------

.. code-block:: python

   import starster

   device = "cuda"

   # List of paths to images
   files = (
      "/path/to/img1.jpg",
      ...
   )

   # Load images with specified resolution (default 512)
   imgs = starster.load_images(files, size=224)

   # Load model from file
   model = starster.Mast3rModel.from_pretrained("/path/to/model.pth").to(device)

   # Reconstruct scene
   scene = starster.reconstruct_scene(model, imgs, files, device)

Use results
-----------

.. code-block:: python

   # Iterate point clouds with respect to each camera.
   num_cameras = len(scene.pts3d)
   for i in range(num_cameras):
       # Point cloud: pts shape is (N, 3); XYZ of each point.
       pts = scene.pts3d[i]
       print(f"Points from camera {i}: {pts.shape}")

       # Point colors: shape is (N, 3); RGB of each point.
       colors = scene.pts3d_colors[i]
       print(f"Colors from camera {i}: {colors.shape}")

   # Iterate all points of all cameras.
   for pt, color in starster.iterate_verts(scene):
       ...

   num_verts = starster.num_verts(scene)
   print("Total verts from all cameras:", num_verts)
