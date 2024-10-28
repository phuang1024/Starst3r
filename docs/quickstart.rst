Quickstart
==========

Please see :ref:`Installation` first.

Mast3r reconstruction
---------------------

Reconstruct:

.. code-block:: python

   import starster

   device = "cuda"

   # List of paths to images
   files = (
      "/path/to/img1.jpg",
      ...
   )
   # Load images with specified resolution (default 224)
   imgs = starster.load_images(files, size=224)

   # Load model from file
   model = starster.Mast3rModel.from_pretrained("/path/to/model.pth").to(device)

   # Reconstruct scene
   scene = starster.Scene()
   scene.add_images(model, images)

Use results:

.. code-block:: python

   # Dense point clouds from each camera (in global XYZ space)
   for i in range(len(scene.dense_pts)):
       pts, colors = scene.dense_pts[i], scene.dense_cols[i]

       # Point cloud: pts shape is (N, 3); XYZ of each point.
       print(f"Points from camera {i}: {pts.shape}")

       # Point colors: shape is (N, 3); RGB of each point.
       print(f"Colors from camera {i}: {colors.shape}")

   # Dense points from all cameras concatenated together
   pts, colors = scene.dense_pts_flat, scene.dense_cols_flat
   print("Total points from all cameras:", pts.shape)

3D Gaussian Splatting refinement
--------------------------------

.. code-block:: python

   # Extend from Mast3r scene. See above
   scene = ...

   # Initialize gaussians
   scene.init_3dgs()

   width, height = 224, 224
   # Render views from original camera poses
   # img (color image render) has shape (N, H, W, 3).
   img, alpha, info = scene.render_3dgs_original(width, height)
   # Render from new camera poses
   img, alpha, info = scene.render_3dgs(world_to_cam, intrinsics, width, height)

   # Run 3DGS optimization for 1000 iters
   scene.run_3dgs_optim(1000, enable_pruning=True, verbose=True)
   # Run without pruning and densification
   scene.run_3dgs_optim(5000, enable_pruning=False, verbose=True)

   # Render again with refined splats
   img, alpha, info = scene.render_3dgs_original(width, height)
