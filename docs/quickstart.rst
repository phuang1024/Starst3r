Quickstart
==========

Download Mast3r model
---------------------

Download the pretrained Mast3r model.

https://github.com/naver/mast3r/?tab=readme-ov-file#checkpoints

Starst3r uses the same Mast3r model internally.

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
   scene = starster.reconstruct_scene(model, imgs, files, device)

Use results:

.. code-block:: python

   # Dense point clouds from each camera (in global XYZ space)
   all_pts = scene.pts_dense()
   for i in range(len(all_pts)):
       pts, colors = all_pts[i]

       # Point cloud: pts shape is (N, 3); XYZ of each point.
       print(f"Points from camera {i}: {pts.shape}")

       # Point colors: shape is (N, 3); RGB of each point.
       print(f"Colors from camera {i}: {colors.shape}")

   # Dense points from all cameras concatenated together
   pts, colors = scene.pts_dense_flat()
   print("Total points from all cameras:", pts.shape)

   # Sparse points (fewer points than dense)
   pts_sparse, colors_sparse = scene.pts_sparse()
   all_pts_sparse = scene.pts_sparse_flat()
   # ... and process similarly

3D Gaussian Splatting refinement
--------------------------------

.. code-block:: python

   # See above
   scene = starster.reconstruct_scene(...)

   gs = starster.GSTrainer(scene, device=device)

   width, height = 224, 224
   # Render views from original camera poses
   # img (color image render) has shape (N, H, W, 3).
   img, alpha, info = gs.render_views_original(width, height)
   # Render from new camera poses
   img, alpha, info = gs.render_views(world_to_cam, intrinsics, width, height)

   # Run 3DGS optimization for 1000 iters
   gs.run_optimization(1000, enable_pruning=True, verbose=True)
   # Run without pruning and densification
   gs.run_optimization(5000, enable_pruning=False, verbose=True)

   # Render again with refined splats
   img, alpha, info = gs.render_views_original(width, height)
