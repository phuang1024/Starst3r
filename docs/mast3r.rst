Mast3r reconstruction
=====================

The Mast3r pipeline takes a set of images and returns:

- Sparse and dense point cloud reconstruction.
- Camera poses.

The Starst3r functions are mostly wrappers around the Mast3r codebase
to improve usability.

Tutorial
--------

1. Download model
^^^^^^^^^^^^^^^^^

Download the official pretrained Mast3r model. See :ref:`Download Mast3r model`.

2. Load images
^^^^^^^^^^^^^^

Load the images using Starst3r functions:

.. code-block:: python

   # These two are sufficient.
   img = starster.load_image("/path/to/img.jpg", size=224)
   imgs = starster.load_images(files)

   # Or, process a preloaded image to pipeline requirements.
   processed_img = starster.process_image(another_img, size=224)

See the docs for return and argument specifications.

2. Run the pipeline
^^^^^^^^^^^^^^^^^^^

Load the PyTorch model and run the reconstruction pipeline.

Due to Mast3r legacy code, the pipeline requires a list of image file paths.

.. code-block:: python

   # Load model from file
   model = starster.Mast3rModel.from_pretrained("/path/to/model.pth").to(device)

   # Reconstruct scene
   scene = starster.reconstruct_scene(model, imgs, files, device)

:func:`starster.reconstruct_scene` returns a :class:`starster.PointCloudScene` object,
which contains the point clouds, camera poses, and more information.

3. Use the results
^^^^^^^^^^^^^^^^^^

You can access sparse or dense points, and all points together or per camera.

Regardless, all XYZ coordinates are in the same global space.

.. code-block:: python

   # (copied from quickstart)

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

About Mast3r
------------
