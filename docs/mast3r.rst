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

   # Load and preprocess from a path.
   img = starster.load_image("/path/to/img.jpg", size=224)
   imgs = starster.load_images(files)

See the docs for return and argument specifications.

2. Run the pipeline
^^^^^^^^^^^^^^^^^^^

Load the PyTorch model and run the reconstruction pipeline.

.. code-block:: python

   # Load model from file
   model = starster.Mast3rModel.from_pretrained("/path/to/model.pth").to(device)

   # Create empty scene
   scene = starster.Scene()

   # Add images and reconstruct.
   scene.add_images(model, images)

The :class:`starster.Scene` object contains the dense points, camera poses, and 3DGS
methods (see 3DGS).

You can use multiple calls of ``add_images`` on the same scene object. This and progressively
reconstructs a scene, using all previous images in each iteration.

3. Use the results
^^^^^^^^^^^^^^^^^^

You can access sparse or dense points, and all points together or per camera.

Regardless, all XYZ coordinates are in the same global space.

.. code-block:: python

   # (copied from quickstart)

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

About Mast3r
------------
