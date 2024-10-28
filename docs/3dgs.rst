3D Gaussian Splatting
=====================

3D Gaussian Splatting (3DGS) refines a 3D scene for visual quality.

In this library, it uses the Mast3r reconstruction as a starting point.

Tutorial
--------

1. Run optimization
^^^^^^^^^^^^^^^^^^^

Run initialization, and optimization.

You need a Mast3r reconstruction as a starting point.

See :ref:`Mast3r reconstruction`.

.. code-block:: python

   # See Mast3r tutorial
   scene = ...
   ...

   # Initialize gaussians
   scene.init_3dgs()

   # Run 3DGS optimization for 500 iters
   scene.run_3dgs_optim(500, enable_pruning=True, verbose=True)
   # Run without pruning and densification
   scene.run_3dgs_optim(100, enable_pruning=False, verbose=True)

2. Render views
^^^^^^^^^^^^^^^

Render views from the refined 3DGS scene.

.. code-block:: python

   width, height = 224, 224

   # Render views from original camera poses
   # img (color image render) has shape (N, H, W, 3).
   img, alpha, info = scene.render_3dgs_original(width, height)

   # Render from new camera poses
   img, alpha, info = gs.render_3dgs(world_to_cam, intrinsics, width, height)

About 3DGS
----------
