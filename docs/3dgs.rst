3D Gaussian Splatting
=====================

3D Gaussian Splatting (3DGS) refines a 3D scene for visual quality.

In this library, it uses the Mast3r reconstruction as a starting point,
dramatically improving speed.

Tutorial
--------

1. Setup trainer
^^^^^^^^^^^^^^^^

Create a trainer object after reconstructing a scene with Mast3r.

See :ref:`Mast3r reconstruction`.

.. code-block:: python

   # See Mast3r tutorials.
   scene = starster.reconstruct_scene(...)

   # Create trainer
   gs = starster.GSTrainer(scene, device=device)

2. Run optimization
^^^^^^^^^^^^^^^^^^^

Run optimization, pruning, and densification.

.. code-block:: python

   # Run 3DGS optimization for 500 iters
   gs.run_optimization(500, enable_pruning=True, verbose=True)
   # Run without pruning and densification
   gs.run_optimization(100, enable_pruning=False, verbose=True)

3. Render views
^^^^^^^^^^^^^^^

Render views from the refined scene.

.. code-block:: python

   width, height = 224, 224

   # Render views from original camera poses
   # img (color image render) has shape (N, H, W, 3).
   img, alpha, info = gs.render_views_original(width, height)

   # Render from new camera poses
   img, alpha, info = gs.render_views(world_to_cam, intrinsics, width, height)

About 3DGS
----------
