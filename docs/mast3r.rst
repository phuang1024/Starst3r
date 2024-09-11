Mast3r reconstruction
=====================

The Mast3r pipeline takes a set of images and returns:

- Sparse and dense point cloud reconstruction.
- Camera poses.

Tutorial
--------

1. Prepare images
^^^^^^^^^^^^^^^^^

Load the images with Starst3r functions:

.. code-block:: py

   starster.load_image("/path/to/img.jpg", size=224)
   starster.load_images(files)
   starster.process_image(img, size=224)

See the docs for return and argument specifications.

There are certain restrictions on the images:

- Width and height must be a multiple of 8.

2. Run the pipeline
^^^^^^^^^^^^^^^^^^^

TODO