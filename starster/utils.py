def num_verts(scene):
    """
    Returns the total number of verts in the scene.

    scene: Result of ``reconstruct_scene``.
    return: Total number of verts.
    """
    return sum(len(pts) for pts in scene.pts3d)


def iterate_verts(scene):
    """
    Yields vert location and color for all verts of all views.

    scene: Result of ``reconstruct_scene``.
    return: Iterable of (vert, color) tuples.
    """
    for i in range(len(scene.pts3d)):
        for j in range(scene.pts3d[i].shape[0]):
            yield scene.pts3d[i][j], scene.pts3d_colors[i][j]
