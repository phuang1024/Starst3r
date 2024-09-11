"""
Reconstructed point cloud wrapper class.
"""

__all__ = (
    "PointCloudScene",
)

import torch


class PointCloudScene:
    """Reconstructed point cloud; output of Mast3r pipeline.

    Use :func:`starster.reconstruct_scene` to get an instance of this class.

    This class has some wrapper functions around Mast3r SparseGA,
    as well as novel functions.
    """

    num_cams: int

    def __init__(self, sparse_ga):
        """
        Parameters
        ----------

        sparse_ga:
            SparseGA instance from Mast3r code.
        """
        self.sparse_ga = sparse_ga
        self.num_cams = len(self.sparse_ga.pts3d)

    @property
    def imgs(self) -> list[torch.Tensor]:
        """Get source image tensors.

        Alias of ``self.sparse_ga.imgs``.

        Returns
        -------

        List of tensors for each camera.

        Each tensor is shape (H, W, 3).
        """
        return self.sparse_ga.imgs

    def pts_sparse(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get reconstructed 3D point coordinates and colors.

        These four functions (pts_sparse, pts_sparse_flat, pts_dense, pts_dense_flat)
        behave similarly.

        Sparse and dense return their respective point clouds.

        Flat returns a concatenated point cloud from all cameras together.

        Non-flat returns a list of point clouds, one for each camera.

        Each point cloud is returned as ``(pts3d, colors)``:

        - pts3d: 3D point coordinates. Shape (N, 3).
        - colors: RGB colors for each point. Shape (N, 3).
        """
        ret = []
        for i in range(self.num_cams):
            ret.append((self.sparse_ga.pts3d[i], self.sparse_ga.pts3d_colors[i]))
        return ret

    def pts_sparse_flat(self) -> tuple[torch.Tensor, torch.Tensor]:
        """See ``pts_sparse``."""
        pts = self.pts_sparse()
        pts3d = torch.cat([p[0] for p in pts], dim=0)
        colors = torch.cat([p[1] for p in pts], dim=0)
        return pts3d, colors

    def pts_dense(self, conf_thres=1.5) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """See ``pts_sparse``."""
        ret = []
        pts, _, confs = self.sparse_ga.get_dense_pts3d(clean_depth=True)
        for i in range(self.num_cams):
            colors = torch.tensor(self.sparse_ga.imgs[i]).reshape(-1, 3)
            mask = (confs[i] > conf_thres).reshape(-1).cpu()
            ret.append((pts[i][mask], colors[mask]))
        return ret

    def pts_dense_flat(self, conf_thres=1.5) -> tuple[torch.Tensor, torch.Tensor]:
        """See ``pts_sparse``."""
        pts = self.pts_dense(conf_thres)
        pts3d = torch.cat([p[0] for p in pts], dim=0)
        colors = torch.cat([p[1] for p in pts], dim=0)
        return pts3d, colors

    def c2w(self) -> torch.Tensor:
        """Returns camera-to-world transformation matrices, shape (C, 4, 4).

        Alias of ``self.sparse_ga.cam2w``.
        """
        return self.sparse_ga.cam2w

    def w2c(self) -> torch.Tensor:
        """Returns world-to-camera transformation matrix (inverse of ``c2w``)."""
        return torch.inverse(self.c2w())

    def intrinsics(self) -> torch.Tensor:
        """Returns camera intrinsic matrices, shape (C, 3, 3).

        Alias of ``self.sparse_ga.intrinsics``.
        """
        return self.sparse_ga.intrinsics
