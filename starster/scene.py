import torch


class PointCloud:
    """
    Reconstructed point cloud.

    Wrapper around Mast3r SparseGA.
    """

    def __init__(self, sparse_ga):
        self.sparse_ga = sparse_ga
        self.num_cams = len(self.sparse_ga.pts3d)

    def pts_sparse(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns (pts3d, colors) for each camera.
        """
        ret = []
        for i in range(self.num_cams):
            ret.append((self.sparse_ga.pts3d[i], self.sparse_ga.pts3d_colors[i]))
        return ret

    def pts_sparse_flat(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns concatenated pts3d and pts3d_colors.
        """
        pts = self.pts_sparse()
        pts3d = torch.cat([p[0] for p in pts], dim=0)
        colors = torch.cat([p[1] for p in pts], dim=0)
        return pts3d, colors

    def pts_dense(self, conf_thres=1.5) -> list[tuple[torch.Tensor, torch.Tensor]]:
        ret = []
        pts, _, confs = self.sparse_ga.get_dense_pts3d(clean_depth=True)
        for i in range(self.num_cams):
            colors = torch.tensor(self.sparse_ga.imgs[i]).reshape(-1, 3)
            mask = (confs[i] > conf_thres).reshape(-1).cpu()
            ret.append((pts[i][mask], colors[mask]))
        return ret

    def pts_dense_flat(self, conf_thres=1.5) -> tuple[torch.Tensor, torch.Tensor]:
        pts = self.pts_dense(conf_thres)
        pts3d = torch.cat([p[0] for p in pts], dim=0)
        colors = torch.cat([p[1] for p in pts], dim=0)
        return pts3d, colors

    def c2w(self) -> torch.Tensor:
        """
        Returns camera-to-world transformation matrix.

        Shape (C, 4, 4).

        Alias of ``self.sparse_ga.cam2w``.
        """
        return self.sparse_ga.cam2w

    def w2c(self) -> torch.Tensor:
        """
        Returns world-to-camera transformation matrix.

        Shape (C, 4, 4).

        Alias of ``self.sparse_ga.w2cam``.
        """
        return torch.inverse(self.c2w())

    def intrinsics(self) -> torch.Tensor:
        """
        Returns camera intrinsics.

        Shape (C, 3, 3).

        Alias of ``self.sparse_ga.intrinsics``.
        """
        return self.sparse_ga.intrinsics
