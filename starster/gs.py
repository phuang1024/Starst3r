"""
Gaussian splatting integration with gsplat.
"""

import gsplat

from .scene import PointCloud


class GSTrainer:
    def __init__(self, scene: PointCloud):
        self.scene = scene
        self.gaussians = {}

        self.init_gaussians()

    def init_gaussians(self):
        pass
