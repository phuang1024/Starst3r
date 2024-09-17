__all__ = (
    "interp_se3",
    "interp_se3_path",
)

import torch


def lerp(a, b, fac):
    return a + (b - a) * fac


def interp_se3(mat1: torch.Tensor, mat2: torch.Tensor, fac: float) -> torch.Tensor:
    """Interpolate between two SE3 matrices.

    Linear interpolation of SO3 and translation components.
    Normalizes and orthogonalizes SO3.

    Parameters
    ----------

    mat1:
        SE3 matrix 1, shape (4,4).

    mat2:
        SE3 matrix 2, shape (4,4).

    fac:
        Interpolation factor, 0 to 1.

    Returns
    -------

    SE3 matrix, shape (4,4).
    """
    ret = torch.zeros_like(mat1)
    ret[3, 3] = 1

    trans = lerp(mat1[:3, 3], mat2[:3, 3], fac)
    ret[:3, 3] = trans

    so3 = lerp(mat1[:3, :3], mat2[:3, :3], fac)

    # Orthogonalize via x -= proj(x, v)
    so3[:, 1] -= so3[:, 0] * so3[:, 0].dot(so3[:, 1])
    so3[:, 2] -= so3[:, 0] * so3[:, 0].dot(so3[:, 2])
    so3[:, 2] -= so3[:, 1] * so3[:, 1].dot(so3[:, 2])

    # Normalize
    so3 /= torch.linalg.norm(so3, dim=0)

    ret[:3, :3] = so3

    return ret


def interp_se3_path(mat1: torch.Tensor, mat2: torch.Tensor, steps: int) -> torch.Tensor:
    """Linear interp between two SE3 matrices with linearly increasing factor.

    Parameters
    ----------

    mat1:
        SE3 matrix 1, shape (4,4).

    mat2:
        SE3 matrix 2, shape (4,4).

    steps:
        Number of interpolation steps.

    Returns
    -------

    List of SE3 matrices, shape (steps, 4, 4).
    """
    facs = torch.linspace(0, 1, steps)
    return torch.stack([interp_se3(mat1, mat2, fac) for fac in facs], dim=0)
