"""
Forward engine: eikonal travel-time solver with Fréchet derivatives.

Solves |∇T|² = s²(x) via fast marching (scikit-fmm), then computes
source and velocity Fréchet derivatives for the inverse problem.
"""

import numpy as np
from scipy import sparse
from typing import List, Tuple, Dict

try:
    import skfmm
except ImportError:
    raise ImportError("scikit-fmm required: pip install scikit-fmm")

from .data import VelocityModel


class EikonalSolver:
    """
    Fast-marching eikonal solver with sub-cell source positioning.

    Key design choices:
    - Signed-distance level-set initialization for sub-cell accuracy
    - Trilinear interpolation for receiver travel times
    - Centered finite-difference source Fréchet derivatives
    - Ray-tracing velocity Fréchet kernels
    """

    def __init__(self, velocity_model: VelocityModel):
        self.vm = velocity_model

    def compute_travel_time_field(
        self, source_xyz: np.ndarray, phase: str
    ) -> np.ndarray:
        """
        Full travel-time field T(x) from a point source.
        Uses signed-distance level set for sub-cell source positioning.
        """
        slow = self.vm.slowness(phase)
        dx = self.vm.spacing

        x, y, z = self._grid_coordinates_3d()
        dist = np.sqrt(
            (x - source_xyz[0]) ** 2 +
            (y - source_xyz[1]) ** 2 +
            (z - source_xyz[2]) ** 2
        )
        radius = 1.01 * float(np.max(dx))
        phi = dist - radius

        # Safety: if source is near grid edge, ensure zero contour exists
        if np.min(phi) >= 0:
            # Source is outside grid or exactly on boundary — fall back to nearest cell
            si = np.round((source_xyz - self.vm.origin) / self.vm.spacing).astype(int)
            si = np.clip(si, 1, np.array(self.vm.shape) - 2)
            phi = np.ones(self.vm.shape)
            phi[si[0], si[1], si[2]] = -1.0

        T = skfmm.travel_time(phi, speed=1.0 / slow, dx=dx)
        return np.asarray(T)

    def travel_times_at_receivers(
        self, source_xyz: np.ndarray,
        receiver_xyz: np.ndarray, phase: str
    ) -> np.ndarray:
        """Travel times from source to receivers via trilinear interpolation."""
        T_field = self.compute_travel_time_field(source_xyz, phase)
        tt = np.zeros(len(receiver_xyz))
        for i, rx in enumerate(receiver_xyz):
            tt[i] = self._interpolate_field(T_field, rx)
        return tt

    def frechet_source(
        self, source_xyz: np.ndarray,
        receiver_xyz: np.ndarray, phase: str,
        dx_perturb: float = None
    ) -> np.ndarray:
        """
        ∂T/∂x_s via centered finite differences.
        Perturbation defaults to 0.5 × grid spacing.

        Returns: G_s (n_receivers, 3)
        """
        if dx_perturb is None:
            dx_perturb = 0.5 * float(np.min(self.vm.spacing))

        n_rec = len(receiver_xyz)
        G_s = np.zeros((n_rec, 3))

        for j in range(3):
            src_plus = source_xyz.copy()
            src_minus = source_xyz.copy()
            src_plus[j] += dx_perturb
            src_minus[j] -= dx_perturb

            tt_p = self.travel_times_at_receivers(src_plus, receiver_xyz, phase)
            tt_m = self.travel_times_at_receivers(src_minus, receiver_xyz, phase)
            G_s[:, j] = (tt_p - tt_m) / (2.0 * dx_perturb)

        return G_s

    def frechet_model(
        self, source_xyz: np.ndarray,
        receiver_xyz: np.ndarray, phase: str
    ) -> sparse.csr_matrix:
        """
        Velocity Fréchet kernel via ray-density approximation.
        δt_i = -∫_{ray_i} (δv / v²) ds

        Returns: G_m (n_receivers, n_model_params) sparse
        """
        T_field = self.compute_travel_time_field(source_xyz, phase)
        slow = self.vm.slowness(phase)
        grad_T = np.gradient(T_field, *self.vm.spacing)

        n_rec = len(receiver_xyz)
        n_model = self.vm.n_cells
        rows, cols, vals = [], [], []

        for i, rx in enumerate(receiver_xyz):
            ray_cells = self._trace_ray(T_field, grad_T, rx, source_xyz)
            for cell_idx, seg_len in ray_cells:
                flat_idx = np.ravel_multi_index(cell_idx, self.vm.shape)
                v_cell = 1.0 / slow[cell_idx[0], cell_idx[1], cell_idx[2]]
                rows.append(i)
                cols.append(flat_idx)
                vals.append(-seg_len / v_cell ** 2)

        return sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_rec, n_model)
        )

    # --- Internal methods ---

    def _grid_coordinates_3d(self):
        x_1d, y_1d, z_1d = self.vm.grid_coordinates()
        return np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

    def _xyz_to_index(self, xyz: np.ndarray) -> Tuple[int, int, int]:
        idx = np.round(
            (xyz - self.vm.origin) / self.vm.spacing
        ).astype(int)
        idx = np.clip(idx, 0, np.array(self.vm.shape) - 1)
        return tuple(idx)

    def _interpolate_field(self, field: np.ndarray, xyz: np.ndarray) -> float:
        """Trilinear interpolation of a 3D field."""
        frac = (xyz - self.vm.origin) / self.vm.spacing
        frac = np.clip(frac, 0, np.array(self.vm.shape) - 1.001)

        i0 = frac.astype(int)
        i1 = np.minimum(i0 + 1, np.array(self.vm.shape) - 1)
        w = frac - i0

        c00 = field[i0[0], i0[1], i0[2]] * (1 - w[0]) + field[i1[0], i0[1], i0[2]] * w[0]
        c01 = field[i0[0], i0[1], i1[2]] * (1 - w[0]) + field[i1[0], i0[1], i1[2]] * w[0]
        c10 = field[i0[0], i1[1], i0[2]] * (1 - w[0]) + field[i1[0], i1[1], i0[2]] * w[0]
        c11 = field[i0[0], i1[1], i1[2]] * (1 - w[0]) + field[i1[0], i1[1], i1[2]] * w[0]
        c0 = c00 * (1 - w[1]) + c10 * w[1]
        c1 = c01 * (1 - w[1]) + c11 * w[1]
        return c0 * (1 - w[2]) + c1 * w[2]

    def _trace_ray(
        self, T_field, grad_T, receiver_xyz, source_xyz,
        step_size=None, max_steps=10000
    ) -> List[Tuple[tuple, float]]:
        """Trace ray from receiver to source following -∇T."""
        if step_size is None:
            step_size = 0.5 * float(np.min(self.vm.spacing))

        pos = np.array(receiver_xyz, dtype=np.float64)
        src = np.array(source_xyz, dtype=np.float64)
        cells: Dict[tuple, float] = {}

        for _ in range(max_steps):
            idx = self._xyz_to_index(pos)
            gx = grad_T[0][idx[0], idx[1], idx[2]]
            gy = grad_T[1][idx[0], idx[1], idx[2]]
            gz = grad_T[2][idx[0], idx[1], idx[2]]
            g = np.array([gx, gy, gz])
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-12:
                break

            pos += step_size * (-g / g_norm)
            cells[idx] = cells.get(idx, 0.0) + step_size

            if np.linalg.norm(pos - src) < step_size * 2:
                break

        return list(cells.items())
