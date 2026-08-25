"""
Structured (r, z) mesh with hyperbolic-tangent stretching.

Provides cell centers, face positions, volumes, and areas for
axisymmetric cylindrical geometry on a (r, z) domain.

Adapted from Stage 10 simulation package.
"""
import numpy as np


class Mesh2D:
    """Axisymmetric (r, z) structured mesh with optional stretching.

    Attributes
    ----------
    Nr, Nz : int
        Number of interior cells in r and z.
    R, L : float
        Domain extents (m).
    rc, zc : 1D arrays (Nr,), (Nz,)
        Cell-center coordinates.
    rf, zf : 1D arrays (Nr+1,), (Nz+1,)
        Face coordinates (rf[0] = 0, rf[-1] = R, etc.).
    dr, dz : 1D arrays (Nr,), (Nz,)
        Cell widths.
    drc, dzc : 1D arrays (Nr-1,), (Nz-1,)
        Distances between adjacent cell centers.
    vol : 2D array (Nr, Nz)
        Cell volumes = 2*pi * r_c * dr * dz.
    """

    def __init__(self, R, L, Nr, Nz, beta_r=1.3, beta_z=1.3):
        self.R, self.L = R, L
        self.Nr, self.Nz = Nr, Nz

        # Face coordinates with stretching
        self.rf = self._stretched_faces(R, Nr, beta_r, two_sided=False)
        self.zf = self._stretched_faces(L, Nz, beta_z, two_sided=True)

        # Cell centers
        self.rc = 0.5 * (self.rf[:-1] + self.rf[1:])
        self.zc = 0.5 * (self.zf[:-1] + self.zf[1:])

        # Cell widths
        self.dr = np.diff(self.rf)
        self.dz = np.diff(self.zf)

        # Distances between adjacent cell centers
        self.drc = np.diff(self.rc)
        self.dzc = np.diff(self.zc)

        # Cell volumes: V = 2*pi * r * dr * dz  (vectorised)
        self.vol = (2.0 * np.pi * self.rc[:, None]
                    * self.dr[:, None] * self.dz[None, :])

    @staticmethod
    def _stretched_faces(L, N, beta, two_sided=False):
        """Generate stretched face coordinates using tanh distribution.

        beta = 1.0: uniform
        beta > 1.0: clustered near boundaries
        """
        xi = np.linspace(0, 1, N + 1)
        if beta <= 1.0 + 1e-10:
            return L * xi
        if two_sided:
            # Stretch toward both z=0 and z=L
            return L * 0.5 * (1.0 + np.tanh(beta * (2 * xi - 1)) / np.tanh(beta))
        else:
            # Stretch toward r=R only (r=0 is axis, needs less resolution)
            return L * (np.tanh(beta * xi) / np.tanh(beta))

    def total_volume(self):
        return np.sum(self.vol)

    def volume_average(self, field):
        return np.sum(field * self.vol) / np.sum(self.vol)

    def radial_at_z(self, field, z_idx):
        """Extract radial profile at a given z index."""
        return self.rc, field[:, z_idx]

    def axial_at_r(self, field, r_idx):
        """Extract axial profile at a given r index."""
        return self.zc, field[r_idx, :]

    def radial_at_wafer(self, field):
        """Extract radial profile at z=0 (wafer, j=0)."""
        return self.rc, field[:, 0]

    def radial_at_midplane(self, field):
        """Extract radial profile at z=L/2."""
        j_mid = self.Nz // 2
        return self.rc, field[:, j_mid]

    def __repr__(self):
        return (f"Mesh2D({self.Nr}x{self.Nz}, R={self.R:.3f}m, L={self.L:.3f}m, "
                f"dr_min={self.dr.min()*1e3:.2f}mm, dz_min={self.dz.min()*1e3:.2f}mm)")
