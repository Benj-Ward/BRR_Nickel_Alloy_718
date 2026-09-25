# surrogate_variance.py
#
# Per-phase surrogate variance lookup for MCMC likelihood evaluation.
#
# Data layout (matches train_surrogate_v3.py output):
#   <run_dir>/
#       <phase>/
#           variance.npz   <- mustrain_edges, hstrain_edges, variance_grid
#           final_model.pkl
#       <phase>/
#           variance.npz
#           ...
#
# The variance grid is (n_mu, n_hs, n_2theta). We bilinearly interpolate
# between bin *centers* given a query (mustrain, hstrain). Queries
# outside the trained range are clamped to the boundary bin.
#
# Primary API is vectorized: sigma_sm_squared_batch takes 1D arrays of
# (mustrain, hstrain) — one entry per walker — and returns a
# (B, n_2theta) array in a single fully vectorized call. The scalar
# convenience method is retained for diagnostic use only.
#
# Usage:
#
#   from surrogate_variance import load_phase_variances
#
#   lookups = load_phase_variances("path/to/run_dir", ["gamma", "delta", ...])
#   # lookups["gamma"].sigma_sm_squared_batch(mu_array, hs_array)
#   #     -> (B, n_2theta)

import os
import numpy as np


class VarianceLookup:
    """One phase's variance table with vectorized query support."""

    def __init__(self, mustrain_edges, hstrain_edges, variance_grid,
                 variance_global=None, phase=None):
        """
        mustrain_edges : (n_mu + 1,)
        hstrain_edges  : (n_hs + 1,)
        variance_grid  : (n_mu, n_hs, n_2theta)
        variance_global : (n_2theta,) optional fallback (unused in the
                          interpolation path; kept for diagnostic access).
        phase : str, name of the phase this table belongs to.
        """
        self.mustrain_edges = np.asarray(mustrain_edges)
        self.hstrain_edges  = np.asarray(hstrain_edges)
        self.variance_grid  = np.asarray(variance_grid)
        self.variance_global = (
            np.asarray(variance_global) if variance_global is not None else None
        )
        self.phase = phase

        # Bin centers — the interpolation treats each bin's variance as
        # the value at its center.
        self.mu_centers = 0.5 * (self.mustrain_edges[:-1] + self.mustrain_edges[1:])
        self.hs_centers = 0.5 * (self.hstrain_edges[:-1]  + self.hstrain_edges[1:])
        self.n_mu     = len(self.mu_centers)
        self.n_hs     = len(self.hs_centers)
        self.n_2theta = self.variance_grid.shape[2]

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------
    @classmethod
    def load(cls, npz_path, phase=None):
        """Load a single phase's variance.npz."""
        data = np.load(npz_path)
        return cls(
            mustrain_edges  = data["mustrain_edges"],
            hstrain_edges   = data["hstrain_edges"],
            variance_grid   = data["variance_grid"],
            variance_global = data["variance_global"] if "variance_global" in data.files else None,
            phase = phase,
        )

    # -----------------------------------------------------------------
    # Vectorized bilinear interpolation (primary API)
    # -----------------------------------------------------------------
    def sigma_sm_squared_batch(self, mustrain, hstrain):
        """
        Vectorized bilinear interpolation.

        Parameters
        ----------
        mustrain : (B,) array of mustrain values, one per walker.
        hstrain  : (B,) array of hstrain  values, one per walker.

        Returns
        -------
        (B, n_2theta) array of variances. Queries beyond the outermost
        bin centers are clamped to the edge bin (no extrapolation).
        """
        mustrain = np.asarray(mustrain, dtype=float)
        hstrain  = np.asarray(hstrain,  dtype=float)
        if mustrain.shape != hstrain.shape:
            raise ValueError(
                f"mustrain shape {mustrain.shape} != hstrain shape "
                f"{hstrain.shape}"
            )

        i0, i1, t_mu = self._bracket_vec(mustrain, self.mu_centers)
        j0, j1, t_hs = self._bracket_vec(hstrain,  self.hs_centers)

        # Each gather: (B, n_2theta).
        v00 = self.variance_grid[i0, j0]
        v10 = self.variance_grid[i1, j0]
        v01 = self.variance_grid[i0, j1]
        v11 = self.variance_grid[i1, j1]

        w_mu = t_mu[:, None]
        w_hs = t_hs[:, None]
        return (
            (1 - w_mu) * (1 - w_hs) * v00
            + w_mu     * (1 - w_hs) * v10
            + (1 - w_mu) * w_hs     * v01
            + w_mu     * w_hs       * v11
        )

    # -----------------------------------------------------------------
    # Scalar convenience (diagnostic use)
    # -----------------------------------------------------------------
    def sigma_sm_squared(self, mustrain, hstrain):
        """Scalar bilinear interpolation. Returns (n_2theta,)."""
        out = self.sigma_sm_squared_batch(
            np.array([mustrain]), np.array([hstrain])
        )
        return out[0]

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    @staticmethod
    def _bracket_vec(values, centers):
        """Locate bracketing bin-center indices for each value.

        Returns (i0, i1, t) such that, for values inside
        [centers[0], centers[-1]],
            value = (1 - t) * centers[i0] + t * centers[i1]
        with i1 = i0 + 1. Out-of-range values are clamped to the
        nearest edge bin (i0 == i1, t = 0) so the interpolation
        evaluates to the boundary variance exactly.
        """
        n = len(centers)
        clamped = np.clip(values, centers[0], centers[-1])

        i1 = np.searchsorted(centers, clamped)
        i1 = np.clip(i1, 1, n - 1)
        i0 = i1 - 1

        denom = centers[i1] - centers[i0]
        t = (clamped - centers[i0]) / denom

        # Collapse i0 and i1 onto the same edge for out-of-range queries
        # so that all four bilinear corners coincide on the boundary
        # bin's value.
        below = values <= centers[0]
        above = values >= centers[-1]
        if np.any(below):
            i0 = np.where(below, 0, i0)
            i1 = np.where(below, 0, i1)
            t  = np.where(below, 0.0, t)
        if np.any(above):
            i0 = np.where(above, n - 1, i0)
            i1 = np.where(above, n - 1, i1)
            t  = np.where(above, 0.0, t)

        return i0, i1, t


# =========================================================================
# CONVENIENCE LOADERS
# =========================================================================
def load_phase_variances(run_dir, phase_names):
    """Load <run_dir>/<phase>/variance.npz for every requested phase.

    Parameters
    ----------
    run_dir : str
        Path to the surrogate run directory (the one that contains
        a subfolder per phase).
    phase_names : iterable of str
        Phases to load. Order is preserved in the returned dict.

    Returns
    -------
    dict {phase: VarianceLookup}
    """
    lookups = {}
    for phase in phase_names:
        path = os.path.join(run_dir, phase, "variance.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing variance file: {path}")
        lookups[phase] = VarianceLookup.load(path, phase=phase)
    return lookups
