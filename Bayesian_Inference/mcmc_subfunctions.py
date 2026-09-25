# mcmc_subfunctions.py
#
# Support functions for the MCMC refactor:
#   - Gaussian process upper-envelope background fit.
#   - Softmax (reference-anchored) <-> phase-fraction transforms.
#   - Initial walker generator that samples unnormalized softmax
#     weights uniformly within their implied range and rejects out-of-
#     box draws.
#   - Restart-position sampler that rejects out-of-bounds proposals.

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.signal import find_peaks


# =========================================================================
# GAUSSIAN PROCESS BACKGROUND
# =========================================================================
def Gaussian_Process_upper_envelope(x_obs, I_obs, n_iter=6,
                                    peak_prominence=2e3,
                                    rbf_length_scale=500.0,
                                    rbf_length_bounds=(64, 1e5),
                                    alpha=1e-6,
                                    edge_buffer=2):
    """Iteratively fit a GP to the log-intensity, keeping only points
    the fit overshoots, to estimate the upper envelope of the background."""
    eps = 1e-12
    I_obs_safe = np.maximum(I_obs, eps)
    log_I = np.log(I_obs_safe)

    peaks, _ = find_peaks(I_obs, prominence=peak_prominence)
    mask_init = np.ones_like(I_obs, dtype=bool)
    for p in peaks:
        mask_init[max(0, p - 4):min(len(mask_init), p + 4)] = False

    X_full = x_obs.reshape(-1, 1)

    orig_idx = np.where(mask_init)[0]
    X = x_obs[orig_idx].reshape(-1, 1)
    y = log_I[orig_idx]

    n_iter_safe = int(np.floor(np.log2(len(y))))
    n_iter = min(n_iter, n_iter_safe)

    amp = np.var(y)
    kernel = (
        ConstantKernel(amp, (1e-3 * amp, 1e3 * amp)) *
        RBF(length_scale=rbf_length_scale, length_scale_bounds=rbf_length_bounds)
    )

    gp = None
    for it in range(n_iter):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=False,
            optimizer=None,
        )
        gp.fit(X, y)
        y_pred = gp.predict(X)

        valid_idx = np.arange(len(y))
        interior = (valid_idx >= edge_buffer) & (valid_idx < len(y) - edge_buffer)

        keep = np.ones_like(y, dtype=bool)
        keep[interior] = y_pred[interior] >= y[interior]
        if np.all(keep):
            break

        X = X[keep]
        y = y[keep]
        orig_idx = orig_idx[keep]

    gp.fit(X, y)
    GP_log, GP_log_std = gp.predict(X_full, return_std=True)

    GP_pred = np.exp(GP_log)
    GP_pred_avg = np.exp(GP_log + 0.5 * GP_log_std ** 2)
    GP_std = np.sqrt(
        (np.exp(GP_log_std ** 2) - 1.0) *
        np.exp(2 * GP_log + GP_log_std ** 2)
    )
    GP_pred_int = np.sum(GP_pred)

    diff = np.min(I_obs - GP_pred)
    GP_pred = GP_pred+diff
    return GP_pred_avg, GP_std, GP_pred, GP_pred_int


# =========================================================================
# REFERENCE-ANCHORED SOFTMAX
# =========================================================================
# Convention:
#   z is a length-N vector with z[ref] = 0 (the anchor).
#   Phase fractions are f_i = simplex_sum * exp(z_i) / sum_j exp(z_j).
#   Walker carries the N-1 non-reference z values only.
#
# z_full_to_fracs: takes a full-length (..., N) z array and returns
#                  fractions; useful internally.
# fracs_to_z_full: inverse; assumes f_ref > 0.
# z_active_to_full / z_full_to_active: pack/unpack the anchor.

def z_full_to_fracs(z, simplex_sum=1.0):
    """z shape (..., N). Returns f shape (..., N) summing to simplex_sum."""
    z = np.asarray(z, dtype=float)
    # Numerically stable softmax: subtract the row max before exponentiating.
    z_shift = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z_shift)
    return simplex_sum * e / np.sum(e, axis=-1, keepdims=True)


def fracs_to_z_full(fracs, ref_index=0, simplex_sum=1.0):
    """fracs shape (..., N). Returns z shape (..., N) with z[..., ref_index] = 0."""
    fracs = np.asarray(fracs, dtype=float)
    if np.any(fracs <= 0):
        raise ValueError("All fractions must be strictly positive.")
    f_ref = fracs[..., ref_index : ref_index + 1]
    return np.log(fracs / f_ref)


def z_active_to_full(z_active, ref_index, N):
    """Insert a zero at ref_index. z_active shape (..., N-1) -> (..., N)."""
    z_active = np.asarray(z_active, dtype=float)
    leading = z_active.shape[:-1]
    z_full = np.empty(leading + (N,), dtype=float)
    # Indices in the full vector that come from z_active.
    other = [i for i in range(N) if i != ref_index]
    z_full[..., ref_index] = 0.0
    z_full[..., other] = z_active
    return z_full


def z_full_to_active(z_full, ref_index):
    """Drop the ref_index column."""
    z_full = np.asarray(z_full, dtype=float)
    N = z_full.shape[-1]
    other = [i for i in range(N) if i != ref_index]
    return z_full[..., other]


# =========================================================================
# WALKER INITIALIZATION
# =========================================================================
def generate_initial_softmax_weights(nwalkers, N, scale_bounds, ref_index,
                                     simplex_sum=1.0, max_iter=10000):
    """
    Draw initial phase fractions by sampling unnormalized softmax
    weights w_i = exp(z_i) uniformly inside their implied bounds,
    then computing the resulting fractions and rejecting any draws
    whose fractions fall outside the per-phase box.

    Implied weight bounds (with w_ref = 1):
        w_i ∈ [low_i / high_ref, high_i / low_ref]   for i != ref_index

    Parameters
    ----------
    nwalkers : int
    N : int
    scale_bounds : (N, 2) array of (low, high) per phase fraction.
    ref_index : int
    simplex_sum : float

    Returns
    -------
    fracs : (nwalkers, N) phase fractions
    """
    scale_bounds = np.asarray(scale_bounds, dtype=float)
    if scale_bounds.shape != (N, 2):
        raise ValueError(
            f"scale_bounds shape mismatch: expected ({N}, 2), "
            f"got {scale_bounds.shape}."
        )
    lows  = scale_bounds[:, 0]
    highs = scale_bounds[:, 1]

    if lows[ref_index] <= 0.0:
        raise ValueError(
            f"Reference phase (index {ref_index}) must have a strictly "
            f"positive lower bound on its scale fraction. Got {lows[ref_index]}."
        )

    other_idx = [i for i in range(N) if i != ref_index]

    # Implied weight bounds.
    w_low  = lows[other_idx]  / highs[ref_index]
    w_high = highs[other_idx] / lows[ref_index]
    # Weights must be strictly positive (lows can be zero; clamp).
    w_low = np.maximum(w_low, 0.0)

    samples = []
    n_collected = 0
    n_attempts = 0
    while n_collected < nwalkers:
        n_needed = nwalkers - n_collected
        # Draw uniform weights in [w_low, w_high].
        w_other = (
            w_low + (w_high - w_low) * np.random.rand(n_needed, N - 1)
        )
        # Reference weight is 1 by construction.
        w_full = np.ones((n_needed, N), dtype=float)
        w_full[:, other_idx] = w_other
        # Convert to fractions.
        f = simplex_sum * w_full / np.sum(w_full, axis=1, keepdims=True)
        # Reject any fraction outside its box.
        valid = np.all((f >= lows) & (f <= highs), axis=1)
        samples.append(f[valid])
        n_collected += int(np.sum(valid))
        n_attempts += n_needed
        if n_attempts > max_iter * nwalkers:
            raise RuntimeError(
                "Softmax-weight rejection sampling failed; feasible "
                "region may be too small or bounds inconsistent."
            )

    fracs = np.vstack(samples)[:nwalkers]
    assert fracs.shape == (nwalkers, N)
    return fracs


def sample_uniform_in_box(nwalkers, bounds):
    """Uniform draws inside (N, 2) per-dimension bounds. Returns (nwalkers, N)."""
    bounds = np.asarray(bounds, dtype=float)
    lows  = bounds[:, 0]
    highs = bounds[:, 1]
    return lows + (highs - lows) * np.random.rand(nwalkers, bounds.shape[0])


def sample_valid_initial_positions(best_mean, perturb, nwalkers, ndim,
                                   lows, highs, max_iter=100000):
    """Reject-sample Gaussian perturbations until they all fall inside
    [lows, highs]. Used when warm-restarting from a previous chain."""
    pos = np.zeros((nwalkers, ndim))
    i = 0
    attempts = 0
    while i < nwalkers:
        trial = best_mean + perturb * np.random.randn(ndim)
        if np.all(trial >= lows) and np.all(trial <= highs):
            pos[i] = trial
            i += 1
        attempts += 1
        if attempts > max_iter:
            raise RuntimeError(
                "Restart sampler could not place all walkers inside "
                "bounds; check perturbation scale and prior box."
            )
    return pos
