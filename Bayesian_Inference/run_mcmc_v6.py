# run_mcmc_v4.py
#
# Refactored MCMC driver for phase-fraction / mustrain / hstrain
# inference against a diffraction pattern.
#
# v4 changes
# ----------
#   * Optional pseudo-marginal mode. Selected via CONFIG["mode"]:
#       - "direct" (default): unchanged from v3. Phase fractions are
#         walker coordinates via softmax-with-reference-anchor.
#       - "pseudo_marginal": walker carries (log_alpha_0, mu, mustrain,
#         hstrain). Phase fractions are marginalized via an inner Monte
#         Carlo over S ~ Dir(alpha_0 * mu); the resulting log-marginal-
#         likelihood is aggregated by log-sum-exp. The per-sample
#         likelihood still uses the same combined variance model and
#         analytic s* Laplace marginalization as direct mode.
#     All v3 behavior is preserved bit-for-bit when mode == "direct".
#
# v3 changes
# ----------
#   * Configuration now lives in a dedicated config.py (default next to
#     this script, overridable with --config) rather than an inline block.
#   * The Gaussian-process background is no longer fit here. It is fit and
#     verified ahead of time by fit_gp.py, which writes gp_fit.npz. This
#     driver loads that file and validates it against the current config
#     (keep_idx and a gp-config hash) before sampling. Run fit_gp.py first.
#
# Parameterization (direct mode)
# ------------------------------
# Walker vector has length 3N - 1, all dimensions approximately N(0,1):
#     [ z_std_active   (N-1 phase-fraction dims, reference-anchored softmax)
#     | mu_std         (N mustrain dims)
#     | h_std          (N hstrain dims) ]
#
# Phase fractions:
#     f_i = simplex_sum * exp(z_i) / sum_j exp(z_j),     z[ref_index] = 0.
#
# Parameterization (pseudo-marginal mode)
# ---------------------------------------
# Walker vector has length 3N, all dimensions approximately N(0,1):
#     [ log_alpha0_std (1)
#     | z_std_active   (N-1 mu dims, reference-anchored softmax, simplex_sum=1)
#     | mu_std         (N mustrain dims)
#     | h_std          (N hstrain dims) ]
#
# The mu component lives on the (N-1)-simplex via the same softmax
# functions as direct mode, but with neutral z_mu=0, z_sigma=1 (no scale
# bounds enter, since mu is unconstrained on the simplex).
#
# Standardization (z, mustrain, hstrain) uses per-dimension affine maps
# built deterministically from the box bounds in CONFIG. The exact maps
# are written to output/<run_name>/standardizer.json so post-processing
# code can reproduce the transform without re-deriving it.
#
# Histogram scale factor s is analytically marginalized via a Laplace
# approximation around s*, with a flat improper prior and s* > 0
# enforced (negative s* -> log_prob = -inf).
#
# Surrogate is Y_calc_i = f_i * g_i(m_i, h_i). The phase-fraction
# multiplication is applied outside the surrogate so linearity in f is
# exact by construction.

import os
import json
import time
import pickle
import hashlib
import argparse
import platform
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import emcee
from scipy.special import logsumexp, gammaln, digamma, polygamma

from mcmc_subfunctions import (
    z_full_to_fracs,
    fracs_to_z_full,
    z_active_to_full,
    z_full_to_active,
    generate_initial_softmax_weights,
    sample_uniform_in_box,
    sample_valid_initial_positions,
)
from surrogate_variance import load_phase_variances

# GP is fit upstream by fit_gp.py; reuse its config/data helpers here so
# the data-range mask and config loading are implemented in exactly one
# place.
from fit_gp import (
    load_config,
    load_observed_pattern,
    build_keep_idx,
    gp_config_hash,
)


# =========================================================================
# CONFIGURATION
# =========================================================================
# Configuration is read from a dedicated config.py (see --config). The GP
# background is fit separately by fit_gp.py.


# =========================================================================
# OUTPUT DIRECTORIES + METADATA
# =========================================================================
def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def setup_outputs(config):
    out_dir = Path("output") / config["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_hash": _git_hash(),
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:12],
        "numpy_version": np.__version__,
        "emcee_version": emcee.__version__,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return out_dir




# =========================================================================
# NOISE BOOST CONFIGURATION
# =========================================================================
def parse_noise_boost_config(config):
    """
    Normalize optional likelihood.noise_boost settings.

    Variance multiplier is exp(eta).  A uniform prior in eta over a finite
    interval is the bounded Jeffreys prior for this multiplicative variance
    scale.  Missing config preserves the original likelihood exactly:
    fixed eta = 0 -> exp(eta) = 1.
    """
    lik = config.get("likelihood", {})
    raw = lik.get("noise_boost", {}) or {}
    mode = raw.get("mode", "fixed")
    if mode not in ("fixed", "mcmc"):
        raise ValueError(
            f"likelihood.noise_boost.mode must be 'fixed' or 'mcmc'; got {mode!r}."
        )

    if mode == "fixed":
        eta = float(raw.get("eta", 0.0))
        if eta < 0.0:
            raise ValueError(
                f"fixed likelihood.noise_boost.eta must be >= 0; got {eta}."
            )
        return {"mode": "fixed", "eta": eta}

    prior = raw.get("prior", "uniform_eta")
    if prior != "uniform_eta":
        raise ValueError(
            "Only likelihood.noise_boost.prior='uniform_eta' is supported. "
            f"Got {prior!r}."
        )
    lo, hi = raw.get("eta_bounds", (0.0, 1.5))
    lo = float(lo)
    hi = float(hi)
    if lo < 0.0 or not lo < hi:
        raise ValueError(
            "likelihood.noise_boost.eta_bounds must satisfy 0 <= low < high; "
            f"got ({lo}, {hi})."
        )
    return {"mode": "mcmc", "eta_bounds": (lo, hi), "prior": prior}


# =========================================================================
# SURROGATE WRAPPER
# =========================================================================
class RFPCASurrogate:
    """
    Surrogate for one phase. Files expected:
        <run_dir>/<phase>/final_model.pkl   -> {model, kept_pcs, ...}
        <pca_dir>/<phase>_pca_full.pkl      -> {pca: PCA}

    The RF was trained on 3 input columns [scale, mustrain, hstrain]
    with scale always equal to 1. Linearity in scale is enforced by
    the caller, which multiplies the predicted pattern by the actual
    phase fraction f_i. This class exposes that g(m, h) prediction.
    """

    def __init__(self, run_dir, pca_dir, phase):
        self.phase = phase

        model_path = os.path.join(run_dir, phase, "final_model.pkl")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"missing surrogate model: {model_path}")
        with open(model_path, "rb") as fh:
            payload = pickle.load(fh)
        self.model    = payload["model"]
        self.kept_pcs = np.asarray(payload["kept_pcs"], dtype=int)

        pca_path = os.path.join(pca_dir, f"{phase}_pca_full.pkl")
        if not os.path.isfile(pca_path):
            raise FileNotFoundError(f"missing PCA file: {pca_path}")
        with open(pca_path, "rb") as fh:
            pca_payload = pickle.load(fh)
        self.pca = pca_payload["pca"]
        self.n_components = self.pca.n_components_

    def predict_g(self, mustrain, hstrain):
        """g_i(m, h) at unit phase fraction. Returns (B, n_2theta)."""
        B = mustrain.shape[0]
        X = np.column_stack([
            np.ones(B, dtype=float),
            mustrain,
            hstrain,
        ])
        Z_kept = self.model.predict(X)
        Z_full = np.zeros((B, self.n_components))
        Z_full[:, self.kept_pcs] = Z_kept
        return self.pca.inverse_transform(Z_full)


# =========================================================================
# STANDARDIZER: standardized walker <-> physical units
# =========================================================================
class Standardizer:
    """
    Owns all transforms between standardized walker space (approx
    N(0,1) per dimension) and physical units.

    Walker layout (length 3N - 1):
        [ z_std_active (N-1) | mu_std (N) | h_std (N) ]

    Phase fractions:
        Reference phase has z_ref = 0 fixed.
        For the active (non-reference) phases i:
            z_i = z_mu[i] + z_sigma[i] * z_std_active[k]
        where the (z_mu, z_sigma) are computed analytically from the
        per-phase scale bounds via:
            c_i  = 0.5 * (low_i + high_i)
            w_i  = high_i - low_i
            z_mu_i    = log(c_i / c_ref)
            z_sigma_i = (w_i / 4) / (c_i * (1 - c_i / simplex_sum))
        Then fractions: softmax over the full N-vector.

    mustrain / hstrain:
        Per-phase midpoint and quarter-range:
            mu_std = (mu_real - midpoint) / quarter_range
        with quarter_range = (high - low) / 4.

    Everything required to reproduce this transform is captured in
    `to_dict()` and saved alongside the chain.
    """

    def __init__(self, config):
        self.config       = config
        self.ref_index    = int(config["reference_phase_index"])
        self.simplex_sum  = float(config["simplex_sum"])
        self.N            = len(config["phases"])

        nb = parse_noise_boost_config(config)
        self.noise_boost = nb
        self.noise_mode = nb["mode"]
        self.has_noise_eta = self.noise_mode == "mcmc"
        if self.has_noise_eta:
            self.eta_low, self.eta_high = nb["eta_bounds"]
            self.eta_mid = 0.5 * (self.eta_low + self.eta_high)
            self.eta_qrange = (self.eta_high - self.eta_low) / 4.0
            self.eta_fixed = None
        else:
            self.eta_low = self.eta_high = None
            self.eta_mid = self.eta_qrange = None
            self.eta_fixed = float(nb["eta"])
        self.ndim         = 3 * self.N - 1 + (1 if self.has_noise_eta else 0)

        # ----- Phase fractions -----
        sb = np.array([p["scale_bounds"] for p in config["phases"]], dtype=float)
        self.scale_low  = sb[:, 0]
        self.scale_high = sb[:, 1]
        c = 0.5 * (self.scale_low + self.scale_high)        # midpoints
        w = self.scale_high - self.scale_low

        if not (0 <= self.ref_index < self.N):
            raise ValueError(f"reference_phase_index out of range: {self.ref_index}")
        if self.scale_low[self.ref_index] <= 0:
            raise ValueError(
                f"Reference phase (index {self.ref_index}) must have a "
                f"strictly positive scale lower bound."
            )

        ref = self.ref_index
        other = np.array([i for i in range(self.N) if i != ref], dtype=int)
        self.active_indices = other

        # Analytic z-space mean and std (only for active phases).
        # z_mu places the active phase's softmax weight at the midpoint
        # ratio; z_sigma is derived by linearizing softmax around z_mu
        # and matching the uniform spread (high-low)/4 in fraction space.
        z_mu    = np.log(c[other] / c[ref])
        # df/dz = f * (1 - f / simplex_sum) at the midpoint c_i:
        df_dz   = c[other] * (1.0 - c[other] / self.simplex_sum)
        # Guard against degenerate denominators.
        df_dz   = np.where(df_dz > 0, df_dz, 1e-12)
        z_sigma = (w[other] / 4.0) / df_dz
        z_sigma = np.where(z_sigma > 0, z_sigma, 1.0)
        self.z_mu    = z_mu          # (N-1,)
        self.z_sigma = z_sigma       # (N-1,)

        # ----- mustrain / hstrain -----
        mb = np.array([p["mustrain_bounds"] for p in config["phases"]], dtype=float)
        hb = np.array([p["hstrain_bounds"]  for p in config["phases"]], dtype=float)
        self.mu_low,  self.mu_high  = mb[:, 0], mb[:, 1]
        self.h_low,   self.h_high   = hb[:, 0], hb[:, 1]
        self.mu_mid     = 0.5 * (self.mu_low + self.mu_high)
        self.h_mid      = 0.5 * (self.h_low  + self.h_high)
        self.mu_qrange  = (self.mu_high - self.mu_low) / 4.0
        self.h_qrange   = (self.h_high  - self.h_low ) / 4.0
        # Defensive against zero ranges (which would be a config error).
        self.mu_qrange = np.where(self.mu_qrange > 0, self.mu_qrange, 1.0)
        self.h_qrange  = np.where(self.h_qrange  > 0, self.h_qrange,  1.0)

    # ------------------------------------------------------------------
    # Walker -> physical
    # ------------------------------------------------------------------
    def unpack(self, theta):
        """
        theta : (nwalkers, ndim)  ->
        fracs    : (nwalkers, N)
        mustrain : (nwalkers, N)
        hstrain  : (nwalkers, N)
        """
        N = self.N
        z_std_active = theta[:, : N - 1]
        mu_std       = theta[:, N - 1 : 2 * N - 1]
        h_std        = theta[:, 2 * N - 1 : 3 * N - 1]

        # z_active (real) = z_mu + z_sigma * z_std
        z_active = self.z_mu + self.z_sigma * z_std_active
        # Insert anchor zero.
        z_full = z_active_to_full(z_active, self.ref_index, N)
        # Softmax to fractions.
        fracs = z_full_to_fracs(z_full, simplex_sum=self.simplex_sum)

        # mustrain / hstrain affine maps.
        mustrain = self.mu_mid + self.mu_qrange * mu_std
        hstrain  = self.h_mid  + self.h_qrange  * h_std

        return fracs, mustrain, hstrain

    def unpack_noise_eta(self, theta):
        """Return physical eta, where Var multiplier is exp(eta)."""
        theta = np.atleast_2d(theta)
        if not self.has_noise_eta:
            return np.full(theta.shape[0], self.eta_fixed, dtype=float)
        eta_std = theta[:, -1]
        return self.eta_mid + self.eta_qrange * eta_std

    def eta_inside_bounds(self, eta):
        if not self.has_noise_eta:
            return np.ones_like(eta, dtype=bool)
        return (eta >= self.eta_low) & (eta <= self.eta_high)

    # ------------------------------------------------------------------
    # Physical -> walker
    # ------------------------------------------------------------------
    def pack(self, fracs, mustrain, hstrain, eta=None):
        """fracs, mustrain, hstrain in physical units -> walker vectors."""
        fracs   = np.atleast_2d(fracs)
        mustrain = np.atleast_2d(mustrain)
        hstrain  = np.atleast_2d(hstrain)

        z_full = fracs_to_z_full(fracs, ref_index=self.ref_index,
                                 simplex_sum=self.simplex_sum)
        z_active = z_full_to_active(z_full, self.ref_index)
        z_std_active = (z_active - self.z_mu) / self.z_sigma

        mu_std = (mustrain - self.mu_mid) / self.mu_qrange
        h_std  = (hstrain  - self.h_mid)  / self.h_qrange

        parts = [z_std_active, mu_std, h_std]
        if self.has_noise_eta:
            if eta is None:
                raise ValueError("eta is required when noise_boost.mode == 'mcmc'.")
            eta = np.atleast_1d(eta).astype(float)
            eta_std = (eta - self.eta_mid) / self.eta_qrange
            parts.append(eta_std[:, None])
        return np.hstack(parts)

    # ------------------------------------------------------------------
    # Hard bounds in real units for box-prior check
    # ------------------------------------------------------------------
    def real_bounds(self):
        lows  = np.concatenate([self.scale_low,  self.mu_low,  self.h_low ])
        highs = np.concatenate([self.scale_high, self.mu_high, self.h_high])
        return lows, highs

    def _noise_boost_to_dict(self):
        if not self.has_noise_eta:
            return {
                "mode": "fixed",
                "eta": float(self.eta_fixed),
                "variance_multiplier": "exp(eta)",
            }
        return {
            "mode": "mcmc",
            "parameter": "eta",
            "prior": "uniform_eta",
            "variance_multiplier": "exp(eta)",
            "eta": {
                "midpoint": float(self.eta_mid),
                "quarter_range": float(self.eta_qrange),
                "low": float(self.eta_low),
                "high": float(self.eta_high),
                "note": "eta_real = midpoint + quarter_range * eta_std",
            },
        }

    # ------------------------------------------------------------------
    # Serialization (for post-processing)
    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "parameterization": "softmax_with_reference_anchor",
            "reference_phase_index": int(self.ref_index),
            "simplex_sum":           float(self.simplex_sum),
            "N":                     int(self.N),
            "ndim":                  int(self.ndim),
            "phase_names":           [p["name"] for p in self.config["phases"]],
            "active_phase_indices":  self.active_indices.tolist(),
            "phase_fractions": {
                "z_mu":    self.z_mu.tolist(),
                "z_sigma": self.z_sigma.tolist(),
                "note":    ("z_active_real = z_mu + z_sigma * z_std_active; "
                            "z_full has 0 at reference_phase_index; "
                            "f = softmax(z_full) * simplex_sum"),
            },
            "mustrain": {
                "midpoint":     self.mu_mid.tolist(),
                "quarter_range": self.mu_qrange.tolist(),
                "low":          self.mu_low.tolist(),
                "high":         self.mu_high.tolist(),
                "note": "real = midpoint + quarter_range * mu_std",
            },
            "hstrain": {
                "midpoint":     self.h_mid.tolist(),
                "quarter_range": self.h_qrange.tolist(),
                "low":          self.h_low.tolist(),
                "high":         self.h_high.tolist(),
                "note": "real = midpoint + quarter_range * h_std",
            },
            "scale_bounds": {
                "low":  self.scale_low.tolist(),
                "high": self.scale_high.tolist(),
            },
            "noise_boost": self._noise_boost_to_dict(),
        }


# =========================================================================
# PSEUDO-MARGINAL STANDARDIZER: walker <-> (log_alpha0, mu, mustrain, hstrain)
# =========================================================================
class PseudoMarginalStandardizer:
    """
    Standardizer for pseudo-marginal mode.
 
    Walker layout (length 3N):
        [ log_alpha0_std (1)
        | z_std_active   (N-1)   -- softmax-with-reference-anchor for mu
        | mu_std         (N)     -- mustrain
        | h_std          (N)     -- hstrain ]
 
    log_alpha0:
        Prior is either a uniform box on log(alpha_0) (with bounds from
        `pseudo_marginal.log_alpha0_bounds`) or a lognormal on alpha_0,
        i.e. Normal(mu, sigma) on log(alpha_0) (with `mu`, `sigma` from
        `pseudo_marginal.log_alpha0_lognormal_params`). The walker
        carries a standardized coordinate; the underlying midpoint /
        quarter-range come from the box for the uniform prior and from
        (mu, sigma) for the lognormal prior (so the standardized coord
        is ~N(0,1) at the prior in both cases).
 
    mu (Dirichlet mean):
        Lives on the (N-1)-simplex with simplex_sum = 1. The walker
        carries N-1 unconstrained coordinates that are mapped through
        the existing softmax-with-reference-anchor:
            z_active_real = z_mu + z_sigma * z_std_active
            z_full        = insert 0 at reference index
            mu            = softmax(z_full) * 1.0
        z_active_real has the interpretation z_i = log(mu_i / mu_ref).
        Under the Dirichlet prior mu ~ Dir(a), this ratio has closed-
        form moments
            E[log(mu_i / mu_ref)]   = digamma(a_i) - digamma(a_ref),
            Var[log(mu_i / mu_ref)] = polygamma(1, a_i) + polygamma(1, a_ref),
        so we set z_mu and z_sigma to those values. The standardized
        walker coords are then ~N(0,1) at the prior, putting them on the
        same scale as mustrain/hstrain (which use a uniform-box affine
        map to the same range).
 
    mustrain / hstrain:
        Identical affine maps as the original Standardizer.
 
    The original Standardizer is left untouched; this class is its
    parallel for the new mode.
    """
 
    def __init__(self, config):
        self.config       = config
        self.ref_index    = int(config["reference_phase_index"])
        # mu lives on the unit simplex by definition.
        self.simplex_sum  = 1.0
        self.N            = len(config["phases"])

        nb = parse_noise_boost_config(config)
        self.noise_boost = nb
        self.noise_mode = nb["mode"]
        self.has_noise_eta = self.noise_mode == "mcmc"
        if self.has_noise_eta:
            self.eta_low, self.eta_high = nb["eta_bounds"]
            self.eta_mid = 0.5 * (self.eta_low + self.eta_high)
            self.eta_qrange = (self.eta_high - self.eta_low) / 4.0
            self.eta_fixed = None
        else:
            self.eta_low = self.eta_high = None
            self.eta_mid = self.eta_qrange = None
            self.eta_fixed = float(nb["eta"])
        self.ndim         = 3 * self.N + (1 if self.has_noise_eta else 0)
 
        pm = config.get("pseudo_marginal", {})
        # ----- log_alpha0 prior -----
        # Two supported forms:
        #   "uniform"   - flat prior on a hard box `log_alpha0_bounds`.
        #                 (low, high) -> mid and qrange via the same
        #                 midpoint / quarter-range convention used
        #                 elsewhere in this file.
        #   "lognormal" - alpha_0 ~ LogNormal(mu, sigma), i.e.
        #                 log(alpha_0) ~ Normal(mu, sigma). Walker
        #                 standardization uses mid=mu, qrange=sigma so
        #                 the standardized coord is ~N(0,1) at the prior.
        #                 No hard bounds are imposed; the Normal density
        #                 itself penalizes extreme values.
        prior_type = pm.get("log_alpha0_prior", "uniform")
        if prior_type not in ("uniform", "lognormal"):
            raise ValueError(
                f"pseudo_marginal.log_alpha0_prior must be 'uniform' or "
                f"'lognormal'; got {prior_type!r}."
            )
        self.log_alpha0_prior_type = prior_type

        if prior_type == "uniform":
            la_low, la_high = pm.get("log_alpha0_bounds", (-2.0, 8.0))
            self.log_alpha0_low  = float(la_low)
            self.log_alpha0_high = float(la_high)
            if not self.log_alpha0_low < self.log_alpha0_high:
                raise ValueError(
                    f"pseudo_marginal.log_alpha0_bounds must satisfy "
                    f"low < high; got "
                    f"({self.log_alpha0_low}, {self.log_alpha0_high})."
                )
            self.log_alpha0_mid    = 0.5 * (self.log_alpha0_low + self.log_alpha0_high)
            self.log_alpha0_qrange = (self.log_alpha0_high - self.log_alpha0_low) / 4.0
            if self.log_alpha0_qrange <= 0:
                self.log_alpha0_qrange = 1.0
            self.log_alpha0_mu    = None
            self.log_alpha0_sigma = None
        else:  # "lognormal"
            params = pm.get("log_alpha0_lognormal_params", None)
            if params is None:
                raise ValueError(
                    "pseudo_marginal.log_alpha0_prior == 'lognormal' "
                    "requires pseudo_marginal.log_alpha0_lognormal_params "
                    "= {'mu': ..., 'sigma': ...}."
                )
            mu_la    = float(params["mu"])
            sigma_la = float(params["sigma"])
            if not sigma_la > 0:
                raise ValueError(
                    "pseudo_marginal.log_alpha0_lognormal_params['sigma'] "
                    f"must be > 0; got {sigma_la}."
                )
            self.log_alpha0_mu     = mu_la
            self.log_alpha0_sigma  = sigma_la
            # Match standardized coord to ~N(0,1) at the prior.
            self.log_alpha0_mid    = mu_la
            self.log_alpha0_qrange = sigma_la
            self.log_alpha0_low    = None
            self.log_alpha0_high   = None
 
        # ----- mu softmax bookkeeping -----
        if not (0 <= self.ref_index < self.N):
            raise ValueError(
                f"reference_phase_index out of range: {self.ref_index}"
            )
        other = np.array(
            [i for i in range(self.N) if i != self.ref_index], dtype=int
        )
        self.active_indices = other
 
        # Prior-derived standardization of the active z coordinates.
        #
        # The active walker coords carry, in real units,
        #     z_i = log(mu_i / mu_ref),  i in active_indices.
        # Under the Dirichlet prior mu ~ Dir(a), each mu_j = X_j / sum X
        # with X_j ~ Gamma(a_j, 1), so the ratio reduces to
        #     z_i = log X_i - log X_ref.
        # The moments of log X_j ~ ExpGamma(a_j) are exactly
        #     E[log X_j]   = digamma(a_j),
        #     Var[log X_j] = polygamma(1, a_j)         (trigamma).
        # Hence
        #     E[z_i]   = digamma(a_i) - digamma(a_ref),
        #     Var[z_i] = polygamma(1, a_i) + polygamma(1, a_ref).
        # Using these as (z_mu, z_sigma) makes the standardized walker
        # coords ~N(0,1) at the prior, putting them on the same scale as
        # mustrain/hstrain so emcee's move parameters are well-matched.
        raw_a = pm.get("mu_dirichlet_prior")
        if raw_a is not None:
            a_vec = np.atleast_1d(np.asarray(raw_a, dtype=float)).ravel()
            if a_vec.size != self.N:
                raise ValueError(
                    f"pseudo_marginal.mu_dirichlet_prior must have length "
                    f"N={self.N}; got length {a_vec.size}."
                )
            if not np.all(a_vec > 0):
                raise ValueError(
                    f"pseudo_marginal.mu_dirichlet_prior entries must be "
                    f"strictly positive; got {a_vec.tolist()}."
                )
            a_ref    = a_vec[self.ref_index]
            a_active = a_vec[other]
            self.z_mu    = digamma(a_active) - digamma(a_ref)
            self.z_sigma = np.sqrt(
                polygamma(1, a_active) + polygamma(1, a_ref)
            )
        else:
            # Defensive fallback: if the config block is missing entirely
            # (e.g. user constructing this class outside the normal flow),
            # fall back to neutral scaling rather than crashing.
            self.z_mu    = np.zeros(self.N - 1)
            self.z_sigma = np.ones(self.N - 1)
 
        # ----- mustrain / hstrain (same convention as Standardizer) -----
        mb = np.array(
            [p["mustrain_bounds"] for p in config["phases"]], dtype=float
        )
        hb = np.array(
            [p["hstrain_bounds"]  for p in config["phases"]], dtype=float
        )
        self.mu_low,  self.mu_high  = mb[:, 0], mb[:, 1]
        self.h_low,   self.h_high   = hb[:, 0], hb[:, 1]
        self.mu_mid     = 0.5 * (self.mu_low + self.mu_high)
        self.h_mid      = 0.5 * (self.h_low  + self.h_high)
        self.mu_qrange  = (self.mu_high - self.mu_low) / 4.0
        self.h_qrange   = (self.h_high  - self.h_low ) / 4.0
        self.mu_qrange = np.where(self.mu_qrange > 0, self.mu_qrange, 1.0)
        self.h_qrange  = np.where(self.h_qrange  > 0, self.h_qrange,  1.0)

    # ------------------------------------------------------------------
    # Walker -> physical
    # ------------------------------------------------------------------
    def unpack(self, theta):
        """
        theta : (nwalkers, ndim)  ->
        log_alpha0 : (nwalkers,)
        mu         : (nwalkers, N)    -- on the unit simplex
        mustrain   : (nwalkers, N)
        hstrain    : (nwalkers, N)
        """
        N = self.N
        la_std       = theta[:, 0]
        z_std_active = theta[:, 1 : N]
        mu_std       = theta[:, N : 2 * N]
        h_std        = theta[:, 2 * N : 3 * N]

        log_alpha0 = self.log_alpha0_mid + self.log_alpha0_qrange * la_std

        z_active = self.z_mu + self.z_sigma * z_std_active
        z_full   = z_active_to_full(z_active, self.ref_index, N)
        mu       = z_full_to_fracs(z_full, simplex_sum=self.simplex_sum)

        mustrain = self.mu_mid + self.mu_qrange * mu_std
        hstrain  = self.h_mid  + self.h_qrange  * h_std

        return log_alpha0, mu, mustrain, hstrain

    def unpack_noise_eta(self, theta):
        """Return physical eta, where Var multiplier is exp(eta)."""
        theta = np.atleast_2d(theta)
        if not self.has_noise_eta:
            return np.full(theta.shape[0], self.eta_fixed, dtype=float)
        eta_std = theta[:, -1]
        return self.eta_mid + self.eta_qrange * eta_std

    def eta_inside_bounds(self, eta):
        if not self.has_noise_eta:
            return np.ones_like(eta, dtype=bool)
        return (eta >= self.eta_low) & (eta <= self.eta_high)

    # ------------------------------------------------------------------
    # Physical -> walker
    # ------------------------------------------------------------------
    def pack(self, log_alpha0, mu, mustrain, hstrain, eta=None):
        """
        Inverse of unpack. Each argument is broadcastable to (nwalkers, ...).
        """
        log_alpha0 = np.atleast_1d(log_alpha0).astype(float)
        mu         = np.atleast_2d(mu).astype(float)
        mustrain   = np.atleast_2d(mustrain).astype(float)
        hstrain    = np.atleast_2d(hstrain).astype(float)

        la_std = ((log_alpha0 - self.log_alpha0_mid)
                  / self.log_alpha0_qrange)

        z_full       = fracs_to_z_full(
            mu, ref_index=self.ref_index, simplex_sum=self.simplex_sum
        )
        z_active     = z_full_to_active(z_full, self.ref_index)
        z_std_active = (z_active - self.z_mu) / self.z_sigma

        mu_std = (mustrain - self.mu_mid) / self.mu_qrange
        h_std  = (hstrain  - self.h_mid)  / self.h_qrange

        parts = [la_std[:, None], z_std_active, mu_std, h_std]
        if self.has_noise_eta:
            if eta is None:
                raise ValueError("eta is required when noise_boost.mode == 'mcmc'.")
            eta = np.atleast_1d(eta).astype(float)
            eta_std = (eta - self.eta_mid) / self.eta_qrange
            parts.append(eta_std[:, None])
        return np.hstack(parts)

    # ------------------------------------------------------------------
    # Hard bounds in real units for box-prior check.
    # Only mustrain and hstrain are boxed here. log_alpha0 has either a
    # uniform-box prior or a lognormal prior, both handled separately
    # via log_alpha0_inside_support / log_alpha0_log_prior. mu is on the
    # simplex by construction.
    # ------------------------------------------------------------------
    def real_bounds_boxed(self):
        lows = np.concatenate([
            self.mu_low,
            self.h_low,
        ])
        highs = np.concatenate([
            self.mu_high,
            self.h_high,
        ])
        return lows, highs

    # ------------------------------------------------------------------
    # log_alpha0 prior dispatch
    # ------------------------------------------------------------------
    def log_alpha0_inside_support(self, log_alpha0):
        """Per-walker support indicator for log(alpha_0). Always True for
        the lognormal prior (support is all of R)."""
        if self.log_alpha0_prior_type == "uniform":
            return (
                (log_alpha0 >= self.log_alpha0_low)
                & (log_alpha0 <= self.log_alpha0_high)
            )
        return np.ones_like(log_alpha0, dtype=bool)

    def log_alpha0_log_prior(self, log_alpha0):
        """Per-walker log prior density on log(alpha_0).

        Uniform: returns -log(width) broadcast to shape, so the joint
        log-prob is fully normalized either way (useful for diagnostics
        and model comparison).
        Lognormal: returns the Normal(mu, sigma) log-density on
        log(alpha_0).
        """
        if self.log_alpha0_prior_type == "uniform":
            width = self.log_alpha0_high - self.log_alpha0_low
            return np.full_like(log_alpha0, -np.log(width), dtype=float)
        mu    = self.log_alpha0_mu
        sigma = self.log_alpha0_sigma
        return (
            -0.5 * np.log(2.0 * np.pi)
            - np.log(sigma)
            - 0.5 * ((log_alpha0 - mu) / sigma) ** 2
        )

    def sample_log_alpha0_initial(self, rng, nwalkers):
        """Sample initial log(alpha_0) positions from the prior."""
        if self.log_alpha0_prior_type == "uniform":
            return rng.uniform(
                self.log_alpha0_low, self.log_alpha0_high, size=nwalkers
            )
        return rng.normal(
            self.log_alpha0_mu, self.log_alpha0_sigma, size=nwalkers
        )

    def log_alpha0_perturb_bounds(self):
        """Practical (low, high) used by the perturb-restart rejection
        sampler. For the lognormal prior this is mu +- 4*sigma, which
        encloses >99.99% of prior mass."""
        if self.log_alpha0_prior_type == "uniform":
            return self.log_alpha0_low, self.log_alpha0_high
        return (
            self.log_alpha0_mu - 4.0 * self.log_alpha0_sigma,
            self.log_alpha0_mu + 4.0 * self.log_alpha0_sigma,
        )

    def _noise_boost_to_dict(self):
        if not self.has_noise_eta:
            return {
                "mode": "fixed",
                "eta": float(self.eta_fixed),
                "variance_multiplier": "exp(eta)",
            }
        return {
            "mode": "mcmc",
            "parameter": "eta",
            "prior": "uniform_eta",
            "variance_multiplier": "exp(eta)",
            "eta": {
                "midpoint": float(self.eta_mid),
                "quarter_range": float(self.eta_qrange),
                "low": float(self.eta_low),
                "high": float(self.eta_high),
                "note": "eta_real = midpoint + quarter_range * eta_std",
            },
        }

    # ------------------------------------------------------------------
    # Serialization (for post-processing)
    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "parameterization": "pseudo_marginal_dirichlet",
            "reference_phase_index": int(self.ref_index),
            "simplex_sum":           float(self.simplex_sum),
            "N":                     int(self.N),
            "ndim":                  int(self.ndim),
            "phase_names":           [p["name"] for p in self.config["phases"]],
            "active_phase_indices":  self.active_indices.tolist(),
            "log_alpha0": {
                "prior_type":    self.log_alpha0_prior_type,
                "midpoint":      float(self.log_alpha0_mid),
                "quarter_range": float(self.log_alpha0_qrange),
                "uniform_low":   (
                    float(self.log_alpha0_low)
                    if self.log_alpha0_low is not None else None
                ),
                "uniform_high":  (
                    float(self.log_alpha0_high)
                    if self.log_alpha0_high is not None else None
                ),
                "lognormal_mu":    (
                    float(self.log_alpha0_mu)
                    if self.log_alpha0_mu is not None else None
                ),
                "lognormal_sigma": (
                    float(self.log_alpha0_sigma)
                    if self.log_alpha0_sigma is not None else None
                ),
                "note": "log_alpha0_real = midpoint + quarter_range * la_std",
            },
            "mu": {
                "z_mu":    self.z_mu.tolist(),
                "z_sigma": self.z_sigma.tolist(),
                "note":    ("z_active_real = z_mu + z_sigma * z_std_active; "
                            "z_full has 0 at reference_phase_index; "
                            "mu = softmax(z_full) (simplex_sum=1)"),
            },
            "mustrain": {
                "midpoint":      self.mu_mid.tolist(),
                "quarter_range": self.mu_qrange.tolist(),
                "low":           self.mu_low.tolist(),
                "high":          self.mu_high.tolist(),
                "note": "real = midpoint + quarter_range * mu_std",
            },
            "hstrain": {
                "midpoint":      self.h_mid.tolist(),
                "quarter_range": self.h_qrange.tolist(),
                "low":           self.h_low.tolist(),
                "high":          self.h_high.tolist(),
                "note": "real = midpoint + quarter_range * h_std",
            },
            "noise_boost": self._noise_boost_to_dict(),
        }


# =========================================================================
# DATA-RANGE MASK
# =========================================================================
# build_keep_idx is imported from fit_gp so the masking logic lives in a
# single place shared by both the GP fit and the MCMC driver.


# =========================================================================
# MODEL: Y = sum_i f_i * g_i(m_i, h_i)
# =========================================================================
def make_model(surrogates, phase_names, keep_idx):
    """
    Y_calc = sum_i f_i * g_i(m_i, h_i), restricted to columns `keep_idx`
    of the full surrogate output.
    """
    n_out = int(keep_idx.size)

    def model(fracs, mustrain, hstrain):
        nwalkers = fracs.shape[0]
        Y_out = np.zeros((nwalkers, n_out))
        for i, phase in enumerate(phase_names):
            g_i = surrogates[phase].predict_g(
                mustrain[:, i], hstrain[:, i]
            )
            Y_out += fracs[:, i:i + 1] * g_i[:, keep_idx]
        return Y_out
    return model


# =========================================================================
# VARIANCE AGGREGATION
# =========================================================================
def make_variance_aggregator(var_lookups, phase_names, keep_idx):
    """
    sigma_SM^2_j(theta) = sum_i f_i^2 * sigma_SM,i,j^2(m_i, h_i),
    restricted to columns `keep_idx` of the full variance output.
    """
    def variance_fn(fracs, mustrain, hstrain):
        first = phase_names[0]
        sig0 = var_lookups[first].sigma_sm_squared_batch(
            mustrain[:, 0], hstrain[:, 0]
        )
        total = (fracs[:, 0:1] ** 2) * sig0[:, keep_idx]
        for i, phase in enumerate(phase_names[1:], start=1):
            sig = var_lookups[phase].sigma_sm_squared_batch(
                mustrain[:, i], hstrain[:, i]
            )
            total += (fracs[:, i:i + 1] ** 2) * sig[:, keep_idx]
        return total
    return variance_fn


# =========================================================================
# LIKELIHOOD WITH ANALYTIC s MARGINALIZATION
# =========================================================================
def make_log_prob(standardizer, lows, highs, model_fn, variance_fn,
                  y_obs, GP_std_sq, epsilon):
    """
    Vectorized joint log-probability for emcee.

    sigma_approx^2 = y_obs + GP_std^2        (s*-independent proxy)
    s* = (sum y_obs y_pred / sigma_approx^2) / (sum y_pred^2 / sigma_approx^2)
    sigma^2 = (s*)^2 * sigma_SM^2 + y_obs + GP_std^2
    log L  = -1/2 sum (y_obs - s* y_pred)^2 / sigma^2
             -1/2 sum log sigma^2 - N/2 log 2pi
             +1/2 log 2pi - 1/2 log A          (Laplace correction)
    Flat improper prior on s, s* > 0 enforced.
    """
    n_data = y_obs.size
    log_2pi = np.log(2.0 * np.pi)
    sigma_approx_sq = np.maximum(y_obs + GP_std_sq, epsilon)

    def joint_log_prob(theta):
        theta = np.atleast_2d(theta)
        nwalkers, _ = theta.shape

        fracs, mustrain, hstrain = standardizer.unpack(theta)
        eta = standardizer.unpack_noise_eta(theta)

        flat = np.hstack([fracs, mustrain, hstrain])
        inside = (
            np.all((flat >= lows) & (flat <= highs), axis=1)
            & standardizer.eta_inside_bounds(eta)
        )

        log_prob = np.full(nwalkers, -np.inf)
        if not np.any(inside):
            return log_prob

        idx = np.where(inside)[0]
        fracs_in    = fracs[idx]
        mustrain_in = mustrain[idx]
        hstrain_in  = hstrain[idx]
        boost_in    = np.exp(eta[idx])

        try:
            y_pred = model_fn(fracs_in, mustrain_in, hstrain_in)
        except Exception as exc:
            print(f"[likelihood] model call failed: {exc}", flush=True)
            return log_prob

        sigma_sm_sq = variance_fn(fracs_in, mustrain_in, hstrain_in)

        sigma_approx_eff_sq = boost_in[:, None] * sigma_approx_sq[None, :]
        A = np.sum(y_pred ** 2     / sigma_approx_eff_sq, axis=1)
        B = np.sum(y_obs * y_pred / sigma_approx_eff_sq, axis=1)
        s_star = np.divide(B, A, out=np.zeros_like(B), where=A > 0)
        pos = s_star > 0.0

        sigma_base_sq = (s_star[:, None] ** 2) * sigma_sm_sq + y_obs[None, :] + GP_std_sq[None, :]

        sigma_sq = np.maximum(boost_in[:, None] * sigma_base_sq, epsilon)
        residuals = y_obs[None, :] - s_star[:, None] * y_pred
        term1 = -0.5 * np.sum(residuals ** 2 / sigma_sq, axis=1)
        term2 = -0.5 * np.sum(np.log(sigma_sq), axis=1)
        term3 = -0.5 * n_data * log_2pi

        A_safe = np.where(A > 0, A, 1.0)
        laplace = 0.5 * log_2pi - 0.5 * np.log(A_safe)

        log_like = term1 + term2 + term3 + laplace
        log_like = np.where(pos, log_like, -np.inf)

        log_prior_const = -np.sum(np.log(highs - lows))

        log_prob[idx] = log_prior_const + log_like
        return log_prob

    return joint_log_prob


# =========================================================================
# PSEUDO-MARGINAL LIKELIHOOD
# =========================================================================
def _per_phase_g_and_var(surrogates, var_lookups, phase_names, keep_idx,
                         mustrain, hstrain):
    """
    Compute g_i(m_i, h_i) and sigma_SM,i^2(m_i, h_i) for each phase, once
    per walker. Both depend only on (mustrain, hstrain), so they do not
    need to be re-evaluated per Dirichlet draw.

    Returns
    -------
    g    : (nwalkers, N, n_out)  -- per-phase surrogate pattern at unit scale
    sig2 : (nwalkers, N, n_out)  -- per-phase surrogate variance
    """
    N = len(phase_names)
    nwalkers = mustrain.shape[0]
    n_out = int(keep_idx.size)
    g    = np.empty((nwalkers, N, n_out))
    sig2 = np.empty((nwalkers, N, n_out))
    for i, phase in enumerate(phase_names):
        g_i = surrogates[phase].predict_g(mustrain[:, i], hstrain[:, i])
        s_i = var_lookups[phase].sigma_sm_squared_batch(
            mustrain[:, i], hstrain[:, i]
        )
        g[:,    i, :] = g_i[:, keep_idx]
        sig2[:, i, :] = s_i[:, keep_idx]
    return g, sig2


def make_log_prob_pseudo_marginal(
    standardizer, surrogates, var_lookups, phase_names, keep_idx,
    y_obs, GP_std_sq, epsilon,
    M, alpha_floor, mu_dirichlet_prior, rng=None,
):
    """
    Vectorized pseudo-marginal joint log-probability for emcee.

    Walker state: (log_alpha0, mu, mustrain, hstrain). For each valid
    walker, draw M samples S^(m) ~ Dir(alpha_0 * mu), evaluate the
    forward model

        y_pred^(m) = sum_i S^(m)_i * g_i(m_i, h_i)

    compute the per-sample log-likelihood l^(m) using the same combined
    variance model and analytic s* Laplace marginalization as direct
    mode, and aggregate

        log p_hat(y | mu, alpha_0, X_D) = logsumexp_m l^(m) - log M.

    Add log p(log_alpha_0) (uniform box), log p(mu) (asymmetric
    Dirichlet with concentration vector `mu_dirichlet_prior`), and
    log p(X_D) (uniform box) to get the joint log-posterior.

    The Dirichlet prior on mu is
        p(mu) = (1 / B(a)) * prod_i mu_i^{a_i - 1},
        log p(mu) = gammaln(sum_i a_i) - sum_i gammaln(a_i)
                  + sum_i (a_i - 1) * log mu_i.
    Ratios of a_i set the prior mean composition (E[mu_i] = a_i / sum a);
    the magnitude sum_i a_i sets the sharpness around that mean.

    The surrogate g_i and variance sigma_SM,i^2 depend only on
    (mustrain, hstrain), so each is computed once per walker rather
    than M times. The Dirichlet draws S^(m) and the resulting per-sample
    likelihood arithmetic are vectorized across (walker, draw, 2theta).
    """
    n_data = y_obs.size
    log_2pi = np.log(2.0 * np.pi)
    sigma_approx_sq = np.maximum(y_obs + GP_std_sq, epsilon)
    log_M = np.log(float(M))

    lows_box, highs_box = standardizer.real_bounds_boxed()
    log_prior_const_X = -np.sum(np.log(highs_box - lows_box))

    N = standardizer.N
    # Asymmetric Dirichlet(a) prior on mu. Validate shape and positivity.
    a_mu = np.atleast_1d(np.asarray(mu_dirichlet_prior, dtype=float)).ravel()
    if a_mu.size != N:
        raise ValueError(
            f"pseudo_marginal.mu_dirichlet_prior must have length "
            f"N={N} (one entry per phase); got length {a_mu.size}."
        )
    if not np.all(a_mu > 0):
        raise ValueError(
            f"pseudo_marginal.mu_dirichlet_prior entries must all be "
            f"strictly positive; got {a_mu.tolist()}."
        )
    # Constant normalization term:  log Gamma(sum a) - sum log Gamma(a_i).
    log_B_mu = gammaln(np.sum(a_mu)) - np.sum(gammaln(a_mu))
    # Whether the data-dependent  sum_i (a_i - 1) log mu_i  contributes.
    a_mu_is_flat = bool(np.all(a_mu == 1.0))

    if rng is None:
        rng = np.random.default_rng()

    def joint_log_prob(theta):
        theta = np.atleast_2d(theta)
        nwalkers = theta.shape[0]

        log_alpha0, mu, mustrain, hstrain = standardizer.unpack(theta)
        eta = standardizer.unpack_noise_eta(theta)

        # Support gate: mustrain/hstrain inside box, log_alpha0 inside
        # its prior support (always True for the lognormal prior).
        # mu is on the simplex by construction (positive, sums to 1).
        boxed = np.column_stack([mustrain, hstrain])
        inside = (
            np.all((boxed >= lows_box) & (boxed <= highs_box), axis=1)
            & standardizer.log_alpha0_inside_support(log_alpha0)
            & standardizer.eta_inside_bounds(eta)
        )

        log_prob = np.full(nwalkers, -np.inf)
        if not np.any(inside):
            return log_prob

        idx = np.where(inside)[0]
        n_in = idx.size
        log_alpha0_in = log_alpha0[idx]
        mu_in         = mu[idx]
        mustrain_in   = mustrain[idx]
        hstrain_in    = hstrain[idx]
        boost_flat    = np.repeat(np.exp(eta[idx]), M)

        # --- Per-phase surrogate evaluation (once per walker) ---
        try:
            g, sig2 = _per_phase_g_and_var(
                surrogates, var_lookups, phase_names, keep_idx,
                mustrain_in, hstrain_in,
            )
        except Exception as exc:
            print(f"[pm-likelihood] surrogate call failed: {exc}", flush=True)
            return log_prob
        # g, sig2 : (n_in, N, n_out)

        # --- Dirichlet draws via Gamma normalization ---
        alpha0 = np.exp(log_alpha0_in)                  # (n_in,)
        alpha  = np.maximum(alpha0[:, None] * mu_in,    # (n_in, N)
                            alpha_floor)
        # Broadcast to (n_in, M, N) for the inner samples.
        alpha_bcast = np.broadcast_to(
            alpha[:, None, :], (n_in, M, N)
        )
        gam = rng.standard_gamma(alpha_bcast)           # (n_in, M, N)
        gam_sum = np.sum(gam, axis=2, keepdims=True)    # (n_in, M, 1)
        # Avoid divide-by-zero from extreme underflow.
        gam_sum = np.where(gam_sum > 0, gam_sum, 1.0)
        S = gam / gam_sum                               # (n_in, M, N)

        # --- Forward model per Dirichlet draw ---
        # y_pred^(m) = sum_i S^(m)_i * g_i,  shape (n_in, M, n_out)
        y_pred = np.einsum('wmi,win->wmn', S, g)
        # sigma_SM^2(theta, S^(m)) = sum_i (S^(m)_i)^2 * sigma_SM,i^2
        sigma_sm_sq = np.einsum('wmi,win->wmn', S ** 2, sig2)

        # --- Per-sample analytic s* (Laplace marginalization over scale) ---
        # Flatten (n_in, M) -> (n_in * M) for the same arithmetic as direct mode.
        ypf = y_pred.reshape(n_in * M, -1)              # (n_in*M, n_out)
        ssf = sigma_sm_sq.reshape(n_in * M, -1)

        sigma_approx_eff_sq = boost_flat[:, None] * sigma_approx_sq[None, :]
        A = np.sum(ypf ** 2     / sigma_approx_eff_sq, axis=1)
        B = np.sum(y_obs * ypf / sigma_approx_eff_sq, axis=1)
        s_star = np.divide(B, A, out=np.zeros_like(B), where=A > 0)
        pos = s_star > 0.0

        sigma_base_sq = (s_star[:, None] ** 2) * ssf + y_obs[None, :] + GP_std_sq[None, :]
        
        sigma_sq = np.maximum(boost_flat[:, None] * sigma_base_sq, epsilon)
        residuals = y_obs[None, :] - s_star[:, None] * ypf
        term1 = -0.5 * np.sum(residuals ** 2 / sigma_sq, axis=1)
        term2 = -0.5 * np.sum(np.log(sigma_sq), axis=1)
        term3 = -0.5 * n_data * log_2pi

        A_safe = np.where(A > 0, A, 1.0)
        laplace = 0.5 * log_2pi - 0.5 * np.log(A_safe)

        ell_flat = term1 + term2 + term3 + laplace
        ell_flat = np.where(pos, ell_flat, -np.inf)
        ell = ell_flat.reshape(n_in, M)                 # (n_in, M)

        # --- Unbiased log-marginal-likelihood via logsumexp ---
        # If every draw is -inf, logsumexp returns -inf cleanly.
        log_p_hat = logsumexp(ell, axis=1) - log_M

        # --- Priors ---
        # log p(log_alpha_0): uniform box OR lognormal (Normal on log).
        # log p(X_D):         uniform box, absorbed into log_prior_const_X.
        # log p(mu):          asymmetric Dirichlet(a_mu).
        log_prior_la = standardizer.log_alpha0_log_prior(log_alpha0_in)
        if a_mu_is_flat:
            log_prior_mu = np.full(n_in, log_B_mu)
        else:
            log_prior_mu = (
                log_B_mu
                + np.sum(
                    (a_mu - 1.0) * np.log(np.maximum(mu_in, 1e-300)),
                    axis=1,
                )
            )

        log_prob[idx] = (
            log_prior_const_X + log_prior_la + log_prior_mu + log_p_hat
        )
        return log_prob

    return joint_log_prob


# =========================================================================
# GP BACKGROUND (loaded from fit_gp.py output)
# =========================================================================
def load_gp_fit(config, out_dir, keep_idx):
    """
    Load the GP background produced by fit_gp.py and validate it against
    the current run.

    Returns (GP_pred, GP_std), both 1-D and aligned to keep_idx:
      * GP_pred : background mean, subtracted from y_obs in the likelihood.
      * GP_std  : background standard deviation; enters the likelihood as
                  GP_std**2.

    Validation guards against silently using a stale or mismatched GP:
      * the gp-config hash must match (gp block + data_ranges + dataset);
      * keep_idx must match exactly;
      * GP_pred and GP_std lengths must match the masked data.
    """
    gp_path = out_dir / "gp_fit.npz"
    if not gp_path.is_file():
        raise FileNotFoundError(
            f"GP fit not found: {gp_path}\n"
            f"Run fit_gp.py for run_name={config['run_name']!r} before MCMC."
        )

    data = np.load(gp_path, allow_pickle=True)
    GP_pred = np.asarray(data["GP_pred"], dtype=float)
    GP_std = np.asarray(data["GP_std"], dtype=float)

    saved_hash = str(data["gp_hash"]) if "gp_hash" in data else None
    expected_hash = gp_config_hash(config)
    if saved_hash != expected_hash:
        raise ValueError(
            f"GP fit is stale: gp_hash mismatch "
            f"(saved={saved_hash}, expected={expected_hash}). The gp/"
            f"data_ranges/dataset settings changed since fit_gp.py was "
            f"run. Re-run fit_gp.py."
        )

    if "keep_idx" in data:
        saved_keep = np.asarray(data["keep_idx"], dtype=int)
        if not np.array_equal(saved_keep, keep_idx):
            raise ValueError(
                "GP fit keep_idx does not match the current data mask. "
                "Re-run fit_gp.py with the current config."
            )

    for name, arr in (("GP_pred", GP_pred), ("GP_std", GP_std)):
        if arr.shape[0] != keep_idx.size:
            raise ValueError(
                f"{name} length ({arr.shape[0]}) does not match the masked "
                f"data length ({keep_idx.size}). Re-run fit_gp.py."
            )

    print(f"Loaded GP fit from {gp_path} (gp_hash={saved_hash}).", flush=True)
    return GP_pred, GP_std


# =========================================================================
# MOVE RESOLUTION
# =========================================================================
def resolve_moves(move_spec):
    out = []
    for name, kwargs, weight in move_spec:
        cls = getattr(emcee.moves, name)
        out.append((cls(**kwargs), weight))
    return out


# =========================================================================
# MAIN
# =========================================================================
def main(config, data_type):
    out_dir = setup_outputs(config)
    print(f"Output directory: {out_dir}", flush=True)

    phase_names = [p["name"] for p in config["phases"]]
    N = len(phase_names)

    # -----------------------------------------------------------------
    # Surrogates and variance tables
    # -----------------------------------------------------------------
    run_dir = config["surrogate"]["run_dir"]
    pca_dir = config["surrogate"]["pca_dir"]
    surrogates = {
        ph: RFPCASurrogate(run_dir, pca_dir, ph) for ph in phase_names
    }
    var_lookups = load_phase_variances(run_dir, phase_names)

    # -----------------------------------------------------------------
    # Observed pattern
    # -----------------------------------------------------------------
    x_obs, y_obs = load_observed_pattern(config, data_type)

    # -----------------------------------------------------------------
    # Data-range mask
    # -----------------------------------------------------------------
    n_2theta_full = int(y_obs.size)
    # Sanity-check: the surrogate's reconstructed pattern must have the
    # same width as the data. Catches grid mismatches between the loaded
    # surrogate and the dataset.
    surrogate_width = int(surrogates[phase_names[0]].pca.n_features_in_)
    if surrogate_width != n_2theta_full:
        raise ValueError(
            f"Surrogate output width ({surrogate_width}) does not match "
            f"data length ({n_2theta_full}). The surrogate and dataset "
            f"must share the same 2theta grid."
        )

    data_ranges = config["dataset"].get("data_ranges")
    keep_idx = build_keep_idx(data_ranges, n_2theta_full)
    np.save(out_dir / "data_mask.npy", keep_idx)
    print(
        f"data_ranges={data_ranges}; using {keep_idx.size} of "
        f"{n_2theta_full} points.",
        flush=True,
    )
    x_obs = x_obs[keep_idx]
    y_obs = y_obs[keep_idx]

    # -----------------------------------------------------------------
    # GP background (fit upstream by fit_gp.py; loaded + validated here)
    # -----------------------------------------------------------------
    GP_pred, GP_std = load_gp_fit(config, out_dir, keep_idx)
    y_obs_peaks = np.maximum(y_obs - GP_pred, 1)

    # -----------------------------------------------------------------
    # Mode dispatch: direct (v3 behavior) vs pseudo_marginal
    # -----------------------------------------------------------------
    run_mode = config.get("mode", "direct")
    if run_mode not in ("direct", "pseudo_marginal"):
        raise ValueError(
            f"Unknown CONFIG['mode']={run_mode!r}; expected 'direct' or "
            f"'pseudo_marginal'."
        )
    print(f"Run mode: {run_mode}", flush=True)

    if run_mode == "direct":
        # -------------------------------------------------------------
        # Standardizer (and persist its parameters)
        # -------------------------------------------------------------
        standardizer = Standardizer(config)
        with open(out_dir / "standardizer.json", "w") as f:
            json.dump(standardizer.to_dict(), f, indent=2)
        lows, highs = standardizer.real_bounds()

        # -------------------------------------------------------------
        # Likelihood
        # -------------------------------------------------------------
        model_fn    = make_model(surrogates, phase_names, keep_idx)
        variance_fn = make_variance_aggregator(
            var_lookups, phase_names, keep_idx
        )
        log_prob_fn = make_log_prob(
            standardizer, lows, highs, model_fn, variance_fn,
            y_obs=y_obs_peaks,
            GP_std_sq=GP_std ** 2,
            epsilon=config["likelihood"]["epsilon"],
        )

        # -------------------------------------------------------------
        # Initial walker positions
        # -------------------------------------------------------------
        ndim = standardizer.ndim
        nwalkers = config["mcmc"]["nwalkers_per_dim"] * ndim

        # Phase fractions: uniform on the softmax-weight space, rejecting
        # any draw whose induced fractions fall outside the per-phase box.
        scale_bounds_arr = np.array(
            [p["scale_bounds"] for p in config["phases"]]
        )
        fracs_init = generate_initial_softmax_weights(
            nwalkers, N, scale_bounds_arr,
            ref_index=standardizer.ref_index,
            simplex_sum=standardizer.simplex_sum,
        )

        # mustrain / hstrain: uniform on the prior box, then standardize.
        mu_bounds_arr = np.array(
            [p["mustrain_bounds"] for p in config["phases"]]
        )
        h_bounds_arr  = np.array(
            [p["hstrain_bounds"]  for p in config["phases"]]
        )
        mustrain_init = sample_uniform_in_box(nwalkers, mu_bounds_arr)
        hstrain_init  = sample_uniform_in_box(nwalkers, h_bounds_arr)
        eta_init = None
        if standardizer.has_noise_eta:
            eta_bounds_arr = np.array([[standardizer.eta_low, standardizer.eta_high]])
            eta_init = sample_uniform_in_box(nwalkers, eta_bounds_arr).ravel()

        initial_pos = standardizer.pack(
            fracs_init, mustrain_init, hstrain_init, eta=eta_init
        )

    else:  # run_mode == "pseudo_marginal"
        # -------------------------------------------------------------
        # Pseudo-marginal config
        # -------------------------------------------------------------
        pm_cfg = config.get("pseudo_marginal", {})
        M = pm_cfg.get("M")
        if M is None:
            M = 20 * N
        M = int(M)
        if M < 1:
            raise ValueError(f"pseudo_marginal.M must be >= 1, got {M}.")
        alpha_floor = float(pm_cfg.get("alpha_floor", 1e-6))
        # Asymmetric Dirichlet concentration vector for the prior on mu.
        # Length must equal N (one entry per phase). Ratios encode the
        # prior mean composition; magnitude controls sharpness.
        raw_a = pm_cfg.get("mu_dirichlet_prior", [1.0] * N)
        mu_dirichlet_prior = np.atleast_1d(
            np.asarray(raw_a, dtype=float)
        ).ravel()
        if mu_dirichlet_prior.size != N:
            raise ValueError(
                f"pseudo_marginal.mu_dirichlet_prior must have length "
                f"N={N} (one entry per phase); got length "
                f"{mu_dirichlet_prior.size}."
            )
        if not np.all(mu_dirichlet_prior > 0):
            raise ValueError(
                f"pseudo_marginal.mu_dirichlet_prior entries must all be "
                f"strictly positive; got {mu_dirichlet_prior.tolist()}."
            )
        print(
            f"pseudo_marginal: M={M}, alpha_floor={alpha_floor}, "
            f"mu_dirichlet_prior={mu_dirichlet_prior.tolist()}",
            flush=True,
        )

        # -------------------------------------------------------------
        # Standardizer (and persist its parameters)
        # -------------------------------------------------------------
        standardizer = PseudoMarginalStandardizer(config)
        with open(out_dir / "standardizer.json", "w") as f:
            json.dump(standardizer.to_dict(), f, indent=2)
        lows, highs = standardizer.real_bounds_boxed()

        # -------------------------------------------------------------
        # Likelihood
        # -------------------------------------------------------------
        # Fresh draws every log-prob call (genuine unbiased pseudo-marginal).
        pm_rng = np.random.default_rng()
        log_prob_fn = make_log_prob_pseudo_marginal(
            standardizer=standardizer,
            surrogates=surrogates,
            var_lookups=var_lookups,
            phase_names=phase_names,
            keep_idx=keep_idx,
            y_obs=y_obs_peaks,
            GP_std_sq=GP_std ** 2,
            epsilon=config["likelihood"]["epsilon"],
            M=M,
            alpha_floor=alpha_floor,
            mu_dirichlet_prior=mu_dirichlet_prior,
            rng=pm_rng,
        )

        # -------------------------------------------------------------
        # Initial walker positions
        # -------------------------------------------------------------
        ndim = standardizer.ndim
        nwalkers = config["mcmc"]["nwalkers_per_dim"] * ndim

        # log_alpha_0: drawn from the configured prior (uniform on the
        # box or Normal on log(alpha_0)).
        log_alpha0_init = standardizer.sample_log_alpha0_initial(
            pm_rng, nwalkers
        )

        # mu: Dirichlet(mu_dirichlet_prior). With an all-ones vector
        # this is uniform on the simplex; asymmetric vectors bias the
        # initial composition toward the configured prior mean.
        mu_init = pm_rng.dirichlet(mu_dirichlet_prior, size=nwalkers)

        # mustrain / hstrain: uniform on the prior box.
        mu_bounds_arr = np.array(
            [p["mustrain_bounds"] for p in config["phases"]]
        )
        h_bounds_arr  = np.array(
            [p["hstrain_bounds"]  for p in config["phases"]]
        )
        mustrain_init = sample_uniform_in_box(nwalkers, mu_bounds_arr)
        hstrain_init  = sample_uniform_in_box(nwalkers, h_bounds_arr)
        eta_init = None
        if standardizer.has_noise_eta:
            eta_bounds_arr = np.array([[standardizer.eta_low, standardizer.eta_high]])
            eta_init = sample_uniform_in_box(nwalkers, eta_bounds_arr).ravel()

        initial_pos = standardizer.pack(
            log_alpha0_init, mu_init, mustrain_init, hstrain_init, eta=eta_init
        )

    # -----------------------------------------------------------------
    # Backend + restart logic
    # -----------------------------------------------------------------
    backend_path = out_dir / "chain.h5"
    backend = emcee.backends.HDFBackend(str(backend_path))

    mode = config["mcmc"]["restart_mode"]
    if backend_path.exists() and backend.iteration > 0:
        if mode == "fresh":
            print("restart_mode=fresh: clearing backend.", flush=True)
            backend.reset(nwalkers, ndim)
        elif mode == "continue":
            print(f"restart_mode=continue: resuming from iteration "
                  f"{backend.iteration}.", flush=True)
            initial_pos = backend.get_last_sample().coords
        elif mode == "perturb":
            print("restart_mode=perturb: reinitializing from best walkers.",
                  flush=True)
            chain = backend.get_chain()
            logp  = backend.get_log_prob()
            final_raw = chain[-1]
            final_lp  = logp[-1]

            n_best = config["mcmc"]["perturb_n_best"]
            best_idx = np.argsort(final_lp)[-n_best:]

            if run_mode == "direct":
                fracs_f, mu_f, h_f = standardizer.unpack(final_raw)
                if standardizer.has_noise_eta:
                    eta_f = standardizer.unpack_noise_eta(final_raw)
                    real_f = np.hstack([fracs_f, mu_f, h_f, eta_f[:, None]])
                    lows_perturb = np.concatenate([lows, np.array([standardizer.eta_low])])
                    highs_perturb = np.concatenate([highs, np.array([standardizer.eta_high])])
                    perturb_ndim = 3 * N + 1
                else:
                    real_f = np.hstack([fracs_f, mu_f, h_f])
                    lows_perturb = lows
                    highs_perturb = highs
                    perturb_ndim = 3 * N

                best_mean = np.mean(real_f[best_idx], axis=0)
                best_std  = np.std(real_f[best_idx], axis=0)
                perturb_scale = config["mcmc"]["perturb_scale"]
                min_frac      = config["mcmc"]["perturb_min_frac"]
                perturb = np.maximum(best_std * perturb_scale,
                                     np.abs(best_mean) * min_frac)

                real_starts = sample_valid_initial_positions(
                    best_mean, perturb, nwalkers, perturb_ndim,
                    lows_perturb, highs_perturb,
                )
                eta_new = real_starts[:, 3 * N] if standardizer.has_noise_eta else None
                initial_pos = standardizer.pack(
                    real_starts[:, :N],
                    real_starts[:, N:2 * N],
                    real_starts[:, 2 * N:3 * N],
                    eta=eta_new,
                )
            else:  # pseudo_marginal
                # Unpack into (log_alpha0, mu, mustrain, hstrain) and
                # perturb in real space. mu is renormalized to the
                # simplex after perturbation; log_alpha0 and mustrain/
                # hstrain are box-clipped by the rejection sampler.
                la_f, mu_f, m_f, h_f = standardizer.unpack(final_raw)
                # Real-space layout for perturb: [log_alpha0 | mu | m | h]
                # plus eta at the end when noise_boost.mode == "mcmc".
                # mu has dimension N but lives on the simplex; we still
                # perturb in its raw coordinates and renormalize below.
                if standardizer.has_noise_eta:
                    eta_f = standardizer.unpack_noise_eta(final_raw)
                    real_f = np.hstack([la_f[:, None], mu_f, m_f, h_f, eta_f[:, None]])
                else:
                    real_f = np.hstack([la_f[:, None], mu_f, m_f, h_f])

                best_mean = np.mean(real_f[best_idx], axis=0)
                best_std  = np.std(real_f[best_idx], axis=0)
                perturb_scale = config["mcmc"]["perturb_scale"]
                min_frac      = config["mcmc"]["perturb_min_frac"]
                perturb = np.maximum(best_std * perturb_scale,
                                     np.abs(best_mean) * min_frac)

                # Box bounds for the rejection sampler: log_alpha0 uses
                # the configured prior's practical envelope (the box for
                # uniform, mu +- 4*sigma for lognormal); mustrain/
                # hstrain are boxed; mu has no box -- give it
                # (epsilon, 1) so the sampler doesn't reject on it, and
                # renormalize afterwards.
                la_perturb_low, la_perturb_high = (
                    standardizer.log_alpha0_perturb_bounds()
                )
                lows_perturb = np.concatenate([
                    np.array([la_perturb_low]),
                    np.full(N, 1e-12),
                    standardizer.mu_low,
                    standardizer.h_low,
                ])
                highs_perturb = np.concatenate([
                    np.array([la_perturb_high]),
                    np.full(N, 1.0),
                    standardizer.mu_high,
                    standardizer.h_high,
                ])
                perturb_ndim = 1 + 3 * N
                if standardizer.has_noise_eta:
                    lows_perturb = np.concatenate([
                        lows_perturb, np.array([standardizer.eta_low])
                    ])
                    highs_perturb = np.concatenate([
                        highs_perturb, np.array([standardizer.eta_high])
                    ])
                    perturb_ndim += 1

                real_starts = sample_valid_initial_positions(
                    best_mean, perturb, nwalkers,
                    perturb_ndim, lows_perturb, highs_perturb,
                )
                la_new   = real_starts[:, 0]
                mu_new   = real_starts[:, 1 : 1 + N]
                mu_new   = mu_new / np.sum(mu_new, axis=1, keepdims=True)
                must_new = real_starts[:, 1 + N : 1 + 2 * N]
                hstr_new = real_starts[:, 1 + 2 * N : 1 + 3 * N]
                eta_new = real_starts[:, 1 + 3 * N] if standardizer.has_noise_eta else None
                initial_pos = standardizer.pack(
                    la_new, mu_new, must_new, hstr_new, eta=eta_new
                )

            backend_path = out_dir / "chain_reint.h5"
            backend = emcee.backends.HDFBackend(str(backend_path))
            backend.reset(nwalkers, ndim)
        else:
            raise ValueError(f"Unknown restart_mode {mode!r}.")
    else:
        print("Starting new MCMC run.", flush=True)
        backend.reset(nwalkers, ndim)

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------
    moves = resolve_moves(config["mcmc"]["moves"])
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob_fn,
        moves=moves, backend=backend, vectorize=True,
    )

    nsteps = config["mcmc"]["nsteps"]
    print(f"Running {nsteps} steps with {nwalkers} walkers "
          f"({ndim}-dim).", flush=True)
    t0 = time.time()
    sampler.run_mcmc(initial_pos, nsteps, progress=True)
    t1 = time.time()
    print(f"Elapsed: {t1 - t0:.1f} s. Total samples: {backend.iteration}",
          flush=True)


def _parse_args(argv=None):
    default_cfg = Path(__file__).resolve().parent / "config.py"
    parser = argparse.ArgumentParser(
        description="Run MCMC inference using a pre-fit GP background "
                    "(see fit_gp.py)."
    )
    parser.add_argument(
        "--config", default=str(default_cfg),
        help=f"Path to the pipeline config.py (default: {default_cfg}).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    config, data_type, _ = load_config(args.config)
    main(config, data_type)
