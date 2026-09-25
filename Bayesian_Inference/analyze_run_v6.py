"""
analyze_run_v5.py

Post-processing for an MCMC run produced by run_mcmc_v5.py (or v4/v3).

Reads the per-run metadata from <run_dir>/{standardizer.json, config.json}
so this script never duplicates configuration that already lives with the
run. Edit only the CONFIG block at the top for paths and toggles.

v5 changes
----------
Adds support for the optional noise-boost parameter eta introduced in
run_mcmc_v5.py. The variance model in the likelihood is

    Var_eff_j = exp(eta) * (s*^2 * sigma_SM^2 + y_obs_j + GP_std_j^2)

with eta either held fixed (mode='fixed', usually 0) or sampled with a
uniform prior on a bounded interval (mode='mcmc'). Detection is
automatic from the 'noise_boost' block in standardizer.json. When eta
is sampled, this script produces:

  * posteriors_eta.pdf  -- two-panel figure: posterior of eta on the
                           left (with the uniform prior overlaid as a
                           flat dotted line), posterior of the variance
                           multiplier exp(eta) on the right (with the
                           induced 1/((b-a)*v) prior overlaid).
  * A 'noise_boost' section in posterior_summary.json with statistics
    on both eta and exp(eta).

When eta is held fixed, no plot is produced; the held value (and its
exp) is still reported in posterior_summary.json. Older runs without a
'noise_boost' block in standardizer.json are treated as fixed eta=0
with a console notice.

This version also:
  * fixes a latent slicing bug in StandardizerDecoder.unpack where
    hstrain was extracted via an open-ended slice (corrupted by the
    appended eta dim when present); it now uses an explicit upper bound,
  * labels trace-plot y-axes with the actual parameter names rather
    than 'param i'.

v4 changes
----------
Adds support for runs in pseudo-marginal mode (Dirichlet hyperparameter
parameterization). Detection is automatic, from the `parameterization`
field of standardizer.json:

    "softmax_with_reference_anchor"  -> direct mode  (v3 behavior, unchanged)
    "pseudo_marginal_dirichlet"      -> pseudo-marginal mode

In pseudo-marginal mode the walker carries (log_alpha_0, mu, mustrain,
hstrain) rather than (fracs, mustrain, hstrain). To produce posteriors
of the realized phase fractions S used by the forward model, we replay
the Dirichlet marginalization: for each post-burn-in walker sample we
draw N_FRAC_DRAWS_PER_SAMPLE samples S ~ Dir(alpha_0 * mu) and stack
them. The resulting S-posterior is what appears in
posteriors_phase_fractions.pdf and what feeds the validators, so the
plot's meaning is the same as in direct mode (realized fractions
compared to PHASE_FRACTION_TRUE). The Dirichlet mean mu gets its own
posterior figure (posteriors_mu.pdf), and log_alpha_0 gets its own
figure (posteriors_log_alpha0.pdf).

Pipeline:

    load chain -> diagnostics -> trace plot
                                    |
                                    v
                          decode raw walker coords
                          to physical units
                                    |
                       +------------+------------+
                       |            |            |
                       v            v            v
              phase-frac plot   mustrain    hstrain plot
                                  plot
                                    |
                       +------------+------------+
                       |                         |
                       v                         v
              surrogate validation       GSAS-II validation
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
import atexit
import glob

from fit_gp import (
    load_config,
    load_observed_pattern,
    build_keep_idx,
    gp_config_hash,
)

# =============================================================================
# CONFIG  --  edit values here. No CLI parsing.
#
# This block is intentionally minimal: anything that also lives in
# config.json (and therefore in <RUN_DIR>/config.json) is read from
# there at runtime, not duplicated here. That includes the X-ray
# wavelength, likelihood epsilon, surrogate/PCA directories, and the
# full phase list. Edit the source-of-truth config.py for those.
# =============================================================================

# Name of the run. Used only to locate RUN_DIR and DATA_DIR below and to
# select the matching true-value overlay block further down. Must agree
# with config["run_name"]; this is verified after the config is loaded.
run_name = "In-situ"

# --- Paths ----------------------------------------------------------------
# Run directory output by run_mcmc.py. Contains chain.h5, config.json,
# standardizer.json, GP_pred.txt, GP_std.txt, metadata.json.
RUN_DIR         = Path("output/"+run_name)
DATA_DIR        = Path("data_dir") 
#DATA_DIR        = Path("data_dir/prescribed_examples/run_20260526_133941_n3")
CHAIN_HDF5      = RUN_DIR / "chain.h5"
CONFIG_JSON     = RUN_DIR / "config.json"
STANDARDIZER_JSON = RUN_DIR / "standardizer.json"
FIGURES_DIR     = RUN_DIR / "figures"

# Observed pattern (read from the same path the MCMC used by default; can
# be overridden here if you've moved the data).
OBSERVED_FXYE  = True   # None -> read dataset.fxye_path from config.json

# --- Burn-in --------------------------------------------------------------
# Variable sets the amount of steps at the end of each walk which should be kept (not burned). 
# This times the number of walkers is the number of accepted samples.
KEEP_LAST_STEPS = 1000

# --- GSAS-II worker -------------------------------------------------------
GSAS_WORKER_SCRIPT      = "model_v1_old.py"
GSAS_WORKER_PYTHON      = r"C:/Users/wardbm1/gsas2main/python.exe" #Insert your GSAS location
GSAS_WORKER_TIMEOUT_S   = 300.0
GSAS_READY_TIMEOUT_S    = 600.0

# --- Toggles --------------------------------------------------------------
RUN_SURROGATE_VALIDATION = False
RUN_GSAS_VALIDATION      = True # Allows you to skip the GSAS-II validation for time. Subsampling not set up here.

# --- Validation draws -----------------------------------------------------
N_VARIANCE_DRAWS = 40
VALIDATION_SEED  = 0

# --- Pseudo-marginal-only knobs -------------------------------------------
# These are silently ignored if the run is in direct mode (i.e. its
# standardizer.json has parameterization == "softmax_with_reference_anchor").

# Number of Dirichlet draws S ~ Dir(alpha_0 * mu) per post-burn-in walker
# sample. The realized-fraction posterior plotted in
# posteriors_phase_fractions.pdf is built from this stack; the validators
# consume the same stack so each S^(k) is paired with the (mustrain,
# hstrain) walker sample it was drawn at.
N_FRAC_DRAWS_PER_SAMPLE = 4

# Number of (alpha_0, mu) prior draws used to estimate the predictive
# phase-fraction prior overlay via KDE. Each draw produces one Dirichlet
# sample S.
N_PRIOR_DRAWS = 5000

# Seed for the analyzer-side resampling (separate from VALIDATION_SEED so
# changing one doesn't shift the other).
FRAC_RESAMPLE_SEED = 1

# --- True values for posterior overlays (optional). Map phase_name -> value.
#     Leave any of these as {} to omit the dashed reference line.
PHASE_FRACTION_TRUE: dict[str, float] = {}
MUSTRAIN_TRUE: dict[str, float] = {}
HSTRAIN_TRUE: dict[str, float] = {}

# --- True Values ----
if run_name == "as-built":
    PHASE_FRACTION_TRUE: dict[str, float] = {
        "gamma":   0.98,  "delta":   0.0,    "gamma1":  0.0,    "gamma2":  0.0,    "laves":   0.012,   "carbide": 0.008,
    }
    MUSTRAIN_TRUE: dict[str, float] = {
        "gamma":   8000.0, "delta":   1000.0, "gamma1":  1000.0, "gamma2":  1000.0, "laves":   19000.0, "carbide": 25000.0,
    }
    HSTRAIN_TRUE: dict[str, float] = {
        "gamma":   0.0025, "delta":   0.0,    "gamma1":  0.0,    "gamma2":  0.0,    "laves":   0.004,   "carbide": 0.002,
    }
elif run_name == "in-situ":
    PHASE_FRACTION_TRUE: dict[str, float] = {
        "gamma":   0.88,   "delta":   0.03,    "gamma1":  0.03,    "gamma2":  0.04,   "laves":   0.01,    "carbide": 0.01,
    }
    MUSTRAIN_TRUE: dict[str, float] = {
        "gamma":   7000.0, "delta":   12000.0, "gamma1":  10000.0, "gamma2":  6000.0, "laves":   17000.0, "carbide": 22000.0,
    }
    HSTRAIN_TRUE: dict[str, float] = {
        "gamma":   0.002,  "delta":   0.0005,  "gamma1": -0.0012,  "gamma2":  0.002,  "laves":   0.003,   "carbide": 0.001,
    }
elif run_name == "homogenized":
    PHASE_FRACTION_TRUE: dict[str, float] = {
        "gamma":   0.81,   "delta":   0.02,    "gamma1":  0.06,   "gamma2":  0.10,    "laves":   1e-8,    "carbide": 0.01,
    }
    MUSTRAIN_TRUE: dict[str, float] = {
        "gamma":   6000.0, "delta":   12000.0, "gamma1":  8000.0, "gamma2":  5000.0, "carbide": 20000.0, #Laves not present
    }
    HSTRAIN_TRUE: dict[str, float] = {
        "gamma":   0.001,  "delta":   0.0003,  "gamma1":  0.0005, "gamma2": -0.0008,   "carbide": 0.0005,
    }
else: #run_name == "ID35" or run_name == "ID35_dirichlet" or run_name == "ID35_dirichlet_boosted":
    PHASE_FRACTION_TRUE: dict[str, float] = {
        "gamma":   0.996,  "laves":  0.002856,    "carbide": 0.0011,
    }
    MUSTRAIN_TRUE: dict[str, float] = {
        "gamma":   9669.6, "laves": 30369.6, "carbide": 14055.2, 
    }
    HSTRAIN_TRUE: dict[str, float] = {
        "gamma":   -0.00446, "laves": 0.000354, "carbide": 0.00225,
    }
#else:
#    raise ValueError(f"unknown run_name: {run_name!r}")

# =============================================================================
# Publication-style matplotlib defaults
# =============================================================================

PUBLICATION_RC = {
    "figure.dpi":       300,
    "font.family":      "serif",
    "mathtext.fontset": "stix",
    "font.size":        14,
    "axes.labelsize":   14,
    "axes.titlesize":   16,
    "legend.fontsize":  14,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "axes.linewidth":   1.2,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.top":        False,
    "ytick.right":      False,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
}

PHASE_LABELS = {
    "gamma":              r"$\gamma$",
    "delta":              r"$\delta$",
    "gamma1":        r"$\gamma'$",
    "gamma2":   r"$\gamma''$",
    "Laves":              "Laves",
    "Carbide":            "Carbide",
}

def setup_publication_style() -> None:
    mpl.rcParams.update(PUBLICATION_RC)


# =============================================================================
# Decoder built from standardizer.json
# =============================================================================

def _read_noise_boost(d: dict, std_path: Path) -> dict:
    """
    Pull the optional 'noise_boost' block out of a parsed standardizer.json
    payload and return a normalized dict the decoders can consume directly.

    Backward compatibility: standardizer.json files written before v5 have
    no 'noise_boost' key. In that case we treat the run as eta-off (fixed
    eta = 0, exp(eta) = 1, ndim unchanged) and emit a one-line console
    notice so the analyzer doesn't silently misread an older chain.

    Returns
    -------
    dict with keys:
        has_noise_eta : bool
        noise_mode    : "fixed" | "mcmc"
        eta_fixed     : float | None    (the held value when fixed)
        eta_low       : float | None    (sampling bounds when mcmc)
        eta_high      : float | None
        eta_mid       : float | None
        eta_qrange    : float | None
    """
    nb = d.get("noise_boost")
    if nb is None:
        print(
            f"No 'noise_boost' block in {std_path}; assuming pre-v5 run "
            f"(eta=0, no boosting)."
        )
        return {
            "has_noise_eta": False,
            "noise_mode":    "fixed",
            "eta_fixed":     0.0,
            "eta_low":       None,
            "eta_high":      None,
            "eta_mid":       None,
            "eta_qrange":    None,
        }

    mode = nb.get("mode", "fixed")
    if mode == "fixed":
        return {
            "has_noise_eta": False,
            "noise_mode":    "fixed",
            "eta_fixed":     float(nb.get("eta", 0.0)),
            "eta_low":       None,
            "eta_high":      None,
            "eta_mid":       None,
            "eta_qrange":    None,
        }
    if mode == "mcmc":
        eta_block = nb["eta"]
        return {
            "has_noise_eta": True,
            "noise_mode":    "mcmc",
            "eta_fixed":     None,
            "eta_low":       float(eta_block["low"]),
            "eta_high":      float(eta_block["high"]),
            "eta_mid":       float(eta_block["midpoint"]),
            "eta_qrange":    float(eta_block["quarter_range"]),
        }
    raise ValueError(
        f"Unknown noise_boost.mode={mode!r} in {std_path}; "
        f"expected 'fixed' or 'mcmc'."
    )


class StandardizerDecoder:
    """
    Inverse of run_mcmc.py's Standardizer.unpack(), built from the
    serialized standardizer.json. Anything that uses the new
    parameterization goes through this — there are no hardcoded specs
    in this file.

    Walker layout (length 3N - 1):
        [ z_std_active (N-1) | mu_std (N) | h_std (N) ]
    """

    parameterization = "softmax_with_reference_anchor"
    is_pseudo_marginal = False

    def __init__(self, std_path: Path):
        with open(std_path) as f:
            d = json.load(f)
        if d.get("parameterization") != "softmax_with_reference_anchor":
            raise ValueError(
                f"unexpected parameterization in {std_path}: "
                f"{d.get('parameterization')!r}"
            )
        self.ref_index    = int(d["reference_phase_index"])
        self.simplex_sum  = float(d["simplex_sum"])
        self.N            = int(d["N"])
        self.ndim         = int(d["ndim"])
        self.phase_names  = list(d["phase_names"])
        self.active_indices = np.asarray(d["active_phase_indices"], dtype=int)

        pf = d["phase_fractions"]
        self.z_mu    = np.asarray(pf["z_mu"], dtype=float)
        self.z_sigma = np.asarray(pf["z_sigma"], dtype=float)

        mu = d["mustrain"]
        self.mu_mid    = np.asarray(mu["midpoint"], dtype=float)
        self.mu_qrange = np.asarray(mu["quarter_range"], dtype=float)

        h = d["hstrain"]
        self.h_mid    = np.asarray(h["midpoint"], dtype=float)
        self.h_qrange = np.asarray(h["quarter_range"], dtype=float)

        # Noise-boost (eta) bookkeeping. See _read_noise_boost above.
        nb = _read_noise_boost(d, std_path)
        self.has_noise_eta = nb["has_noise_eta"]
        self.noise_mode    = nb["noise_mode"]
        self.eta_fixed     = nb["eta_fixed"]
        self.eta_low       = nb["eta_low"]
        self.eta_high      = nb["eta_high"]
        self.eta_mid       = nb["eta_mid"]
        self.eta_qrange    = nb["eta_qrange"]

        # Cross-check ndim against has_noise_eta. In direct mode the
        # walker has length 3N - 1 without eta and 3N with eta.
        expected_ndim = 3 * self.N - 1 + (1 if self.has_noise_eta else 0)
        if self.ndim != expected_ndim:
            raise ValueError(
                f"standardizer.json ndim={self.ndim} but expected "
                f"{expected_ndim} given N={self.N} and "
                f"has_noise_eta={self.has_noise_eta}."
            )

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        theta : (M, ndim) walker vectors.

        Returns
        -------
        fracs    : (M, N)
        mustrain : (M, N)
        hstrain  : (M, N)
        """
        theta = np.atleast_2d(theta)
        N = self.N
        # Explicit upper bound on h_std so the optional eta column at the
        # end (when has_noise_eta) does not bleed into hstrain. The v4
        # decoder used h_std = theta[:, 2*N - 1 :], which silently
        # corrupted the last hstrain dim when eta was present.
        z_std_active = theta[:, : N - 1]
        mu_std       = theta[:, N - 1     : 2 * N - 1]
        h_std        = theta[:, 2 * N - 1 : 3 * N - 1]

        # z_active (real) = z_mu + z_sigma * z_std
        z_active = self.z_mu + self.z_sigma * z_std_active

        # Insert anchor zero at ref_index.
        M = theta.shape[0]
        z_full = np.empty((M, N), dtype=float)
        z_full[:, self.ref_index] = 0.0
        z_full[:, self.active_indices] = z_active

        # Stable softmax.
        z_shift = z_full - np.max(z_full, axis=-1, keepdims=True)
        e = np.exp(z_shift)
        fracs = self.simplex_sum * e / np.sum(e, axis=-1, keepdims=True)

        mustrain = self.mu_mid + self.mu_qrange * mu_std
        hstrain  = self.h_mid  + self.h_qrange  * h_std
        return fracs, mustrain, hstrain

    def unpack_eta(self, theta: np.ndarray) -> np.ndarray:
        """
        Physical eta per walker sample, shape (M,). When the run had
        noise_boost.mode == 'mcmc', eta is the last walker coordinate
        and is decoded via the affine standardization. Otherwise we
        return a constant array filled with the held value eta_fixed.

        This mirrors PseudoMarginalStandardizer.unpack_noise_eta and
        Standardizer.unpack_noise_eta in run_mcmc_v5 exactly.
        """
        theta = np.atleast_2d(theta)
        if not self.has_noise_eta:
            return np.full(theta.shape[0], self.eta_fixed, dtype=float)
        eta_std = theta[:, -1]
        return self.eta_mid + self.eta_qrange * eta_std

    @property
    def param_names(self) -> list[str]:
        """
        Human-readable name per walker dimension, for trace-plot labels.

        Direct-mode layout:
            z_active[<active_phase_0>], ..., z_active[<active_phase_{N-2}>],
            mustrain[<phase_0>], ..., mustrain[<phase_{N-1}>],
            hstrain[<phase_0>], ..., hstrain[<phase_{N-1}>],
            (eta, if has_noise_eta)
        """
        names: list[str] = []
        for idx in self.active_indices:
            names.append(f"z_active[{self.phase_names[int(idx)]}]")
        for ph in self.phase_names:
            names.append(f"mustrain[{ph}]")
        for ph in self.phase_names:
            names.append(f"hstrain[{ph}]")
        if self.has_noise_eta:
            names.append("eta")
        return names


class PseudoMarginalStandardizerDecoder:
    """
    Inverse of run_mcmc_v4.py's PseudoMarginalStandardizer.unpack(),
    built from the serialized standardizer.json.

    Walker layout (length 3N):
        [ log_alpha0_std (1)
        | z_std_active   (N-1)  -- softmax-with-reference-anchor for mu
        | mu_std         (N)    -- mustrain
        | h_std          (N)    -- hstrain ]

    `unpack` returns (log_alpha0, mu, mustrain, hstrain) -- realized
    phase fractions S are NOT walker coordinates and must be obtained
    by Dirichlet resampling (see resample_phase_fractions).
    """

    parameterization = "pseudo_marginal_dirichlet"
    is_pseudo_marginal = True

    def __init__(self, std_path: Path):
        with open(std_path) as f:
            d = json.load(f)
        if d.get("parameterization") != "pseudo_marginal_dirichlet":
            raise ValueError(
                f"unexpected parameterization in {std_path}: "
                f"{d.get('parameterization')!r}"
            )
        self.ref_index    = int(d["reference_phase_index"])
        self.simplex_sum  = float(d["simplex_sum"])
        self.N            = int(d["N"])
        self.ndim         = int(d["ndim"])
        self.phase_names  = list(d["phase_names"])
        self.active_indices = np.asarray(d["active_phase_indices"], dtype=int)

        la = d["log_alpha0"]
        self.log_alpha0_mid    = float(la["midpoint"])
        self.log_alpha0_qrange = float(la["quarter_range"])
        # Prior type and parameters. New runs save prior_type plus the
        # relevant params; older runs only carried "low"/"high" for a
        # uniform prior, which we accept as a fallback.
        self.log_alpha0_prior_type = str(la.get("prior_type", "uniform"))
        if self.log_alpha0_prior_type == "uniform":
            low_key  = "uniform_low"  if "uniform_low"  in la else "low"
            high_key = "uniform_high" if "uniform_high" in la else "high"
            self.log_alpha0_low    = float(la[low_key])
            self.log_alpha0_high   = float(la[high_key])
            self.log_alpha0_mu     = None
            self.log_alpha0_sigma  = None
        elif self.log_alpha0_prior_type == "lognormal":
            self.log_alpha0_mu     = float(la["lognormal_mu"])
            self.log_alpha0_sigma  = float(la["lognormal_sigma"])
            self.log_alpha0_low    = None
            self.log_alpha0_high   = None
        else:
            raise ValueError(
                f"unknown log_alpha0 prior_type in standardizer.json: "
                f"{self.log_alpha0_prior_type!r}"
            )

        mu = d["mu"]
        self.z_mu    = np.asarray(mu["z_mu"],    dtype=float)
        self.z_sigma = np.asarray(mu["z_sigma"], dtype=float)

        ms = d["mustrain"]
        self.mu_mid    = np.asarray(ms["midpoint"],      dtype=float)
        self.mu_qrange = np.asarray(ms["quarter_range"], dtype=float)

        h = d["hstrain"]
        self.h_mid    = np.asarray(h["midpoint"],      dtype=float)
        self.h_qrange = np.asarray(h["quarter_range"], dtype=float)

        # Noise-boost (eta) bookkeeping. See _read_noise_boost above.
        nb = _read_noise_boost(d, std_path)
        self.has_noise_eta = nb["has_noise_eta"]
        self.noise_mode    = nb["noise_mode"]
        self.eta_fixed     = nb["eta_fixed"]
        self.eta_low       = nb["eta_low"]
        self.eta_high      = nb["eta_high"]
        self.eta_mid       = nb["eta_mid"]
        self.eta_qrange    = nb["eta_qrange"]

        # Cross-check ndim against has_noise_eta. In pseudo-marginal mode
        # the walker has length 3N without eta and 3N + 1 with eta.
        expected_ndim = 3 * self.N + (1 if self.has_noise_eta else 0)
        if self.ndim != expected_ndim:
            raise ValueError(
                f"standardizer.json ndim={self.ndim} but expected "
                f"{expected_ndim} given N={self.N} and "
                f"has_noise_eta={self.has_noise_eta}."
            )

    def unpack(self, theta: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        theta : (M, ndim) walker vectors.

        Returns
        -------
        log_alpha0 : (M,)
        mu         : (M, N)    on the simplex (sums to 1)
        mustrain   : (M, N)
        hstrain    : (M, N)
        """
        theta = np.atleast_2d(theta)
        N = self.N
        la_std       = theta[:, 0]
        z_std_active = theta[:, 1 : N]
        mu_std       = theta[:, N : 2 * N]
        h_std        = theta[:, 2 * N : 3 * N]

        log_alpha0 = self.log_alpha0_mid + self.log_alpha0_qrange * la_std

        # mu via softmax-with-reference-anchor (z_mu=0, z_sigma=1 by
        # construction in pseudo-marginal mode).
        z_active = self.z_mu + self.z_sigma * z_std_active
        M = theta.shape[0]
        z_full = np.empty((M, N), dtype=float)
        z_full[:, self.ref_index] = 0.0
        z_full[:, self.active_indices] = z_active
        z_shift = z_full - np.max(z_full, axis=-1, keepdims=True)
        e = np.exp(z_shift)
        mu = self.simplex_sum * e / np.sum(e, axis=-1, keepdims=True)

        mustrain = self.mu_mid + self.mu_qrange * mu_std
        hstrain  = self.h_mid  + self.h_qrange  * h_std
        return log_alpha0, mu, mustrain, hstrain

    def unpack_eta(self, theta: np.ndarray) -> np.ndarray:
        """
        Physical eta per walker sample, shape (M,). When the run had
        noise_boost.mode == 'mcmc', eta is the last walker coordinate
        and is decoded via the affine standardization. Otherwise we
        return a constant array filled with the held value eta_fixed.

        Same semantics as StandardizerDecoder.unpack_eta; the only
        difference is the walker-layout offset, which is absorbed by
        taking theta[:, -1].
        """
        theta = np.atleast_2d(theta)
        if not self.has_noise_eta:
            return np.full(theta.shape[0], self.eta_fixed, dtype=float)
        eta_std = theta[:, -1]
        return self.eta_mid + self.eta_qrange * eta_std

    @property
    def param_names(self) -> list[str]:
        """
        Human-readable name per walker dimension, for trace-plot labels.

        Pseudo-marginal-mode layout:
            log_alpha0,
            z_active[<active_phase_0>], ..., z_active[<active_phase_{N-2}>],
            mustrain[<phase_0>], ..., mustrain[<phase_{N-1}>],
            hstrain[<phase_0>], ..., hstrain[<phase_{N-1}>],
            (eta, if has_noise_eta)
        """
        names: list[str] = ["log_alpha0"]
        for idx in self.active_indices:
            names.append(f"z_active[{self.phase_names[int(idx)]}]")
        for ph in self.phase_names:
            names.append(f"mustrain[{ph}]")
        for ph in self.phase_names:
            names.append(f"hstrain[{ph}]")
        if self.has_noise_eta:
            names.append("eta")
        return names


def load_decoder(std_path: Path
                 ) -> StandardizerDecoder | PseudoMarginalStandardizerDecoder:
    """
    Factory: read standardizer.json, branch on the `parameterization`
    field, return the matching decoder.
    """
    with open(std_path) as f:
        d = json.load(f)
    par = d.get("parameterization")
    if par == "softmax_with_reference_anchor":
        return StandardizerDecoder(std_path)
    if par == "pseudo_marginal_dirichlet":
        return PseudoMarginalStandardizerDecoder(std_path)
    raise ValueError(
        f"Unknown parameterization in {std_path}: {par!r}. "
        f"Expected 'softmax_with_reference_anchor' or "
        f"'pseudo_marginal_dirichlet'."
    )


# =============================================================================
# Chain loading & diagnostics
# =============================================================================

@dataclass
class ChainData:
    chain:    np.ndarray   # (nsteps, nwalkers, ndim)
    log_prob: np.ndarray   # (nsteps, nwalkers)
    blobs:    Any
    nsteps:   int
    nwalkers: int
    ndim:     int


def load_chain(path: Path, expected_ndim: int) -> ChainData:
    backend = emcee.backends.HDFBackend(str(path))
    chain    = backend.get_chain(flat=False)
    log_prob = backend.get_log_prob(flat=False)
    try:
        blobs = backend.get_blobs(flat=False)
    except AttributeError:
        blobs = None

    nsteps, nwalkers, ndim = chain.shape
    if ndim != expected_ndim:
        raise ValueError(
            f"Chain has ndim={ndim} but expected ndim={expected_ndim} "
            f"(from standardizer.json). Refusing to silently reshape."
        )
    return ChainData(chain, log_prob, blobs, nsteps, nwalkers, ndim)


def acceptance_mask(chain: np.ndarray) -> np.ndarray:
    """Vectorized: True where the walker moved since the prior step."""
    if chain.shape[0] < 2:
        return np.zeros(chain.shape[:2], dtype=bool)
    diff_any = np.any(chain[1:] != chain[:-1], axis=2)   # (nsteps-1, nwalkers)
    mask = np.zeros(chain.shape[:2], dtype=bool)
    mask[1:] = diff_any
    return mask


def gelman_rubin(chains: np.ndarray) -> float:
    """R-hat for one parameter; chains shape (nsteps, m_chains)."""
    n, _ = chains.shape
    chain_means = chains.mean(axis=0)
    b = n * np.var(chain_means, ddof=1)
    w = np.mean(np.var(chains, axis=0, ddof=1))
    var_hat = (1 - 1 / n) * w + (1 / n) * b
    return float(np.sqrt(var_hat / w))


def split_rhat(x: np.ndarray) -> float:
    n = x.shape[0] // 2
    chains = np.concatenate([x[:n], x[n:2 * n]], axis=1)
    return gelman_rubin(chains)


def print_diagnostics(cd: ChainData, chain_path: Path, burn_in: int) -> None:
    mask = acceptance_mask(cd.chain)

    overall_acc = mask.sum() / (cd.nsteps * cd.nwalkers)
    print(f"Overall acceptance fraction (full chain): {overall_acc:.4f}")

    per_walker = mask.sum(axis=0) / cd.nsteps
    for i, frac in enumerate(per_walker):
        print(f"  Walker {i:2d}: acceptance fraction = {frac:.3f}")

    burned_mask = mask[burn_in:]
    if burned_mask.shape[0] > 0:
        burned_acc = burned_mask.sum() / (burned_mask.shape[0] * cd.nwalkers)
        print(f"Overall acceptance fraction (post burn-in): {burned_acc:.4f}")

    try:
        backend = emcee.backends.HDFBackend(str(chain_path))
        tau = backend.get_autocorr_time(tol=0)
        print(f"Autocorrelation time per param: {tau}")
    except Exception as e:
        print(f"Autocorr time unavailable: {e}")

    samples = cd.chain[burn_in:]
    print("Gelman-Rubin R-hat (per parameter):")
    for i in range(cd.ndim):
        print(f"  param {i}: R-hat = {gelman_rubin(samples[:, :, i]):.4f}")
    print("Split R-hat (per parameter):")
    for i in range(cd.ndim):
        print(f"  param {i}: split R-hat = {split_rhat(samples[:, :, i]):.4f}")


# =============================================================================
# Decoded posterior
# =============================================================================

@dataclass
class DecodedPosterior:
    """
    Decoded post-burn-in samples in physical units.

    Common to both modes:
        phase_fractions : (n_eff, N)  realized fractions S used by the
                                       forward model. In direct mode
                                       these are walker coordinates; in
                                       pseudo-marginal mode they are
                                       Dirichlet draws S ~ Dir(alpha_0*mu)
                                       resampled from the chain.
        mustrain        : (n_eff, N)
        hstrain         : (n_eff, N)
        raw_burned      : (n_walker_samples, ndim)   raw post-burn-in walkers

    For pseudo-marginal mode, mustrain/hstrain are broadcast to (n_eff, N)
    by repeating each walker sample N_FRAC_DRAWS_PER_SAMPLE times so each
    realized S^(k) is paired with the (m, h) it was conditioned on.

    Pseudo-marginal-only (None in direct mode):
        mu              : (n_walker_samples, N)  Dirichlet mean
        log_alpha0      : (n_walker_samples,)    Dirichlet concentration
        k_draws         : int                    N_FRAC_DRAWS_PER_SAMPLE

    Noise-boost (v5):
        noise_mode      : "fixed" | "mcmc"
        eta_fixed       : float | None   value used when noise_mode == "fixed"
        eta             : (n_walker_samples,) | None
                          posterior samples of eta when noise_mode == "mcmc";
                          None otherwise. Each sample is paired 1:1 with
                          raw_burned (and, in pseudo-marginal mode, with
                          mu / log_alpha0).
    """
    phase_fractions: np.ndarray
    mustrain:        np.ndarray
    hstrain:         np.ndarray
    raw_burned:      np.ndarray
    mu:              np.ndarray | None = None
    log_alpha0:      np.ndarray | None = None
    k_draws:         int | None = None
    noise_mode:      str = "fixed"
    eta_fixed:       float | None = 0.0
    eta:             np.ndarray | None = None


def decode_burned(cd: ChainData,
                  decoder: StandardizerDecoder | PseudoMarginalStandardizerDecoder,
                  burn_in: int,
                  k_draws: int = 1,
                  alpha_floor: float = 1e-6,
                  seed: int = 0,
                  ) -> DecodedPosterior:
    """
    Apply burn-in and unpack all post-burn-in samples.

    Direct mode: one vectorized standardizer call; k_draws is ignored.

    Pseudo-marginal mode: unpack walker samples to (log_alpha0, mu,
    mustrain, hstrain), then draw k_draws realized fractions S ~
    Dir(alpha_0 * mu) per walker sample and broadcast mustrain/hstrain
    to match. We keep ALL post-burn-in samples (not only accepted moves)
    because rejected steps are repeats at the current point, which is
    the correct weight for high-density regions.
    """
    burned_chain = cd.chain[burn_in:]
    raw = burned_chain.reshape(-1, cd.ndim)

    # Noise-boost eta: same recipe for both modes (eta is always the
    # trailing walker coordinate when it's a coord at all). decoder
    # returns a constant array when noise_mode == 'fixed', so the field
    # is always populated and downstream consumers can branch on
    # noise_mode rather than on None-ness.
    noise_mode = decoder.noise_mode
    if decoder.has_noise_eta:
        eta_samples = decoder.unpack_eta(raw)
        eta_fixed_val: float | None = None
    else:
        eta_samples = None
        eta_fixed_val = float(decoder.eta_fixed)

    if not decoder.is_pseudo_marginal:
        fracs, mu, h = decoder.unpack(raw)
        return DecodedPosterior(
            phase_fractions=fracs, mustrain=mu, hstrain=h, raw_burned=raw,
            noise_mode=noise_mode,
            eta_fixed=eta_fixed_val,
            eta=eta_samples,
        )

    # Pseudo-marginal: draw realized phase fractions via Dirichlet.
    log_alpha0, mu_chain, mustrain, hstrain = decoder.unpack(raw)
    n = raw.shape[0]
    N = decoder.N
    rng = np.random.default_rng(seed)

    alpha0 = np.exp(log_alpha0)                            # (n,)
    alpha  = np.maximum(alpha0[:, None] * mu_chain,        # (n, N)
                        alpha_floor)
    # Broadcast to (n, k_draws, N) and draw Gammas, then normalize.
    alpha_bcast = np.broadcast_to(alpha[:, None, :], (n, k_draws, N))
    gam = rng.standard_gamma(alpha_bcast)                  # (n, k_draws, N)
    gam_sum = np.sum(gam, axis=2, keepdims=True)
    gam_sum = np.where(gam_sum > 0, gam_sum, 1.0)
    S = gam / gam_sum                                      # (n, k_draws, N)

    # Flatten (n, k_draws) -> (n*k_draws). Each S^(k) is paired with the
    # (mustrain, hstrain) walker sample it was drawn at.
    fracs_flat = S.reshape(n * k_draws, N)
    mustrain_flat = np.repeat(mustrain, k_draws, axis=0)
    hstrain_flat  = np.repeat(hstrain,  k_draws, axis=0)

    return DecodedPosterior(
        phase_fractions=fracs_flat,
        mustrain=mustrain_flat,
        hstrain=hstrain_flat,
        raw_burned=raw,
        mu=mu_chain,
        log_alpha0=log_alpha0,
        k_draws=int(k_draws),
        noise_mode=noise_mode,
        eta_fixed=eta_fixed_val,
        eta=eta_samples,
    )


# =============================================================================
# Posterior summary (JSON dump for tables / downstream)
# =============================================================================

def write_posterior_summary(decoded: DecodedPosterior,
                            phase_names: list[str],
                            savepath: Path) -> None:
    def stats_for(arr: np.ndarray) -> dict:
        return {
            "mean":     float(np.mean(arr)),
            "std":      float(np.std(arr)),
            "median":   float(np.median(arr)),
            "p2.5":     float(np.percentile(arr, 2.5)),
            "p16":      float(np.percentile(arr, 16)),
            "p84":      float(np.percentile(arr, 84)),
            "p97.5":    float(np.percentile(arr, 97.5)),
        }

    out: dict[str, dict] = {"phases": {}}
    for j, ph in enumerate(phase_names):
        entry = {
            "phase_fraction": stats_for(decoded.phase_fractions[:, j]),
            "mustrain":       stats_for(decoded.mustrain[:, j]),
            "hstrain":        stats_for(decoded.hstrain[:, j]),
        }
        if decoded.mu is not None:
            entry["mu"] = stats_for(decoded.mu[:, j])
        out["phases"][ph] = entry
    out["n_samples"] = int(decoded.phase_fractions.shape[0])

    if decoded.log_alpha0 is not None:
        out["log_alpha0"] = stats_for(decoded.log_alpha0)
        out["dirichlet_resampling"] = {
            "k_draws_per_walker_sample": int(decoded.k_draws),
            "n_walker_samples":          int(decoded.raw_burned.shape[0]),
        }

    # Noise-boost (eta) reporting. Always emitted so downstream
    # consumers can tell what variance model the chain ran under.
    # exp(eta) is the actual multiplicative variance scale and is more
    # interpretable than eta itself, so we always include both.
    if decoded.noise_mode == "mcmc" and decoded.eta is not None:
        out["noise_boost"] = {
            "mode": "mcmc",
            "variance_multiplier": "exp(eta)",
            "eta":                          stats_for(decoded.eta),
            "variance_multiplier_exp_eta":  stats_for(np.exp(decoded.eta)),
        }
    else:
        held = float(decoded.eta_fixed if decoded.eta_fixed is not None else 0.0)
        out["noise_boost"] = {
            "mode": "fixed",
            "variance_multiplier": "exp(eta)",
            "eta": held,
            "variance_multiplier_exp_eta": float(np.exp(held)),
        }

    with open(savepath, "w") as f:
        json.dump(out, f, indent=2)


def update_posterior_summary(savepath: Path, updates: dict) -> None:
    """
    Merge `updates` into the existing posterior summary JSON and rewrite
    it. Used to append validation-time results (e.g. R-factors) after
    the summary has already been written. Each call rewrites the file,
    so a later crash never erases earlier additions.

    Merge semantics: for each top-level key in `updates`, if both the
    existing value and the new value are dicts, the dicts are merged
    (new values win on key collision); otherwise the value is replaced.
    """
    with open(savepath) as f:
        data = json.load(f)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k] = {**data[k], **v}
        else:
            data[k] = v
    with open(savepath, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# Plotting -- traces, posteriors, validation
# =============================================================================

def plot_traces(cd: ChainData, savepath: Path,
                param_names: list[str] | None = None) -> None:
    """
    Per-dimension trace plot with optional human-readable y-labels.

    If `param_names` is supplied, its length must equal cd.ndim, and each
    entry becomes the y-label for the corresponding subplot. Otherwise
    we fall back to 'param i'.
    """
    if param_names is not None and len(param_names) != cd.ndim:
        raise ValueError(
            f"param_names has length {len(param_names)} but chain has "
            f"ndim={cd.ndim}; they must match."
        )

    fig, axes = plt.subplots(cd.ndim + 1, 1,
                             figsize=(10, 2.5 * (cd.ndim + 1)),
                             sharex=True)
    for i in range(cd.ndim):
        for w in range(cd.nwalkers):
            axes[i].plot(cd.chain[:, w, i], alpha=0.3)
        label = param_names[i] if param_names is not None else f"param {i}"
        axes[i].set_ylabel(label)
    for w in range(cd.nwalkers):
        axes[-1].plot(cd.log_prob[:, w], alpha=0.3)
    axes[-1].set_ylabel("log prob")
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close(fig)


def _plot_posteriors_publication(samples: np.ndarray,
                                 phase_names: list[str],
                                 xlabels: dict[str, str],
                                 priors: dict[str, Any],
                                 true_values: dict[str, float] | None,
                                 savepath: Path,
                                 legend_labels: tuple[str, str, str] = (
                                     "Posterior", "Prior", "Reference",
                                 )) -> None:
    """Posterior subplot grid that adapts to any number of phases.

    `priors[ph]` may be either:
      * (lo, hi)                   -- uniform box, drawn as a flat line
                                       at height 1/(hi-lo) over [lo, hi].
      * ("uniform", (lo, hi))      -- explicit form of the above.
      * ("samples", array_of_samples) -- KDE'd as the prior curve.

    Grid layout:
        N == 1            -> 1x1
        N == 2            -> 1x2
        3 <= N <= 4       -> 2x2
        5 <= N <= 6       -> 2x3
        7 <= N <= 9       -> 3x3
        N >= 10           -> ceil(sqrt(N)) cols, ceil(N/cols) rows
    Unused axes are hidden so figures stay clean regardless of N.
    """
    n_params = len(phase_names)
    if n_params <= 0:
        raise ValueError("phase_names is empty; nothing to plot.")
    if n_params <= 3:
        nrows, ncols = 1, n_params
    elif n_params <= 4:
        nrows, ncols = 2, 2
    elif n_params <= 6:
        nrows, ncols = 2, 3
    elif n_params <= 9:
        nrows, ncols = 3, 3
    else:
        ncols = int(np.ceil(np.sqrt(n_params)))
        nrows = int(np.ceil(n_params / ncols))
    figsize = (max(2.5 * ncols + 0.5, 3.0), max(1.5 * nrows + 0.5, 2.0))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    posterior_color = sns.color_palette("husl", n_params)[0]

    for j, ph in enumerate(phase_names):
        ax = axes[j]
        sns.kdeplot(samples[:, j], color=posterior_color, linewidth=2, ax=ax)
        ax.hist(samples[:, j], bins=50, density=True,
                alpha=0.6, color=posterior_color, edgecolor="none")

        # --- Prior overlay (uniform box OR KDE of supplied samples) ---
        prior_spec = priors[ph]
        kind, payload = _normalize_prior_spec(prior_spec)
        if kind == "uniform":
            lo, hi = payload
            ax.hlines(1.0 / (hi - lo), lo, hi, colors="red",
                      linestyles="dotted", linewidth=3)
        else:  # "samples"
            sns.kdeplot(payload, color="red", linestyle="dotted",
                        linewidth=2, ax=ax)

        if true_values is not None and ph in true_values:
            ax.axvline(true_values[ph], color="black",
                       linestyle="dashed", linewidth=2)

        ax.set_xlabel(xlabels[ph], labelpad=3)
        ax.set_ylabel("Density", labelpad=3)
        # # Compact x-tick formatting — sci notation for small-magnitude
        # # params (e.g. HStrain ~ 1e-2) and a hard cap on tick count.
        # max_abs = np.max(np.abs(samples[:, j]))
        # if 0 < max_abs < 0.1:
        #     ax.ticklabel_format(axis="x", style="sci",
        #                         scilimits=(-2, 2), useMathText=True)
        # ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # Hide unused axes when the grid is larger than n_params.
    for k in range(n_params, len(axes)):
        axes[k].axis("off")

    posterior_proxy = Line2D([0], [0], color=posterior_color, linewidth=2)
    prior_proxy     = Line2D([0], [0], color="red", linestyle="dotted", linewidth=2)
    true_proxy      = Line2D([0], [0], color="black", linestyle="dashed", linewidth=2)
    fig.legend(
        handles=[posterior_proxy, prior_proxy, true_proxy],
        labels=list(legend_labels),
        loc="lower center", ncol=3, #fontsize=14,
        frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, -0.25),
    )
    plt.savefig(savepath, bbox_inches="tight")
    plt.close(fig)


def _normalize_prior_spec(spec) -> tuple[str, Any]:
    """
    Accept any of the prior payloads the publication plotter understands
    and return a uniform ('uniform', (lo, hi)) / ('samples', array) pair.
    """
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
        kind, payload = spec
        if kind not in ("uniform", "samples"):
            raise ValueError(f"unknown prior kind: {kind!r}")
        return kind, payload
    # Bare (lo, hi) tuple: treat as uniform.
    return "uniform", tuple(spec)


def _make_xlabels(phase_names: list[str], suffix: str) -> dict[str, str]:
    """Map each phase to '<LaTeX phase> <suffix>', falling back to the
    raw phase name if it isn't in PHASE_LABELS."""
    return {
        ph: f"{PHASE_LABELS.get(ph, ph)} {suffix}".strip()
        for ph in phase_names
    }


def plot_posteriors_phase_fractions(decoded: DecodedPosterior,
                                    phase_names: list[str],
                                    priors: dict[str, tuple[float, float]],
                                    true_values: dict[str, float],
                                    savepath: Path) -> None:
    xlabels = _make_xlabels(phase_names, "")
    _plot_posteriors_publication(
        decoded.phase_fractions, phase_names, xlabels, priors,
        true_values if true_values else None, savepath,
    )


def plot_posteriors_mustrain(decoded: DecodedPosterior,
                             phase_names: list[str],
                             priors: dict[str, tuple[float, float]],
                             true_values: dict[str, float],
                             savepath: Path) -> None:
    xlabels = _make_xlabels(phase_names, "mustrain")
    _plot_posteriors_publication(
        decoded.mustrain, phase_names, xlabels, priors,
        true_values if true_values else None, savepath,
    )


def plot_posteriors_hstrain(decoded: DecodedPosterior,
                            phase_names: list[str],
                            priors: dict[str, tuple[float, float]],
                            true_values: dict[str, float],
                            savepath: Path) -> None:
    xlabels = _make_xlabels(phase_names, "hstrain")
    _plot_posteriors_publication(
        decoded.hstrain, phase_names, xlabels, priors,
        true_values if true_values else None, savepath,
    )


# =============================================================================
# Pseudo-marginal-only: Dirichlet predictive prior, mu/alpha_0 plots
# =============================================================================

def sample_phase_fraction_prior(a_mu: np.ndarray,
                                log_alpha0_prior: dict,
                                n_draws: int,
                                seed: int = 0) -> np.ndarray:
    """
    Monte Carlo draws from the predictive prior on phase fractions S
    induced by

        log_alpha_0 ~ <log_alpha0_prior>,
        mu          ~ Dir(a_mu),
        S | mu, alpha_0 ~ Dir(alpha_0 * mu).

    `log_alpha0_prior` is a dict with key "type" that selects the prior
    on log(alpha_0):
        {"type": "uniform",   "low": ..., "high": ...}
        {"type": "lognormal", "mu":  ..., "sigma": ...}

    Returns (n_draws, N).
    """
    rng = np.random.default_rng(seed)
    N = a_mu.size
    ptype = log_alpha0_prior["type"]
    if ptype == "uniform":
        log_alpha0 = rng.uniform(
            log_alpha0_prior["low"], log_alpha0_prior["high"], size=n_draws
        )
    elif ptype == "lognormal":
        log_alpha0 = rng.normal(
            log_alpha0_prior["mu"], log_alpha0_prior["sigma"], size=n_draws
        )
    else:
        raise ValueError(f"unknown log_alpha0_prior type: {ptype!r}")
    mu = rng.dirichlet(a_mu, size=n_draws)              # (n_draws, N)
    alpha = np.exp(log_alpha0)[:, None] * mu            # (n_draws, N)
    # Element-wise Gamma; broadcast over the (n_draws, N) shape.
    gam = rng.standard_gamma(alpha)
    gam_sum = np.sum(gam, axis=1, keepdims=True)
    gam_sum = np.where(gam_sum > 0, gam_sum, 1.0)
    return gam / gam_sum


def plot_posteriors_phase_fractions_pm(decoded: DecodedPosterior,
                                       phase_names: list[str],
                                       a_mu: np.ndarray,
                                       log_alpha0_prior: dict,
                                       true_values: dict[str, float],
                                       n_prior_draws: int,
                                       prior_seed: int,
                                       savepath: Path) -> None:
    """
    Pseudo-marginal phase-fraction posterior plot. The prior overlay is
    a KDE of the predictive prior on S (not a flat box).
    """
    S_prior = sample_phase_fraction_prior(
        a_mu, log_alpha0_prior, n_prior_draws, seed=prior_seed
    )
    priors = {ph: ("samples", S_prior[:, j])
              for j, ph in enumerate(phase_names)}
    xlabels = _make_xlabels(phase_names, "")
    _plot_posteriors_publication(
        decoded.phase_fractions, phase_names, xlabels, priors,
        true_values if true_values else None, savepath,
    )


def plot_posteriors_mu(decoded: DecodedPosterior,
                       phase_names: list[str],
                       a_mu: np.ndarray,
                       savepath: Path) -> None:
    """
    Posterior over the Dirichlet mean mu, with the marginal Dirichlet
    prior overlaid. The marginal of Dir(a) on mu_i is Beta(a_i, sum_{j!=i} a_j),
    drawn analytically.
    """
    from scipy.stats import beta as _beta
    if decoded.mu is None:
        raise ValueError("decoded.mu is None; not a pseudo-marginal run.")

    # Build a per-phase prior payload: ('samples', beta_draws_i).
    rng = np.random.default_rng(0)
    a_sum = float(np.sum(a_mu))
    priors: dict[str, Any] = {}
    n_prior = 20000
    for j, ph in enumerate(phase_names):
        a_i = float(a_mu[j])
        b_i = a_sum - a_i
        priors[ph] = ("samples", _beta.rvs(a_i, b_i, size=n_prior, random_state=rng))

    xlabels = {ph: rf"$\mu_{{{ph}}}$" for ph in phase_names}
    _plot_posteriors_publication(
        decoded.mu, phase_names, xlabels, priors,
        None, savepath,
    )


def plot_posteriors_log_alpha0(decoded: DecodedPosterior,
                               log_alpha0_prior: dict,
                               savepath: Path) -> None:
    """
    Single-panel posterior over log(alpha_0), with the prior overlaid.

    `log_alpha0_prior` is a dict with key "type":
        {"type": "uniform",   "low": ..., "high": ...}  -> flat dotted line
        {"type": "lognormal", "mu":  ..., "sigma": ...} -> Normal density curve
    """
    if decoded.log_alpha0 is None:
        raise ValueError("decoded.log_alpha0 is None; not a pseudo-marginal run.")

    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    color = sns.color_palette("husl", 1)[0]
    sns.kdeplot(decoded.log_alpha0, color=color, linewidth=2, ax=ax)
    ax.hist(decoded.log_alpha0, bins=50, density=True,
            alpha=0.6, color=color, edgecolor="none")

    ptype = log_alpha0_prior["type"]
    if ptype == "uniform":
        lo = float(log_alpha0_prior["low"])
        hi = float(log_alpha0_prior["high"])
        width = hi - lo
        if width > 0:
            ax.hlines(1.0 / width, lo, hi,
                      colors="red", linestyles="dotted", linewidth=3)
    elif ptype == "lognormal":
        mu_la    = float(log_alpha0_prior["mu"])
        sigma_la = float(log_alpha0_prior["sigma"])
        # Cover the union of the prior bulk and the posterior support.
        post_lo, post_hi = float(decoded.log_alpha0.min()), float(decoded.log_alpha0.max())
        x_lo = min(post_lo, mu_la - 4.0 * sigma_la)
        x_hi = max(post_hi, mu_la + 4.0 * sigma_la)
        x = np.linspace(x_lo, x_hi, 400)
        pdf = (
            np.exp(-0.5 * ((x - mu_la) / sigma_la) ** 2)
            / (sigma_la * np.sqrt(2.0 * np.pi))
        )
        ax.plot(x, pdf, color="red", linestyle="dotted", linewidth=3)
    else:
        raise ValueError(f"unknown log_alpha0_prior type: {ptype!r}")

    ax.set_xlabel(r"$\log \alpha_0$")
    ax.set_ylabel("Density")

    posterior_proxy = Line2D([0], [0], color=color, linewidth=2)
    prior_proxy     = Line2D([0], [0], color="red", linestyle="dotted", linewidth=2)
    fig.legend(
        handles=[posterior_proxy, prior_proxy],
        labels=["Posterior", "Prior"],
        loc="lower center", ncol=2, #fontsize=13,
        frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, -0.15),
    )
    plt.savefig(savepath, bbox_inches="tight")
    plt.close(fig)

def plot_posterior_eta(decoded: DecodedPosterior,
                       eta_low: float,
                       eta_high: float,
                       savepath: Path) -> None:
    """
    Single-panel posterior over the noise-boost parameter eta, with the
    uniform prior on [eta_low, eta_high] overlaid as a flat dotted line
    at height 1/(eta_high - eta_low).
    Only called when decoded.noise_mode == "mcmc".
    """
    if decoded.eta is None or decoded.noise_mode != "mcmc":
        raise ValueError(
            "plot_posterior_eta requires a noise_mode == 'mcmc' run with "
            "sampled eta; got noise_mode="
            f"{decoded.noise_mode!r}, eta is None={decoded.eta is None}."
        )
    eta_samples = decoded.eta
    color = sns.color_palette("husl", 1)[0]
    fig, ax_eta = plt.subplots(figsize=(4, 3), constrained_layout=True)
    # --- eta posterior + uniform prior ----------------------------------
    sns.kdeplot(eta_samples, color=color, linewidth=2, ax=ax_eta)
    ax_eta.hist(eta_samples, bins=50, density=True,
                alpha=0.6, color=color, edgecolor="none")
    width = eta_high - eta_low
    if width > 0:
        ax_eta.hlines(1.0 / width, eta_low, eta_high,
                      colors="red", linestyles="dotted", linewidth=3)
    ax_eta.set_xlabel(r"$\eta$")
    ax_eta.set_ylabel("Density")
    posterior_proxy = Line2D([0], [0], color=color, linewidth=2)
    prior_proxy     = Line2D([0], [0], color="red", linestyle="dotted",
                             linewidth=2)
    fig.legend(
        handles=[posterior_proxy, prior_proxy],
        labels=["Posterior", "Prior"],
        loc="lower center", ncol=2,
        frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, -0.15),
    )
    plt.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
# def plot_posterior_eta(decoded: DecodedPosterior,
#                        eta_low: float,
#                        eta_high: float,
#                        savepath: Path) -> None:
#     """
#     Two-panel posterior over the noise-boost parameter.

#     Left:  eta itself, with the uniform prior on [eta_low, eta_high]
#            overlaid as a flat dotted line at height 1/(eta_high-eta_low).
#     Right: the variance multiplier v = exp(eta), with the *induced*
#            prior overlaid. If eta ~ Uniform(a, b), then v = exp(eta) has
#            density
#                 p_v(v) = 1 / ((b - a) * v),    v in [exp(a), exp(b)],
#            i.e. the bounded log-uniform / Jeffreys form. This is plotted
#            as a dotted curve (not a flat line) so the prior comparison
#            on the right is honest.

#     Only called when decoded.noise_mode == "mcmc".
#     """
#     if decoded.eta is None or decoded.noise_mode != "mcmc":
#         raise ValueError(
#             "plot_posterior_eta requires a noise_mode == 'mcmc' run with "
#             "sampled eta; got noise_mode="
#             f"{decoded.noise_mode!r}, eta is None={decoded.eta is None}."
#         )

#     eta_samples = decoded.eta
#     v_samples = np.exp(eta_samples)
#     color = sns.color_palette("husl", 1)[0]

#     fig, (ax_eta, ax_v) = plt.subplots(
#         1, 2, figsize=(11, 3.8), constrained_layout=True
#     )

#     # --- Left panel: eta posterior + uniform prior ----------------------
#     sns.kdeplot(eta_samples, color=color, linewidth=2, ax=ax_eta)
#     ax_eta.hist(eta_samples, bins=50, density=True,
#                 alpha=0.6, color=color, edgecolor="none")
#     width = eta_high - eta_low
#     if width > 0:
#         ax_eta.hlines(1.0 / width, eta_low, eta_high,
#                       colors="red", linestyles="dotted", linewidth=3)
#     ax_eta.set_xlabel(r"$\eta$")
#     ax_eta.set_ylabel("Density")

#     # --- Right panel: exp(eta) posterior + induced prior ----------------
#     sns.kdeplot(v_samples, color=color, linewidth=2, ax=ax_v)
#     ax_v.hist(v_samples, bins=50, density=True,
#               alpha=0.6, color=color, edgecolor="none")
#     if width > 0:
#         v_lo = float(np.exp(eta_low))
#         v_hi = float(np.exp(eta_high))
#         # Dense grid through the prior support; plotted as a dotted curve.
#         v_grid = np.linspace(v_lo, v_hi, 400)
#         prior_density = 1.0 / (width * v_grid)
#         ax_v.plot(v_grid, prior_density,
#                   color="red", linestyle="dotted", linewidth=3)
#     ax_v.set_xlabel(r"$\exp(\eta)$  (variance multiplier)")
#     ax_v.set_ylabel("Density")

#     posterior_proxy = Line2D([0], [0], color=color, linewidth=2)
#     prior_proxy     = Line2D([0], [0], color="red", linestyle="dotted",
#                              linewidth=2)
#     fig.legend(
#         handles=[posterior_proxy, prior_proxy],
#         labels=["Posterior", "Prior"],
#         loc="lower center", ncol=2, #fontsize=13,
#         frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, -0.20),
#     )
#     plt.savefig(savepath, bbox_inches="tight")
#     plt.close(fig)


def plot_validation_pattern(Q: np.ndarray,
                            mean_intensity: np.ndarray,
                            std_intensity: np.ndarray,
                            observed: np.ndarray,
                            savepath: Path,
                            wavelength: float) -> None:
    """Q-axis validation plot: mean +/- 1 sigma vs. observed."""
    def q_to_d(q):
        q = np.asarray(q)
        return np.where(q == 0, np.inf, 2 * np.pi / q)

    def d_to_q(d):
        d = np.asarray(d)
        return np.where(d == 0, np.inf, 2 * np.pi / d)

    plt.rcParams.update({"figure.figsize": (6, 4)})
    fig, ax = plt.subplots()

    ax.plot(Q, mean_intensity, label="Mean intensity")
    ax.fill_between(Q,
                    mean_intensity - std_intensity,
                    mean_intensity + std_intensity,
                    alpha=0.3, label=r"$\pm 1\sigma$")
    ax.plot(Q, observed, "k--", label="XRD Data")

    ax.set_xlabel(r"$Q\ (\mathrm{\AA^{-1}})$")
    ax.set_ylabel("Intensity (counts)")
    ax.legend()

    secax = ax.secondary_xaxis("top", functions=(q_to_d, d_to_q))
    secax.set_xlabel(r"$d\ (\mathrm{\AA})$", labelpad=10)
    d_ticks = np.array([2, 1])
    qmin, qmax = ax.get_xlim()
    q_ticks = 2 * np.pi / d_ticks
    mask = (q_ticks >= qmin) & (q_ticks <= qmax)
    secax.set_xticks(d_ticks[mask])
    secax.set_xticklabels([f"{d:g}" for d in d_ticks[mask]])

    ax.set_yscale("function",
                  functions=(lambda x: np.sqrt(np.abs(x)), lambda x: x ** 2))
    #yticks = [0, 5e4, 2e5, 5e5, 1e6]
    #ylabels = ["0", "5e4", "2e5", "5e5", "1e6"]
    yticks = [0, 1e5, 5e5, 2e6, 4e6] 
    ylabels =["0", "1e5", "5e5", "2e6", "4e6"] 
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(0, 4e6)

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# s* rescaling (matches run_mcmc.py likelihood exactly)
# =============================================================================

def s_star(y_obs: np.ndarray, y_pred: np.ndarray,
           sigma_approx_sq: np.ndarray) -> np.ndarray:
    """
    Per-sample analytic histogram-scale factor.
        s* = sum(y_obs * y_pred / sigma_approx^2) / sum(y_pred^2 / sigma_approx^2)
    y_pred shape (M, n_bins); returns (M,).

    Matches the formula in make_log_prob in run_mcmc.py.
    """
    A = np.sum(y_pred ** 2     / sigma_approx_sq[None, :], axis=1)
    B = np.sum(y_obs * y_pred / sigma_approx_sq[None, :], axis=1)
    return np.divide(B, A, out=np.zeros_like(B), where=A > 0)


def build_sigma_approx_sq(y_obs: np.ndarray, GP_std: np.ndarray,
                          epsilon: float) -> np.ndarray:
    return np.maximum(y_obs + GP_std ** 2, epsilon)


# =============================================================================
# Surrogate validation (new payload format)
# =============================================================================

class RFPCASurrogate:
    """Loads the new-format surrogate (final_model.pkl + separate PCA file)
    and exposes g_i(m, h) at unit phase fraction. Mirrors RFPCASurrogate in
    run_mcmc.py exactly."""

    def __init__(self, run_dir: Path, pca_dir: Path, phase: str):
        self.phase = phase

        model_path = run_dir / phase / "final_model.pkl"
        if not model_path.is_file():
            raise FileNotFoundError(f"missing surrogate model: {model_path}")
        with open(model_path, "rb") as fh:
            payload = pickle.load(fh)
        self.model    = payload["model"]
        self.kept_pcs = np.asarray(payload["kept_pcs"], dtype=int)

        pca_path = pca_dir / f"{phase}_pca_full.pkl"
        if not pca_path.is_file():
            raise FileNotFoundError(f"missing PCA file: {pca_path}")
        with open(pca_path, "rb") as fh:
            self.pca = pickle.load(fh)["pca"]
        self.n_components = self.pca.n_components_

    def predict_g(self, mustrain: np.ndarray, hstrain: np.ndarray) -> np.ndarray:
        B = mustrain.shape[0]
        X = np.column_stack([np.ones(B, dtype=float), mustrain, hstrain])
        Z_kept = self.model.predict(X)
        Z_full = np.zeros((B, self.n_components))
        Z_full[:, self.kept_pcs] = Z_kept
        return self.pca.inverse_transform(Z_full)


def predict_surrogate_pattern(surrogates: dict[str, RFPCASurrogate],
                              phase_names: list[str],
                              fracs: np.ndarray,
                              mustrain: np.ndarray,
                              hstrain: np.ndarray) -> np.ndarray:
    """Y_calc = sum_i f_i * g_i(m_i, h_i). Returns (M, n_2theta)."""
    Y = None
    for i, ph in enumerate(phase_names):
        g = surrogates[ph].predict_g(mustrain[:, i], hstrain[:, i])
        if Y is None:
            Y = np.zeros((fracs.shape[0], g.shape[1]))
        Y += fracs[:, i:i + 1] * g
    return Y


def run_surrogate_validation(decoded: DecodedPosterior,
                             phase_names: list[str],
                             observed_minus_bg: np.ndarray,
                             #y_obs: np.ndarray,
                             GP_std: np.ndarray,
                             run_dir: Path,
                             pca_dir: Path,
                             keep_idx: np.ndarray,
                             epsilon: float,
                             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Surrogate forward predictions for every post-burn-in sample, each
    scaled by its own analytic s* (same formula as the likelihood).
    Returns (mean_intensity, std_intensity) over the s*-scaled stack,
    masked to keep_idx so the result aligns with y_obs / observed_minus_bg.
    """
    surrogates = {
        ph: RFPCASurrogate(run_dir, pca_dir, ph) for ph in phase_names
    }
    y_pred_full = predict_surrogate_pattern(
        surrogates, phase_names,
        decoded.phase_fractions, decoded.mustrain, decoded.hstrain,
    )                                                          # (M, n_2theta_full)
    y_pred = y_pred_full[:, keep_idx]                          # (M, n_keep)

    sigma_approx_sq = build_sigma_approx_sq(observed_minus_bg, GP_std, epsilon)
    s = s_star(observed_minus_bg, y_pred, sigma_approx_sq)            # (M,)
    y_scaled = y_pred * s[:, None]
    return y_scaled.mean(axis=0), y_scaled.std(axis=0)


# =============================================================================
# GSAS-II forward model worker (subprocess wrapper)
# =============================================================================

class GSASWorker:
    """Spawn model_v1.py and talk to it over stdin/stdout. Context manager.

    Protocol:
      * Worker prints 'READY' on stdout when initialisation finishes.
      * Send 'RUN_JOB <json-dict>\\n' to stdin.
      * Worker replies on stdout with:
            JSON_START
            <one-line json array of y_calc>
            JSON_END
        ...or a single-line {"error": "..."} JSON dict on failure.
      * Send 'EXIT\\n' to terminate cleanly.
    """

    def __init__(self,
                 worker_script: Path,
                 python_exe: str = GSAS_WORKER_PYTHON,
                 ready_timeout_s: float = GSAS_READY_TIMEOUT_S,
                 job_timeout_s: float = GSAS_WORKER_TIMEOUT_S):
        self.worker_script = Path(worker_script)
        self.python_exe = python_exe
        self.ready_timeout_s = ready_timeout_s
        self.job_timeout_s = job_timeout_s
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "GSASWorker":
        if not self.worker_script.exists():
            raise FileNotFoundError(f"worker script not found: {self.worker_script}")
        self.proc = subprocess.Popen(
            [self.python_exe, str(self.worker_script)],
            cwd=str(self.worker_script.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._wait_for_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None and self.proc.stdin is not None:
                try:
                    self.proc.stdin.write("EXIT\n")
                    self.proc.stdin.flush()
                except (BrokenPipeError, ValueError):
                    pass
            try:
                self.proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        finally:
            self.proc = None

    def _wait_for_ready(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"GSAS-II worker exited during startup (rc={self.proc.returncode})."
                )
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            line = line.strip()
            if line == "READY":
                return
            print(f"Worker output (before READY): {line}", file=sys.stderr, flush=True)
        raise TimeoutError(
            f"GSAS-II worker did not print READY within {self.ready_timeout_s:.0f}s"
        )

    def run(self, params: dict) -> np.ndarray:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("worker not running")
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"GSAS-II worker died (rc={self.proc.returncode})."
            )

        payload = json.dumps(params)
        self.proc.stdin.write(f"RUN_JOB {payload}\n")
        self.proc.stdin.flush()

        deadline = time.monotonic() + self.job_timeout_s
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"GSAS-II worker died mid-job (rc={self.proc.returncode})."
                    )
                continue
            line = line.strip()
            if line == "JSON_START":
                break
            if line.startswith("{"):
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    print(f"IGNORED worker stdout: {line}", file=sys.stderr, flush=True)
                    continue
                if isinstance(parsed, dict) and "error" in parsed:
                    raise RuntimeError(f"GSAS-II worker error: {parsed['error']}")
            else:
                print(f"IGNORED worker stdout: {line}", file=sys.stderr, flush=True)
        else:
            raise TimeoutError(
                f"GSAS-II job timed out after {self.job_timeout_s:.0f}s waiting for JSON_START"
            )

        json_lines: list[str] = []
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"GSAS-II worker died mid-job (rc={self.proc.returncode})."
                    )
                continue
            line = line.strip()
            if line == "JSON_END":
                response = "\n".join(json_lines)
                try:
                    payload = json.loads(response)

                    if isinstance(payload, dict):
                        return np.asarray(payload["ycalc"], dtype=float)

                    return np.asarray(payload, dtype=float)

                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"failed to parse JSON from worker: {e}\n"
                        f"worker stdout: {response[:500]}"
                    )
            json_lines.append(line)

        raise TimeoutError(
            f"GSAS-II job timed out after {self.job_timeout_s:.0f}s reading JSON payload"
        )


# =============================================================================
# GSAS-II validation
# =============================================================================

def build_gsas_param_dict(phase_names: list[str],
                          phase_fractions: np.ndarray,
                          mustrain: np.ndarray,
                          hstrain: np.ndarray,
                          all_phases: list[str]) -> dict[str, float]:
    """
    Translate one decoded posterior point into the keyed dict
    model_v1.py wants.

    Emits per phase, using MCMC phase names verbatim:
        scale_<phase>
        mustrain_<phase>
        hstrain_<phase>_D11

    `all_phases` is the full phase list known to the GSAS-II project
    template (typically the names from config["phases"]). Any phase in
    all_phases that the MCMC did not fit gets scale_<phase> = 0.0 so
    GSAS-II evaluates a clean zero contribution.
    """
    params: dict[str, float] = {}
    for j, ph in enumerate(phase_names):
        params[f"scale_{ph}"]          = float(phase_fractions[j])
        params[f"mustrain_{ph}"]       = float(mustrain[j])
        params[f"hstrain_{ph}_D11"]    = float(hstrain[j])
    for ph in all_phases:
        if ph in phase_names:
            continue
        params[f"scale_{ph}"]       = 0.0

    return params


def run_gsas_validation(decoded: DecodedPosterior,
                        phase_names: list[str],
                        y_obs: np.ndarray,
                        GP_std: np.ndarray,
                        n_variance_draws: int,
                        seed: int,
                        keep_idx: np.ndarray,
                        epsilon: float,
                        all_phases: list[str],
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean-parameter calc line + variance band from n_variance_draws random
    posterior samples. Each calc is rescaled by its own analytic s*
    against (y_obs, sigma_approx) — same formula as the likelihood and
    surrogate validation, so the two validation figures are comparable.

    `y_obs` and `GP_std` are masked-length (length keep_idx.size). The
    GSAS-II worker returns full-length calc patterns; we index them by
    keep_idx before computing s*. The returned calc_line and std_band
    are full-length (n_2theta_full) so callers can plot the full pattern
    or index by keep_idx as they see fit.
    """
    sigma_approx_sq = build_sigma_approx_sq(y_obs, GP_std, epsilon)

    # Mean parameter set.
    mean_pf       = decoded.phase_fractions.mean(axis=0)
    mean_mustrain = decoded.mustrain.mean(axis=0)
    mean_hstrain  = decoded.hstrain.mean(axis=0)

    rng = np.random.default_rng(seed)
    n_total = decoded.phase_fractions.shape[0]
    if n_total < n_variance_draws:
        raise ValueError(
            f"Only {n_total} post-burn-in samples available; "
            f"need {n_variance_draws} for variance band."
        )
    idx = rng.choice(n_total, size=n_variance_draws, replace=False)

    print(f"GSAS-II validation: 1 mean + {n_variance_draws} variance draws "
          f"(out of {n_total} post-burn-in samples).")

    with GSASWorker(GSAS_WORKER_SCRIPT) as worker:
        # Mean parameter -> calc line.
        params = build_gsas_param_dict(
            phase_names, mean_pf, mean_mustrain, mean_hstrain, all_phases
        )
        t0 = time.perf_counter()
        y_calc_mean = worker.run(params)
        print(f"  mean-param call: {time.perf_counter() - t0:.2f}s, "
              f"{y_calc_mean.size} bins")

        # s* for the mean calc, computed on the masked region where y_obs
        # and sigma_approx_sq live, then applied to the full calc line.
        s_mean = s_star(y_obs, y_calc_mean[None, keep_idx], sigma_approx_sq)[0]
        calc_line = y_calc_mean * s_mean

        # Variance draws.
        n_bins = y_calc_mean.size
        stack = np.empty((n_variance_draws, n_bins))
        for k, i in enumerate(idx):
            params_i = build_gsas_param_dict(
                phase_names,
                decoded.phase_fractions[i],
                decoded.mustrain[i],
                decoded.hstrain[i],
                all_phases,
            )
            t0 = time.perf_counter()
            y_calc_i = worker.run(params_i)
            s_i = s_star(y_obs, y_calc_i[None, keep_idx], sigma_approx_sq)[0]
            stack[k] = y_calc_i * s_i
            if (k + 1) % 10 == 0 or k == n_variance_draws - 1:
                print(f"  draw {k + 1}/{n_variance_draws} "
                      f"({time.perf_counter() - t0:.2f}s)")

    std_band = stack.std(axis=0)
    return calc_line, std_band


# =============================================================================
# Q-axis helpers + observed-pattern loader
# =============================================================================

def two_theta_to_Q(two_theta_deg: np.ndarray,
                   wavelength: float) -> np.ndarray:
    theta_rad = np.radians(two_theta_deg / 2.0)
    return 4 * np.pi * np.sin(theta_rad) / wavelength

def load_observed_arrays(run_name: str,
                         trim: int | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load observed (x_2theta_deg, y_intensity) for `run_name`.

    Looks for .npy pair in DATA_DIR first, then falls back to a .fxye
    file in the parent data_dir/ folder. Exactly one of these must exist.

    .npy:  DATA_DIR/<run_name>_x_spacing.npy + <run_name>_pattern.npy
    .fxye: data_dir/<run_name>.fxye  (x stored as 100 * two_theta)
    """
    npy_x = DATA_DIR / f"{run_name}_x_spacing.npy"
    npy_y = DATA_DIR / f"{run_name}_pattern.npy"
    if npy_x.is_file() and npy_y.is_file():
        x = np.asarray(np.load(npy_x), dtype=float)
        y = np.asarray(np.load(npy_y), dtype=float)
    else:
        fxye = DATA_DIR / f"ED_ID35_reduced.fxye"
        if not fxye.is_file():
            raise FileNotFoundError(
                f"No observed data for run {run_name!r}. "
                f"Tried {npy_x} / {npy_y} and {fxye}."
            )
        x = np.loadtxt(fxye, usecols=0, skiprows=3, encoding="latin1") / 100.0
        y = np.loadtxt(fxye, usecols=1, skiprows=3, encoding="latin1")

    if trim is not None:
        x = x[:trim]
        y = y[:trim]
    return x, y

# =============================================================================
# Prior bounds from config.json
# =============================================================================

def priors_from_config(config: dict
                       ) -> tuple[dict[str, tuple[float, float]],
                                  dict[str, tuple[float, float]],
                                  dict[str, tuple[float, float]]]:
    """Per-phase prior boxes from config.json. Single source of truth."""
    pf, mu, h = {}, {}, {}
    for entry in config["phases"]:
        name = entry["name"]
        pf[name] = tuple(entry["scale_bounds"])
        mu[name] = tuple(entry["mustrain_bounds"])
        h[name]  = tuple(entry["hstrain_bounds"])
    return pf, mu, h

#Clean up GSAS-II workers
def cleanup_worker_files(workdir=None):
    """Delete per-process GSAS-II artifacts left by the worker."""
    workdir = workdir or os.getcwd()
    patterns = [
        "sample_pid*.gpx",
        "sample_pid*.bak*.gpx",
        "sample_pid*.lst",
    ]
    deleted = 0
    for pat in patterns:
        for path in glob.glob(os.path.join(workdir, pat)):
            try:
                os.remove(path)
                deleted += 1
            except OSError as e:
                print(f"  could not delete {path}: {e}")
    if deleted:
        print(f"Cleanup: removed {deleted} worker artifact file(s) from {workdir}")

# Error Statistic
def r_pattern_factor(y_obs: np.ndarray, y_calc: np.ndarray) -> float:
    """
    Profile R-factor (Rp), as a percentage:

        Rp = 100 * sum_i |y_obs_i - y_calc_i| / sum_i |y_obs_i|

    Both inputs must already be on the same scale (i.e. y_calc has the
    scale factor s* applied) and on the same masked grid. Pass the
    background-subtracted observed pattern for consistency with the
    verification plot.
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_calc = np.asarray(y_calc, dtype=float)
    denom = np.sum(np.abs(y_obs))
    if denom <= 0:
        raise ValueError("sum(|y_obs|) is non-positive; cannot compute Rp.")
    return 100.0 * np.sum(np.abs(y_obs - y_calc)) / denom

def load_gp_fit(config, out_dir, keep_idx, strict: bool = True):
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
    When strict=False, the gp-hash and keep_idx checks are downgraded to
    warnings rather than errors. The length check remains fatal because a
    length mismatch would break the likelihood computation.
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
        msg = (
            f"GP fit is stale: gp_hash mismatch "
            f"(saved={saved_hash}, expected={expected_hash}). The gp/"
            f"data_ranges/dataset settings changed since fit_gp.py was "
            f"run. Re-run fit_gp.py."
        )
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg} Proceeding anyway (strict=False).", flush=True)
    if "keep_idx" in data:
        saved_keep = np.asarray(data["keep_idx"], dtype=int)
        if not np.array_equal(saved_keep, keep_idx):
            msg = (
                "GP fit keep_idx does not match the current data mask. "
                "Re-run fit_gp.py with the current config."
            )
            if strict:
                raise ValueError(msg)
            print(f"WARNING: {msg} Proceeding anyway (strict=False).", flush=True)
    for name, arr in (("GP_pred", GP_pred), ("GP_std", GP_std)):
        if arr.shape[0] != keep_idx.size:
            raise ValueError(
                f"{name} length ({arr.shape[0]}) does not match the masked "
                f"data length ({keep_idx.size}). Re-run fit_gp.py."
            )
    print(f"Loaded GP fit from {gp_path} (gp_hash={saved_hash}).", flush=True)
    return GP_pred, GP_std

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    atexit.register(cleanup_worker_files)
    setup_publication_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load run metadata --------------------------------------------
    print(f"Loading run metadata from {RUN_DIR} ...")
    with open(CONFIG_JSON) as f:
        config = json.load(f)

    # Sanity-check the script-local run_name against the config copy.
    cfg_run_name = config.get("run_name")
    if cfg_run_name != run_name:
        raise ValueError(
            f"run_name mismatch: script has {run_name!r} but "
            f"{CONFIG_JSON} has {cfg_run_name!r}. Fix the script's "
            f"run_name (and RUN_DIR) to match the run you're analyzing."
        )

    # Single source of truth: everything below comes from config.json.
    wavelength = float(config["dataset"]["Xray_wavelength"])
    epsilon    = float(config["likelihood"]["epsilon"])
    #all_phases = [p["name"] for p in config["phases"]] #Code isn't equiped to handle absent phases like this due to asumptions in model_old.py
    all_phases = ["gamma","delta", "gamma1", "gamma2", "laves", "carbide"]
    surrogate_run_dir = Path(config["surrogate"]["run_dir"])
    surrogate_pca_dir = Path(config["surrogate"]["pca_dir"])

    decoder = load_decoder(STANDARDIZER_JSON)
    phase_names = decoder.phase_names
    is_pm = decoder.is_pseudo_marginal

    # Cross-check: if the standardizer says pseudo-marginal, config.json
    # must carry a pseudo_marginal block (and vice versa). Catches the
    # case of a mismatched/stale standardizer.json.
    has_pm_cfg = "pseudo_marginal" in config and config.get(
        "mode", "direct"
    ) == "pseudo_marginal"
    if is_pm and not has_pm_cfg:
        raise ValueError(
            f"standardizer.json says pseudo-marginal but config.json has "
            f"mode={config.get('mode', 'direct')!r} or no pseudo_marginal "
            f"block. Run directory is inconsistent."
        )
    if (not is_pm) and has_pm_cfg:
        raise ValueError(
            f"config.json is pseudo-marginal but standardizer.json says "
            f"{decoder.parameterization!r}. Run directory is inconsistent."
        )

    print(f"  mode:    {'pseudo-marginal' if is_pm else 'direct'}")
    print(f"  phases ({decoder.N}): {phase_names}")
    print(f"  ndim:  {decoder.ndim}")
    print(f"  ref:   {phase_names[decoder.ref_index]} "
          f"(index {decoder.ref_index})")

    pf_priors, mu_priors, h_priors = priors_from_config(config)

    # --- 2. Load chain & diagnostics -------------------------------------
    print(f"Loading chain from {CHAIN_HDF5} ...")
    cd = load_chain(CHAIN_HDF5, expected_ndim=decoder.ndim)
    print(f"  shape: nsteps={cd.nsteps}, nwalkers={cd.nwalkers}, ndim={cd.ndim}")

    # Burn-in is set algorithmically: keep the last KEEP_LAST_STEPS, burn
    # the rest. Guarded so that short chains just keep everything.
    burn_in = max(0, cd.nsteps - KEEP_LAST_STEPS)
    print(f"Burn-in: discarding first {burn_in} of {cd.nsteps} steps "
          f"(keeping last {cd.nsteps - burn_in}).")

    print_diagnostics(cd, CHAIN_HDF5, burn_in)

    # --- 3. Trace plot ----------------------------------------------------
    plot_traces(cd, FIGURES_DIR / "traces.pdf",
                param_names=decoder.param_names)

    # --- 4. Decode posterior ---------------------------------------------
    if is_pm:
        alpha_floor = float(
            config.get("pseudo_marginal", {}).get("alpha_floor", 1e-6)
        )
        decoded = decode_burned(
            cd, decoder, burn_in,
            k_draws=N_FRAC_DRAWS_PER_SAMPLE,
            alpha_floor=alpha_floor,
            seed=FRAC_RESAMPLE_SEED,
        )
        print(f"Walker post-burn-in samples: {decoded.raw_burned.shape[0]}")
        print(f"Dirichlet resamples (k={decoded.k_draws}): "
              f"phase-fraction stack of {decoded.phase_fractions.shape[0]} rows")
    else:
        decoded = decode_burned(cd, decoder, burn_in)
        print(f"Post-burn-in samples: {decoded.raw_burned.shape[0]}")

    print(f"Mean phase fractions: {decoded.phase_fractions.mean(axis=0)}")
    print(f"Mean mustrain:        {decoded.mustrain.mean(axis=0)}")
    print(f"Mean hstrain:         {decoded.hstrain.mean(axis=0)}")
    if is_pm:
        print(f"Mean mu:              {decoded.mu.mean(axis=0)}")
        print(f"Mean log_alpha_0:     {decoded.log_alpha0.mean():.3f}")

    write_posterior_summary(
        decoded, phase_names, RUN_DIR / "posterior_summary.json"
    )
    summary_path = RUN_DIR / "posterior_summary.json"
    # Seed the r_factors section so consumers can tell "didn't run" from
    # "ran but result missing". Each validator below overwrites its slot.
    update_posterior_summary(summary_path, {"r_factors": {
        "surrogate": None,
        "gsas":      None,
    }})

    # --- 5. Posterior subplots --------------------------------------------
    if is_pm:
        pm_cfg = config["pseudo_marginal"]
        a_mu = np.asarray(pm_cfg["mu_dirichlet_prior"], dtype=float)
        # Build the log_alpha0 prior spec from config. Defaults to
        # uniform for backwards compatibility with older config files.
        la_prior_type = pm_cfg.get("log_alpha0_prior", "uniform")
        if la_prior_type == "uniform":
            la_low, la_high = pm_cfg["log_alpha0_bounds"]
            log_alpha0_prior = {
                "type": "uniform",
                "low":  float(la_low),
                "high": float(la_high),
            }
        elif la_prior_type == "lognormal":
            params = pm_cfg["log_alpha0_lognormal_params"]
            log_alpha0_prior = {
                "type":  "lognormal",
                "mu":    float(params["mu"]),
                "sigma": float(params["sigma"]),
            }
        else:
            raise ValueError(
                f"unknown pseudo_marginal.log_alpha0_prior: {la_prior_type!r}"
            )
        plot_posteriors_phase_fractions_pm(
            decoded, phase_names, a_mu, log_alpha0_prior,
            PHASE_FRACTION_TRUE,
            n_prior_draws=N_PRIOR_DRAWS,
            prior_seed=FRAC_RESAMPLE_SEED,
            savepath=FIGURES_DIR / "posteriors_phase_fractions.pdf",
        )
        plot_posteriors_mu(
            decoded, phase_names, a_mu,
            FIGURES_DIR / "posteriors_mu.pdf",
        )
        plot_posteriors_log_alpha0(
            decoded, log_alpha0_prior,
            FIGURES_DIR / "posteriors_log_alpha0.pdf",
        )
    else:
        plot_posteriors_phase_fractions(
            decoded, phase_names, pf_priors, PHASE_FRACTION_TRUE,
            FIGURES_DIR / "posteriors_phase_fractions.pdf",
        )
    plot_posteriors_mustrain(
        decoded, phase_names, mu_priors, MUSTRAIN_TRUE,
        FIGURES_DIR / "posteriors_mustrain.pdf",
    )
    plot_posteriors_hstrain(
        decoded, phase_names, h_priors, HSTRAIN_TRUE,
        FIGURES_DIR / "posteriors_hstrain.pdf",
    )

    # Noise-boost (eta) posterior. Only produced when the run actually
    # sampled eta; the fixed-eta case is recorded in posterior_summary
    # but doesn't get a plot. Identical handling for direct and
    # pseudo-marginal modes because eta is the same trailing coordinate
    # either way.
    if decoded.noise_mode == "mcmc":
        plot_posterior_eta(
            decoded,
            eta_low=float(decoder.eta_low),
            eta_high=float(decoder.eta_high),
            savepath=FIGURES_DIR / "posteriors_eta.pdf",
        )
        print(
            f"eta posterior: mean={float(np.mean(decoded.eta)):.3f}, "
            f"std={float(np.std(decoded.eta)):.3f}; "
            f"exp(eta) mean={float(np.mean(np.exp(decoded.eta))):.3f}"
        )
    else:
        held = float(decoded.eta_fixed if decoded.eta_fixed is not None else 0.0)
        print(
            f"noise_boost.mode=fixed: eta held at {held:.3f} "
            f"(variance multiplier exp(eta)={np.exp(held):.3f}); "
            f"no eta posterior plotted."
        )

    # --- 6. Observed pattern + GP --------------------------------------
    x_2theta_full, y_obs_full = load_observed_arrays(run_name)
    n_2theta_full = int(x_2theta_full.size)

    data_ranges = config["dataset"].get("data_ranges")
    keep_idx = build_keep_idx(data_ranges, n_2theta_full)
    np.save(RUN_DIR / "data_mask.npy", keep_idx)
    GP_pred_masked, GP_std_masked = load_gp_fit(config, RUN_DIR, keep_idx, strict=False)

    x_2theta_masked, y_obs_masked = x_2theta_full[keep_idx], y_obs_full[keep_idx]
    Q                 = two_theta_to_Q(x_2theta_masked, wavelength)
    observed_minus_bg = y_obs_masked - GP_pred_masked

    # --- 7. Surrogate validation -----------------------------------------
    # run_surrogate_validation returns arrays already aligned to keep_idx
    # (length keep_idx.size), so they index 1:1 against observed_minus_bg
    # and Q.
    if RUN_SURROGATE_VALIDATION:
        print("Running surrogate (RF-PCA) validation ...")
        try:
            mean_int, std_int = run_surrogate_validation(
                decoded, phase_names, observed_minus_bg, GP_std_masked,
                surrogate_run_dir, surrogate_pca_dir,
                keep_idx, epsilon,
            )
            R_pattern_surrogate = r_pattern_factor(observed_minus_bg, mean_int)
            print(f"R_p statistic with the surrogate model: {R_pattern_surrogate}")
            update_posterior_summary(summary_path, {
                "r_factors": {"surrogate": float(R_pattern_surrogate)},
            })
            plot_validation_pattern(
                Q, mean_int, std_int, observed_minus_bg,
                FIGURES_DIR / "validation_surrogate.pdf",
                wavelength=wavelength,
            )
        except FileNotFoundError as e:
            print(f"  surrogate not available: {e}; skipping.")

    # --- 8. GSAS-II validation -------------------------------------------
    # run_gsas_validation returns full-length calc patterns (n_2theta_full);
    # we index by keep_idx for the R-factor and plot.
    if RUN_GSAS_VALIDATION:
        print("Running GSAS-II validation ...")
        calc_line, std_band = run_gsas_validation(
            decoded, phase_names,
            observed_minus_bg, GP_std_masked,
            n_variance_draws=N_VARIANCE_DRAWS, seed=VALIDATION_SEED,
            keep_idx=keep_idx, epsilon=epsilon, all_phases=all_phases,
        )
        R_pattern_gsas = r_pattern_factor(observed_minus_bg, calc_line[keep_idx])
        print(f"R_p statistic with the gsas model: {R_pattern_gsas}")
        update_posterior_summary(summary_path, {
            "r_factors": {"gsas": float(R_pattern_gsas)},
        })
        plot_validation_pattern(
            Q, calc_line[keep_idx], std_band[keep_idx], observed_minus_bg,
            FIGURES_DIR / "validation_gsas.pdf",
            wavelength=wavelength,
        )

    print("Done.")
    cleanup_worker_files()


if __name__ == "__main__":
    main()
