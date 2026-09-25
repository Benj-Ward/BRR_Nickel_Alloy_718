# train_surrogate_v3.py
#
# Stage 2 of the surrogate pipeline (stage 1 = prep_and_pca.py).
#
# For each phase:
#   1. Load X_train/Y_train/X_val/Y_val from data_splits/<run>/<phase>/
#      and the full-rank PCA from pca_transforms/<run>/.
#   2. Variance-truncate PCs: keep PCs [0..K-1] where K is set per
#      phase (PHASE_N_PCS). Variance-ordered, since R²-ordered did
#      not outperform it in our comparisons.
#   3. Train a single multi-output RandomForestRegressor (n_estimators
#      fixed at 200) to predict the kept PC scores from the 3 input
#      columns [scale, mustrain, hstrain].
#   4. Predict on val, reconstruct full patterns via inverse-PCA.
#   5. Compute validation metrics per sample: Rp, Rwp (uniform and
#      Poisson), Rexp, chi², and masked-APE @90%. Save per-sample CSV
#      and a summary JSON.
#   6. Build the 8×8 (mustrain × hstrain × 2θ) variance grid of the
#      val residuals, smoothing low-count bins by averaging
#      neighbors with global per-2θ variance as the last fallback.
#      Save as <phase>/variance.npz for MCMC likelihood use.
#   7. Plot error heatmaps over (mustrain, hstrain) for Rwp uniform,
#      Rwp Poisson, and masked-APE.
#   8. Plot variance diagnostics: standardized residual histogram and
#      Q-Q plot against the unit normal.
#
# Inputs:
#   data_splits/<run>/<phase>/X_{train,val}.npy, Y_{train,val}.npy
#   pca_transforms/<run>/<phase>_pca_full.pkl
#
# Outputs (in surrogates_final/<run>/<phase>/):
#   final_model.pkl
#   variance.npz
#   per_sample_metrics.csv
#   metrics_summary.json
#   error_heatmap_rwp_uniform.png
#   error_heatmap_rwp_poisson.png
#   error_heatmap_ape.png
#   variance_diagnostics/residual_hist.png
#   variance_diagnostics/qq_plot.png
#   variance_diagnostics/coverage.json
# At the run level:
#   surrogates_final/<run>/_config.json
#   surrogates_final/<run>/_summary.json

import os
import sys
import json
import pickle
import argparse
import glob
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from scipy.stats import binned_statistic_2d
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import DataConversionWarning

# Silence sklearn's "DataConversionWarning: A column-vector y was passed".
# We never pass column-vector targets, but the warning class is imported
# here to keep behavior consistent with train_surrogate_v2.py.
warnings.filterwarnings("ignore", category=DataConversionWarning)


# =========================================================================
# CONFIGURATION
# =========================================================================
PHASE_NAMES = ["gamma", "delta", "gamma1", "gamma2", "laves", "carbide"]

# Variance-truncated PC counts per phase.
PHASE_N_PCS = {
    "gamma":   240,
    "delta":   240,
    "gamma1":  240,
    "gamma2":  240,
    "laves":   240,
    "carbide": 240,
}

# RF hyperparameters. Fixed at n_estimators=200 — the sweep in v2
# showed marginal accuracy differences at 300/400 and we prefer the
# faster inference.
RF_KWARGS_BASE = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=105,
)

# Validation metric knobs.
APE_FRACTION = 1.0      # masked-APE top-energy fraction
WEIGHT_FLOOR = 1.0      # w_i = 1 / max(y_i, WEIGHT_FLOOR) for Poisson Rwp
HEATMAP_BINS = 8       # bins for the per-phase error heatmaps

# Inference / load timing: warmup once, then average wall-clock over
# this many repeats. The first call after a fresh load is consistently
# slowest due to memory layout / cache warmup and skews single-call
# estimates.
TIMING_REPEATS = 5

# Variance grid knobs — same defaults as compute_surrogate_variance.py.
N_BINS_MUSTRAIN = 8
N_BINS_HSTRAIN  = 8
MIN_COUNT_FOR_VARIANCE = 8
MAX_SMOOTHING_PASSES = 4

# Per-phase input bounds for the variance grid. Should match the data
# generator (driver_lhs_v2.py).
PHASE_BOUNDS = {
    "gamma":   {"mustrain": (1000.0, 41000.0), "hstrain": (-0.01, 0.01)},
    "delta":   {"mustrain": (1000.0, 41000.0), "hstrain": (-0.01, 0.01)},
    "gamma1":  {"mustrain": (1000.0, 41000.0), "hstrain": (-0.01, 0.01)},
    "gamma2":  {"mustrain": (1000.0, 41000.0), "hstrain": (-0.01, 0.01)},
    "laves":   {"mustrain": (1000.0, 41000.0), "hstrain": (-0.01, 0.01)},
    "carbide": {"mustrain": (1000.0, 41000.0), "hstrain": (-0.01, 0.01)},
}

# Roots.
SPLITS_ROOT     = "data_splits"
PCA_ROOT        = "pca_transforms"
SURROGATES_ROOT = "surrogates_final"


# =========================================================================
# PUBLICATION FIGURE SETTINGS
# =========================================================================
PUBLICATION_RC = {
    "figure.dpi":       300,
    "font.family":      "serif",
    "mathtext.fontset": "stix",
    "font.size":        14,
    "axes.labelsize":   16,
    "axes.titlesize":   16,
    "legend.fontsize":  13,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "axes.linewidth":   1.2,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.top":        False,
    "ytick.right":      False,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
}

# Display labels for the per-phase panels.
PHASE_DISPLAY = {
    "gamma":   r"$\gamma$",
    "delta":   r"$\delta$",
    "gamma1":  r"$\gamma'$",
    "gamma2":  r"$\gamma''$",
    "laves":   "Laves",
    "carbide": "Carbide",
}

# Cache populated as a side effect of plot_rp_heatmap_single and
# consumed by plot_combined_rp_heatmaps in main(). Keys are phase
# names; values are dicts with keys 'rp_grid', 'mu_edges', 'hs_edges'.
RP_HEATMAP_DATA = {}


# =========================================================================
# UTILITIES
# =========================================================================
def auto_detect_run(splits_root):
    cands = sorted(
        [d for d in glob.glob(os.path.join(splits_root, "run_*"))
         if os.path.isdir(d)],
        key=os.path.getmtime, reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"no run_* found under {splits_root}")
    return os.path.basename(cands[0])


def load_phase_assets(run_name, phase, splits_root, pca_root):
    """Load X/Y train and val, plus the saved PCA."""
    psd = os.path.join(splits_root, run_name, phase)
    X_train = np.load(os.path.join(psd, "X_train.npy"))
    Y_train = np.load(os.path.join(psd, "Y_train.npy"))
    #X_test   = np.load(os.path.join(psd, "X_test.npy"))
    #Y_test   = np.load(os.path.join(psd, "Y_test.npy"))
    X_val   = np.load(os.path.join(psd, "X_test.npy"))
    Y_val   = np.load(os.path.join(psd, "Y_test.npy"))
    with open(os.path.join(pca_root, run_name, f"{phase}_pca_full.pkl"), "rb") as fh:
        pca = pickle.load(fh)["pca"]
    return X_train, Y_train, X_val, Y_val, pca


def reconstruct(Z_pred_kept, kept_pcs, pca):
    """Zero-pad to full PCA rank, then pca.inverse_transform."""
    Z_full = np.zeros((Z_pred_kept.shape[0], pca.n_components_))
    Z_full[:, kept_pcs] = Z_pred_kept
    return pca.inverse_transform(Z_full)


# =========================================================================
# VALIDATION METRICS (from validate_surrogates.py)
# =========================================================================
def r_profile(y_true, y_pred):
    """Per-sample Rp in percent."""
    num = np.abs(y_true - y_pred).sum(axis=1)
    den = y_true.sum(axis=1)
    out = np.full_like(num, np.nan, dtype=float)
    nz = den != 0
    out[nz] = 100.0 * num[nz] / den[nz]
    return out


def poisson_weights(y_true, floor=WEIGHT_FLOOR):
    """w_i = 1 / max(y_i, floor)."""
    return 1.0 / np.maximum(y_true, floor)


def r_weighted_profile_uniform(y_true, y_pred):
    """Per-sample Rwp with uniform weights (normalized L2 residual)."""
    num = ((y_true - y_pred) ** 2).sum(axis=1)
    den = (y_true ** 2).sum(axis=1)
    out = np.full_like(num, np.nan, dtype=float)
    nz = den > 0
    out[nz] = 100.0 * np.sqrt(num[nz] / den[nz])
    return out


def r_weighted_profile_poisson(y_true, y_pred, w):
    """Per-sample Rwp with Poisson weights w = 1/max(y, floor)."""
    num = (w * (y_true - y_pred) ** 2).sum(axis=1)
    den = (w * y_true ** 2).sum(axis=1)
    out = np.full_like(num, np.nan, dtype=float)
    nz = den > 0
    out[nz] = 100.0 * np.sqrt(num[nz] / den[nz])
    return out


def r_expected(y_true, w, n_params):
    """Per-sample Rexp using Poisson weights. P = n_kept_pcs."""
    n_bins = y_true.shape[1]
    dof = max(n_bins - n_params, 1)
    den = (w * y_true ** 2).sum(axis=1)
    out = np.full_like(den, np.nan, dtype=float)
    nz = den > 0
    out[nz] = 100.0 * np.sqrt(dof / den[nz])
    return out


def chi_squared(rwp, rexp):
    out = np.full_like(rwp, np.nan, dtype=float)
    nz = (rexp > 0) & np.isfinite(rexp) & np.isfinite(rwp)
    out[nz] = (rwp[nz] / rexp[nz]) ** 2
    return out


def top_contributors_mask(y, fraction):
    """Bool mask of bins making up the top `fraction` of total intensity."""
    y = np.asarray(y)
    sorted_idx = np.argsort(y)[::-1]
    cum = np.cumsum(y[sorted_idx])
    total = y.sum()
    n = int(np.searchsorted(cum, fraction * total) + 1)
    mask = np.zeros_like(y, dtype=bool)
    mask[sorted_idx[:n]] = True
    return mask


def masked_ape(y_true, y_pred, fraction=APE_FRACTION):
    """Per-sample masked APE in percent. Mask built from y_true."""
    n = y_true.shape[0]
    out = np.empty(n)
    for i in range(n):
        m = top_contributors_mask(y_true[i], fraction)
        denom = y_true[i, m].sum()
        if denom == 0:
            out[i] = np.nan
        else:
            out[i] = 100.0 * np.abs(y_true[i, m] - y_pred[i, m]).sum() / denom
    return out


# =========================================================================
# VARIANCE GRID CONSTRUCTION (from compute_surrogate_variance.py)
# =========================================================================
def build_variance_grid(mustrain, hstrain, residuals,
                        mustrain_edges, hstrain_edges):
    """Per-bin (mustrain × hstrain × 2θ) population variance grid.

    mustrain, hstrain : (n_val,) — input coordinates per val sample
    residuals         : (n_val, n_2θ) — y_pred - y_true
    """
    n_mu = len(mustrain_edges) - 1
    n_hs = len(hstrain_edges) - 1
    n_2theta = residuals.shape[1]

    mu_idx = np.clip(np.digitize(mustrain, mustrain_edges) - 1, 0, n_mu - 1)
    hs_idx = np.clip(np.digitize(hstrain,  hstrain_edges)  - 1, 0, n_hs - 1)

    variance_grid = np.full((n_mu, n_hs, n_2theta), np.nan)
    count_grid    = np.zeros((n_mu, n_hs), dtype=int)
    for i in range(n_mu):
        for j in range(n_hs):
            mask = (mu_idx == i) & (hs_idx == j)
            n_in = int(mask.sum())
            count_grid[i, j] = n_in
            if n_in >= 2:
                variance_grid[i, j] = residuals[mask].var(axis=0, ddof=0)
    return variance_grid, count_grid


def smooth_sparse_bins(variance_grid, count_grid, min_count, max_passes):
    """Replace bins with count < min_count by the mean variance of their
    8-connected neighbors that DO have enough counts. Iterated up to
    max_passes times to propagate values across holes."""
    out = variance_grid.copy()
    sparse_mask = count_grid < min_count
    filled_by_smoothing = np.zeros_like(sparse_mask)
    n_mu, n_hs, _ = out.shape

    for _ in range(max_passes):
        if not sparse_mask.any():
            break
        new_values = {}
        for i in range(n_mu):
            for j in range(n_hs):
                if not sparse_mask[i, j]:
                    continue
                neighbor_vals = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n_mu and 0 <= nj < n_hs:
                            if not sparse_mask[ni, nj]:
                                neighbor_vals.append(out[ni, nj])
                if neighbor_vals:
                    new_values[(i, j)] = np.mean(neighbor_vals, axis=0)
        if not new_values:
            break
        for (i, j), val in new_values.items():
            out[i, j] = val
            sparse_mask[i, j] = False
            filled_by_smoothing[i, j] = True

    return out, filled_by_smoothing, sparse_mask


def apply_global_fallback(variance_grid, still_sparse, global_variance):
    """For bins smoothing couldn't fill, substitute the global per-2θ
    variance."""
    out = variance_grid.copy()
    n_mu, n_hs, _ = out.shape
    n_fallback = 0
    for i in range(n_mu):
        for j in range(n_hs):
            if still_sparse[i, j]:
                out[i, j] = global_variance
                n_fallback += 1
    return out, n_fallback


def gaussian_coverage(residuals, sigma_per_sample):
    """Fraction of |z| within 1, 2, 3 σ. Targets: 68.3 / 95.4 / 99.7%."""
    valid = sigma_per_sample > 0
    z = np.zeros_like(residuals)
    z[valid] = residuals[valid] / sigma_per_sample[valid]
    abs_z = np.abs(z[valid])
    return {
        "n_valid_points": int(valid.sum()),
        "frac_within_1sigma":  float((abs_z <= 1.0).mean()),
        "frac_within_2sigma":  float((abs_z <= 2.0).mean()),
        "frac_within_3sigma":  float((abs_z <= 3.0).mean()),
        "frac_beyond_5sigma":  float((abs_z > 5.0).mean()),
        "frac_beyond_10sigma": float((abs_z > 10.0).mean()),
    }


def assign_sigma_to_samples(variance_grid, mustrain_edges, hstrain_edges,
                            mustrain, hstrain):
    """Nearest-bin σ lookup for diagnostic purposes."""
    n_mu = len(mustrain_edges) - 1
    n_hs = len(hstrain_edges) - 1
    mu_idx = np.clip(np.digitize(mustrain, mustrain_edges) - 1, 0, n_mu - 1)
    hs_idx = np.clip(np.digitize(hstrain,  hstrain_edges)  - 1, 0, n_hs - 1)
    sigma2 = variance_grid[mu_idx, hs_idx]
    return np.sqrt(np.maximum(sigma2, 0.0))


# =========================================================================
# TIMING
# =========================================================================
def time_load(final_model_path, pca_path, repeats=TIMING_REPEATS):
    """Wall-clock time (in seconds) to load model + PCA from disk.

    We unpickle both files in sequence — the minimum needed to do
    inference, matching the (b) load definition: model + PCA. The
    variance grid is NOT loaded here; the MCMC loads that separately
    through surrogate_variance.py.

    A warmup load is performed first (OS file-cache state, Python's
    pickle dispatch tables, sklearn import caches, etc.). The reported
    time is the mean of `repeats` subsequent loads.
    """
    # Warmup.
    with open(final_model_path, "rb") as fh:
        _ = pickle.load(fh)
    with open(pca_path, "rb") as fh:
        _ = pickle.load(fh)

    elapsed = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with open(final_model_path, "rb") as fh:
            _ = pickle.load(fh)
        with open(pca_path, "rb") as fh:
            _ = pickle.load(fh)
        elapsed.append(time.perf_counter() - t0)
    return float(np.mean(elapsed))


def time_inference(rf, kept_pcs, pca, X_query, repeats=TIMING_REPEATS):
    """Wall-clock time (in seconds) for the full inference path:
    rf.predict → inverse-PCA reconstruction.

    X_query has shape (n_samples, 3) — already the right form for
    .predict. Performs one warmup call (first call is slowest due to
    cache warmup and any lazy initialization), then averages `repeats`
    timed calls.
    """
    # Warmup.
    Z_pred = rf.predict(X_query)
    Z_full = np.zeros((Z_pred.shape[0], pca.n_components_))
    Z_full[:, kept_pcs] = Z_pred
    _ = pca.inverse_transform(Z_full)

    elapsed = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        Z_pred = rf.predict(X_query)
        Z_full = np.zeros((Z_pred.shape[0], pca.n_components_))
        Z_full[:, kept_pcs] = Z_pred
        _ = pca.inverse_transform(Z_full)
        elapsed.append(time.perf_counter() - t0)
    return float(np.mean(elapsed))


# =========================================================================
# PLOTTING
# =========================================================================
def plot_rp_heatmap_single(X_val, rp, phase, path_no_ext, bins=HEATMAP_BINS):
    """Per-phase mean-Rp heatmap over (mustrain, hstrain).

    Bin edges are taken from PHASE_BOUNDS so the heatmap shares its
    spatial discretization with the variance grid. The binned grid is
    cached in RP_HEATMAP_DATA so plot_combined_rp_heatmaps can assemble
    the multi-panel appendix figure at the end of main().

    Saves both <path_no_ext>.pdf (for LaTeX) and <path_no_ext>.png
    (for quick inspection).
    """
    mustrain = X_val[:, 1]
    hstrain  = X_val[:, 2]

    b = PHASE_BOUNDS[phase]
    mu_edges = np.linspace(b["mustrain"][0], b["mustrain"][1], bins + 1)
    hs_edges = np.linspace(b["hstrain"][0],  b["hstrain"][1],  bins + 1)

    stat, _, _, _ = binned_statistic_2d(
        mustrain, hstrain, rp,
        statistic="mean", bins=[mu_edges, hs_edges],
    )
    counts, _, _, _ = binned_statistic_2d(
        mustrain, hstrain, rp,
        statistic="count", bins=[mu_edges, hs_edges],
    )
    stat = np.where(counts > 0, stat, np.nan)

    # Cache for the combined appendix figure.
    RP_HEATMAP_DATA[phase] = {
        "rp_grid":  stat,
        "mu_edges": mu_edges,
        "hs_edges": hs_edges,
    }

    with plt.rc_context(PUBLICATION_RC):
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("lightgrey")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(
            stat.T, origin="lower", aspect="auto",
            extent=[mu_edges[0], mu_edges[-1], hs_edges[0], hs_edges[-1]],
            cmap=cmap, interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"mean $R_p$ (%)")
        ax.set_xlabel(r"Mustrain $\epsilon_{iso}$")
        ax.set_ylabel(r"HStrain $D_{11}$")
        ax.set_title(PHASE_DISPLAY.get(phase, phase))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
        fig.tight_layout()
        fig.savefig(path_no_ext + ".pdf")
        fig.savefig(path_no_ext + ".png", dpi=PUBLICATION_RC["figure.dpi"])
        plt.close(fig)


def plot_combined_rp_heatmaps(path_no_ext, phase_order=None):
    """Assemble the multi-phase mean-Rp heatmap figure for the appendix.

    Uses the cached grids in RP_HEATMAP_DATA. Color scale is shared
    across all panels. Layout is 3 rows x 2 columns at 6.5 x 8.5 in.
    """
    if not RP_HEATMAP_DATA:
        print("  no Rp heatmap data collected; skipping combined figure")
        return
    if phase_order is None:
        phase_order = [p for p in PHASE_NAMES if p in RP_HEATMAP_DATA]
    n_panels = len(phase_order)

    # Shared color limits.
    all_vals = np.concatenate([
        RP_HEATMAP_DATA[p]["rp_grid"][
            np.isfinite(RP_HEATMAP_DATA[p]["rp_grid"])
        ]
        for p in phase_order
    ])
    vmin = float(all_vals.min()) if all_vals.size else 0.0
    vmax = float(all_vals.max()) if all_vals.size else 1.0

    with plt.rc_context(PUBLICATION_RC):
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("lightgrey")
        fig, axes = plt.subplots(
            3, 2, figsize=(6.5, 8.5),
            sharex=True, sharey=True,
            constrained_layout=True,
        )
        ims = []
        for i, phase in enumerate(phase_order):
            ax = axes.flat[i]
            d = RP_HEATMAP_DATA[phase]
            im = ax.imshow(
                d["rp_grid"].T, origin="lower", aspect="auto",
                extent=[d["mu_edges"][0], d["mu_edges"][-1],
                        d["hs_edges"][0], d["hs_edges"][-1]],
                cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation="nearest",
            )
            ims.append(im)
            ax.set_title(PHASE_DISPLAY.get(phase, phase))
            row, col = i // 2, i % 2
            if row == 2:
                ax.set_xlabel(r"Mustrain $\epsilon_{iso}$")
            if col == 0:
                ax.set_ylabel(r"HStrain $D_{11}$")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))

        # Hide unused axes if fewer than 6 phases were processed.
        for ax in axes.flat[n_panels:]:
            ax.set_visible(False)

        cbar = fig.colorbar(
            ims[0], ax=axes, location="right",
            shrink=0.85, aspect=30, pad=0.02,
        )
        cbar.set_label(r"mean $R_p$ (%)")
        fig.savefig(path_no_ext + ".pdf")
        fig.savefig(path_no_ext + ".png",
                    dpi=PUBLICATION_RC["figure.dpi"])
        plt.close(fig)


def plot_variance_diagnostics(residuals, sigma_per_sample, phase, out_dir):
    """Histogram of standardized residuals + Q-Q plot vs. N(0,1)."""
    valid = sigma_per_sample > 0
    z = (residuals[valid] / sigma_per_sample[valid]).ravel()
    z = z[np.isfinite(z)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-6, 6, 121)
    ax.hist(np.clip(z, -6, 6), bins=bins, density=True, alpha=0.6,
            color="tab:blue", label="standardized residuals")
    xs = np.linspace(-6, 6, 400)
    ax.plot(xs, stats.norm.pdf(xs), "r-", lw=1.5, label="N(0, 1)")
    ax.set_xlabel("z = (y_pred - y_true) / sigma_SM")
    ax.set_ylabel("density")
    ax.set_title(f"{phase}: standardized residuals vs. unit normal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "residual_hist.png"), dpi=130)
    plt.close(fig)

    z_sample = z if z.size <= 50000 else np.random.choice(z, 50000, replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(z_sample, dist="norm", plot=ax)
    ax.set_title(f"{phase}: Q-Q plot of standardized residuals")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "qq_plot.png"), dpi=130)
    plt.close(fig)


# =========================================================================
# PER-PHASE
# =========================================================================
def process_phase(phase, run_name, splits_root, pca_root, surrogates_root):
    print(f"\n{'='*72}\nPHASE: {phase}\n{'='*72}")
    X_train, Y_train, X_val, Y_val, pca = load_phase_assets(
        run_name, phase, splits_root, pca_root
    )
    n_train = X_train.shape[0]
    n_val   = X_val.shape[0]
    n_pcs_total = pca.n_components_
    print(f"    train n={n_train}, val n={n_val}, PCA rank {n_pcs_total}")

    # PC selection: variance-ordered top-K.
    if phase not in PHASE_N_PCS:
        raise KeyError(f"PHASE_N_PCS has no entry for {phase}")
    K = PHASE_N_PCS[phase]
    if K > n_pcs_total:
        print(f"    WARNING: PHASE_N_PCS[{phase}]={K} > PCA rank "
              f"{n_pcs_total}; capping at PCA rank")
        K = n_pcs_total
    kept_pcs = np.arange(K, dtype=int)
    print(f"    keeping leading {K} PCs (variance-ordered)")
    exp_var = pca.explained_variance_ratio_[kept_pcs]
    cum_var = np.sum(exp_var)
    print(f"    Explained variance of kept pcs: {cum_var}")

    # Train (timed).
    Z_train_kept = pca.transform(Y_train)[:, kept_pcs]
    print(f"    training RF (n_estimators="
          f"{RF_KWARGS_BASE['n_estimators']}) on {K} PCs...")
    rf = RandomForestRegressor(**RF_KWARGS_BASE)
    t0 = time.perf_counter()
    rf.fit(X_train, Z_train_kept)
    train_seconds = float(time.perf_counter() - t0)
    print(f"    training wall time: {train_seconds:.2f} s")

    # Predict on val + reconstruct full patterns.
    Z_pred_kept = rf.predict(X_val)
    Y_pred = reconstruct(Z_pred_kept, kept_pcs, pca)
    residuals = Y_pred - Y_val   # (n_val, n_2θ)

    # ----- Validation metrics -----
    w = poisson_weights(Y_val, floor=WEIGHT_FLOOR)
    rp        = r_profile(Y_val, Y_pred)
    rwp_unif  = r_weighted_profile_uniform(Y_val, Y_pred)
    rwp_pois  = r_weighted_profile_poisson(Y_val, Y_pred, w)
    rexp      = r_expected(Y_val, w, n_params=K)
    chi2      = chi_squared(rwp_pois, rexp)
    ape       = masked_ape(Y_val, Y_pred, fraction=APE_FRACTION)

    def med_p95(x):
        return float(np.nanmedian(x)), float(np.nanmean(x)), float(np.nanpercentile(x, 95))
    rp_med, rp_mean, rp_p95    = med_p95(rp)
    rwp_unif_med, _, rwp_unif_p95 = med_p95(rwp_unif)
    rwp_pois_med, _, rwp_pois_p95 = med_p95(rwp_pois)
    rexp_med, _,     rexp_p95     = med_p95(rexp)
    chi2_med,_,     chi2_p95     = med_p95(chi2)
    ape_med,_,      ape_p95      = med_p95(ape)

    print(f"    Rp            median={rp_med:.4f}      mean={rp_mean:.4f}   p95={rp_p95:.4f}  (%)")
    print(f"    Rwp (uniform) median={rwp_unif_med:.4f}    p95={rwp_unif_p95:.4f}  (%)")
    print(f"    Rwp (Poisson) median={rwp_pois_med:.4f}    p95={rwp_pois_p95:.4f}  (%)")
    print(f"    Rexp          median={rexp_med:.4f}    p95={rexp_p95:.4f}  (%)")
    print(f"    chi^2         median={chi2_med:.4f}    p95={chi2_p95:.4f}")
    print(f"    APE@{int(APE_FRACTION*100)}%       median={ape_med:.4f}    "
          f"p95={ape_p95:.4f}  (%)")

    # ----- Variance grid -----
    mustrain = X_val[:, 1]
    hstrain  = X_val[:, 2]
    b = PHASE_BOUNDS[phase]
    mu_edges = np.linspace(b["mustrain"][0], b["mustrain"][1],
                           N_BINS_MUSTRAIN + 1)
    hs_edges = np.linspace(b["hstrain"][0],  b["hstrain"][1],
                           N_BINS_HSTRAIN  + 1)

    var_grid, count_grid = build_variance_grid(
        mustrain, hstrain, residuals, mu_edges, hs_edges,
    )
    sparse_before = int((count_grid < MIN_COUNT_FOR_VARIANCE).sum())
    total_bins = count_grid.size
    print(f"    variance grid: {N_BINS_MUSTRAIN} x {N_BINS_HSTRAIN}, "
          f"sparse (< {MIN_COUNT_FOR_VARIANCE} samples): "
          f"{sparse_before}/{total_bins}, "
          f"counts min={int(count_grid.min())} "
          f"median={int(np.median(count_grid))} "
          f"max={int(count_grid.max())}")

    global_var = residuals.var(axis=0, ddof=0)
    var_grid_smooth, filled_by_smoothing, still_sparse = smooth_sparse_bins(
        var_grid, count_grid,
        min_count=MIN_COUNT_FOR_VARIANCE,
        max_passes=MAX_SMOOTHING_PASSES,
    )
    n_smoothed = int(filled_by_smoothing.sum())
    n_still_sparse = int(still_sparse.sum())
    var_grid_final, n_fallback = apply_global_fallback(
        var_grid_smooth, still_sparse, global_var,
    )
    print(f"    variance smoothing: {n_smoothed} bins filled by "
          f"smoothing, {n_fallback} fell back to global per-2theta")

    # ----- Save outputs -----
    out_dir = os.path.join(surrogates_root, run_name, phase)
    diag_dir = os.path.join(out_dir, "variance_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    # Final model bundle.
    val_summary = {
        "rp_median":           rp_med, "rp_mean":  rp_mean,        "rp_p95":           rp_p95,
        "rwp_uniform_median":  rwp_unif_med,  "rwp_uniform_p95":  rwp_unif_p95,
        "rwp_poisson_median":  rwp_pois_med,  "rwp_poisson_p95":  rwp_pois_p95,
        "rexp_median":         rexp_med,      "rexp_p95":         rexp_p95,
        "chi2_median":         chi2_med,      "chi2_p95":         chi2_p95,
        "ape_median":          ape_med,       "ape_p95":          ape_p95,
    }
    with open(os.path.join(out_dir, "final_model.pkl"), "wb") as fh:
        pickle.dump({
            "model":         rf,
            "kept_pcs":      kept_pcs,
            "rf_kwargs":     RF_KWARGS_BASE,
            "phase":         phase,
            "n_pcs_kept":    int(K),
            "truncation":    "variance_ordered",
            "n_estimators":  RF_KWARGS_BASE["n_estimators"],
            "val_summary":   val_summary,
        }, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # ----- Timings -----
    # Load time: model + PCA from disk. Now that final_model.pkl is
    # written, we can time the real disk path the MCMC would take.
    final_model_path = os.path.join(out_dir, "final_model.pkl")
    pca_path = os.path.join(pca_root, run_name, f"{phase}_pca_full.pkl")
    load_seconds = time_load(final_model_path, pca_path,
                             repeats=TIMING_REPEATS)

    # Inference: 1-sample and 100-sample. We pick X_val samples for
    # the inputs; values are drawn from the trained input range so the
    # tree traversal cost is representative.
    X_one     = X_val[:1]
    X_hundred = X_val[:100] if X_val.shape[0] >= 100 else X_val
    infer_1_seconds   = time_inference(rf, kept_pcs, pca, X_one,
                                       repeats=TIMING_REPEATS)
    infer_100_seconds = time_inference(rf, kept_pcs, pca, X_hundred,
                                       repeats=TIMING_REPEATS)
    print(f"    load (model+PCA): {load_seconds*1000:.2f} ms")
    print(f"    inference (1 sample):   {infer_1_seconds*1000:.3f} ms")
    print(f"    inference (100 samples): {infer_100_seconds*1000:.3f} ms")

    timings = {
        "timing_repeats": int(TIMING_REPEATS),
        "train_seconds":              float(train_seconds),
        "load_model_pca_seconds":     float(load_seconds),
        "inference_1_sample_seconds": float(infer_1_seconds),
        "inference_100_sample_seconds": float(infer_100_seconds),
    }
    with open(os.path.join(out_dir, "timings.json"), "w") as fh:
        json.dump(timings, fh, indent=2)

    # Variance .npz — same schema as compute_surrogate_variance.py output
    # so the existing surrogate_variance.py library can load it unchanged.
    np.savez_compressed(
        os.path.join(out_dir, "variance.npz"),
        mustrain_edges     = mu_edges,
        hstrain_edges      = hs_edges,
        variance_grid      = var_grid_final,
        count_grid         = count_grid,
        variance_global    = global_var,
        filled_by_smoothing = filled_by_smoothing,
        used_global_fallback = still_sparse,
    )

    # Per-sample metrics CSV.
    per_sample_df = pd.DataFrame({
        "mustrain":      mustrain,
        "hstrain":       hstrain,
        "rp":            rp,
        "rwp_uniform":   rwp_unif,
        "rwp_poisson":   rwp_pois,
        "rexp":          rexp,
        "chi2":          chi2,
        "masked_ape_90": ape,
    })
    per_sample_df.to_csv(os.path.join(out_dir, "per_sample_metrics.csv"),
                         index=False)

    # Metrics summary JSON.
    summary_dict = {
        "phase": phase,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_pcs_total": int(n_pcs_total),
        "n_pcs_kept": int(K),
        "truncation": "variance_ordered",
        "rf_kwargs": RF_KWARGS_BASE,
        "val_metrics": val_summary,
        "timings": timings,
        "variance_grid": {
            "n_bins_mustrain": int(N_BINS_MUSTRAIN),
            "n_bins_hstrain":  int(N_BINS_HSTRAIN),
            "min_count_for_variance": int(MIN_COUNT_FOR_VARIANCE),
            "n_bins_sparse_before_smoothing": sparse_before,
            "n_bins_filled_by_smoothing": n_smoothed,
            "n_bins_used_global_fallback": n_fallback,
            "bin_count_stats": {
                "min":    int(count_grid.min()),
                "max":    int(count_grid.max()),
                "median": float(np.median(count_grid)),
                "mean":   float(count_grid.mean()),
            },
        },
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as fh:
        json.dump(summary_dict, fh, indent=2)

    # Mean R_p heatmap over (mustrain, hstrain). The binned grid is
    # also cached so the combined appendix figure can be assembled in
    # main() after all phases are processed.
    plot_rp_heatmap_single(
        X_val, rp, phase,
        os.path.join(out_dir, "error_heatmap_rp"),
    )

    # Variance diagnostics: histogram + Q-Q on standardized residuals.
    sigma_per_sample = assign_sigma_to_samples(
        var_grid_final, mu_edges, hs_edges, mustrain, hstrain,
    )
    coverage = gaussian_coverage(residuals, sigma_per_sample)
    print(f"    Gaussian coverage (target 68.3 / 95.4 / 99.7 %): "
          f"{coverage['frac_within_1sigma']*100:.2f} / "
          f"{coverage['frac_within_2sigma']*100:.2f} / "
          f"{coverage['frac_within_3sigma']*100:.2f}")
    print(f"    fraction > 5 sigma: "
          f"{coverage['frac_beyond_5sigma']*100:.3f}%")
    with open(os.path.join(diag_dir, "coverage.json"), "w") as fh:
        json.dump({
            "phase": phase,
            "n_2theta_bins": int(residuals.shape[1]),
            **coverage,
        }, fh, indent=2)
    plot_variance_diagnostics(residuals, sigma_per_sample, phase, diag_dir)

    print(f"    saved everything to {out_dir}")
    return summary_dict


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--splits-root",     default=SPLITS_ROOT)
    parser.add_argument("--pca-root",        default=PCA_ROOT)
    parser.add_argument("--surrogates-root", default=SURROGATES_ROOT)
    parser.add_argument("--phases", nargs="+", default=PHASE_NAMES)
    args = parser.parse_args()

    run_name = args.run_name or auto_detect_run(args.splits_root)
    print(f"Run name: {run_name}")
    print(f"PHASE_N_PCS: {PHASE_N_PCS}")
    print(f"RF kwargs: {RF_KWARGS_BASE}")

    out_root = os.path.join(args.surrogates_root, run_name)
    os.makedirs(out_root, exist_ok=True)

    config = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "run_name": run_name,
        "phase_n_pcs": PHASE_N_PCS,
        "truncation": "variance_ordered",
        "rf_kwargs": RF_KWARGS_BASE,
        "ape_fraction": APE_FRACTION,
        "weight_floor": WEIGHT_FLOOR,
        "heatmap_bins": HEATMAP_BINS,
        "variance_grid": {
            "n_bins_mustrain": N_BINS_MUSTRAIN,
            "n_bins_hstrain":  N_BINS_HSTRAIN,
            "min_count_for_variance": MIN_COUNT_FOR_VARIANCE,
            "max_smoothing_passes": MAX_SMOOTHING_PASSES,
            "phase_bounds": PHASE_BOUNDS,
        },
    }
    with open(os.path.join(out_root, "_config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    summaries = []
    for phase in args.phases:
        try:
            s = process_phase(
                phase, run_name, args.splits_root, args.pca_root,
                args.surrogates_root,
            )
            summaries.append(s)
        except FileNotFoundError as e:
            print(f"  SKIP {phase}: {e}")
        except Exception as e:
            print(f"  ERROR {phase}: {e}")
            import traceback
            traceback.print_exc()

    # Combined multi-phase mean-Rp heatmap for the appendix figure.
    plot_combined_rp_heatmaps(
        os.path.join(out_root, "error_heatmap_rp_combined"),
        phase_order=[p for p in args.phases if p in RP_HEATMAP_DATA],
    )

    # Run-level summary table.
    print(f"\n{'='*120}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'='*120}")
    header = (
        f"{'phase':<10} {'n_val':>6} {'n_pc':>5}   "
        f"{'Rp_med':>9} {'Rwp_u_med':>10} {'Rwp_p_med':>10} "
        f"{'chi2_med':>10} {'APE_med':>9}   "
        f"{'Rp_p95':>9} {'Rwp_u_p95':>10} {'Rwp_p_p95':>10} "
        f"{'chi2_p95':>10} {'APE_p95':>9}"
    )
    print(header)
    print("-" * len(header))
    lines = [header, "-" * len(header)]
    for s in summaries:
        m = s["val_metrics"]
        line = (
            f"{s['phase']:<10} {s['n_val']:>6} {s['n_pcs_kept']:>5}   "
            f"{m['rp_median']:>9.4f} "
            f"{m['rwp_uniform_median']:>10.4f} "
            f"{m['rwp_poisson_median']:>10.4f} "
            f"{m['chi2_median']:>10.4f} "
            f"{m['ape_median']:>9.4f}   "
            f"{m['rp_p95']:>9.4f} "
            f"{m['rwp_uniform_p95']:>10.4f} "
            f"{m['rwp_poisson_p95']:>10.4f} "
            f"{m['chi2_p95']:>10.4f} "
            f"{m['ape_p95']:>9.4f}"
        )
        print(line)
        lines.append(line)

    with open(os.path.join(out_root, "_summary.json"), "w") as fh:
        json.dump(summaries, fh, indent=2)
    with open(os.path.join(out_root, "_summary_table.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote summary to {os.path.join(out_root, '_summary.json')}")

    # Run-level timing summary. All durations in milliseconds except
    # training (in seconds — typically 10s of seconds to minutes).
    print(f"\n{'='*80}")
    print("TIMING SUMMARY")
    print(f"{'='*80}")
    t_header = (
        f"{'phase':<10} {'n_pc':>5}   "
        f"{'train(s)':>10} {'load(ms)':>10} "
        f"{'infer_1(ms)':>13} {'infer_100(ms)':>15}"
    )
    print(t_header)
    print("-" * len(t_header))
    t_lines = [t_header, "-" * len(t_header)]
    timing_rows = []
    for s in summaries:
        t = s["timings"]
        line = (
            f"{s['phase']:<10} {s['n_pcs_kept']:>5}   "
            f"{t['train_seconds']:>10.2f} "
            f"{t['load_model_pca_seconds']*1000:>10.2f} "
            f"{t['inference_1_sample_seconds']*1000:>13.3f} "
            f"{t['inference_100_sample_seconds']*1000:>15.3f}"
        )
        print(line)
        t_lines.append(line)
        timing_rows.append({
            "phase": s["phase"],
            "n_pcs_kept": s["n_pcs_kept"],
            "train_seconds":                t["train_seconds"],
            "load_model_pca_seconds":       t["load_model_pca_seconds"],
            "inference_1_sample_seconds":   t["inference_1_sample_seconds"],
            "inference_100_sample_seconds": t["inference_100_sample_seconds"],
        })

    pd.DataFrame(timing_rows).to_csv(
        os.path.join(out_root, "_timings.csv"), index=False)
    with open(os.path.join(out_root, "_timings_table.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(t_lines) + "\n")
    print(f"\nWrote timing summary to "
          f"{os.path.join(out_root, '_timings.csv')}")


if __name__ == "__main__":
    main()
