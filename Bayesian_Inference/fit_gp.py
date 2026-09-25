# fit_gp.py
#
# Standalone Gaussian-process background fit for the diffraction
# inference pipeline. This step is intentionally separated from MCMC so
# the background can be tuned and verified before any sampling is run.
#
# Responsibilities
# ----------------
#   1. Read the dedicated config file (default ./config.py, overridable
#      with --config) and copy it verbatim into the output directory so
#      the fit is reproducible from its own run folder.
#   2. Load the observed pattern and apply the SAME data-range mask that
#      MCMC will use, so the GP is fit on byte-identical data.
#   3. Fit the GP upper-envelope background via
#      Gaussian_Process_upper_envelope.
#   4. Save the fit:
#        - gp_fit.npz : self-describing bundle (x, y, GP_pred, GP_std,
#          GP_pred_avg, GP_pred_int, keep_idx, gp_hash, data_ranges).
#          run_mcmc_v3.py validates against this before sampling.
#        - GP_pred.txt / GP_std.txt : legacy text outputs, unchanged
#          format, for backward compatibility.
#   5. Write publication-ready figures of the background fit.
#
# Note on physics: the MCMC likelihood consumes only GP_std**2 (the
# background *variance*); the background *mean* GP_pred does not enter the
# residual. This file fits and saves both, and plots the mean for visual
# verification, but does not change how the background is used downstream.
#
# Run order: fit_gp.py  ->  run_mcmc_v3.py

import os
import sys
import json
import shutil
import hashlib
import argparse
import importlib.util
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe; no display needed
import matplotlib.pyplot as plt

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.signal import find_peaks

def two_theta_to_Q(two_theta_deg: np.ndarray, wavelength: float) -> np.ndarray:
    theta_rad = np.radians(two_theta_deg / 2.0)
    return 4 * np.pi * np.sin(theta_rad) / wavelength

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

    # ---- FINAL mask: only points actually used ----
    final_mask = np.zeros_like(I_obs, dtype=bool)
    final_mask[orig_idx] = True

    diff = np.min(I_obs - GP_pred)
    GP_pred = GP_pred+diff
    return GP_pred_avg, GP_std, GP_pred, GP_pred_int, final_mask


# =========================================================================
# SHARED HELPERS (imported by run_mcmc_v3.py as well)
# =========================================================================
def load_config(config_path):
    """
    Import a config.py given by path and return (CONFIG, data_type,
    resolved_path). Done via importlib so an arbitrary --config path works
    rather than relying on it being on sys.path.
    """
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("_pipeline_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"{config_path} does not define CONFIG.")
    data_type = getattr(module, "data_type", "fxye")
    return module.CONFIG, data_type, config_path


def gp_config_hash(config):
    """
    Stable short hash over the inputs that determine the GP fit: the gp
    block, the data_ranges, and the dataset paths/loader. Used so the MCMC
    driver can detect a stale or mismatched GP file.
    """
    payload = {
        "gp": config["gp"],
        "data_ranges": config["dataset"].get("data_ranges"),
        "x_data": config["dataset"].get("x_data"),
        "y_data": config["dataset"].get("y_data"),
        "fxye_path": config["dataset"].get("fxye_path"),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def load_observed_pattern(config, data_type):
    """Load the full (unmasked) observed pattern as (x_obs, y_obs)."""
    if data_type == "numpy":
        x_obs = np.load(config["dataset"]["x_data"]) * 100
        y_obs = np.load(config["dataset"]["y_data"])
    elif data_type == "fxye":
        fxye = config["dataset"]["fxye_path"]
        x_obs = np.loadtxt(fxye, skiprows=3, usecols=0)
        y_obs = np.loadtxt(fxye, skiprows=3, usecols=1)
    else:
        raise ValueError(f"unknown data_type {data_type!r} (expected 'numpy' or 'fxye').")
    return x_obs, y_obs


def build_keep_idx(data_ranges, n_2theta_full):
    """
    Build a 1-D index array selecting the columns of the full pattern
    that participate in the likelihood.

    data_ranges : list of [start, stop_exclusive] pairs, or None.
                  If None, the full range [0, n_2theta_full) is used.
                  Pairs must be sorted, non-overlapping, and in-bounds.

    Returns a contiguous np.ndarray of int indices.
    """
    if data_ranges is None:
        return np.arange(n_2theta_full, dtype=int)

    if not isinstance(data_ranges, (list, tuple)) or len(data_ranges) == 0:
        raise ValueError(
            "data_ranges must be None or a non-empty list of "
            "[start, stop_exclusive] pairs."
        )

    prev_stop = -1
    pieces = []
    for k, pair in enumerate(data_ranges):
        if len(pair) != 2:
            raise ValueError(
                f"data_ranges[{k}] must be a [start, stop_exclusive] pair, "
                f"got {pair!r}."
            )
        start, stop = int(pair[0]), int(pair[1])
        if not (0 <= start < stop <= n_2theta_full):
            raise ValueError(
                f"data_ranges[{k}] = [{start}, {stop}] out of bounds for "
                f"pattern length {n_2theta_full}; require "
                f"0 <= start < stop <= {n_2theta_full}."
            )
        if start < prev_stop:
            raise ValueError(
                f"data_ranges[{k}] = [{start}, {stop}] overlaps or "
                f"precedes previous range (ends at {prev_stop}). "
                f"Ranges must be sorted and non-overlapping."
            )
        pieces.append(np.arange(start, stop, dtype=int))
        prev_stop = stop

    return np.concatenate(pieces)


# =========================================================================
# FIGURES
# =========================================================================
def _publication_style():
    """A clean, publication-oriented rcParams context."""
    return  {
    "figure.dpi":       300,
    "figure.autolayout": True,
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

def make_figures(x_obs, y_obs, GP_pred, GP_std, fig_dir, label, training_points):
    """
    Publication-ready figure of the background fit, with two square
    side-by-side subplots:
      (left)  observed pattern with fitted background envelope and a
              +/- 1 GP_std band;
      (right) GP standard deviation vs Q.
    Saved as both PDF (vector) and PNG.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with plt.rc_context(_publication_style()):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(6.0, 3.0))

        # ---- (left) background overlay ----
        ax_l.plot(x_obs, y_obs, 
            #color="0.25", 
            lw=1.2, label=r"$y^{obs}_i$"
        )
        # ax_l.plot(x_obs, GP_pred, 
        #     #color="tab:orange", 
        #     lw=2.2,
        #     label=r"$y^b_i$"
        # )
        # ax_l.scatter(
        #     x_obs[training_points],
        #     y_obs[training_points],
        #     color="black",
        #     s=2,
        #     zorder=5,
        #     label=r"$\mathcal{T}^{\,\,(n)}$"
        # )
        #ax_l.set_yscale("log")
        ax_l.set_yscale("function",
              functions=(lambda x: np.sqrt(np.abs(x)), lambda x: x ** 2))
        ax_l.set_xlabel(r"$Q\ (\mathrm{\AA^{-1}})$")
        ax_l.set_ylabel("Intensity (counts)")
        ax_l.set_xlim(x_obs.min(), x_obs.max())
        ax_l.ticklabel_format(axis="y", style="scientific", scilimits=(-5, 4), useMathText=True)
        #ax_l.set_ylim(bottom=9e3)  # 9e3 for ID35
        ax_l.legend(frameon=True, loc="best")
        ax_l.set_box_aspect(1)

        # ---- (right) GP standard deviation ----
        ax_r.plot(x_obs, GP_std, 
            #color="tab:red", 
            lw=1.2, 
            label= r"$\sigma_{b,i}$"
        )
        ax_r.set_xlabel(r"$Q\ (\mathrm{\AA^{-1}})$")
        ax_r.set_xlim(x_obs.min(), x_obs.max())
        ax_r.set_ylim(bottom=0.0)
        ax_r.yaxis.tick_right()
        ax_r.yaxis.set_label_position("right")
        ax_r.legend(frameon=True, loc="best")
        ax_r.set_box_aspect(1)

        for ext in ("pdf", "png"):
            p = fig_dir / f"gp_background_fit.{ext}"
            fig.savefig(p)
            saved.append(p)
        plt.close(fig)
    return saved
# def make_figures(x_obs, y_obs, GP_pred, GP_std, fig_dir, label):
#     """
#     Publication-ready figures of the background fit:
#       (1) observed pattern with fitted background envelope and a
#           +/- 1 GP_std band;
#       (2) GP standard deviation vs 2theta.
#     Each figure is saved as both PDF (vector) and PNG.
#     """
#     fig_dir = Path(fig_dir)
#     fig_dir.mkdir(parents=True, exist_ok=True)
#     saved = []

#     with plt.rc_context(_publication_style()):
#         # ---- (1) background overlay ----
#         fig, ax = plt.subplots(figsize=(7.0, 4.2))
#         ax.plot(x_obs, y_obs, color="0.25", lw=0.9, label="Observed")
#         ax.fill_between(
#             x_obs, GP_pred - GP_std, GP_pred + GP_std,
#             color="tab:orange", alpha=0.25, linewidth=0,
#             label=r"Background $\pm 1\sigma_{\mathrm{GP}}$",
#         )
#         ax.plot(x_obs, GP_pred, color="tab:orange", lw=1.4,
#                 label="GP background")
#         ax.set_yscale("log")
#         ax.set_xlabel(r"$Q\ (\mathrm{\AA^{-1}})$")
#         ax.set_ylabel("Intensity (counts)")
#         #ax.set_title(f"GP background fit — {label}")
#         ax.set_xlim(x_obs.min(), x_obs.max())
#         ax.set_ylim(bottom=9e3) #9e3 for ID35
#         ax.legend(frameon=False, loc="best")
#         for ext in ("pdf", "png"):
#             p = fig_dir / f"gp_background_fit.{ext}"
#             fig.savefig(p)
#             saved.append(p)
#         plt.close(fig)

#         # ---- (2) GP standard deviation ----
#         fig, ax = plt.subplots(figsize=(7.0, 3.4))
#         ax.plot(x_obs, GP_std, color="tab:red", lw=1.2)
#         ax.set_xlabel(r"$Q\ (\mathrm{\AA^{-1}})$")
#         ax.set_ylabel(r"$\sigma_{\mathrm{GP}}$")
#         #ax.set_title(f"GP background standard deviation — {label}")
#         ax.set_xlim(x_obs.min(), x_obs.max())
#         ax.set_ylim(bottom=0.0)
#         for ext in ("pdf", "png"):
#             p = fig_dir / f"gp_std.{ext}"
#             fig.savefig(p)
#             saved.append(p)
#         plt.close(fig)

#     return saved


# =========================================================================
# MAIN
# =========================================================================
def main(config_path):
    print("Config taken from:", config_path)
    config, data_type, resolved_cfg = load_config(config_path)
    WAVELENGTH = config["dataset"]["Xray_wavelength"]

    out_dir = Path("output") / config["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}", flush=True)

    # Copy the config verbatim into the run folder for reproducibility.
    dst = out_dir / "config_used.py"
    if dst.exists() and os.path.samefile(resolved_cfg, dst):
        print(f"Config already in place at {dst}; skipping copy.", flush=True)
    else:
        shutil.copyfile(resolved_cfg, dst)
        print(f"Copied config -> {dst}", flush=True)

    # -----------------------------------------------------------------
    # Observed pattern + identical data-range mask
    # -----------------------------------------------------------------
    x_obs, y_obs = load_observed_pattern(config, data_type)
    #x_obs = 100*x_obs
    n_2theta_full = int(y_obs.size)

    data_ranges = config["dataset"].get("data_ranges")
    keep_idx = build_keep_idx(data_ranges, n_2theta_full)
    print(
        f"data_ranges={data_ranges}; using {keep_idx.size} of "
        f"{n_2theta_full} points.",
        flush=True,
    )
    x_obs = x_obs[keep_idx]
    y_obs = y_obs[keep_idx]

    # -----------------------------------------------------------------
    # GP background fit
    # -----------------------------------------------------------------
    gp_cfg = config["gp"]
    GP_pred_avg, GP_std, GP_pred, GP_pred_int, training_points = Gaussian_Process_upper_envelope(
        x_obs, y_obs,
        n_iter=gp_cfg["n_iter"],
        peak_prominence=gp_cfg["peak_prominence"],
        rbf_length_scale=gp_cfg["rbf_length_scale"],
        rbf_length_bounds=gp_cfg["rbf_length_bounds"],
    )
    GP_pred = np.asarray(GP_pred, dtype=float)
    GP_std = np.asarray(GP_std, dtype=float)

    # -----------------------------------------------------------------
    # Save: self-describing bundle + legacy text outputs
    # -----------------------------------------------------------------
    gp_hash = gp_config_hash(config)
    np.savez(
        out_dir / "gp_fit.npz",
        x_obs=x_obs,
        y_obs=y_obs,
        GP_pred=GP_pred,
        GP_std=GP_std,
        GP_pred_avg=np.asarray(GP_pred_avg, dtype=float),
        GP_pred_int=np.asarray(GP_pred_int, dtype=float),
        keep_idx=keep_idx,
        gp_hash=gp_hash,
        data_ranges=np.array(json.dumps(data_ranges)),  # stored as JSON scalar
        n_2theta_full=n_2theta_full,
    )
    # Legacy outputs, unchanged format.
    # np.savetxt(out_dir / "GP_pred.txt", GP_pred)
    # np.savetxt(out_dir / "GP_std.txt",  GP_std)
    # print(f"Saved gp_fit.npz (gp_hash={gp_hash}), GP_pred.txt, GP_std.txt.",
    #       flush=True)

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    Q = two_theta_to_Q(two_theta_deg=x_obs/100, wavelength=WAVELENGTH) #convert from units of centidegrees
    fig_dir = out_dir / "figures"
    saved = make_figures(
        Q, y_obs, GP_pred, GP_std, fig_dir,
        label=config["dataset"].get("label", config["run_name"]),
        training_points=training_points,
    )
    print("Saved figures:", flush=True)
    for p in saved:
        print(f"  {p}", flush=True)


def _parse_args(argv=None):
    default_cfg = Path(__file__).resolve().parent / "config.py"
    parser = argparse.ArgumentParser(
        description="Fit and verify the GP background ahead of MCMC."
    )
    parser.add_argument(
        "--config", default=str(default_cfg),
        help=f"Path to the pipeline config.py (default: {default_cfg}).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(args.config)
