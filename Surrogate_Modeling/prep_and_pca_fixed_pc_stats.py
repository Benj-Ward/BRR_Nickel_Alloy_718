# prep_and_pca.py
#
# Stage 1 of the surrogate-model training pipeline.
#
# For each phase in a training-data run produced by driver_lhs_v0.py:
#   1. Load <phase>_inputs.csv and <phase>_patterns.npy.
#   2. Drop samples whose patterns are all-NaN (worker failures).
#   3. Build a 70/15/15 train/val/test split with the rules:
#        - All corner samples are pinned to the training set.
#        - main_lhs samples are split 70/15/15.
#        - boundary_lhs samples (if any exist for this phase) are split
#          70/15/15 the same way. Gamma typically has none, but the rule
#          is "if present, split it" — no phase is hard-coded to skip.
#   4. Write per-phase split files to data_splits/<run_name>/<phase>/:
#        X_train.npy, X_val.npy, X_test.npy   — 3 active-phase params
#        Y_train.npy, Y_val.npy, Y_test.npy   — full diffraction patterns
#        split_metadata.json                  — sample_ids, source breakdown
#   5. Fit PCA on Y_train only, with all components retained (no
#      truncation at this stage). Save to pca_transforms/<run_name>/
#      <phase>_pca_full.pkl.
#   6. Compute validation Rp (R-profile) and Rwp (uniform-weight
#      weighted profile) as a function of how many leading PCs are
#      retained, save the curves to disk and plot per-phase PNGs with
#      both metrics in stacked panels.
#
# Run directory is auto-detected as the most-recent training_data/run_*
# subdirectory, unless overridden by --run-dir.

import os
import sys
import json
import pickle
import argparse
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# =========================================================================
# CONFIGURATION
# =========================================================================
SEED = 123

# Split fractions. Corners are pinned to train regardless; these apply to
# main_lhs and boundary_lhs.
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# Phase order (must match the driver). The 3 active-phase columns extracted
# from the inputs CSV are scale_<phase>, mustrain_<phase>, hstrain_<phase>_D11.
PHASE_NAMES = ["gamma", "delta", "gamma1", "gamma2", "laves", "carbide"]

# Default roots. Both are relative to the working directory.
TRAINING_DATA_ROOT = "training_data_NIST"
SPLITS_ROOT        = "data_splits"
PCA_ROOT           = "pca_transforms"

# How densely to sample the "PCs retained vs. error" curve. Up through
# PC_CURVE_DENSE we evaluate every PC count; after that, every step.
PC_CURVE_DENSE = 100
PC_CURVE_STEP  = 5

# Fixed PCA truncation used for validation reconstruction-error reporting.
# Gamma gets 160 PCs; all other phases get 240 PCs.
FIXED_PC_BY_PHASE = {
    "gamma": 240,
    "delta": 240,
    "gamma1": 240,
    "gamma2": 240,
    "laves": 240,
    "carbide": 240,
}


# =========================================================================
# UTILITIES
# =========================================================================
def auto_detect_run_dir(root):
    """Find the most-recently-modified run_* subdirectory under root."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{root} does not exist")
    candidates = sorted(
        glob.glob(os.path.join(root, "run_*")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no run_* directories found under {root}")
    return candidates[0]


def load_phase(run_dir, phase):
    """Load the inputs DataFrame and patterns array for one phase."""
    phase_dir = os.path.join(run_dir, phase)
    csv_path  = os.path.join(phase_dir, f"{phase}_inputs.csv")
    npy_path  = os.path.join(phase_dir, f"{phase}_patterns.npy")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"missing {csv_path}")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"missing {npy_path}")
    df = pd.read_csv(csv_path)
    Y = np.load(npy_path)
    if len(df) != Y.shape[0]:
        raise ValueError(
            f"{phase}: CSV rows ({len(df)}) != patterns rows ({Y.shape[0]})"
        )
    return df, Y


def filter_failed(df, Y):
    """Drop samples whose pattern is all-NaN (failed worker jobs).

    Note: the driver writes successful patterns as full rows of finite
    values and missing/failed patterns as rows of NaN. Partial-NaN rows
    aren't expected, but we still treat any NaN as a failure to be safe.
    """
    has_nan = np.isnan(Y).any(axis=1)
    if has_nan.any():
        n_dropped = int(has_nan.sum())
        df = df.loc[~has_nan].reset_index(drop=True)
        Y  = Y[~has_nan]
        print(f"    dropped {n_dropped} failed sample(s)")
    return df, Y


def extract_X(df, phase):
    """Pull the 3 active-phase columns out of the 18-column param block.

    Column names in the CSV follow the canonical order set by the driver:
    scale_<p>, mustrain_<p>, hstrain_<p>_D11 for each phase in PHASE_NAMES.
    """
    cols = [f"scale_{phase}", f"mustrain_{phase}", f"hstrain_{phase}_D11"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{phase}: CSV missing expected columns: {missing}")
    return df[cols].to_numpy(dtype=float), cols


def split_indices(df_indices, sources, seed):
    """
    Build train/val/test index arrays.

    df_indices : 1D array of integer positions into the (post-filter)
                 inputs DataFrame for this phase.
    sources    : 1D array of strings (same length), one of
                 "main_lhs", "boundary_lhs", "corner".
    """
    train_parts = []
    val_parts   = []
    test_parts  = []

    # Corners → train, in their entirety.
    corner_mask = (sources == "corner")
    train_parts.append(df_indices[corner_mask])

    # main_lhs and boundary_lhs each get their own 70/15/15 split. Doing
    # them separately (rather than stratifying a single call) keeps the
    # ratio exact within each source category, which is what we want
    # given the small absolute count of boundary samples.
    for src in ("main_lhs", "boundary_lhs"):
        src_mask = (sources == src)
        n_src = int(src_mask.sum())
        if n_src == 0:
            continue
        src_idx = df_indices[src_mask]

        # First split off test, then split the remainder into train/val.
        # train_test_split's test_size is a fraction of the input it sees.
        idx_trainval, idx_test = train_test_split(
            src_idx,
            test_size=TEST_FRAC,
            random_state=seed,
            shuffle=True,
        )
        # Within train+val, val occupies VAL_FRAC of the original total,
        # which is VAL_FRAC / (TRAIN_FRAC + VAL_FRAC) of train+val.
        val_size_within = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=val_size_within,
            random_state=seed,
            shuffle=True,
        )
        train_parts.append(idx_train)
        val_parts.append(idx_val)
        test_parts.append(idx_test)

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    val_idx   = np.concatenate(val_parts)   if val_parts   else np.array([], dtype=int)
    test_idx  = np.concatenate(test_parts)  if test_parts  else np.array([], dtype=int)

    # Sort each so file order is deterministic given the seed.
    train_idx.sort()
    val_idx.sort()
    test_idx.sort()
    return train_idx, val_idx, test_idx


def r_profile(y_true, y_pred):
    """Per-sample Rp (R-profile) in percent.

        Rp = 100 * sum |y_obs - y_calc| / sum y_obs

    The standard XRD whole-pattern unweighted residual. Computed per row;
    returns a 1-D array of length y_true.shape[0].
    """
    num = np.abs(y_true - y_pred).sum(axis=1)
    den = y_true.sum(axis=1)
    out = np.full_like(num, np.nan, dtype=float)
    nz = den != 0
    out[nz] = 100.0 * num[nz] / den[nz]
    return out


def r_weighted_profile(y_true, y_pred):
    """Per-sample Rwp with uniform weights, in percent.

        Rwp = 100 * sqrt( sum (y_obs - y_calc)^2 / sum y_obs^2 )

    Uniform weights are appropriate here because the patterns are
    computed from the GSAS-II forward model — there is no Poisson
    counting noise, so the standard w = 1/y_obs weighting (which derives
    from the counting-statistics noise model) doesn't apply. Uniform-
    weight Rwp reduces to a normalized L2 residual, expressed as a
    percentage to match the convention of the unweighted Rp.
    """
    num = ((y_true - y_pred) ** 2).sum(axis=1)
    den = (y_true ** 2).sum(axis=1)
    out = np.full_like(num, np.nan, dtype=float)
    nz = den != 0
    out[nz] = 100.0 * np.sqrt(num[nz] / den[nz])
    return out


def reconstruct_with_k_components(pca, Z_full, k):
    """Inverse-transform using only the leading k PCs.

    Avoids sklearn's "build a truncated PCA object" dance: we just zero
    out everything past column k and run the standard inverse_transform,
    which adds the mean back. Equivalent and faster.
    """
    Z_trunc = Z_full.copy()
    Z_trunc[:, k:] = 0.0
    return pca.inverse_transform(Z_trunc)


# =========================================================================
# PER-PHASE WORK
# =========================================================================
def process_phase(phase, run_dir, splits_dir, pca_dir, plots_dir):
    print(f"\n{'='*72}\nPHASE: {phase}\n{'='*72}")

    # -- Load and clean.
    df, Y = load_phase(run_dir, phase)
    df, Y = filter_failed(df, Y)
    sources = df["source"].to_numpy()
    indices = np.arange(len(df))
    print(f"    {len(df)} samples after filtering: "
          f"{(sources=='main_lhs').sum()} main_lhs, "
          f"{(sources=='boundary_lhs').sum()} boundary_lhs, "
          f"{(sources=='corner').sum()} corner")

    # -- Extract active-phase X.
    X_all, x_cols = extract_X(df, phase)

    # -- Build splits.
    train_idx, val_idx, test_idx = split_indices(indices, sources, seed=SEED)
    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
    print(f"    split: train={n_train}, val={n_val}, test={n_test}")

    X_train, X_val, X_test = X_all[train_idx], X_all[val_idx], X_all[test_idx]
    Y_train, Y_val, Y_test = Y[train_idx],     Y[val_idx],     Y[test_idx]

    # -- Save splits.
    phase_split_dir = os.path.join(splits_dir, phase)
    os.makedirs(phase_split_dir, exist_ok=True)
    np.save(os.path.join(phase_split_dir, "X_train.npy"), X_train)
    np.save(os.path.join(phase_split_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(phase_split_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(phase_split_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(phase_split_dir, "Y_val.npy"),   Y_val)
    np.save(os.path.join(phase_split_dir, "Y_test.npy"),  Y_test)

    # Map back to sample_ids in the original CSV for traceability.
    sample_ids = df["sample_id"].to_numpy()
    metadata = {
        "phase": phase,
        "seed": SEED,
        "x_columns": x_cols,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "sample_ids_train": sample_ids[train_idx].tolist(),
        "sample_ids_val":   sample_ids[val_idx].tolist(),
        "sample_ids_test":  sample_ids[test_idx].tolist(),
        "source_breakdown": {
            "train": {s: int((sources[train_idx] == s).sum())
                      for s in ("main_lhs", "boundary_lhs", "corner")},
            "val":   {s: int((sources[val_idx]   == s).sum())
                      for s in ("main_lhs", "boundary_lhs", "corner")},
            "test":  {s: int((sources[test_idx]  == s).sum())
                      for s in ("main_lhs", "boundary_lhs", "corner")},
        },
    }
    with open(os.path.join(phase_split_dir, "split_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    # -- Fit PCA on training set, full rank.
    print(f"    fitting PCA (full)...")
    pca = PCA(n_components=None)
    pca.fit(Y_train)
    n_pcs_total = pca.n_components_
    print(f"    PCA fit complete: {n_pcs_total} components, "
          f"cumulative variance covered: "
          f"{pca.explained_variance_ratio_.sum():.6f}")

    pca_payload = {
        "pca": pca,
        "phase": phase,
        "n_components_total": n_pcs_total,
        "y_train_shape": Y_train.shape,
        "seed": SEED,
    }
    pca_path = os.path.join(pca_dir, f"{phase}_pca_full.pkl")
    with open(pca_path, "wb") as fh:
        pickle.dump(pca_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    saved {pca_path}")

    # -- Validation Rp and Rwp vs. number of PCs retained.
    Z_val = pca.transform(Y_val)
    pc_counts = list(range(1, min(PC_CURVE_DENSE, n_pcs_total) + 1))
    pc_counts += list(range(PC_CURVE_DENSE + PC_CURVE_STEP,
                            n_pcs_total + 1, PC_CURVE_STEP))
    if pc_counts[-1] != n_pcs_total:
        pc_counts.append(n_pcs_total)

    rp_median  = np.empty(len(pc_counts))
    rp_p95     = np.empty(len(pc_counts))
    rwp_median = np.empty(len(pc_counts))
    rwp_p95    = np.empty(len(pc_counts))

    print(f"    sweeping PC count over {len(pc_counts)} values...")
    for i, k in enumerate(pc_counts):
        Y_val_recon = reconstruct_with_k_components(pca, Z_val, k)
        rp  = r_profile(Y_val, Y_val_recon)
        rwp = r_weighted_profile(Y_val, Y_val_recon)
        rp_median[i]  = np.nanmedian(rp)
        rp_p95[i]     = np.nanpercentile(rp, 95)
        rwp_median[i] = np.nanmedian(rwp)
        rwp_p95[i]    = np.nanpercentile(rwp, 95)

    # Save the curve as a CSV.
    curve_path = os.path.join(phase_split_dir, f"{phase}_pca_val_error_curve.csv")
    pd.DataFrame({
        "n_components": pc_counts,
        "rp_median":  rp_median,
        "rp_p95":     rp_p95,
        "rwp_median": rwp_median,
        "rwp_p95":    rwp_p95,
    }).to_csv(curve_path, index=False)
    print(f"    saved {curve_path}")

    # -- Fixed-PC validation reconstruction statistics.
    requested_fixed_k = FIXED_PC_BY_PHASE[phase]
    fixed_k = min(requested_fixed_k, n_pcs_total)
    if fixed_k < requested_fixed_k:
        print(f"    WARNING: requested {requested_fixed_k} PCs for {phase}, "
              f"but only {n_pcs_total} are available; using {fixed_k}")

    Y_val_recon_fixed = reconstruct_with_k_components(pca, Z_val, fixed_k)
    rp_fixed = r_profile(Y_val, Y_val_recon_fixed)
    cum_var_fixed = float(pca.explained_variance_ratio_[:fixed_k].sum())
    fixed_stats = {
        "phase": phase,
        "requested_n_components": int(requested_fixed_k),
        "n_components_used": int(fixed_k),
        "n_validation_samples": int(len(Y_val)),
        "cumulative_variance_explained": cum_var_fixed,
        "rp_mean": float(np.nanmean(rp_fixed)),
        "rp_median": float(np.nanmedian(rp_fixed)),
        "rp_p95": float(np.nanpercentile(rp_fixed, 95)),
    }

    fixed_stats_path = os.path.join(
        phase_split_dir, f"{phase}_pca_val_rpattern_stats_fixed_pc.json"
    )
    with open(fixed_stats_path, "w") as fh:
        json.dump(fixed_stats, fh, indent=2)
    print(f"    saved {fixed_stats_path}")

    # -- Plot: two stacked panels, Rp on top, Rwp below.
    fig, (ax_rp, ax_rwp) = plt.subplots(
        2, 1, figsize=(7.5, 7.5), sharex=True
    )
    ax_rp.plot(pc_counts, rp_median, "-o", ms=3, label="median")
    ax_rp.plot(pc_counts, rp_p95,    "-s", ms=3, label="95th percentile")
    ax_rp.set_ylabel("Rp (%)")
    ax_rp.set_yscale("log")
    ax_rp.set_title(f"{phase}: PCA reconstruction error on validation set")
    ax_rp.grid(True, which="both", alpha=0.3)
    ax_rp.legend()

    ax_rwp.plot(pc_counts, rwp_median, "-o", ms=3, label="median")
    ax_rwp.plot(pc_counts, rwp_p95,    "-s", ms=3, label="95th percentile")
    ax_rwp.set_ylabel("Rwp (%, uniform weights)")
    ax_rwp.set_xlabel("Number of PCs retained")
    ax_rwp.set_yscale("log")
    ax_rwp.grid(True, which="both", alpha=0.3)
    ax_rwp.legend()

    fig.tight_layout()
    plot_path = os.path.join(plots_dir, f"{phase}_pca_val_error.png")
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"    saved {plot_path}")

    return {
        "phase": phase,
        "n_train": n_train,
        "n_val":   n_val,
        "n_test":  n_test,
        "n_pcs_total":     n_pcs_total,
        "rp_median_full":  float(rp_median[-1]),
        "rp_p95_full":     float(rp_p95[-1]),
        "rwp_median_full": float(rwp_median[-1]),
        "rwp_p95_full":    float(rwp_p95[-1]),
        "fixed_pc_rp_stats": fixed_stats,
    }


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None,
                        help="Path to a training_data/run_* directory. "
                             "If omitted, the most recent one is used.")
    parser.add_argument("--training-root", default=TRAINING_DATA_ROOT,
                        help="Root containing run_* directories.")
    parser.add_argument("--splits-root", default=SPLITS_ROOT)
    parser.add_argument("--pca-root", default=PCA_ROOT)
    args = parser.parse_args()

    run_dir = args.run_dir or auto_detect_run_dir(args.training_root)
    run_name = os.path.basename(os.path.normpath(run_dir))
    print(f"Run directory: {run_dir}")
    print(f"Run name:      {run_name}")

    splits_dir = os.path.join(args.splits_root, run_name)
    pca_dir    = os.path.join(args.pca_root,    run_name)
    plots_dir  = os.path.join(splits_dir, "_plots")
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(pca_dir,    exist_ok=True)
    os.makedirs(plots_dir,  exist_ok=True)

    # Top-level config dump for traceability.
    config = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "run_dir": os.path.abspath(run_dir),
        "seed": SEED,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "phase_names": PHASE_NAMES,
        "pc_curve_dense": PC_CURVE_DENSE,
        "pc_curve_step":  PC_CURVE_STEP,
        "fixed_pc_by_phase": FIXED_PC_BY_PHASE,
    }
    with open(os.path.join(splits_dir, "_config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    summaries = []
    for phase in PHASE_NAMES:
        try:
            summary = process_phase(phase, run_dir, splits_dir, pca_dir, plots_dir)
            summaries.append(summary)
        except FileNotFoundError as e:
            print(f"  SKIP {phase}: {e}")
        except Exception as e:
            print(f"  ERROR {phase}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined fixed-PC validation R-pattern statistics.
    fixed_stats = [s["fixed_pc_rp_stats"] for s in summaries
                   if "fixed_pc_rp_stats" in s]
    fixed_stats_json = os.path.join(
        splits_dir, "_fixed_pc_validation_rpattern_stats.json"
    )
    with open(fixed_stats_json, "w") as fh:
        json.dump(fixed_stats, fh, indent=2)

    fixed_stats_csv = os.path.join(
        splits_dir, "_fixed_pc_validation_rpattern_stats.csv"
    )
    if fixed_stats:
        pd.DataFrame(fixed_stats).to_csv(fixed_stats_csv, index=False)
    print(f"\nWrote fixed-PC validation R-pattern statistics to {fixed_stats_json}")
    if fixed_stats:
        print(f"Wrote fixed-PC validation R-pattern statistics to {fixed_stats_csv}")

    # Print final summary table.
    print(f"\n{'='*84}\nSUMMARY (validation, all PCs retained)\n{'='*84}")
    print(f"{'phase':<10} {'train':>6} {'val':>5} {'test':>5} {'n_pcs':>6} "
          f"{'Rp_med':>10} {'Rp_p95':>10} {'Rwp_med':>10} {'Rwp_p95':>10}")
    for s in summaries:
        print(f"{s['phase']:<10} {s['n_train']:>6} {s['n_val']:>5} "
              f"{s['n_test']:>5} {s['n_pcs_total']:>6} "
              f"{s['rp_median_full']:>10.4e} "
              f"{s['rp_p95_full']:>10.4e} "
              f"{s['rwp_median_full']:>10.4e} "
              f"{s['rwp_p95_full']:>10.4e}")

    print(f"\n{'='*84}\nFIXED-PC VALIDATION R-PATTERN SUMMARY\n{'='*84}")
    print(f"{'phase':<10} {'requested':>9} {'used':>6} {'val':>7} "
          f"{'Rp_mean':>12} {'Rp_median':>12} {'Rp_p95':>12}")
    for fs in fixed_stats:
        print(f"{fs['phase']:<10} {fs['requested_n_components']:>9} "
              f"{fs['n_components_used']:>6} "
              f"{fs['n_validation_samples']:>7} "
              f"{fs['rp_mean']:>12.4e} "
              f"{fs['rp_median']:>12.4e} "
              f"{fs['rp_p95']:>12.4e}")

    summary_path = os.path.join(splits_dir, "_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
