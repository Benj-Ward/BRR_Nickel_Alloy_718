# driver_lhs_v2.py
#
# Parallel training-data generator for surrogate models of the GSAS-II
# forward model (model_v1_old.py). For each of the 6 IN718 phases, samples
# a 2D parameter space (mustrain, hstrain_D11) at FIXED SCALE = 1.0 and
# saves the resulting diffraction patterns alongside the input parameters
# in a form directly consumable by surrogate-model training.
#
# Scale is held fixed because it is a direct multiplicative factor in the
# physics model (pattern = scale * unit_pattern). Including scale as a
# surrogate input wastes capacity on a relationship that is exact and
# can be applied analytically at inference time — the surrogate predicts
# the unit pattern, the caller multiplies by whatever scale they want.
# This produces unit patterns directly, with no division and no MIN_SCALE
# noise-amplification concerns.
#
# Sampling strategy per phase:
#   1. Main Latin Hypercube — N_MAIN_LHS samples uniform across the
#      2D (mustrain, hstrain) box.
#   2. Corner points — all 4 vertices of the 2D box, included verbatim.
#   3. Optional Beta(alpha, alpha) marginal transform on the main LHS
#      (set USE_BETA_MARGINALS=True). Default alpha=0.7 produces a mild
#      U-shape that concentrates points at both ends of each marginal.
#      Corners are not transformed (already at edges).
#
# Boundary LHS is gone: its purpose was to densify near scale=0, which
# no longer applies.
#
# State carryover (approach a): every job sends a full dict covering all
# 6 phases. Only the active phase's mustrain and hstrain vary per sample;
# scale_<active> is held at 1.0; the other five phases are pinned to
# REFERENCE_VALUES. This makes each sample's job specification fully
# explicit, with no dependence on what the worker happened to be set to
# from the previous job.
#
# Outputs (in TRAINING_DATA_DIR/<phase>/):
#   <phase>_inputs.csv      - sample matrix, one row per sample, header
#                             names every column. Includes all 18
#                             parameter columns (6 phases x 3 params)
#                             plus sample_id, active_phase, source. The
#                             active phase's scale column is 1.0 in every
#                             row; only mustrain and hstrain vary.
#   <phase>_patterns.npy    - stacked y_calc patterns (unit patterns,
#                             since scale=1.0), shape
#                             (n_samples, n_two_theta), rows aligned
#                             with <phase>_inputs.csv by sample_id.
#   <phase>_failed.txt      - sample_ids that returned None.
#   <phase>_patterns_partial/ - per-sample .npy files written during
#                             the run; assembled into the final .npy
#                             when the phase finishes. Persists if the
#                             run crashes, so progress isn't lost.
# Plus a single run_metadata.json at the top of TRAINING_DATA_DIR.
#
# Usage: python driver_lhs_v2.py
# Edit the CONFIGURATION block to tune sample counts and runtime.

import os
import sys
import json
import time
import subprocess
import atexit
import glob
import csv
from datetime import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import qmc, beta as beta_dist


# =========================================================================
# CONFIGURATION
# =========================================================================
WORKER_SCRIPT = "model_v1_old.py"
# INSERT YOUR GSAS-II INSTALL LOCATION
WORKER_PYTHON = #r"C:/Users/wardbm1/gsas2main/python.exe"

N_WORKERS = 6

# Sampling counts (per phase).
N_MAIN_LHS = 10000

# Scale is held fixed at this value for every sample. The patterns
# produced are therefore unit patterns (or, more precisely, unit *
# FIXED_SCALE patterns); the surrogate predicts these and the caller
# applies the desired scale at inference time as a single multiplication.
FIXED_SCALE = 1.0

# Optional U-shape marginal transform on the main LHS only.
USE_BETA_MARGINALS = True
BETA_ALPHA = 0.7              # alpha=beta=0.5 -> classic arcsine U-shape # alpha=beta=1 -> uniform

SEED = 123 

# Output root. A timestamped subdirectory will be created inside.
OUTPUT_ROOT = "training_data"


# =========================================================================
# PHASE DEFINITIONS
# =========================================================================
# Order matters for the parameter column layout in the CSV header. This is
# the canonical phase order used everywhere in this file.
PHASE_NAMES = ["gamma", "delta", "gamma1", "gamma2", "laves", "carbide"]
# Subselection of these phases requires editing in model_old as well
#PHASE_NAMES = ["gamma", "laves", "carbide"]

# Per-phase 2D box: (mustrain_low, mustrain_high), (hstrain_low,
# hstrain_high). Scale is no longer sampled — it is fixed at FIXED_SCALE
# for every sample. The previous per-phase scale ranges are recorded
# below as comments for reference but are not used in this driver.
PHASE_BOUNDS = {
    "gamma":   {"mustrain": (1000.0, 41000.0), "hstrain": (-0.01, 0.01)},  # was scale=(0.6,  1.0) #No scale factor required anymore.
    "delta":   {"mustrain": (1000.0, 41000.0), "hstrain": (-0.01, 0.01)},  # was scale=(0.0,  0.2)
    "gamma1":  {"mustrain": (1000.0, 41000.0), "hstrain": (-0.01, 0.01)},  # was scale=(0.0,  0.25)
    "gamma2":  {"mustrain": (1000.0, 41000.0), "hstrain": (-0.01, 0.01)},  # was scale=(0.0,  0.30)
    "laves":   {"mustrain": (1000.0, 41000.0), "hstrain": (-0.01, 0.01)},  # was scale=(0.0,  0.1)
    "carbide": {"mustrain": (1000.0, 41000.0), "hstrain": (-0.01, 0.01)},  # was scale=(0.0,  0.1)
}

# Reference values for inactive phases (approach a: full dict every job). Critically mustrain > 0 even if scale is 0.
REFERENCE_VALUES = {
    "scale":    1e-8,    # effectively zero contribution; avoids exact-zero pathology
    "mustrain": 1000.0,  # lower end of the bounds
    "hstrain":  0.0,     # no isotropic strain
}

# Canonical column ordering for the parameter dict and the CSV header.
PARAM_TYPES = ["scale", "mustrain", "hstrain"]

def param_key(phase, ptype):
    """Build the JSON dict key the worker expects."""
    if ptype == "hstrain":
        return f"hstrain_{phase}_D11"
    return f"{ptype}_{phase}"

# Full ordered list of 18 column names: scale_gamma, mustrain_gamma,
# hstrain_gamma_D11, scale_delta, ...
PARAM_COLUMNS = [param_key(phase, ptype)
                 for phase in PHASE_NAMES
                 for ptype in PARAM_TYPES]


# =========================================================================
# SAMPLING
# =========================================================================
def _maybe_beta_transform(unit_samples, alpha):
    """Apply Beta(alpha, alpha) inverse-CDF to unit-cube samples in place
    of the identity. Pushes mass toward 0 and 1 when alpha < 1."""
    return beta_dist.ppf(unit_samples, alpha, alpha)


def main_lhs_for_phase(phase, n, seed, use_beta, beta_alpha):
    """N samples uniform over the full 2D (mustrain, hstrain) box for
    this phase. If use_beta is True, the unit-cube draw is reshaped
    through a Beta(alpha, alpha) inverse-CDF before scaling to bounds,
    concentrating mass at both edges of each marginal.
    """
    bounds = PHASE_BOUNDS[phase]
    l_bounds = np.array([bounds["mustrain"][0], bounds["hstrain"][0]])
    u_bounds = np.array([bounds["mustrain"][1], bounds["hstrain"][1]])
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n=n)
    if use_beta:
        # Beta inverse-CDF on the (0,1) draws, then scale to bounds.
        unit = _maybe_beta_transform(unit, beta_alpha)
    scaled = qmc.scale(unit, l_bounds, u_bounds)
    return scaled


def corner_points_for_phase(phase):
    """All 4 vertices of the 2D box (mustrain × hstrain), in a
    deterministic order."""
    bounds = PHASE_BOUNDS[phase]
    m = bounds["mustrain"]
    h = bounds["hstrain"]
    corners = []
    for mv in m:
        for hv in h:
            corners.append([mv, hv])
    return np.array(corners)


def build_samples_for_phase(phase):
    """
    Combine main LHS + corners into a single sample matrix for this
    phase. Returns a list of dicts with keys 'pair' (2-vector
    [mustrain, hstrain]) and 'source' (str label).

    Boundary LHS is gone — its purpose was densifying near scale=0,
    which no longer applies in the 2D-input setup.
    """
    out = []

    # Main LHS
    main = main_lhs_for_phase(
        phase, N_MAIN_LHS, seed=SEED,
        use_beta=USE_BETA_MARGINALS, beta_alpha=BETA_ALPHA,
    )
    for row in main:
        out.append({"pair": row, "source": "main_lhs"})

    # Corners
    corners = corner_points_for_phase(phase)
    for row in corners:
        out.append({"pair": row, "source": "corner"})

    return out


# =========================================================================
# JOB DICT BUILDING
# =========================================================================
def reference_dict():
    """Return a fresh dict of reference values for all 6 phases (all 18
    keys). The active phase's pair (and FIXED_SCALE) will overwrite
    three of these."""
    d = {}
    for phase in PHASE_NAMES:
        for ptype in PARAM_TYPES:
            d[param_key(phase, ptype)] = REFERENCE_VALUES[ptype]
    return d


def make_job_dict(active_phase, pair):
    """Reference values for all phases, then the active phase's two
    sampled parameters from the pair plus scale fixed at FIXED_SCALE.

    `pair` is [mustrain, hstrain]. The active phase's scale is set to
    FIXED_SCALE (1.0) rather than the REFERENCE_VALUES placeholder so
    the worker computes a real unit pattern for this phase.
    """
    d = reference_dict()
    d[param_key(active_phase, "scale")]    = float(FIXED_SCALE)
    d[param_key(active_phase, "mustrain")] = float(pair[0])
    d[param_key(active_phase, "hstrain")]  = float(pair[1])
    return d


def dict_to_row(job_dict):
    """Flatten a job dict into the canonical 18-column row order."""
    return [job_dict[col] for col in PARAM_COLUMNS]


# =========================================================================
# WORKER WRAPPER
# =========================================================================
class GSASWorker:
    """Persistent subprocess wrapper around model_v1_old.py."""
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen(
            [WORKER_PYTHON, WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"Worker {self.worker_id} exited before READY"
                )
            if line.strip() == "READY":
                return

    def run(self, params_dict):
        """Send one job, return y_calc as np.ndarray, or None on failure."""
        msg = json.dumps(params_dict)
        self.proc.stdin.write(f"RUN_JOB {msg}\n")
        self.proc.stdin.flush()

        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if line == "JSON_START":
                break
            if line.startswith("{") and "error" in line:
                return None

        json_lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if line == "JSON_END":
                break
            json_lines.append(line)

        try:
            data = json.loads("\n".join(json_lines))
            return np.asarray(data, dtype=float)
        except json.JSONDecodeError:
            return None

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write("EXIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()


# Module-level worker cache so each pool process starts its GSAS-II
# subprocess exactly once. Reused across all jobs in this Python process.
_worker_instance = None


def _init_worker():
    global _worker_instance
    pid = os.getpid()
    _worker_instance = GSASWorker(worker_id=pid)
    _worker_instance.start()


def run_sample(sample_index, params_dict, partial_dir):
    """Top-level pool function. Runs one job and writes the result to
    partial_dir/<sample_index>.npy on success. Returns (sample_index,
    success_bool). Pattern data is NOT returned through the queue —
    writing to disk avoids a multi-MB pickle round-trip per sample."""
    global _worker_instance
    if _worker_instance is None:
        _init_worker()
    y = _worker_instance.run(params_dict)
    if y is None:
        return sample_index, False
    np.save(os.path.join(partial_dir, f"{sample_index}.npy"), y)
    return sample_index, True


# =========================================================================
# OUTPUT HELPERS
# =========================================================================
def write_inputs_csv(path, samples, active_phase):
    """Write the inputs CSV. Columns: sample_id, active_phase, source,
    then PARAM_COLUMNS in canonical order."""
    header = ["sample_id", "active_phase", "source"] + PARAM_COLUMNS
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, s in enumerate(samples):
            row = [i, active_phase, s["source"]] + [
                f"{v:.10g}" for v in dict_to_row(s["job_dict"])
            ]
            writer.writerow(row)


def assemble_patterns_npy(partial_dir, n_samples):
    """Read every <i>.npy from partial_dir (i = 0 .. n_samples-1) and
    stack them into a single (n_samples, n_two_theta) array. Missing
    files become rows of NaN. Returns (stacked, failed_ids)."""
    failed = []
    # Find one successful sample to learn the row width.
    width = None
    for i in range(n_samples):
        path = os.path.join(partial_dir, f"{i}.npy")
        if os.path.exists(path):
            try:
                width = len(np.load(path))
                break
            except Exception:
                continue
    if width is None:
        # No successful samples at all.
        return None, list(range(n_samples))

    stacked = np.full((n_samples, width), np.nan)
    for i in range(n_samples):
        path = os.path.join(partial_dir, f"{i}.npy")
        if not os.path.exists(path):
            failed.append(i)
            continue
        try:
            stacked[i] = np.load(path)
        except Exception:
            failed.append(i)
    return stacked, failed


def cleanup_worker_files(workdir=None):
    """Delete per-process GSAS-II artifacts left by the workers."""
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


# =========================================================================
# PER-PHASE RUN
# =========================================================================
def run_one_phase(phase, phase_dir):
    """Generate samples, dispatch to the worker pool, write outputs."""
    print(f"\n=== Phase: {phase} ===")
    os.makedirs(phase_dir, exist_ok=True)
    partial_dir = os.path.join(phase_dir, f"{phase}_patterns_partial")
    os.makedirs(partial_dir, exist_ok=True)

    # Build samples and decorate with full job dicts.
    samples = build_samples_for_phase(phase)
    for s in samples:
        s["job_dict"] = make_job_dict(phase, s["pair"])
    n_samples = len(samples)

    n_main = sum(1 for s in samples if s["source"] == "main_lhs")
    n_corn = sum(1 for s in samples if s["source"] == "corner")
    print(f"  {n_samples} samples: {n_main} main_lhs, "
          f"{n_corn} corner (fixed scale = {FIXED_SCALE})")

    # Write inputs CSV upfront — doesn't depend on model output.
    inputs_csv = os.path.join(phase_dir, f"{phase}_inputs.csv")
    write_inputs_csv(inputs_csv, samples, phase)
    print(f"  wrote {inputs_csv}")

    # Dispatch to workers.
    print(f"  launching {N_WORKERS} workers...")
    t0 = time.time()
    completed = 0
    failed = 0
    with ProcessPoolExecutor(
        max_workers=N_WORKERS, initializer=_init_worker
    ) as ex:
        futures = {
            ex.submit(run_sample, i, samples[i]["job_dict"], partial_dir): i
            for i in range(n_samples)
        }
        for fut in as_completed(futures):
            i, ok = fut.result()
            if not ok:
                failed += 1
            completed += 1
            if completed % 50 == 0 or completed == n_samples:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (n_samples - completed) / rate if rate > 0 else float("inf")
                print(f"    {completed}/{n_samples} done "
                      f"({failed} failed), "
                      f"{rate:.2f} samples/s, ETA {eta:.0f}s")

    print(f"  forward model runs complete: {completed} total, {failed} failed.")

    # Assemble final outputs.
    stacked, failed_ids = assemble_patterns_npy(partial_dir, n_samples)
    if stacked is not None:
        out_npy = os.path.join(phase_dir, f"{phase}_patterns.npy")
        np.save(out_npy, stacked)
        print(f"  wrote {out_npy} (shape {stacked.shape})")
    else:
        print(f"  no successful samples for {phase}; no patterns saved")

    if failed_ids:
        with open(os.path.join(phase_dir, f"{phase}_failed.txt"), "w") as fh:
            fh.write("\n".join(str(i) for i in failed_ids) + "\n")
        print(f"  wrote {phase}_failed.txt ({len(failed_ids)} ids)")


# =========================================================================
# MAIN
# =========================================================================
def main():
    atexit.register(cleanup_worker_files)

    # Timestamped run directory.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        OUTPUT_ROOT,
        f"run_{timestamp}_seed{SEED}_main{N_MAIN_LHS}_2D"
        + ("_beta" if USE_BETA_MARGINALS else "")
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Save run metadata.
    metadata = {
        "timestamp": timestamp,
        "seed": SEED,
        "n_main_lhs": N_MAIN_LHS,
        "fixed_scale": FIXED_SCALE,
        "sampled_dimensions": ["mustrain", "hstrain"],
        "use_beta_marginals": USE_BETA_MARGINALS,
        "beta_alpha": BETA_ALPHA,
        "n_workers": N_WORKERS,
        "phase_names": PHASE_NAMES,
        "phase_bounds": PHASE_BOUNDS,
        "reference_values": REFERENCE_VALUES,
        "param_columns": PARAM_COLUMNS,
        "worker_script": WORKER_SCRIPT,
        "worker_python": WORKER_PYTHON,
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    # Loop over phases. Each phase gets its own subdirectory and its own
    # ProcessPoolExecutor (workers shut down between phases — clean state,
    # and the Python-level worker startup cost is small compared to GSAS).
    for phase in PHASE_NAMES:
        phase_dir = os.path.join(run_dir, phase)
        run_one_phase(phase, phase_dir)

    print(f"\nAll phases complete. Results in {run_dir}/")
    cleanup_worker_files()


if __name__ == "__main__":
    main()
