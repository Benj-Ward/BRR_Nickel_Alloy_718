# driver_three_examples.py
#
# Configurable 3-example driver for model_v1_old.py.
#
# This is intentionally NOT a sampler. It reuses the GSAS-II worker launch
# and stdin/stdout protocol from the LHS training-data driver, but runs only
# the explicitly prescribed examples in EXAMPLES below.
#
# Workflow:
#   1. Start one persistent model_v1_old.py worker.
#   2. Wait for READY.
#   3. For each example, send one complete RUN_JOB JSON dict containing all
#      phase parameters.
#   4. Read y_calc from JSON_START / JSON_END.
#   5. Save inputs and patterns in a timestamped output directory.
#
# Usage:
#   python driver_three_examples.py
#
# Edit EXAMPLES to prescribe your three cases.

import os
import sys
import json
import csv
import time
import glob
import subprocess
import atexit
from datetime import datetime

import numpy as np


# =========================================================================
# CONFIGURATION
# =========================================================================
WORKER_SCRIPT = "model_v1_old.py"
WORKER_PYTHON = r"C:/Users/wardbm1/gsas2main/python.exe"

OUTPUT_ROOT = "prescribed_examples"

# Canonical phase order used by model_v1_old.py and the LHS driver.
PHASE_NAMES = ["gamma", "delta", "gamma1", "gamma2", "laves", "carbide"]

# The worker expects exactly these parameter types for the current file-4
# parameter set: scale_<phase>, mustrain_<phase>, hstrain_<phase>_D11.
PARAM_TYPES = ["scale", "mustrain", "hstrain"]

ADD_COUNTING_NOISE = True
NOISE_RANDOM_SEED = 12345
NOISE_COUNT_SCALE = 1.0
if ADD_COUNTING_NOISE:
    print("Will add counting noise!")


def param_key(phase, ptype):
    """Build the JSON dict key expected by model_v1_old.py."""
    if ptype == "hstrain":
        return f"hstrain_{phase}_D11"
    return f"{ptype}_{phase}"


PARAM_COLUMNS = [
    param_key(phase, ptype)
    for phase in PHASE_NAMES
    for ptype in PARAM_TYPES
]

# -------------------------------------------------------------------------
# EDIT THIS BLOCK
# -------------------------------------------------------------------------
# Prescribe exactly the values you want for each example. Placeholders below
# are valid numeric values and are deliberately easy to find/change.
#
# Units/meaning are the same as in model_v1_old.py:
#   scale    -> phase histogram Scale
#   mustrain -> isotropic Mustrain value
#   hstrain  -> HStrain D11 value, sent as hstrain_<phase>_D11
#
# Every phase should be present in every example. The driver sends a full
# dict every time so there is no state carryover from a previous run.
EXAMPLES = [
    {
        "example_id": "as-built",
        "description": "Three phases",
        "phases": {
            "gamma":   {"scale": 0.98,  "mustrain": 8000.0,  "hstrain": 0.0025},
            "delta":   {"scale": 0.0,  "mustrain": 1000.0,  "hstrain": 0.0},
            "gamma1":  {"scale": 0.0,  "mustrain": 1000.0,  "hstrain": 0.0},
            "gamma2":  {"scale": 0.0,  "mustrain": 1000.0,  "hstrain": 0.0},
            "laves":   {"scale": 0.012,  "mustrain": 19000.0,  "hstrain": 0.004},
            "carbide": {"scale": 0.008,  "mustrain": 25000.0,  "hstrain": 0.002},
        },
    },
    {
        "example_id": "in-situ",
        "description": "Six phases",
        "phases": {
            "gamma":   {"scale": 0.88,  "mustrain": 7000.0,  "hstrain": 0.002},
            "delta":   {"scale": 0.03, "mustrain": 12000.0,  "hstrain": 0.0005},
            "gamma1":  {"scale": 0.03, "mustrain": 10000.0,  "hstrain": -0.0012},
            "gamma2":  {"scale": 0.04, "mustrain": 6000.0,  "hstrain": 0.002},
            "laves":   {"scale": 0.01, "mustrain": 17000.0,  "hstrain": 0.003},
            "carbide": {"scale": 0.01, "mustrain": 22000.0,  "hstrain": 0.001},
        },
    },
    {
        "example_id": "homogenized",
        "description": "Five phases",
        "phases": {
            "gamma":   {"scale": 0.81,  "mustrain": 6000.0, "hstrain": 0.001},
            "delta":   {"scale": 0.02,  "mustrain": 12000.0, "hstrain": 0.0003},
            "gamma1":  {"scale": 0.06,  "mustrain": 8000.0, "hstrain": 0.0005},
            "gamma2":  {"scale": 0.1,  "mustrain": 5000.0, "hstrain": -0.0008},
            "laves":   {"scale": 1e-8, "mustrain": 15000.0, "hstrain": 0.002},
            "carbide": {"scale": 0.01, "mustrain": 20000.0, "hstrain": 0.0005},
        },
    },
]


# =========================================================================
# JOB DICT BUILDING
# =========================================================================
def validate_example(example):
    """Fail early if an example is missing phases or parameter values."""
    if "example_id" not in example:
        raise ValueError("Each example must have an example_id")
    if "phases" not in example or not isinstance(example["phases"], dict):
        raise ValueError(f"{example['example_id']} must contain a phases dict")

    missing_phases = [p for p in PHASE_NAMES if p not in example["phases"]]
    extra_phases = [p for p in example["phases"] if p not in PHASE_NAMES]
    if missing_phases:
        raise ValueError(f"{example['example_id']} missing phases: {missing_phases}")
    if extra_phases:
        raise ValueError(f"{example['example_id']} has unknown phases: {extra_phases}")

    for phase in PHASE_NAMES:
        phase_values = example["phases"][phase]
        missing = [ptype for ptype in PARAM_TYPES if ptype not in phase_values]
        extra = [ptype for ptype in phase_values if ptype not in PARAM_TYPES]
        if missing:
            raise ValueError(
                f"{example['example_id']} phase {phase} missing values: {missing}"
            )
        if extra:
            raise ValueError(
                f"{example['example_id']} phase {phase} has unknown values: {extra}"
            )


def make_job_dict(example):
    """Convert nested example config into the flat JSON dict for the worker."""
    validate_example(example)
    d = {}
    for phase in PHASE_NAMES:
        for ptype in PARAM_TYPES:
            d[param_key(phase, ptype)] = float(example["phases"][phase][ptype])
    return d


def dict_to_row(job_dict):
    """Flatten a job dict into the canonical 18-column row order."""
    return [job_dict[col] for col in PARAM_COLUMNS]


# =========================================================================
# WORKER WRAPPER
# =========================================================================
class GSASWorker:
    """Persistent subprocess wrapper around model_v1_old.py."""

    def __init__(self, worker_id="single"):
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
                raise RuntimeError(f"Worker {self.worker_id} exited before READY")
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

            if isinstance(data, dict):
                return data

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


# =========================================================================
# OUTPUT HELPERS
# =========================================================================
def write_inputs_csv(path, examples_with_jobs):
    """Write one CSV row per example with the same flat parameter columns."""
    header = ["sample_id", "example_id", "description"] + PARAM_COLUMNS
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, item in enumerate(examples_with_jobs):
            example = item["example"]
            job_dict = item["job_dict"]
            writer.writerow(
                [
                    i,
                    example["example_id"],
                    example.get("description", ""),
                ]
                + [f"{v:.10g}" for v in dict_to_row(job_dict)]
            )


def write_metadata(path, run_dir):
    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir,
        "worker_script": WORKER_SCRIPT,
        "worker_python": WORKER_PYTHON,
        "phase_names": PHASE_NAMES,
        "param_types": PARAM_TYPES,
        "param_columns": PARAM_COLUMNS,
        "n_examples": len(EXAMPLES),
        "notes": (
            "Prescribed examples only. No Latin hypercube sampling. "
            "Each job sends a full 18-parameter dict to avoid state carryover."
        ),
    }
    with open(path, "w") as fh:
        json.dump(metadata, fh, indent=2)


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


# =========================================================================
# MAIN
# =========================================================================
def main():
    rng = np.random.default_rng(NOISE_RANDOM_SEED)
    atexit.register(cleanup_worker_files)

    examples_with_jobs = []
    for example in EXAMPLES:
        examples_with_jobs.append(
            {
                "example": example,
                "job_dict": make_job_dict(example),
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_ROOT, f"run_{timestamp}_n{len(EXAMPLES)}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    inputs_csv = os.path.join(run_dir, "prescribed_inputs.csv")
    write_inputs_csv(inputs_csv, examples_with_jobs)
    print(f"Wrote {inputs_csv}")

    write_metadata(os.path.join(run_dir, "run_metadata.json"), run_dir)

    worker = GSASWorker(worker_id="prescribed_examples")
    print("Starting GSAS-II worker...")
    worker.start()
    print("Worker ready.")

    patterns = []
    x_spacing = []
    failed = []
    t0 = time.time()
    try:
        for i, item in enumerate(examples_with_jobs):
            example = item["example"]
            example_id = example["example_id"]
            print(f"Running {i + 1}/{len(examples_with_jobs)}: {example_id}")
            result = worker.run(item["job_dict"])
            x = np.asarray(result["x"], dtype=float)
            y = np.asarray(result["ycalc"], dtype=float)

            if y is not None and ADD_COUNTING_NOISE:
                lam = np.clip(y * NOISE_COUNT_SCALE, 0.0, None)
                y = rng.poisson(lam).astype(float) / NOISE_COUNT_SCALE
            if y is None:
                failed.append(i)
                print(f"  FAILED: {example_id}")
                continue

            patterns.append((i, y))
            per_example_path = os.path.join(run_dir, f"{example_id}_pattern.npy")
            np.save(per_example_path, y)
            print(f"  wrote {per_example_path} shape={y.shape}")

            x_spacing.append((i, x))
            per_example_path_X = os.path.join(run_dir, f"{example_id}_x_spacing.npy")
            np.save(per_example_path_X, x)
            print(f"  wrote {per_example_path} shape={y.shape}")
    finally:
        worker.stop()

    if patterns:
        width = len(patterns[0][1])
        stacked = np.full((len(EXAMPLES), width), np.nan)
        for i, y in patterns:
            stacked[i] = y
        stacked_path = os.path.join(run_dir, "prescribed_patterns.npy")
        np.save(stacked_path, stacked)
        print(f"Wrote {stacked_path} shape={stacked.shape}")
    else:
        print("No successful patterns; prescribed_patterns.npy was not written.")

    if failed:
        failed_path = os.path.join(run_dir, "failed.txt")
        with open(failed_path, "w") as fh:
            fh.write("\n".join(str(i) for i in failed) + "\n")
        print(f"Wrote {failed_path} with {len(failed)} failed example id(s).")

    elapsed = time.time() - t0
    print(f"Done. Results in {run_dir}/. Elapsed: {elapsed:.1f}s")
    #cleanup_worker_files()


if __name__ == "__main__":
    main()
