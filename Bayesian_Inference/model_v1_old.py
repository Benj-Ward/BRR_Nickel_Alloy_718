# model_v1_old.py
#
# Forward model worker for GSAS-II surrogate-model training data generation.
#
# This is a v1-style worker (dict-based parameter interface, JSON_START/
# JSON_END handshake, per-PID gpx for parallel safety, max cyc = 0 for
# pure pattern calculation) but configured with the phases, CIF files, and
# diffraction data from phase_fractions_sample_4.py.
#
# Interface: accepts JSON dicts on stdin with keys like
#     scale_<phase>, mustrain_<phase>, hstrain_<phase>_D11
# for each phase in:
#     gamma, delta, gamma1, gamma2, laves, carbide
#
# Any parameter omitted from the input dict is not applied; GSAS-II keeps
# whatever value it currently holds (CIF default or value left by a prior
# job). The driver is responsible for sending a full dict every job if it
# wants explicit control of all phases — see driver_lhs_v0.py.
#
# March-Dollase preferred orientation is NOT installed in this worker. The
# apply_MD machinery is preserved for future use; to enable it, add an
# "MD_hkl" key to each entry of PHASES_CONFIG and uncomment the
# install_march_dollase loop in profile_setup.
#
# Workflow:
#   1. Startup: create project, load histogram, add phases, set background
#      to zero, set histogram scale, do an initial pattern calculation,
#      print READY.
#   2. Per job: parse dict, apply parameters, run pattern calculation,
#      return y_calc as JSON between JSON_START / JSON_END sentinels.

import sys
import os
import json
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    print("ERROR importing matplotlib.pyplot", flush=True)
    sys.exit(1)

# Add the folder containing the GSASII package
sys.path.insert(0, r"C:\Users\wardbm1\gsas2main\GSAS-II")
try:
    from GSASII import GSASIIscriptable as G2sc
except ModuleNotFoundError as e:
    print(f"ERROR: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

print("Debug: GSAS run started, setup initializing.", flush=True)


# =========================================================================
# CONFIGURATION
# =========================================================================
# Phases for the IN718 surrogate-model training set. Paths and names match
# phase_fractions_sample_4.py.
PHASEDIR = os.path.join(os.getcwd(), "data_dir/phase_dir/IN718 Phases/IN718 Phases")
PHASES_CONFIG = [
    {
        "name": "gamma",
        "cif": os.path.join(PHASEDIR, "Ni_mp-23_symmetrized.cif"),
    },
    {
        "name": "delta",
        "cif": os.path.join(PHASEDIR, "delta/NbNi3_mp-1451_conventional_standard.cif"),
    },
    {
        "name": "gamma1",
        "cif": os.path.join(PHASEDIR, "gamma prime/NbNi3_mp-999188_conventional_standard.cif"),
    },
    {
        "name": "gamma2",
        "cif": os.path.join(PHASEDIR, "gamma double prime/1NbNi3_mp-11513_conventional_standard.cif"),
    },
    {
        "name": "laves",
        "cif": os.path.join(PHASEDIR, "laves/NbNi2_mp-1191285_conventional_standard.cif"),
    },
    {
        "name": "carbide",
        "cif": os.path.join(PHASEDIR, "carbide/1NbC_mp-910_conventional_standard.cif"),
    },
]

DATADIR = os.path.join(os.getcwd(), "data_dir")
HIST_FXYE = "In-situ.fxye"
HIST_PRM = "In-situ.prm"


# =========================================================================
# UTILITIES
# =========================================================================
def phase_by_name(gpx, name):
    """Return the G2Phase object with the given name, or None."""
    for p in gpx.phases():
        if p.name == name:
            return p
    return None


def elements_in_phase(phase):
    """Return the sorted list of unique element symbols in a phase."""
    try:
        return sorted({atom.type for atom in phase.atoms()})
    except Exception:
        ap = phase.data["General"]["AtomPtrs"]
        ct = ap[1]
        return sorted({a[ct] for a in phase.data["Atoms"]})


def install_march_dollase(phase, hkl):
    """
    Set March-Dollase preferred orientation for this phase on every linked
    histogram. Currently unused — left in for future enablement.
    """
    for hist_name in phase.data["Histograms"]:
        hap = phase.data["Histograms"][hist_name]
        hap["Pref.Ori."] = ["MD", 1.0, False, list(hkl), 0, {}, []]


def calc_pattern(gpx):
    """Trigger a pattern calculation without refining anything."""
    gpx.do_refinements([{}])


# =========================================================================
# PARAMETER APPLICATION
# =========================================================================
# Each applier reads the subset of `params` relevant to it and silently
# skips keys that aren't present.

def apply_scales(gpx, params):
    for p in gpx.phases():
        key = f"scale_{p.name}"
        if key in params:
            for hist in p.data["Histograms"]:
                p.data["Histograms"][hist]["Scale"][0] = float(params[key])


def apply_mustrain(gpx, params):
    for p in gpx.phases():
        key = f"mustrain_{p.name}"
        if key not in params:
            continue
        mu = float(params[key])
        for hist in p.data["Histograms"]:
            mus_struct = p.data["Histograms"][hist]["Mustrain"]
            if mus_struct[0] != "isotropic":
                mus_struct[0] = "isotropic"
            mus_struct[1][0] = mu
            mus_struct[1][1] = mu  # harmless for isotropic
            mus_struct[1][2] = 1.0


def apply_hstrain(gpx, params):
    # HStrain D11 only in this worker (file-4 parameter set).
    for p in gpx.phases():
        for hist in p.data["Histograms"]:
            hs = p.data["Histograms"][hist]["HStrain"]
            ncomp = len(hs[0])
            k11 = f"hstrain_{p.name}_D11"
            k33 = f"hstrain_{p.name}_D33"
            if k11 in params:
                hs[0][0] = float(params[k11])
            if k33 in params and ncomp >= 2:
                hs[0][1] = float(params[k33])


def apply_uiso(gpx, params):
    for p in gpx.phases():
        by_elem = {}
        for atom in p.atoms():
            by_elem.setdefault(atom.type, []).append(atom)
        for elem, atoms in by_elem.items():
            key = f"uiso_{p.name}_{elem}"
            if key not in params:
                continue
            u = float(params[key])
            for atom in atoms:
                try:
                    atom.uiso = u
                except AttributeError:
                    ap = p.data["General"]["AtomPtrs"]
                    cia = ap[3]
                    atom._data[cia + 1] = u


def apply_lattice(gpx, params):
    for p in gpx.phases():
        cell = p.data["General"]["Cell"]
        system = p.data["General"]["SGData"].get("SGSys", "").lower()
        a_key = f"a_{p.name}"
        c_key = f"c_{p.name}"
        if a_key in params:
            a = float(params[a_key])
            cell[1] = a
            if system in ("cubic", "tetragonal", "trigonal", "hexagonal"):
                cell[2] = a
            if system == "cubic":
                cell[3] = a
        if c_key in params:
            if system == "cubic":
                pass
            else:
                cell[3] = float(params[c_key])


def apply_MD(gpx, params):
    for p in gpx.phases():
        key = f"MD_R_{p.name}"
        if key not in params:
            continue
        R = float(params[key])
        for hist in p.data["Histograms"]:
            pref = p.data["Histograms"][hist]["Pref.Ori."]
            pref[1] = R


# =========================================================================
# EXPECTED KEYS (introspection-based)
# =========================================================================
def build_expected_keys(gpx):
    """
    Build the master list of recognised parameter keys based on the loaded
    phases. Used to warn about unknown keys in incoming jobs. For the
    file-4 parameter set we expose only scale, mustrain, and hstrain D11.
    """
    keys = set()
    for p in gpx.phases():
        name = p.name
        keys.add(f"scale_{name}")
        keys.add(f"mustrain_{name}")
        keys.add(f"hstrain_{name}_D11")
        # The following keys are recognised by the apply_* functions but are
        # NOT part of the file-4 parameter set, so they're omitted from
        # EXPECTED_KEYS to make typos visible. To enable any of them, add
        # the appropriate keys here:
        #   keys.add(f"hstrain_{name}_D33")  # non-cubic phases
        #   keys.add(f"a_{name}")
        #   keys.add(f"c_{name}")            # non-cubic phases
        #   keys.add(f"MD_R_{name}")         # requires install_march_dollase
        #   for elem in elements_in_phase(p): keys.add(f"uiso_{name}_{elem}")
    return keys


# =========================================================================
# PROJECT SETUP
# =========================================================================
def profile_setup(datadir, phases_config):
    """Create the project, load data, add phases, init."""
    proj_path = os.path.join(os.getcwd(), f"sample_pid{os.getpid()}.gpx")
    gpx = G2sc.G2Project(newgpx=proj_path)
    gpx.data['Controls']['data']['Save copy of .gpx file on Save'] = False

    hist1 = gpx.add_powder_histogram(
        os.path.join(datadir, HIST_FXYE),
        os.path.join(datadir, HIST_PRM),
    )

    for cfg in phases_config:
        gpx.add_phase(cfg["cif"], phasename=cfg["name"], histograms=[hist1])

    # No refinement cycling — we only want pattern calculations.
    gpx.data["Controls"]["data"]["max cyc"] = 0

    # Background off, histogram scale fixed (matches phase_fractions_sample_4).
    gpx.do_refinements([
        {"set": {"Background": {"coeffs": [0.0, 0, 0], "refine": False}}}
    ], histogram=hist1)
    hist1.data["Sample Parameters"]["Scale"] = [50.0, 0.0]

    # March-Dollase intentionally NOT installed for the file-4 parameter set.
    # To enable: add "MD_hkl" to each PHASES_CONFIG entry and uncomment:
    # for cfg in phases_config:
    #     p = phase_by_name(gpx, cfg["name"])
    #     if p is None:
    #         raise RuntimeError(f"phase {cfg['name']} not added correctly")
    #     install_march_dollase(p, cfg["MD_hkl"])

    # Initial pattern calculation.
    calc_pattern(gpx)
    return gpx, hist1


# =========================================================================
# MAIN
# =========================================================================
try:
    gpx, hist1 = profile_setup(DATADIR, PHASES_CONFIG)
except Exception as e:
    print(f"ERROR in profile setup: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

EXPECTED_KEYS = build_expected_keys(gpx)
print("Debug: GSAS setup complete.", flush=True)
print(f"Debug: expected parameter keys ({len(EXPECTED_KEYS)}): "
      f"{sorted(EXPECTED_KEYS)}", file=sys.stderr, flush=True)

# Signal worker ready
print("READY", flush=True)

# =========================================================================
# JOB LOOP
# =========================================================================
while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.strip()
    if line == "EXIT":
        print("Debug: Worker exiting", file=sys.stderr, flush=True)
        break
    if not line.startswith("RUN_JOB"):
        continue

    # Parse payload as a dict.
    try:
        _, payload = line.split(" ", 1)
        params = json.loads(payload)
        if not isinstance(params, dict):
            raise ValueError("payload must be a JSON object (dict)")
    except Exception as e:
        print(json.dumps({"error": f"invalid JSON payload: {e}"}), flush=True)
        continue

    # Warn (but don't fail) on unknown keys.
    unknown = set(params) - EXPECTED_KEYS
    if unknown:
        print(f"Debug: unknown keys in job, ignoring: {sorted(unknown)}",
              file=sys.stderr, flush=True)

    # Apply parameters and compute the pattern.
    try:
        apply_lattice(gpx, params)     # lattice first; affects peak positions
        apply_scales(gpx, params)
        apply_mustrain(gpx, params)
        apply_hstrain(gpx, params)
        apply_uiso(gpx, params)
        apply_MD(gpx, params)

        calc_pattern(gpx)

        hist = gpx.histogram(0)
        x_calc = hist.getdata("x")
        y_calc = hist.getdata("ycalc")

    except Exception as e:
        print(json.dumps({"error": f"refinement failed: {e}"}), flush=True)
        print(
            f"Debug: refinement failed: {e}",
            file=sys.stderr,
            flush=True,
        )
        continue

    x_out = x_calc.tolist() if isinstance(x_calc, np.ndarray) else x_calc
    y_out = y_calc.tolist() if isinstance(y_calc, np.ndarray) else y_calc

    print("JSON_START", flush=True)
    print(json.dumps({
        "x": x_out,
        "ycalc": y_out,
    }), flush=True)
    print("JSON_END", flush=True)
