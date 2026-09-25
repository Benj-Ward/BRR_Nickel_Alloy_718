# plot_phase_fractions_multirun.py
# ================================
# Publication figure: phase-fraction posteriors from multiple MCMC runs
# overlaid on a shared set of subplots (one panel per phase). Posteriors
# are drawn as KDE lines so multiple runs remain legible without bar
# clutter.
#
# Priors:
#   - Direct-mode runs share a single uniform prior overlay derived from
#     the first direct run's scale_bounds.
#   - Pseudo-marginal runs share a single Dirichlet-predictive prior
#     overlay derived from the first PM run's (mu_dirichlet_prior,
#     log_alpha0_prior).
# Both, either, or neither overlay can be present and the legend reflects
# whichever ones are drawn.
#
# Reads <output/run_name>/{standardizer.json, chain.h5, config.json} for
# each run, identical to the analyzer's conventions, and reuses the
# decoder / decoder / prior-sampling helpers from analyze_run_v5 so this
# file stays small.

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from analyze_run_v6 import (
    setup_publication_style,
    load_decoder,
    load_chain,
    decode_burned,
)


# =============================================================================
# USER CONFIG
# =============================================================================

# Runs to overlay. Each entry is (run_name, display_label). run_name must
# match the subdirectory under output/ (i.e. output/<run_name>/chain.h5).
# display_label is what appears in the legend.
RUNS: list[tuple[str, str]] = [
    ("ID35",                  "Standard"),
    ("ID35_boosted",          "Standard & Rescaled"),
    ("ID35_dirichlet",        "Dirichlet"),
    ("ID35_dirichlet_boosted4","Dirichlet & Rescaled"),
]

# Output PDF path. Parent directory is created if needed.
OUTPUT_PATH = Path("output/posteriors_phase_fractions_multirun.pdf")

# Per-axis x-limits. Three modes:
#   None                   -> default: per-phase bounds from the uniform
#                             prior (scale_bounds) of the first direct
#                             run; if no direct runs, fall back to the
#                             union of all runs' posterior support with
#                             5% padding.
#   (lo, hi)               -> apply this single range to every phase.
#   {phase_name: (lo, hi)} -> per-phase override; any phase not in the
#                             dict falls back to the default rule above.
XLIM_OVERRIDE = None

# Burn-in policy: keep the last KEEP_LAST_STEPS samples of each chain.
# Matches analyze_run_v5.py.
KEEP_LAST_STEPS = 500

# Pseudo-marginal-only knobs (silently ignored for direct runs).
N_FRAC_DRAWS_PER_SAMPLE = 4
N_PRIOR_DRAWS           = 5000
PRIOR_SEED              = 1

# Optional true-value overlay: per-phase reference line drawn as a black
# dashed vertical. Leave as {} to omit. Phases absent from the dict are
# simply not drawn.
PHASE_FRACTION_TRUE: dict[str, float] = {
    "gamma":   0.996037,
    "laves":   0.002856,
    "carbide": 0.001107,
}
PHASE_LABELS = {
    "gamma":              r"$\gamma$",
    "delta":              r"$\delta$",
    "gamma_prime":        r"$\gamma'$",
    "gamma_double_prime": r"$\gamma''$",
    "Laves":              "Laves",
    "Carbide":            "Carbide",
}

# =============================================================================
# Helpers
# =============================================================================

def _load_run(run_name: str) -> dict:
    """Load config, decoder, and decoded posterior for one run."""
    run_dir = Path("output") / run_name
    config_path = run_dir / "config.json"
    std_path    = run_dir / "standardizer.json"
    chain_path  = run_dir / "chain.h5"
    for p in (config_path, std_path, chain_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"missing file for run {run_name!r}: {p}"
            )
    with open(config_path) as f:
        config = json.load(f)
    decoder = load_decoder(std_path)
    cd = load_chain(chain_path, expected_ndim=decoder.ndim)
    burn_in = max(0, cd.nsteps - KEEP_LAST_STEPS)
    if decoder.is_pseudo_marginal:
        alpha_floor = float(
            config.get("pseudo_marginal", {}).get("alpha_floor", 1e-6)
        )
        decoded = decode_burned(
            cd, decoder, burn_in,
            k_draws=N_FRAC_DRAWS_PER_SAMPLE,
            alpha_floor=alpha_floor,
            seed=PRIOR_SEED,
        )
    else:
        decoded = decode_burned(cd, decoder, burn_in)
    return {
        "name":    run_name,
        "config":  config,
        "decoder": decoder,
        "decoded": decoded,
    }


def _build_log_alpha0_prior(pm_cfg: dict) -> dict:
    """Translate the pseudo_marginal config block into the prior-spec
    dict that _sample_phase_fraction_prior consumes."""
    ptype = pm_cfg.get("log_alpha0_prior", "uniform")
    if ptype == "uniform":
        lo, hi = pm_cfg["log_alpha0_bounds"]
        return {"type": "uniform", "low": float(lo), "high": float(hi)}
    if ptype == "lognormal":
        p = pm_cfg["log_alpha0_lognormal_params"]
        return {
            "type":  "lognormal",
            "mu":    float(p["mu"]),
            "sigma": float(p["sigma"]),
        }
    raise ValueError(f"unknown pseudo_marginal.log_alpha0_prior: {ptype!r}")


def _sample_phase_fraction_prior(a_mu: np.ndarray,
                                 log_alpha0_prior: dict,
                                 n_draws: int,
                                 seed: int = 0) -> np.ndarray:
    """Monte Carlo draws from the Dirichlet predictive prior:
        log_alpha_0 ~ <log_alpha0_prior>,
        mu          ~ Dir(a_mu),
        S | mu, alpha_0 ~ Dir(alpha_0 * mu).
    Inlined here so this script does not depend on which version of
    analyze_run_v5.py is on disk. Returns (n_draws, N).
    """
    rng = np.random.default_rng(seed)
    ptype = log_alpha0_prior["type"]
    if ptype == "uniform":
        log_alpha0 = rng.uniform(
            log_alpha0_prior["low"], log_alpha0_prior["high"], size=n_draws,
        )
    elif ptype == "lognormal":
        log_alpha0 = rng.normal(
            log_alpha0_prior["mu"], log_alpha0_prior["sigma"], size=n_draws,
        )
    else:
        raise ValueError(f"unknown log_alpha0_prior type: {ptype!r}")
    mu = rng.dirichlet(a_mu, size=n_draws)                  # (n_draws, N)
    alpha = np.exp(log_alpha0)[:, None] * mu                # (n_draws, N)
    gam = rng.standard_gamma(alpha)
    gam_sum = np.sum(gam, axis=1, keepdims=True)
    gam_sum = np.where(gam_sum > 0, gam_sum, 1.0)
    return gam / gam_sum


def _grid_shape(n: int) -> tuple[int, int]:
    """Subplot grid (nrows, ncols) for n panels. Matches the convention
    used in analyze_run_v5._plot_posteriors_publication."""
    if n <= 3:
        return 1, n
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    if n <= 9:
        return 3, 3
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    setup_publication_style()

    if not RUNS:
        raise ValueError("RUNS is empty; nothing to plot.")

    print(f"Loading {len(RUNS)} runs...")
    loaded = []
    for run_name, label in RUNS:
        entry = _load_run(run_name)
        entry["label"] = label
        loaded.append(entry)
        mode = "pm" if entry["decoder"].is_pseudo_marginal else "direct"
        print(f"  {run_name!r} ({mode}) -> {entry['decoded'].phase_fractions.shape[0]} samples")

    # All runs must share the same phase set, in the same order, for the
    # overlay to make sense.
    phase_names = loaded[0]["decoder"].phase_names
    for entry in loaded[1:]:
        if entry["decoder"].phase_names != phase_names:
            raise ValueError(
                f"run {entry['name']!r} has phases "
                f"{entry['decoder'].phase_names!r}, but the first run has "
                f"{phase_names!r}; cannot overlay runs with different "
                f"phase sets."
            )
    N = len(phase_names)

    # Split runs by mode to pick prior overlays.
    direct_runs = [e for e in loaded if not e["decoder"].is_pseudo_marginal]
    pm_runs     = [e for e in loaded if     e["decoder"].is_pseudo_marginal]
    has_uniform_prior   = len(direct_runs) > 0
    has_dirichlet_prior = len(pm_runs)     > 0

    if len(direct_runs) > 1:
        print(
            f"Note: {len(direct_runs)} direct-mode runs; using "
            f"{direct_runs[0]['name']!r}'s scale_bounds for the uniform "
            f"prior overlay (assumed shared)."
        )
    if len(pm_runs) > 1:
        print(
            f"Note: {len(pm_runs)} pseudo-marginal runs; using "
            f"{pm_runs[0]['name']!r}'s Dirichlet prior parameters for the "
            f"predictive prior overlay (assumed shared)."
        )

    # Uniform-prior box per phase, from the first direct run.
    if has_uniform_prior:
        uniform_scale_bounds = {
            entry["name"]: tuple(entry["scale_bounds"])
            for entry in direct_runs[0]["config"]["phases"]
        }
    else:
        uniform_scale_bounds = None

    # Dirichlet predictive prior samples (Monte Carlo), from the first PM run.
    if has_dirichlet_prior:
        first_pm_cfg = pm_runs[0]["config"]
        pm_block = first_pm_cfg["pseudo_marginal"]
        a_mu = np.asarray(pm_block["mu_dirichlet_prior"], dtype=float)
        log_alpha0_prior = _build_log_alpha0_prior(pm_block)
        S_prior = _sample_phase_fraction_prior(
            a_mu, log_alpha0_prior, N_PRIOR_DRAWS, seed=PRIOR_SEED,
        )
    else:
        S_prior = None

    # ---- xlim resolution -------------------------------------------------
    if isinstance(XLIM_OVERRIDE, dict):
        user_xlim: dict[str, tuple[float, float]] = {
            ph: tuple(v) for ph, v in XLIM_OVERRIDE.items()
        }
    elif XLIM_OVERRIDE is not None:
        user_xlim = {ph: tuple(XLIM_OVERRIDE) for ph in phase_names}
    else:
        user_xlim = {}

    def xlim_for(j: int, ph: str) -> tuple[float, float]:
        if ph in user_xlim:
            return user_xlim[ph]
        if uniform_scale_bounds is not None and ph in uniform_scale_bounds:
            return uniform_scale_bounds[ph]
        # Fallback: union of all posteriors with 5% padding.
        all_data = np.concatenate([
            e["decoded"].phase_fractions[:, j] for e in loaded
        ])
        lo, hi = float(all_data.min()), float(all_data.max())
        pad = 0.05 * max(hi - lo, 1e-6)
        return (max(0.0, lo - pad), hi + pad)

    # ---- Figure ----------------------------------------------------------
    nrows, ncols = _grid_shape(N)
    figsize = (8, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    run_colors = sns.color_palette("husl", len(loaded))

    for j, ph in enumerate(phase_names):
        ax = axes[j]
        xlim = xlim_for(j, ph)

        # Posteriors: one KDE line per run.
        for entry, color in zip(loaded, run_colors):
            sns.kdeplot(
                entry["decoded"].phase_fractions[:, j],
                color=color, linewidth=2, ax=ax, clip=xlim,
            )

        # Uniform prior: flat dotted line over [lo, hi].
        if has_uniform_prior and ph in uniform_scale_bounds:
            lo, hi = uniform_scale_bounds[ph]
            if hi > lo:
                ax.hlines(
                    1.0 / (hi - lo), lo, hi,
                    colors="red", linestyles="dotted", linewidth=2.5,
                )

        # Dirichlet predictive prior: KDE of MC samples.
        if has_dirichlet_prior:
            sns.kdeplot(
                S_prior[:, j],
                color="red", linestyle="dashed", linewidth=2,
                ax=ax, clip=xlim,
            )

        # True value (optional): black dashed vertical line.
        if ph in PHASE_FRACTION_TRUE:
            ax.axvline(
                PHASE_FRACTION_TRUE[ph],
                color="black", linestyle="dashed", linewidth=2,
            )

        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))
        ax.set_xlim(*xlim)
        ax.set_xlabel(PHASE_LABELS.get(ph, ph), labelpad=3)

    for k in range(N, len(axes)):
        axes[k].axis("off")

    # One shared y-label for the figure (sharey hides the redundant ticks).
    #fig.supylabel("Density", fontsize=16)

    # ---- Legend ----------------------------------------------------------
    handles: list[Line2D] = []
    labels:  list[str]    = []
    for entry, color in zip(loaded, run_colors):
        handles.append(Line2D([0], [0], color=color, linewidth=2))
        labels.append(entry["label"])
    if has_uniform_prior:
        handles.append(Line2D(
            [0], [0], color="red", linestyle="dotted", linewidth=2,
        ))
        labels.append("Uniform prior")
    if has_dirichlet_prior:
        handles.append(Line2D(
            [0], [0], color="red", linestyle="dashed", linewidth=2,
        ))
        labels.append("Dirichlet prior")
    if PHASE_FRACTION_TRUE:
        handles.append(Line2D(
            [0], [0], color="black", linestyle="dashed", linewidth=2,
        ))
        labels.append("Reference")

    ncol = min(len(handles), 4)
    fig.legend(
        handles=handles, labels=labels,
        loc="lower center", ncol=ncol, fontsize=10,
        frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, -0.3),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
