# config.py
#
# Single source of truth for the phase-fraction / mustrain / hstrain
# inference pipeline. Read (and copied to the output directory) by:
#     fit_gp.py        - fits and verifies the GP background
#     run_mcmc_v3.py   - imports the completed GP fit and runs MCMC
#
# Edit this file for a new dataset. The driver scripts should not require
# modification under normal use.
#
# Both scripts accept --config to point at an alternate copy of this file;
# the default is the config.py sitting next to the scripts.

from pathlib import Path

# How the observed pattern is loaded: "fxye" or "numpy".
data_type = "numpy"
name = "in-situ"

CONFIG = {
    "run_name": name,
    # "direct"          - original parameterization: phase fractions are
    #                     walker coordinates via softmax-with-reference-anchor.
    # "pseudo_marginal" - phase fractions are marginalized out via an inner
    #                     Monte Carlo over Dir(alpha_0 * mu); walker carries
    #                     (log_alpha_0, mu, mustrain, hstrain).
    "mode": "direct",
    "reference_phase_index": 0,
    "simplex_sum": 1.0,
    "dataset": {
        "label": "ID35",
        "x_data": "data_dir/prescribed_examples/run_20260526_133941_n3/"+name+"_x_spacing.npy",
        "y_data": "data_dir/prescribed_examples/run_20260526_133941_n3/"+name+"_pattern.npy",
        "fxye_path": "data_dir/In-situ.fxye",
        # Optional: restrict the fit to a subset of 2theta indices. Each
        # entry is [start, stop_exclusive] into the full pattern of length
        # surrogate.n_2theta. Multiple ranges are supported (e.g. to mask
        # a bad region in the middle: [[0, 200], [240, 584]]).
        # Set to None or omit to use the full pattern.
        "data_ranges": [[0, 624]],
        "Xray_wavelength": 0.172973
    },
    "surrogate": {
        # Directory containing <phase>/final_model.pkl and
        # <phase>/variance.npz per phase.
        "run_dir": str(
            Path.cwd().parent /
            "training_data" / "syncreton_training" / "surrogates_final" /
            "run_20260519_174722_seed700_main10000_2D_beta" #insert your run of the surrogate
        ),
        # Directory containing <phase>_pca_full.pkl per phase.
        "pca_dir": str(
            Path.cwd().parent /
            "training_data" / "syncreton_training" / "pca_transforms" /
            "run_20260519_174722_seed700_main10000_2D_beta" #insert your run of the surrogate
        ),
    },
    "phases": [
        {
            "name": "gamma",
            "scale_bounds":    (0.98,   1.0),
            "mustrain_bounds": (1000,  21000),
            "hstrain_bounds":  (-0.0075, 0.0075),
        },
        {
            "name": "delta",
            "scale_bounds":    (1e-6,  0.01),
            "mustrain_bounds": (1000,  21000),
            "hstrain_bounds":  (-0.005, 0.005),
        },
        {
            "name": "gamma1",
            "scale_bounds":    (1e-6,  0.01),
            "mustrain_bounds": (1000,  21000),
            "hstrain_bounds":  (-0.005, 0.005),
        },
        {
            "name": "gamma2",
            "scale_bounds":    (1e-6,  0.01),
            "mustrain_bounds": (1000,  21000),
            "hstrain_bounds":  (-0.005, 0.005),
        },
        {
            "name": "laves",
            "scale_bounds":    (1e-6,  0.01),
            "mustrain_bounds": (11000,  41000),
            "hstrain_bounds":  (-0.0075, 0.0075),
        },
        {
            "name": "carbide",
            "scale_bounds":    (1e-6,  0.01),
            "mustrain_bounds": (11000,  41000),
            "hstrain_bounds":  (-0.0075, 0.0075),
        },
    ],
    "mcmc": {
        "nwalkers_per_dim": 8,
        "nsteps": 50000,
        "moves": [
            ("StretchMove",   {"a": 1.2}, 0.2),
            ("DEMove",        {},         0.6),
            ("DESnookerMove", {},         0.2),
        ],
        # "fresh"    - reset backend if it exists, start a new chain.
        # "continue" - resume from last sample if backend exists.
        # "perturb"  - take the best walkers from the previous run,
        #              perturb them, write to a new HDF5 file.
        "restart_mode": "fresh",
        "perturb_n_best": 10,
        "perturb_scale": 0.15,
        "perturb_min_frac": 0.01,
    },
    "gp": {
        "n_iter": 4,
        "peak_prominence": 2e6,
        "rbf_length_scale": 900.0,
        "rbf_length_bounds": (64, 1e5),
    },
    "synthetic": { #Allows the synthetic background of the examples to be plotted for visual clarity besides the gp fit.
        "enabled": True,                 # set False (or omit the block) for real data
        "cheb_coeffs": [50000,-30000,20000], # GSAS-II order, first-kind
    },
    "likelihood": {
        "epsilon": 100.0,  # variance floor
        "noise_boost": {
            # "fixed" preserves the current likelihood dimensionality.
            # "mcmc" adds eta as one additional walker coordinate.
            "mode": "fixed",
            # Variance multiplier is exp(eta). eta=0.0 gives no boost.
            "eta": 0.0,
            # Used only when mode == "mcmc". Uniform eta over this interval
            # is the bounded Jeffreys prior for exp(eta).
            "eta_bounds": (0.0, 10.0),
            "prior": "uniform_eta",
        },
    },
    "pseudo_marginal": {
        # Inner Monte Carlo samples per log-prob call.
        # If None, defaults to 20 * N_phases at runtime.
        "M": None,

        # Prior family for log(alpha_0). Either:
        #   "uniform"   - flat prior on a hard box. Uses log_alpha0_bounds;
        #                 log_alpha0_lognormal_params is ignored.
        #   "lognormal" - alpha_0 ~ LogNormal(mu, sigma), i.e.
        #                 log(alpha_0) ~ Normal(mu, sigma). Uses
        #                 log_alpha0_lognormal_params; log_alpha0_bounds is
        #                 ignored. Support is all of R (no hard truncation);
        #                 the Normal density penalizes extreme values.
        "log_alpha0_prior": "lognormal",

        # Hard bounds on log(alpha_0) for the uniform prior. alpha_0
        # controls Dirichlet concentration: large -> mu strongly enforced;
        # small -> S can depart from mu freely. Used ONLY when
        # log_alpha0_prior == "uniform".
        "log_alpha0_bounds": (2.0, 10.0),

        # Normal(mu, sigma) parameters for the lognormal prior on alpha_0,
        # i.e. log(alpha_0) ~ Normal(mu, sigma). Used ONLY when
        # log_alpha0_prior == "lognormal". With mu=9, sigma=2 the prior
        # centers alpha_0 around exp(9) ~ 8100 and places ~95% of mass
        # between exp(5) ~ 150 and exp(13) ~ 4.4e5.
        "log_alpha0_lognormal_params": {"mu": 12.0, "sigma": 4.0},

        # Asymmetric Dirichlet concentration vector for the prior on mu.
        # Length must equal len(phases) and entries must be > 0. Ratios
        # encode the prior mean composition (E[mu_i] = a_i / sum_j a_j);
        # the magnitude sum_j a_j sets the prior sharpness. Use all-ones
        # for a flat (uniform) prior on the simplex.
        "mu_dirichlet_prior": [80.0, 1.0, 1.0],

        # Floor on alpha_i = alpha_0 * mu_i to keep Gamma draws stable.
        "alpha_floor": 1e-6,
    },
}
