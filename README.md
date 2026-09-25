# BRR_Nickel_Alloy_718
Codebase for the scientific paper "Bayesian Rietveld refinement to estimate phase fractions of additively manufactured Nickel Alloy 718"

--- Objective --- 
The objective of this code is to perform Bayesian Rietveld Refinement for the Nickel Alloy 718 system using a surrogate model. In the paper, synthetic and laboratory examples are analyzed. In this codebase, we provide only the synthetic data, as our laboratory data is proprietary to the Tao Sun Group at Northwestern University. However, the scripts for both kinds of data are essentially the same.

--- Codebase Outline ---
This project contains two substantial computational efforts:
1. "Surrogate_Modeling"
2. "Bayesian_Inference"

--- Surrogate_Modeling ---
The second effort depends on the first. We provide 3 scripts, 1 dependency, and a data directory to train a random forest surrogate model for X-ray diffraction of Nickel Alloy 718. 

The data directory in Surrogate_Modeling contains a parameter file containing info on the laboratory setup and a phase directory with CIF files, which encode our assumptions about the crystalographic features of Nickel Alloy 718.

Before running the scripts, install GSAS-II. Provide the GSAS-II python install location to driver_lhs_v1 as the variable WORKER_PYTHON.

Run the scripts in this order: driver_lhs_v1 -> prep_and_pca_fixed_pc_stats -> train_surrogate_v4

model_v1_old is the dependency that runs GSAS-II as a subprocess. driver_lhs_v1 launches 6 workers by default, which may be too many for your machine due to the high RAM use of GSAS-II.

--- Bayesian Inference ---
We provide 6 scripts, 2 dependencies, a config file, a data directory, and a sub-selection of output files (everything except the raw output of MCMC). 

The data directory contains the parameter file and CIF files in addition to a script (driver_three_examples) to create synthetic X-ray diffraction data using GSAS-II. We provide the output of this script in a sub-directory called prescribed_examples.

Before running the scripts, update the config_v6 file with the name of your data, and the location and name of your surrogate model. Before running the analysis of the results, update analyze_run_v6 with your GSAS-II install location.

Run the scripts in this order: fit_gp -> run_mcmc_v6 -> analyze_run_v6
Optionally, fit_gp_synthetic plots the background fit versus a ground truth and plot_phase_fractions_multirun plots phase scale factor posteriors of multiple runs on one figure.

The output files will include the MCMC chain as a h5 file, metadata, the background fit, the validation metrics, a copy of the config file, and figures with posteriors, validation figures, and diagnostic figures for MCMC.
