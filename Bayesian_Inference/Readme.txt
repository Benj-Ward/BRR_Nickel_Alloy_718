The MCMC is controlled through the config file. Many aspects of the data analysis and plots are not specified in the config and have variables in-line of their own files.

We provide in the data directory our synthetic examples' data files and the script used to make them. The other scripts hook up with the existing run.

To run these files, make sure to specify the location of your trained surrogate models and your installation of GSAS-II. The MCMC doesn't require GSAS-II (thanks surrogate model), but the validation of the results requires GSAS-II to be run for metrics and predicted diffractograms. There is a variable to use only the surrogate for the validation, but this is not as rigorous. 