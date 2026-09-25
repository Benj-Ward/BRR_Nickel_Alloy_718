driver_lhs_v1 has to be run first. Critically, you must install GSAS-II and provide the path to that install location in the file.
It will launch parallel subprocesses running GSAS-II to collect data for the training of the surrogate model.
These GSAS-II workers are RAM heavy (~2 GB each). The current setting of 6 cores may use more ram than available on some computers. 

The training data collects quickly with multiple cores (O(1 hour)). You can reduce the amount of data (e.g. 10000 -> 10) to test the piping.

Next, prep_and_pca_fixed_pc_stats has to be run. It divides up the data into training, testing, and validation. It performs PCA.

Afterwards, train_surrogate_v4 can be run. It will produce very large output files. The surrogates of all 6 phases take up combined ~20 gB.