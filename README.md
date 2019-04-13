# Overview
This repository contains the code for the analyses whose results are shown in Figure 9 panels A, B, D, F and G of Frechter et al. 2019.

# Requirements
The main analyses are performed in Python 3.6.8. The required packages are listed in `requirements.txt` and can be installed using `pip install -r requirements.txt`. The data required for the analyses is provided in the `input_data.tar.gz` compressed tar file, but can also be prepared from scratch using the instructions below. Doing so requires the [gphys](https://github.com/jefferis/gphys), [physplitdata](https://github.com/jefferislab/physplitdata) and [physplit.analysis](https://github.com/sfrechter/physplit.analysis) packages. The code has been developed and tested on macOS High Sierra and CentOS 7, and should run with little modification on similar *nix systems.

# Usage
## Input Data
The data required for the analyses can be decompressed from `input_data.tar.gz` or built from scratch. Decompressing the file in the top level repository folder will yield an `input_data` folder containing the required files. Alternatively, it can be built from scratch by:
1. Creating a folder `input_data` at the top level of the repository.
2. Installing the R packages [gphys](https://github.com/jefferis/gphys), [physplitdata](https://github.com/jefferislab/physplitdata) and [physplit.analysis](https://github.com/sfrechter/physplit.analysis) packages.
3. Running `prepare_data.R` from the top level folder.
4. Running `prepare_data.py` from the top level folder.
## AUC Analysis
The AUC analysis whose results are shown in Figures 9B and C can be performed by running `compute_and_plot_auc.py`. This will generate the box and whisker plot of Figure 9B, and a csv file of AUC values that can be used to plot Figure 9C.
## PCA Analysis
The PCA plots on the raw data shown in Figure 9A and for the divisively normalized data shown in Figure 9D are performed in `compute_and_plot_pca.py`.
## Identity and Category Decoding
The identity and category decoding analysis whose results are shown in Figure 9F and 9G can be performed as follows:
1. The `runs` folder at the top level of the repository will contain the results. It is initialized with a `config.json` describing the parameters of the runs to be generated.
2. Execute `python ../gen_run_scripts.py` from inside the `runs` folder. This will generate a number of shell scripts of the form `job*.sh`. Each of these scripts is suitable for submitting to the SLURM cluster manager and contains instructions for running a single instance of the analysis.
3. The analysis itself is run by calling `single.py`.
4. The results are processed by `process_results.py`.

# Questions and Comments
All questions and comments regarding the analysis should be directed to sina.tootoonian@gmail.com









