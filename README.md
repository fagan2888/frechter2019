# Overview
This repository contains the code for the analyses whose results are shown in Figure 9 panels A, B, D, F and G of Frechter et al. 2019.

# Requirements
The main analyses are performed in Python 3.6.8. The required packages are listed in `requirements.txt` and can be installed using 
```
pip install -r requirements.txt
``` 
The data required for the analyses is provided in the `input_data.tar.gz` compressed tar file, but can also be prepared from scratch using the instructions below. Doing so requires the [gphys](https://github.com/jefferis/gphys), [physplitdata](https://github.com/jefferislab/physplitdata) and [physplit.analysis](https://github.com/sfrechter/physplit.analysis) packages. The code has been developed and tested on macOS High Sierra and CentOS 7, and should run with little modification on similar *nix systems. Finally, [gnuplot](http://www.gnuplot.info/) is used to make some diagnostic plots for the population decoding analyses.

# Usage
## Input Data
The data required for the analyses can be decompressed from `input_data.tar.gz` or built from scratch. Decompressing the file in the top level repository folder will yield an `input_data` folder containing the required files. Alternatively, it can be built from scratch by:
1. Creating a folder `input_data` at the top level of the repository.
2. Installing the R packages [gphys](https://github.com/jefferis/gphys), [physplitdata](https://github.com/jefferislab/physplitdata) and [physplit.analysis](https://github.com/sfrechter/physplit.analysis) packages.
3. Running `prepare_data.R` from the top level folder.
4. Running `prepare_data.py` from the top level folder.
## AUC Analysis
The AUC analysis whose results are shown in **Figures 9B and C** can be performed by running `compute_and_plot_auc.py`. This will generate the box and whisker plot of Figure 9B, and a csv file of AUC values that can be used to plot Figure 9C. The plot and csv file are written to the `output` folder.
## PCA Analysis
The PCA plots on the raw data shown in **Figure 9A** and for the divisively normalized data shown in **Figure 9D** are performed in `compute_and_plot_pca.py`. The plot is written to the `output` folder.
## Identity and Category Decoding
Identity and category decoding is run as a parameter sweep over populations, number of classes, shuffles, etc. The sweep can be performed as follows:

1) A folder is created at the top level to contain the results of the runs. The name can be arbitrary; it's set to `runs` here.  

2) The folder is initialized with a `config.json` describing the parameters of the runs to be generated. 

3) The script `gen_run_scripts.py` is executed from inside the `runs` folder using 
```
python ../gen_run_scripts.py
```

This will generate shell scripts of the form `job*.sh`, to be submitted to the SLURM job scheduler. Each script contains instructions for running a single instance of the analysis by calling the `single.py` script with the parameters of the run. In the absence of SLURM, the python call to `single.py` inside each job script can also be run manually. For example:
```
python -u ../single.py 0 PN 1 identity 0 1 0
```
See `single.py` for the meaining of the input paramters. Note that `single.py` uses gnuplot to generate a summary plot after the run is complete, so gnuplot should be available on the system.

4) Each job writes a number of files to the `runs` directory. These files are prefixed as e.g. `seed12orig`, indicating that the job used random seed 12 and used the original (unshuffled) data. The files are as follows:
   - `seed(...).csv`: Each row contains the accuracy at each time bin for a particular value of the linear SVM parameter `C`. 
   - `seed(...).pdf`: The accuracy time course data in the csv file plotted using gnuplot.
   - `seed(...)_summary.txt`: A plain text file listing the parameters of the run as well as the peak accuracy achieved.   
   - `seed(...)_summary.p`: The summary data as in `...summary.txt` but stored as pickled python dictionary.
   - `seed(...)_acc.npy`: A `num_C_values x num_time_bins x num_trials` numpy array containing the decoding accuracy for each time bin, trial, and value of C.   

5) The results stored in the `runs` folder are analyzed by `process_results.py` by running 
```
python process_results.py runs
``` 

from the top level folder. 
   - This script will first assemble the pickled summary data generated by each run into a pandas dataframe. The dataframe will b e written to `runs/all_acc.df.p.bz2`. Subsequent calls to `process_results.py` will look to load the dataframe from this file in favour of recomputing it, as the computation takes a a few minutes.
   - Once the data a processed, a number of performance plots are generated and written to the `./output` folder. The most important is `id_vs_cat_w_shuff.pdf` which shows the identity and category decoding performance of each population as in **Figures 9F and 9G** in the paper. The remaining plots show the time course of accuracy for each population, as well as the best SVM parameter `C` the yielded the best performance at each time bin.
# Questions and Comments
All questions and comments regarding the analysis should be directed to sina.tootoonian@gmail.com









