# frechter2019
This repository contains the code for the analyses whose results are shown in Figure 9 panels A, B, D, F and G of Frechter et al. 2019.

## Requirements
The main analyses are performed in Python 3.6.8. The required packages are listed in `requirements.txt` and can be installed using `pip install -r requirements.txt`. The data required for the analyses is provided in the `input_data.tar.gz` compressed tar file, but can also be prepared from scratch using the instructions below. Doing so requires executing `prepare_data.R` and requires the [gphys](https://github.com/jefferis/gphys), [physplitdata](https://github.com/jefferislab/physplitdata) and [physplit.analysis](https://github.com/sfrechter/physplit.analysis) packages. The code has been developed and tested on macOS High Sierra and CentOS 7, and should run with little modification on similar *nix systems.

## Usage
### Input Data
The data required for the analyses can be decompressed from `input_data.tar.gz` or built from scratch. Decompressing the file in the top level repository folder will yield an `input_data` folder containing the required files. Alternatively, it can be built from scratch by:
1. Creating a folder `input_data` at the top level of the repository.
2. Installing the R packages [gphys](https://github.com/jefferis/gphys), [physplitdata](https://github.com/jefferislab/physplitdata) and [physplit.analysis](https://github.com/sfrechter/physplit.analysis) packages.
3. Running `prepare_data.R` from the top level folder.
4. Running `prepare_data.py` from the top level folder.







