# MBSSFCC
# Detecting the locus of auditory attention based on the spectro-spatial-temporal analysis of EEG

This repository contains the python scripts developed as a part of the work presented in the paper "Detecting the locus of auditory attention based on the spectro-spatial-temporal analysis of EEG"

## Getting Started

### Dataset

The public [KUL dataset](https://zenodo.org/record/3997352#.YUGaZdP7R6q) [DTU dataset](https://zenodo.org/record/1199011#.Yx6eHKRBxPa) 
[PKU dataset](https://disk.pku.edu.cn/#/link/73833D62682190304756CE3AFAB71C88)are used in the paper. The dataset themself come with matlab processing program, please adjust them according to your own needs.

### Prerequisites

- python 3.7.9
- tensorflow-gpu 2.2.0
- keras 2.4.3

### Run the Code

1. Download the preprocessed data from [here](https://mailscuteducn-my.sharepoint.com/:f:/g/personal/202021058399_mail_scut_edu_cn/Evu3JoynOJxJlYtpKft2UfIBcZuNbkSrbymvDHLNdpiK9w?e=gWx9J0).

2. Modify the `args.data_document_path` variable in model.py to point to the downloaded data folder

3. Run the model:

   ```powershell
   python model.py
   ```

4. If you want to run multiple subjects in parallel, you can modify the variable `path` in multi_processing.py and run:

   ```powershell
   python multi_processing.py
   ```

