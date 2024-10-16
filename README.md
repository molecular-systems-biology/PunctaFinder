# PunctaFinder

### Introduction
PunctaFinder is a Python package for the detection of puncta, _i.e._ small bright spots, in fluorescence microscopy images of cells. Besides detecting puncta, the algorithm can also estimate their size. It can be applied to both single images and z-stacks.

### Dependencies
PunctaFinder was developed in Python version 3.9 using the following modules:

- pandas 1.4.4
- numpy 1.24.4
- skimage (scikit-image) 0.22.0
- matplotlib 3.5.2
- math
- random
- copy

Installation of PunctaFinder (detailed below) creates a virtual environment with these specific versions of the modules.

### Available files
This repository contains the following files:

- _demo_data.tif:_ example data that are used in the demo notebook. This image stack contains a bright-field image, a GFP fluorescence image and cell masks of six cells that express Pln1-mNeonGreen (a marker protein for lipid droplets).

- _PunctaFinder_Demo.ipynb:_ tutorial notebook for first-time PunctaFinder users. A stepwise guide that provides a short and practical demonstration of:
    1. creation of a validated dataset;
    2. determination of detection threshold values;
    3. running PunctaFinder with the obtained thresholds.
    

- _PF_Dataset_Creation_and_Threshold_Optimisation.ipynb:_ this file contains a detailed stepwise guide for:
    1. creating a validated dataset;
    2. obtaining optimal punctum detection thresholds **with** a bootstrapping approach;
    3. visualisation of the optimisation outcomes.

- _PunctaFinder.zip:_ the PunctaFinder Python package
 

### Installation
**Step 1.** First, clone the repository with _e.g._ Git Bash:
```sh
git clone https://github.com/molecular-systems-biology/PunctaFinder
```
**Step 2.** In the Conda command line, navigate to the folder in which you cloned the PunctaFinder repository. Use (for example):
```sh
cd C:\Users\name\PunctaFinder
```
**Step 3.** Create a virtual environment with all required modules and their correct versions:
```sh
conda env create -f environment.yml
```

**Step 4.** Activate this environment:
```sh
conda activate punctafinder_env
```

**Step 5.** Add the PunctaFinder environment to your Jupyter Notebook as a kernel:
```sh
python -m ipykernel install --user --name=punctafinder_env --display-name "PunctaFinder"
```

**Step 6.** Start Jupyter lab or Jupyter notebook with
```sh
jupyter-lab
```
or
```sh
jupyter notebook
```

**Step 7.** Create a new notebook and select the PunctaFinder kernel.
