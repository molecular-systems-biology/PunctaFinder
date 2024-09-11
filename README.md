# Introduction
PunctaFinder is a (collection of) Python function(s) for the detection of puncta, i.e. small bright spots, in fluorescence microscopy images of cells. Besides detecting puncta, the algorithm can also estimate their size; it can be applied to both single images and z-stacks.

# Dependencies
PunctaFinder was developed in Python version 3.9 using the following modules:
- pandas 1.4.4
- numpy 1.24.4
- skimage (scikit-image) 0.22.0
- matplotlib 3.5.2
- math
- random
- copy

# Available files
- `demo_data.tif`: example data used in the demo notebook. bright-field image, GFP fluorescence image and cell masks of 6 cells that express Pln1-mNeonGreen (a marker protein for lipid droplets).
- `PunctaFinder_Demo.ipynb`: tutorial notebook for first-time PunctaFinder users. A stepwise guide that provides a practical demonstration of: (1) creation of a validated dataset; (2) determination of optimal threshold values; (3) running PunctaFinder with the obtained thresholds.
- `PunctaFinde_Code_Dataset_Thresholds.ipynb`: this file contains code for: (1) creating a validated dataset; (2) obtaining optimal punctum detection thresholds with a bootstrapping approach; (3) visualisation of the optimisation outcomes.
- `PunctaFinder_Functions.ipynb`: all PunctaFinder functions as described in Table S1 of the manuscript PunctaFinder: 'An algorithm for automated spot detection in fluorescence microscopy images'. An abridged version of this table is included below.

- 
