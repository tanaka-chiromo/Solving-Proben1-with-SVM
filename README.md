Solving The Proben1 Problems With SVM (RBF kernel)
==================================================

Proben1 is a collection of datasets compiled for neural network classification algorithms. It contains 15 data sets from 
12 different domains. 

This python implementation evaluates the Proben1 datasets using Support
Vector Machines instead of neural networks. The LIBSVM library for python was used for the implementation.

Instructions
============

Place the Proben1 problem file that is to be evaluated into the directory 
labellded \data_files. Only the file types ".dt" are compatible with the 
implementation. The implementation will iterate through all the compatible 
file types in this directory.

Some files from the Proben1 problems are already included, the full datasets can
be retrieved from https://github.com/jeffheaton/proben1

Running The File
================

Run the script... evaluate_proben1_svm.py

Output
======

The evaluated test accuracy for each Proben1 dataset.
