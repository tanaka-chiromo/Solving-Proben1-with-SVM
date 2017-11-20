Solving The Proben1 Problems With SVM (RBF kernel)
==================================================

Proben1 is a collection of problems for neural network learning 
in the realm of pattern classification and function approximation 
plus a set of rules and conventions for carrying out benchmark tests 
with these or similar problems. Proben1 contains 15 data sets from 
12 different domains. 

This python implementation evaluates the Proben1 problems with Support
Vector Machines. The LIBSVM library for python was used for the implementation.

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

The evaluated test results for each Proben1 problem.
