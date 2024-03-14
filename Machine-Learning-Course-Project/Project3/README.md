# Project 3: Kernel SVM

## Introduction
This project entails the implementation of a kernel SVM (Support Vector Machine) for classification tasks. By leveraging the mathematical foundations of SVM and kernel methods, this project explores the capabilities of SVMs in complex, non-linear classification scenarios.

## Key Components

### Scripts to Edit
- `l2distance.py`: Efficient computation of Euclidean distances between vectors.
- `computeK.py`: Kernel matrix computation based on input vectors and a chosen kernel type.
- `generateQP.py`: Preparation of quadratic programming problem variables for the SVM solver.
- `recoverBias.py`: Calculation of the bias term after solving the SVM optimization problem.
- `crossvalidate.py`: Optimization of kernel parameters and regularization constants via cross-validation.
- `createsvmclassifier.py`: Production of a classifier function from trained SVM parameters.
- `main.py`: Main testing script that consolidates the implementation and runs the SVM classifier.
