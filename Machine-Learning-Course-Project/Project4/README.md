# Project 4: Neural Networks

## Introduction
This project is dedicated to implementing a Neural Network to predict housing prices using the Boston data set. The goal is to understand the fundamentals of neural networks, including data preprocessing, forward and backward passes, and applying gradient descent for optimization.

## Key Components
- `preprocess.py`: Standardize the training data to have zero mean and a standard deviation of 1. The preprocessing step is vital for consistent network performance.
- `grdescent.py`: Utilize the gradient descent algorithm to minimize the loss function during the training of the neural network.
- `forward_pass.py`: Execute a forward pass through the neural network, calculating the output for the given input data.
- `compute_loss.py`: Compute the loss of the network's predictions compared to the actual values, an essential step for backpropagation.
- `backprop.py`: Perform the backpropagation algorithm, computing the gradient of the network's weights with respect to the loss.
