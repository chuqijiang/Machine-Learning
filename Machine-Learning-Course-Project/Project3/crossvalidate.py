"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""

import numpy as np
import math
from trainsvm import trainsvm

def crossvalidate(xTr, yTr, ktype, Cs, paras):
    bestC, bestP, lowest_error = 0, 0, 0
    errors = np.zeros((len(paras),len(Cs)))
    
    # YOUR CODE HERE
    n = xTr.shape[1]
    cutoff = int(n * 0.8)  # using 80% of data for training and 20% for validation
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:cutoff], indices[cutoff:]

    xTrain, yTrain = xTr[:, train_idx], yTr[:, train_idx]
    xVal, yVal = xTr[:, val_idx], yTr[:, val_idx]

    bestC, bestP, lowest_error = 0, 0, float('inf')
    errors = np.zeros((len(Cs), len(paras)))

    for i, C in enumerate(Cs):
        for j, par in enumerate(paras):
            # Train the SVM
            alphas, bias = trainsvm(xTrain, yTrain, C, ktype, par)

            # Define the classifier
            svmclassify = createsvmclassifier(xTrain, yTrain, alphas, bias, ktype, par)

            # Validate
            preds = svmclassify(xVal)
            error = np.mean(preds != yVal.reshape(-1, 1))

            errors[i, j] = error

            # Update the best parameters if this is the lowest error we've seen
            if error < lowest_error:
                lowest_error = error
                bestC, bestP = C, par

    
    return bestC, bestP, lowest_error, errors


    