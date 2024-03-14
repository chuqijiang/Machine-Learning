"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    bias = 0
    
    # YOUR CODE HERE
    sv_indices = np.where((alphas > 0) & (alphas < C))[0]

    biases = []
    for i in sv_indices:
        bias_i = yTr[i] - np.sum(alphas * yTr * K[i, :])
        biases.append(bias_i)

    bias = np.mean(biases) if biases else 0
    
    return bias 
    
