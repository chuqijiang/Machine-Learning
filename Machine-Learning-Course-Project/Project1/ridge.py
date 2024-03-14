
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # YOUR CODE HERE

    loss = np.sum((np.dot(w.T, xTr) - yTr)**2) + lambdaa * np.sum(w**2)

    gradient = -2 * np.dot(xTr, (yTr-np.dot(w.T, xTr)).T) + 2 * lambdaa * w

    return loss,gradient
