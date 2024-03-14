from numpy import maximum
import numpy as np


def hinge(w, xTr, yTr, lambdaa):
    #
    #
    # INPUT:
    # xTr dxn matrix (each column is an input vector)
    # yTr 1xn matrix (each entry is a label)
    # lambda: regularization constant
    # w weight vector (default w=0)
    #
    # OUTPUTS:
    #
    # loss = the total loss obtained with w on xTr and yTr
    # gradient = the gradient at w

    # YOUR CODE HERE
    d, n = xTr.shape
    margin = yTr * np.dot(w.T, xTr)
    loss = np.maximum(0, np.ones_like(margin) - margin)
    loss = np.sum(loss) + lambdaa * np.sum(w ** 2)

    gradient = np.zeros_like(w)
    mask = (margin < 1)
    gradient += -np.dot(yTr * xTr, mask.T)
    gradient += 2 * lambdaa * w

    return loss, gradient
