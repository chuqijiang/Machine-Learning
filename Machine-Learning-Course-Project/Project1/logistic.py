import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic(w,xTr,yTr):

    # YOUR CODE HERE

    z = np.dot(w.T, xTr)
    y_pred = sigmoid(yTr * z)
    loss = -np.sum(np.log(y_pred))

    gradient = -np.dot(yTr * xTr, 1 / (1 + np.exp(yTr * z)).T)

    return loss,gradient
