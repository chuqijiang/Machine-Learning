
import numpy as np
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14  #minimum step size for gradient descent

    # YOUR CODE HERE
    w = w0
    stepsize = 1e-1

    for i in range(maxiter):
        loss, gradient = func(w)
        w = w - stepsize * gradient

        if np.linalg.norm(gradient) < tolerance:
            break

        # 在每个迭代步骤中更新步长
        stepsize = stepsize * 0.9935
    return w
