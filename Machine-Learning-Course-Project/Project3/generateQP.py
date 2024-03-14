"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
C : regularization constant

Output:
Q,p,G,h,A,b as defined in qpsolvers.solve_qp

A call of qpsolvers.solve_qp(Q, p, G, h, A, b) should return the optimal nx1 vector of alphas
of the SVM specified by K, yTr, C. Just make these variables np arrays.

"""
import numpy as np

def generateQP(K, yTr, C):
    yTr = yTr.astype(np.double)
    n = yTr.shape[0]
    
    # YOUR CODE HERE
    Q = (yTr @ yTr.T) * K

    p = -np.ones(n)

    G_top = np.eye(n)
    G_bottom = -np.eye(n)
    G = np.vstack([G_top, G_bottom])

    h_top = C * np.ones(n)
    h_bottom = np.zeros(n)
    h = np.hstack([h_top, h_bottom])

    A = yTr.T
    b = np.array([0.0])

            
    return Q, p, G, h, A, b

