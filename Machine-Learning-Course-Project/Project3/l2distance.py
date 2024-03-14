import numpy as np

"""
function D=l2distance(X,Z)
	
Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""

def l2distance(X,Z):
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
    
    D = np.zeros((n, m))
    
    # YOUR CODE HERE
    # Compute the squared norms for each data point
    X2 = np.sum(X ** 2, axis=0).reshape((n, 1))
    Z2 = np.sum(Z ** 2, axis=0).reshape((1, m))

    # Compute the pairwise squared distances
    D2 = X2 + Z2 - 2 * X.T.dot(Z)

    # Compute the Euclidean distances
    D = np.sqrt(D2)
    
    return D
