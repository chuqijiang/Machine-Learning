# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:30:29 2019

@author: remus
"""
import numpy as np
def forward_pass(W, xTr, trans_func):
#% function [as,zs]=forward_pass(W,xTr,trans_func)
#%
#% INPUT:
#% W weights (list of numpy array)
#% xTr dxn numpy array (each column is an input vector)
#% trans_func transition function to apply for inner layers
#%
#% OUTPUTS:
#%
#% as = result of forward pass 
#% zs = result of forward pass (zs[0] output layer of the forward pass) 
#%
    n = np.shape(xTr)[1]
    
    ## CHECK!  -JERRY
    
    # First, we add the constant weight
    zzs = [None]*(len(W)+1);   zzs[-1] = np.vstack((xTr, np.ones([1, n])))
    aas = [None]*(len(W)+1);   aas[-1] = xTr
    
    # Do the forward process here
    for i in range(len(W)-1, -1, -1):
        # INSERT CODE
        # zs_next = np.dot(W[i], zzs[i + 1])
        # zzs[i] = np.vstack((zs_next, np.ones([1, n])))
        # aas[i] = trans_func(zs_next)
        aas[i] = W[i] @ zzs[i + 1]

        # Apply transition function except for the last layer
        if i > 0:
            zzs[i] = trans_func(aas[i])
            zzs[i] = np.vstack((zzs[i], np.ones([1, n])))  # add bias for next layer
        else:
            zzs[i] = aas[i]
        
    # INSERT CODE: (last one is special, no transition function)
    # aas[0] = np.dot(W[0], zzs[1])
    
    return aas, zzs
