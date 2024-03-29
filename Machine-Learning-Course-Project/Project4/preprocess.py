# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:05:03 2019

@author: Jerry Xing
"""
import numpy as np
def preprocess(xTr,xTe):
# function [xTr,xTe,u,m]=preprocess(xTr,xTe);
#
# Preproces the data to make the training features have zero-mean and
# standard-deviation 1
# input:
# xTr - raw training data as d by n_train numpy ndarray 
# xTe - raw test data as d by n_test numpy ndarray
    
# output:
# xTr - pre-processed training data 
# xTe - pre-processed testing data
#
# u,m - any other data should be pre-processed by x-> u*(x-m)
#       where u is d by d ndnumpy array and m is d by 1 numpy ndarray
    
    d, _ = np.shape(xTr)
    # m = np.zeros((d,1))
    # u = np.zeros((d,d))
    ## << Remove 2 lines above and insert your solution here

    ## >>


    m = np.mean(xTr, axis=1, keepdims=True)

    sigma = np.std(xTr, axis=1, keepdims=True)

    sigma[sigma == 0] = 1e-10

    u = np.diag(1.0 / sigma.flatten())

    xTr = np.dot(u.T, (xTr - m))
    xTe = np.dot(u.T, (xTe - m))
    return xTr, xTe, u, m