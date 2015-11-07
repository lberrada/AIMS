""" converted from Matlab code
source: http://www.robots.ox.ac.uk/~fwood/teaching/AIMS_CDT_ML_2015/homework/HW_2_em/
"""

import numpy as np

def m_step_linear_regression(X, Y, m, s):
    """% M-step of EM algorithm
    %
    % @param X      : design matrix for regression (n x d, includes intercept)
    % @param Y      : target vector
    % @param m      : mean of weight vector
    % @param s      : covariance matrix of weight vector
    %
    % @return alpha : weight precision = 1/(weight variance)
    % @return beta  : noise precision = 1 / (noise variance)"""
    
    M = np.shape(X)[1]
    
    aux = float(np.dot(m.T, m) + np.trace(s))
    alpha = M / aux
    
    N = np.shape(X)[0]
    
    Xxm = np.dot(X, m)
    aux = float(np.sum(np.power(Y, 2) - 2.*Y * Xxm) + np.dot(Xxm.T, Xxm) + np.trace(np.dot(np.dot(X, s), X.T)))
    beta = N / aux
    
    return [alpha, beta]
