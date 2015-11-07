""" converted from Matlab code
source: http://www.robots.ox.ac.uk/~fwood/teaching/AIMS_CDT_ML_2015/homework/HW_2_em/
"""

import numpy as np

def e_step_linear_regression(X, Y, alpha, beta):
    """% E-step of EM algorithm
    %
    % @param X      : design matrix for regression (n x d, includes intercept)
    % @param Y      : target vector
    % @param alpha  : weight precision = 1/(weight variance)
    % @param beta   : noise precision = 1 / (noise variance)
    %
    % @return m     : posterior mean of weight vector
    % @return s     : posterior covariance matrix of weight vector"""

    S_inv = alpha + beta * np.dot(X.T, X) 
    S = np.linalg.inv(S_inv)
    
    XTxY = np.dot(X.T, Y)
    m = beta * np.dot(S, XTxY)
    
    return [m, S]

