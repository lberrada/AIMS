""" converted from Matlab code
source: http://www.robots.ox.ac.uk/~fwood/teaching/AIMS_CDT_ML_2015/homework/HW_2_em/
"""

import scipy.stats
import numpy as np

def log_likelihood_linear_regression(X, Y, m, beta):
    """% This function evaluates an approximation of the joint log likelihood.
    % The estimate uses the posterior mean of the weights.  This quantity
    % should increase as the algorithm progresses."""
    
    lik = scipy.stats.norm.pdf(Y - np.dot(X, m), 0, np.sqrt(1. / beta));
    ll = np.sum(np.log(lik));
    
    return ll

