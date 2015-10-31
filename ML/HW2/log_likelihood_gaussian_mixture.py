""" converted from Matlab code
source: http://www.robots.ox.ac.uk/~fwood/teaching/AIMS_CDT_ML_2015/homework/HW_2_em/
"""

import scipy.stats
import numpy as np

def log_likelihood_gaussian_mixture(data, mu, sigma, pi):
    """% Calculates the log likelihood of the data given the parameters of the
    % model
    %
    % @param data   : each row is a d dimensional data point
    % @param mu     : a d x k dimensional matrix with columns as the means of
    % each cluster
    % @param sigma  : a cell array of the cluster covariance matrices
    % @param pi     : a column matrix of probabilities for each cluster
    %
    % @return ll    : the log likelihood of the data (scalar)"""
    
    ll = 0.
    k = len(pi)
    n = len(data)
    
    for nn in range(n):
        likelihood_n = 0.
        for kk in range(k):
            likelihood_n += pi[kk] * scipy.stats.multivariate_normal.pdf(data[nn],
                                                                         mean=mu[kk],
                                                                         cov=sigma[kk])
        ll += np.log(likelihood_n)
    
    return ll
