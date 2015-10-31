""" converted from Matlab code
source: http://www.robots.ox.ac.uk/~fwood/teaching/AIMS_CDT_ML_2015/homework/HW_2_em/
"""

import numpy as np
import scipy.stats

def e_step_gaussian_mixture(data, pi, mu, sigma):

    """ Returns a matrix of responsibilities.
    %
    % @param    data : data matrix n x d with rows as elements of data
    % @param    pi   : column vector of probabilities for each class
    % @param    mu   : d x k matrix of class centers listed as columns
    % @param    sigma: cell array of class covariance matrices (d x d)
    %
    % @return   gamma: n x k matrix of responsibilities
    """
    
    k = len(pi)
    n = len(data)
    
    gamma = np.zeros((n, k))
    
    for nn in range(n):
        for kk in range(k):
            gamma[nn, kk] = pi[kk] * scipy.stats.multivariate_normal.pdf(data[nn],
                                                                         mean=mu[kk],
                                                                         cov=sigma[kk])
        
    rows_sum = np.sum(gamma, axis=1)
    gamma /= rows_sum[:, None]
    
    return gamma
    
