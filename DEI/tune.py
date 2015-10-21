""" Description here

Author: Leonard Berrada
Date: 21 Oct 2015
"""

import numpy as np
import scipy.stats
import scipy.optimize
from kernels import gaussian_kernel

def optimize_hyperparameters(X=None,
                             Y=None):
    
    print("optimizing hyper-parameters...")
    
    def get_neg_log_likelihood(params, *args):
        
        sigma_f, sigma_n, l = params
        
        K = gaussian_kernel(X1=X[None, :],
                            X2=X[:, None],
                            sigma_f=sigma_f,
                            sigma_n=sigma_n,
                            scale=l)
        
        L = np.linalg.cholesky(K)
        aux_u = np.linalg.solve(L, Y)
        u = np.linalg.solve(L.T, aux_u)
        log_det_K = 2 * np.trace(np.log(L))
        
        log_likelihood = -0.5 * np.dot(Y.T, u) - 0.5 * log_det_K
        
        
        neg_log_likelihood = -log_likelihood
        
        return neg_log_likelihood
    
    mean_params = np.array([5., 1., 25.])
    sigma_params = np.array([3., 1., 10.])
    
    def get_neg_log_posterior(params, *args):
        
        params = np.abs(params)
        
        log_likelihood = -np.array(get_neg_log_posterior(params))
        log_prior = np.sum(scipy.stats.norm.logpdf(params,
                                                loc=mean_params,
                                                scale=sigma_params)) - np.sum(np.log(np.abs(params)))
        
        log_posterior = log_likelihood + log_prior
        
        neg_log_posterior = -log_posterior
        
        return neg_log_posterior
        
        
        
    
    init_theta = np.array([10., 0.5, 25])
    theta = scipy.optimize.fmin_cg(get_neg_log_likelihood,
                                   init_theta)
    
    
    
    print('done')
    print('Parameters found:')
    print('sigma_f :', theta[0])
    print('sigma_n :', theta[1])
    print('l :', theta[2])
    print('-' * 50)
    
    return theta.tolist()
