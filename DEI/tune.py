""" Description here

Author: Leonard Berrada
Date: 21 Oct 2015
"""

import numpy as np
import scipy.stats
import scipy.optimize
from kernels import gaussian_kernel, gaussian_kernel_2, locally_periodic_kernel

def optimize_hyperparameters(X=None,
                             Y=None,
                             use_kernel="gaussian"):
    
    print("optimizing hyper-parameters...")
    
    if use_kernel == "gaussian":
        kernel = gaussian_kernel
        mean_params = np.array([1., 1., 10.])
        sigma_params = np.array([10., 10., 10.])
        init_theta = np.array([10., 0.5, 25])
        
    elif use_kernel == "gaussian_2":
        kernel = gaussian_kernel_2
        mean_params = np.array([1., 1., 10., 0.1, 248.])
        sigma_params = np.array([10., 10., 10., 10., 10.])
        init_theta = np.array([10., 0.5, 25, 1., 250.])
        
    elif use_kernel == "locally_periodic":
        kernel = locally_periodic_kernel
        mean_params = np.array([1., 1., 10., 2, 120])
        sigma_params = np.array([10., 10., 10., 10., 10.])
        init_theta = np.array([10., 0.5, 25, 2., 100.])
        
    else:
        raise ValueError("%s kernel not implemented:" % use_kernel)
        
    bounds = [(1e-2, None)] * len(init_theta)
    
    def get_neg_log_likelihood(params,
                               *args,
                               **kwargs):
        
        K = kernel(X1=X[None, :],
                   X2=X[:, None],
                   params=params)
        
        L = np.linalg.cholesky(K)
        aux_u = np.linalg.solve(L, Y)
        u = np.linalg.solve(L.T, aux_u)
        log_det_K = 2 * np.trace(np.log(L))
        
        log_likelihood = -0.5 * np.dot(Y.T, u) - 0.5 * log_det_K
        
        
        neg_log_likelihood = -log_likelihood
        
        return neg_log_likelihood
    
    
    
    def get_neg_log_posterior(params,
                              *args,
                              **kwargs):
        
        log_likelihood = -np.array(get_neg_log_likelihood(params))
        log_prior = np.sum(scipy.stats.norm.logpdf(np.log(params),
                                                loc=np.log(mean_params),
                                                scale=sigma_params)) - np.sum(np.log(params))
        
        log_posterior = log_likelihood + log_prior
        
        neg_log_posterior = -log_posterior
        
        return neg_log_posterior
        
    
    

    theta = scipy.optimize.fmin_l_bfgs_b(get_neg_log_posterior,
                                         init_theta,
                                         approx_grad=True,
                                         bounds=bounds)
    
    params_found = theta[0]
    
    print('done')
    print('Parameters found:')
    print('sigma_f :', params_found[0])
    print('sigma_n :', params_found[1])
    print('scale :', params_found[2])
    if use_kernel == "gaussian_2":
        print('sigma_f_2 :', params_found[3])
        print('scale_2 :', params_found[4])
    elif use_kernel == "locally_periodic":
        print('p :', params_found[3])
        print('scale_2 :', params_found[4])
    print('-' * 50)
    
    return params_found.tolist()
