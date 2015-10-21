""" Description here

Author: Leonard Berrada
Date: 21 Oct 2015
"""

import numpy as np
import scipy.stats
import scipy.optimize
import csv
from kernels import gaussian_kernel, gaussian_kernel_2, locally_periodic_kernel,\
    matern_kernel

def optimize_hyperparameters(Xtraining=None,
                             Ytraining=None,
                             use_kernel="gaussian",
                             estimator="MAP",
                             variable=None):
    
    print("optimizing hyper-parameters...")
    
    if use_kernel == "gaussian":
        kernel = gaussian_kernel
        if variable == "temperature":
            mean_params = np.array([1., 1., 10.])
            sigma_params = np.array([10., 10., 10.])
            init_theta = np.array([10., 0.5, 25])
        
        elif variable=="tide":
            mean_params = np.array([1., 1., 10.])
            sigma_params = np.array([10., 10., 10.])
            init_theta = np.array([10., 0.5, 25])
            
    elif use_kernel == "gaussian_2":
        kernel = gaussian_kernel_2
        if variable == "temperature":
            mean_params = np.array([1., 1., 10., 0.1, 248.])
            sigma_params = np.array([10., 10., 10., 10., 10.])
            init_theta = np.array([10., 0.5, 25, 1., 250.])
            
        elif variable=="tide":
            mean_params = np.array([1., 1., 10., 0.1, 100.])
            sigma_params = np.array([10., 10., 10., 10., 20.])
            init_theta = np.array([10., 0.5, 25, 1., 100.])
        
    elif use_kernel == "locally_periodic":
        kernel = locally_periodic_kernel
        if variable == "temperature":
            mean_params = np.array([1., 1., 10., 2, 120])
            sigma_params = np.array([10., 10., 10., 10., 10.])
            init_theta = np.array([10., 0.5, 25, 2., 100.])
            
        elif variable=="tide":
            mean_params = np.array([1., 1., 10., 2, 50])
            sigma_params = np.array([10., 10., 10., 10., 20.])
            init_theta = np.array([10., 0.5, 25, 2., 50.])
    
    elif use_kernel == "matern":
        kernel = matern_kernel
        if variable == "temperature":
            mean_params = np.array([1., 1., 10., 3])
            sigma_params = np.array([10., 10., 10., 1.])
            init_theta = np.array([10., 0.5, 25, 3.])
        
        elif variable=="tide":
            mean_params = np.array([1., 1., 10., 3.])
            sigma_params = np.array([10., 10., 10., 1.])
            init_theta = np.array([10., 0.5, 25, 3.])
    
    else:
        raise ValueError("%s kernel not implemented:" % use_kernel)
        
    bounds = [(1e-2, None)] * len(init_theta)
    
    def get_neg_log_likelihood(params,
                               *args,
                               **kwargs):
        
        K = kernel(X1=Xtraining[None, :],
                   X2=Xtraining[:, None],
                   params=params)
        
        L = np.linalg.cholesky(K)
        aux_u = np.linalg.solve(L, Ytraining)
        u = np.linalg.solve(L.T, aux_u)
        log_det_K = 2 * np.trace(np.log(L))
        
        log_likelihood = -0.5 * np.dot(Ytraining.T, u) - 0.5 * log_det_K
        
        
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
        
    if estimator.lower() == "mle":
        fitness_function = get_neg_log_likelihood
        
    elif estimator.lower() == "map":
        fitness_function = get_neg_log_posterior
    
    else:
        raise ValueError("%s estimator is not implemented: should be 'MLE' or 'MAP'" % estimator)

    theta = scipy.optimize.fmin_l_bfgs_b(fitness_function,
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
    
    filename = use_kernel + "-" + estimator + "-" + variable + ".csv"
            
    with open(filename, 'w', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter='\t',
                               quoting=csv.QUOTE_MINIMAL)
        my_writer.writerow(np.round(params_found, 3).tolist())
    
    return params_found.tolist()
