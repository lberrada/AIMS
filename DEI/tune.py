""" Description here

Author: Leonard Berrada
Date: 21 Oct 2015
"""

import numpy as np
import scipy.stats
import scipy.optimize
import csv
from build import mu_K
from params import get_params

def optimize_hyperparameters(Xtraining=None,
                             Ytraining=None,
                             use_kernels="gaussian",
                             estimator="MAP",
                             variable=None,
                             use_means="constant"):
    
    print("optimizing hyper-parameters...")
    
    my_params = get_params(use_kernels, use_means)
    
    init_params = my_params["init"]
    mean_params = my_params["means"]
    mean_stds = my_params["stds"]
    bounds = [(1e-6, None)] * len(my_params["init"])
    
    def get_neg_log_likelihood(params,
                               *args,
                               **kwargs):
        
        mu, K = mu_K(use_means=use_means,
                     use_kernels=use_kernels,
                     X1=Xtraining[None, :],
                     X2=Xtraining[:, None],
                     Xtesting=Xtraining,
                     params=params)
        
        Ycentered = Ytraining - mu
        
        L = np.linalg.cholesky(K)
        
        aux_u = np.linalg.solve(L, Ycentered)
        u = np.linalg.solve(L.T, aux_u)
        log_det_K = 2 * np.trace(np.log(L))
        
        log_likelihood = -0.5 * np.dot(Ycentered.T, u) - 0.5 * log_det_K
        
        
        neg_log_likelihood = -log_likelihood
        
        return neg_log_likelihood
    
    
    def get_neg_log_posterior(params,
                              *args,
                              **kwargs):
        
        log_likelihood = -np.array(get_neg_log_likelihood(params))
        log_prior = np.sum(scipy.stats.norm.logpdf(np.log(params),
                                                   loc=np.log(mean_params),
                                                   scale=mean_stds)) - np.sum(np.log(params))
        
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
                                         init_params,
                                         approx_grad=True,
                                         bounds=bounds)
    
    params_found = theta[0]
    
    print('done')
    print('Parameters found:')
    for k in range(len(my_params["names"])):
        print(my_params["names"][k] + " : " + str(params_found[k]))
    print('-' * 50)
    
    filename = "./out/" + use_kernels + "-" + use_means + "-" + estimator + "-" + variable + ".csv"
            
    with open(filename, 'w', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter='\t',
                               quoting=csv.QUOTE_MINIMAL)
        my_writer.writerow(list(np.round(params_found, 3)))
    
    return params_found.tolist()
