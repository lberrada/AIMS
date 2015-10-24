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
    
    init_params = np.array(my_params["init"])
    mean_params = my_params["means"]
    mean_stds = my_params["stds"]
    bounds = my_params["bounds"]
    use_log = my_params["use_log"]
    
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
        log_det_K = 2 * np.trace(np.log(L))
        
        aux = np.linalg.solve(L, Ycentered.T)
        YcenteredTxK_inv = np.linalg.solve(L.T, aux).T
        
        log_likelihood = -0.5 * np.dot(YcenteredTxK_inv, Ycentered) - 0.5 * log_det_K
        
        neg_log_likelihood = -log_likelihood
        
        return neg_log_likelihood
    
    
    def get_neg_log_posterior(params,
                              *args,
                              **kwargs):
        
        log_likelihood = -np.array(get_neg_log_likelihood(params))

        log_prior = 0
        for i in range(len(params)):
            if use_log[i]:
                x = np.log(params[i])
                mu = np.log(mean_params[i])
                log_prior -= x
            else:
                x = params[i]
                mu = mean_params[i]
            
            log_prior += scipy.stats.norm.logpdf(x,
                                               loc=mu,
                                               scale=mean_stds[i])
        
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
    if estimator == "MLE":
        score = get_neg_log_likelihood(params_found)
    else:
        score = get_neg_log_posterior(params_found)
    
    print('done')
    print('Parameters found:')
    for k in range(len(my_params["names"])):
        print(my_params["names"][k] + " : " + str(params_found[k]))
    print("Score found : " + str(score))
    print('-' * 50)
    
#     filename = "../out/" + use_kernels + "-" + use_means + "-" + estimator + "-" + variable + ".csv"
    filename = "../out/results_v3-3.csv"
            
#     with open(filename, 'w', newline='') as csvfile:
    with open(filename, 'a', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter='\t',
                               quoting=csv.QUOTE_MINIMAL)
        my_writer.writerow(['-' * 50])
        my_writer.writerow([use_kernels, use_means, variable, estimator])
        my_writer.writerow(list(my_params["names"]))
        my_writer.writerow(list(np.round(params_found, 4)))
        my_writer.writerow([estimator, score])
    
    return params_found.tolist()
