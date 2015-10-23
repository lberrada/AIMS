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
from GP_model import predict

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
    
    n_validation = int(0.3 * len(Xtraining))
    validation_indices = np.random.choice(range(len(Xtraining)),
                                          size=n_validation,
                                          replace=False)
    training_indices = [i for i in range(len(Xtraining)) if i not in validation_indices]
    
    XXtraining = Xtraining[training_indices]
    XXvalidation = Xtraining[validation_indices]
    YYtraining = Ytraining[training_indices]
    YYvalidation = Ytraining[validation_indices]
    
    def get_neg_log_likelihood(params,
                               *args,
                               **kwargs):
        
        mu, K = mu_K(use_means=use_means,
                     use_kernels=use_kernels,
                     X1=XXtraining[None, :],
                     X2=XXtraining[:, None],
                     Xtesting=XXtraining,
                     params=params)
        
        Ycentered = YYtraining - mu

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
    
    r2 = predict(Xtraining=XXtraining,
                 Ytraining=YYtraining,
                 Xtesting=XXvalidation,
                 Ytestingtruth=YYvalidation,
                 params=params_found,
                 use_kernels=use_kernels,
                 use_means=use_means,
                 sequential_mode=False,
                 variable=variable,
                 estimator=estimator,
                 show_plot=False,
                 validation=True)
    
    print('done')
    print('Parameters found:')
    for k in range(len(my_params["names"])):
        print(my_params["names"][k] + " : " + str(params_found[k]))
    print('-' * 50)
    print("R2 score : " + str(r2))
    print('-' * 50)
    
    filename = "./out/" + use_kernels + "-" + use_means + "-" + estimator + "-" + variable + ".csv"
            
    with open(filename, 'w', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter='\t',
                               quoting=csv.QUOTE_MINIMAL)
        my_writer.writerow(list(np.round(params_found, 3)))
        my_writer.writerow([r2])
    
    return params_found.tolist()
