""" Description here

Author: Leonard Berrada
Date: 21 Oct 2015
"""

import numpy as np
import scipy.stats
import scipy.optimize
import csv
from build import train_on
from params import get_params

def optimize_hyperparameters(self):
    
    print("optimizing hyper-parameters...")
    
    my_params = get_params(self.use_kernels, 
                           self.use_means)
    
    init_params = np.array(my_params["init"])
    mean_params = my_params["means"]
    mean_stds = my_params["stds"]
    bounds = my_params["bounds"]
    use_log = my_params["use_log"]
    
    def get_neg_log_likelihood(params,
                               *args,
                               **kwargs):
        
        mu, K = train_on(self,
                         X1=self.Xtraining[None, :],
                         X2=self.Xtraining[:, None],
                         Xtesting=self.Xtraining)
        
        Ycentered = self.Ytraining - mu

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
        
    if self.estimator.lower() == "mle":
        fitness_function = get_neg_log_likelihood
        
    elif self.estimator.lower() == "map":
        fitness_function = get_neg_log_posterior
    
    else:
        raise ValueError("%s estimator is not implemented: should be 'MLE' or 'MAP'" % self.estimator)

    theta = scipy.optimize.fmin_l_bfgs_b(fitness_function,
                                         init_params,
                                         approx_grad=True,
                                         bounds=bounds)
    
    params_found = theta[0]
    if self.estimator == "MLE":
        score = get_neg_log_likelihood(params_found)
    else:
        score = get_neg_log_posterior(params_found)
    
    print('done')
    print('Parameters found:')
    for k in range(len(my_params["names"])):
        print(my_params["names"][k] + " : " + str(params_found[k]))
    print("Score found : " + str(score))
    print('-' * 50)
    
    filename = "../out/results.csv"
            
    with open(filename, 'a', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter='\t',
                               quoting=csv.QUOTE_MINIMAL)
        my_writer.writerow(['-' * 50])
        my_writer.writerow([self.use_kernels, 
                            self.use_means, 
                            self.variable, 
                            self.estimator])
        my_writer.writerow(list(my_params["names"]))
        my_writer.writerow(list(np.round(params_found, 4)))
        my_writer.writerow([self.estimator, 
                            score])
    
    return params_found.tolist()
