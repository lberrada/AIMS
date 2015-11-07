""" Description here

Author: Leonard Berrada
Date: 21 Oct 2015
"""

import numpy as np
import scipy.stats
import scipy.optimize
import csv
from params import get_params

def optimize_hyperparameters(self, out=""):
    
    my_params = get_params(self.use_kernels,
                           self.use_means)
    
    print("optimizing hyper-parameters (%i)..." % len(my_params['means']))
    
    mean_params = my_params["means"]
    std_params = my_params["stds"]
    bounds = my_params["bounds"]
    use_log = my_params["use_log"]
    
    init_params = self.params
    
    print("initialization of parameters :")
    print(my_params["names"])
    print(np.round(self.params, 2).tolist())
    
    
    def get_neg_log_likelihood(params,
                               *args,
                               **kwargs):
        
        self.params = list(params)
        
        mu = self.compute_mu(Xtesting=self.X_training())
        K = self.compute_K(XX=self.X_training())
        
        Ycentered = self.Y_training() - mu
        
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
                                               scale=std_params[i])
        
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
                                         bounds=bounds,
                                         maxiter=300,
                                         disp=1)
    
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
    
    if out:
            
        with open(out, 'a', newline='') as csvfile:
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
