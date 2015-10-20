""" Data, Estimation & Inference module

kernels.py : definition of kernels in a dictionary

Author: Leonard Berrada
Date: 19 Oct 2015
"""

import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

def optimize_hp_gaussian(X=None,
                         Y=None):
    
    print("optimizing hyper-parameters...")
    
    n = len(X)
    same_x = [np.arange(n), np.arange(n)]
    D = X[None, :] - X[:, None]
    
    def get_neg_log_likelihood(params, *args):
        
        sigma_f, sigma_n, l = params
        
        K = sigma_f ** 2 * np.exp(-np.power(D, 2) / (2 * l ** 2))
        K[same_x] += sigma_n ** 2
        
        L = np.linalg.cholesky(K)
        aux_u = np.linalg.solve(L, Y)
        u = np.linalg.solve(L.T, aux_u)
        log_det_K = 2 * np.trace(np.log(L))
        
        log_likelihood = -0.5 * np.dot(Y.T, u) - 0.5 * log_det_K
        
        neg_log_likelihood = -log_likelihood
        
        return neg_log_likelihood
    
    mean_params = np.array([1., 1., 20])
    sigma_params = np.ones(3)
    log_mean_params = np.log(mean_params)
    
    def get_neg_posterior_probability(params, *args):
        
        log_likelihood = -np.array(get_neg_log_likelihood(params))
        log_params = np.log(params)
        log_prior = -0.5 * np.sum(sigma_params * np.power((log_params - log_mean_params), 2)) -np.sum(np.abs(log_params))
#         log_normalization = scipy.integrate.nquad(np.exp(-get_neg_log_likelihood(params))*scipy.stats.norm(log_params-log_mean_params),
#                                                  -np.infty*np.ones_like(params),
#                                                  np.infty*np.ones_like(params),
#                                                  args=params)
        
#         log_posterior = log_likelihood + log_prior - log_normalization
        log_posterior = log_likelihood + log_prior
        
        neg_log_posterior = -log_posterior
        print(neg_log_posterior)
        
        return neg_log_posterior
        
        
        
    
    init_theta = np.array([1., 1., 25])
    theta = scipy.optimize.fmin_cg(get_neg_log_likelihood,
                                   init_theta)
    
    
    
    print('done')
    print('Parameters found:')
    print('sigma_f :', theta[0])
    print('sigma_n :', theta[1])
    print('l :', theta[2])
    print('-' * 50)
    
    return theta.tolist()
    

def gaussian_kernel(X=None,
                    Y=None,
                    xstar=None,
                    sigma_f=None,
                    sigma_n=None,
                    l=None,
                    truth=None,
                    jitter=1e-6):
    
    print("predicting data...")
    
    n = len(X)
    same_x = [np.arange(n), np.arange(n)]
        
    D = X[None, :] - X[:, None]
    K = sigma_f ** 2 * np.exp(-np.power(D, 2) / (2 * l ** 2))
    K[same_x] += sigma_n ** 2
    
    diag_indices = [np.arange(n), np.arange(n)]
    K[diag_indices] += jitter
    inv_K = np.linalg.inv(K)

    def get_y(xxstar):

        Xstar = xxstar * np.ones_like(X)
        D = X - Xstar
        Ks = sigma_f ** 2 * np.exp(-np.power(D, 2) / (2 * l ** 2))
        
        Kss = sigma_f ** 2 * np.exp(-np.power(xxstar - xxstar, 2) / (2 * l ** 2)) + sigma_n ** 2
        

        aux_K = np.dot(Ks, inv_K)
        
        yy_mean = np.dot(aux_K, Y)
        yy_var = Kss - np.dot(aux_K, Ks.T)
        
        return yy_mean, yy_var

    if not hasattr(xstar, "__len__"):
        y_mean, y_var = get_y(xstar)

    else:
        y_mean = []
        y_var = []
        for xxstar in xstar:
            
            yy_mean, yy_var = get_y(xxstar)
            y_mean.append(yy_mean)
            y_var.append(yy_var)
            
    print("done")
    print("-"*50)
    
    print("displaying results...")
    
    plt.plot(X,
             Y,
             'go',
             alpha=0.5)
            
    plt.plot(xstar,
             truth,
             'bo',
             alpha=0.5)
    
    plt.errorbar(xstar,
                 y_mean,
                 c='red',
                 yerr=1.96 * np.sqrt(y_var),
                 alpha=0.5)
    plt.show()
    
    print("done")

    return y_mean, y_var

kernels_dict = dict()

kernels_dict['gaussian'] = gaussian_kernel
