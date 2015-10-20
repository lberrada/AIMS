""" Data, Estimation & Inference module

kernels.py : definition of kernels in a dictionary

Author: Leonard Berrada
Date: 19 Oct 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


def gaussian_kernel(X=None,
                    Y=None,
                    xstar=None,
                    sigma_f=None,
                    sigma_n=None,
                    l=None,
                    truth=None,
                    epsilon=1e-6):

    same_x = np.where(X[None, :] == X[:, None])
        
    D = X[None, :] - X[:, None]
    K = np.round(sigma_f ** 2 * np.exp(-np.power(D, 2) / (2 * l ** 2)), 3)
    K[same_x] += sigma_n ** 2
    

    n = len(X)
    diag_indices = [np.arange(n), np.arange(n)]
    K[diag_indices] += epsilon
    inv_K = np.linalg.inv(K)

    def get_y(xxstar):

        Xstar = xxstar * np.ones_like(X)
        same_x = np.where(X == Xstar)
        D = X - Xstar
        Ks = sigma_f ** 2 * np.exp(-np.power(D, 2) / (2 * l ** 2))
        Ks[same_x] += sigma_n ** 2
        
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
            
    plt.plot(xstar, 
             truth, 
             'bo')
    plt.errorbar(xstar, 
                 y_mean, 
                 c='red', 
                 yerr=1.96*np.sqrt(y_var), 
                 alpha=0.3)
    plt.show()

    return y_mean, y_var

kernels_dict = dict()

kernels_dict['gaussian'] = gaussian_kernel
