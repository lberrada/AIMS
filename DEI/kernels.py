""" Data, Estimation & Inference module

kernels.py : definition of kernels in a dictionary

Author: Leonard Berrada
Date: 19 Oct 2015
"""

import numpy as np


def gaussian_kernel(X=None,
                    Y=None,
                    xstar=None,
                    sigma_f=None,
                    sigma_n=None,
                    l=None,
                    epsilon=1e-6):

    def kernel(x, xprime):
        if x.dtype == "datetime64[ns]":
            same_x = np.where(x[None, :] == xprime[:, None])
        else:
            same_x = np.isclose(x[None, :], xprime[:, None], rtol=1e-3)
        D = x[None, :] - xprime[:, None]
        k = sigma_f * np.exp(- np.power(D, 2) / (2 * l**2))
        k[same_x] += sigma_n

        return k

    K = kernel(X, X)
    n = len(X)
    diag_indices = [np.arange(n), np.arange(n)]
    K[diag_indices] += epsilon
    inv_K = np.linalg.inv(K)

    def get_y(xxstar):
        Xstar = xxstar * np.ones_like(X)
        
        same_x = np.isclose(X, Xstar, rtol=1e-3)
        D = X - Xstar
        Ks = sigma_f * np.exp(- np.power(D, 2) / (2 * l**2))
        Ks[same_x] += sigma_n
        
        Kss = sigma_f * np.exp(- np.power(xxstar - xxstar, 2) / (2 * l**2))
        

        aux_K = np.dot(Ks, inv_K)
        
        yy_mean = np.dot(aux_K, Y)
        yy_var = Kss - np.dot(aux_K, Ks.T)
        
        print(yy_mean, yy_var)

        return yy_mean, yy_var

    if not hasattr(xstar, "__len__"):
        y_mean, y_var = get_y(xstar)

    else:
        y_mean = []
        y_var = []
        for xxstar in xstar:
            
            print(xxstar)
            
            yy_mean, yy_var = get_y(xxstar)
            y_mean.append(yy_mean)
            y_var.append(yy_var)

    return y_mean, y_var

kernels_dict = dict()

kernels_dict['gaussian'] = gaussian_kernel
