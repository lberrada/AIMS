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

    assert(X != None and Y!=None and sigma_f != None 
           and sigma_n != None and l != None)

    def k(x, xprime):
        same_x = np.isclose(x[None, :], xprime[:, None], rtol=1e-3)
        D = x[None, :] - xprime[:, None]
        k = sigma_f * np.exp(- np.power(D, 2) / (2 * l**2))
        k[same_x] += sigma_n

        return k

    K = k(X, X)
    Xstar = xstar * np.ones_like(X)
    Ks = k(X, Xstar)
    Kss = sigma_f * np.exp(- np.power(xstar - xstar, 2) / (2 * l**2))
    
    aux_K = np.dot(Ks, np.linalg.inv(K))
    
    y_mean = np.dot(aux_K, Y)
    y_var = Kss - np.dot(aux_K, Ks.T)
    
    return y_mean, y_var

kernels_dict = dict()

kernels_dict['gaussian'] = gaussian_kernel
