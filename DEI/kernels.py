""" Data, Estimation & Inference module

kernels.py : definition of kernels in a dictionary

Author: Leonard Berrada
Date: 19 Oct 2015
"""

import numpy as np

def gaussian_kernel(X1=None,
                    X2=None,
                    params=None,
                    **kwargs):
    
    
    sigma_n, sigma_f, scale = params
    
    D = X1 - X2
    K = sigma_f ** 2 * np.exp(-np.power(D, 2) / (2 * scale ** 2))
    
    if len(D.shape) == 2:
        n = len(D)
        same_x = [np.arange(n), np.arange(n)]
        K[same_x] += sigma_n ** 2
        
    elif not hasattr(D, "__len__"):
        K += sigma_n ** 2
    
    return K

def gaussian_kernel_2(X1=None,
                      X2=None,
                      params=None,
                      **kwargs):
    
    sigma_n, sigma_f, scale, sigma_f_2, scale_2 = params
    params_1 = (sigma_n, sigma_f, scale)
        
    K = gaussian_kernel(X1,
                        X2,
                        params_1)
    
    D = X1 - X2
    K += sigma_f_2 ** 2 * np.exp(-np.power(D, 2) / (2 * scale_2 ** 2))
    
    return K

def locally_periodic_kernel(X1=None,
                            X2=None,
                            params=None,
                            **kwargs):
    
    sigma, sigma_n, p, scale, scale_2 = params
    
    D = X1 - X2
    K = sigma ** 2 * np.exp(-np.power(D, 2) / (2 * scale ** 2))
    K *= np.exp(-2 * np.sin(np.pi * np.abs(D) / p) ** 2 / (scale_2 ** 2))
    
    if len(D.shape) == 2:
        n = len(D)
        same_x = [np.arange(n), np.arange(n)]
        K[same_x] += sigma_n ** 2
        
    elif not hasattr(D, "__len__"):
        K += sigma_n ** 2
    
    return K
    
    pass

