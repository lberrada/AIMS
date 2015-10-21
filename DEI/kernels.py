""" Data, Estimation & Inference module

kernels.py : definition of kernels in a dictionary

Author: Leonard Berrada
Date: 19 Oct 2015
"""

import numpy as np

def gaussian_kernel(X1=None,
                    X2=None,
                    sigma_n=None,
                    sigma_f=None,
                    scale=None):
    
        
    D = X1 - X2
    K = sigma_f ** 2 * np.exp(-np.power(D, 2) / (2 * scale ** 2))
    
    if len(D.shape) == 2:
        n = len(D)
        same_x = [np.arange(n), np.arange(n)]
        K[same_x] += sigma_n ** 2
        
    elif not hasattr(D, "__len__"):
        K += sigma_n ** 2
    
    return K

