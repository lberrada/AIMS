""" Description here

Author: Leonard Berrada
Date: 22 Oct 2015
"""

import numpy as np

def get_mean(mean_name):
    
    if mean_name=="constant":
        return constant_mean
    
    if mean_name=="linear":
        return linear_mean
    
    if mean_name=="periodic":
        return periodic_mean
    
    raise ValueError("%s mean function not implemented" % mean_name)

def constant_mean(alpha,
                  Xtesting):
    
    if not hasattr(Xtesting, "__len__"):
        return alpha
    
    return alpha * np.ones_like(Xtesting)

def linear_mean(alpha,
                beta,
                Xtest):
    
    return alpha * Xtest + beta

def periodic_mean(scale,
                  period,
                  Xtest):
    
    return scale * np.sin(2.*np.pi * Xtest / period)
    
    
