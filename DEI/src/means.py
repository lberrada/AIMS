""" Description here

Author: Leonard Berrada
Date: 22 Oct 2015
"""

import numpy as np


def get_mean(mean_name):
    
    if mean_name == "constant":
        return constant_mean
    
    if mean_name == "linear":
        return linear_mean
    
    if mean_name == "periodic":
        return periodic_mean
    
    raise ValueError("%s mean function not implemented" % mean_name)

def constant_mean(Xtesting,
                  params,
                  **kwargs):
    
    alpha = params.pop(0)
    
    if not hasattr(Xtesting, "__len__"):
        return alpha
    
    return alpha * np.ones_like(Xtesting)

def linear_mean(Xtesting,
                params,
                **kwargs):
    
    alpha = params.pop(0)
    beta = params.pop(0)
    
    return alpha * Xtesting + beta

def periodic_mean(Xtesting,
                  params,
                  **kwargs):
    
    scale = params.pop(0)
    period = params.pop(0)
    
    return scale * np.sin(2.*np.pi * Xtesting / period)
    
    
