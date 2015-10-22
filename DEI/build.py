""" Description here

Author: Leonard Berrada
Date: 22 Oct 2015
"""

import numpy as np

from kernels import get_kernel
from means import get_mean
import copy

def mu_K(use_kernels=None,
         use_means=None,
         X1=None,
         X2=None,
         Xtesting=None,
         params=None,
         **kwargs):
    
    # ensure aux_params has the right data structure
    aux_params = copy.copy(params)
    aux_params = list(aux_params)
    
    sigma_n = aux_params.pop(0)
    
    # parse string
    temp_str = use_kernels.replace("*", "+")
    all_kernels = temp_str.split("+")

    use_kernels = use_kernels[len(all_kernels[0]):]
    K = get_kernel(all_kernels.pop(0))(X1=X1,
                                       X2=X2,
                                       params=aux_params)

    while len(all_kernels):
        op = use_kernels[0]
        use_kernels = use_kernels[len(all_kernels[0]) + 1:]
        if op == "+":
            K += get_kernel(all_kernels.pop(0))(X1=X1,
                                                X2=X2,
                                                params=aux_params)
        
        elif op == "*":
            K *= get_kernel(all_kernels.pop(0))(X1=X1,
                                                X2=X2,
                                                params=aux_params)
        else:
            raise ValueError("shit happened : %s" % op)
        
    if len(K.shape) == 2:
        n = len(K)
        same_x = [np.arange(n), np.arange(n)]
        K[same_x] += sigma_n ** 2
        
    elif not hasattr(K, "__len__"):
        K += sigma_n ** 2
        
    mu = 0
    
    if not hasattr(Xtesting, "__len__") and Xtesting == None:
        return mu, K  
        
        
    temp_str = use_means.replace("*", "+")
    all_means = temp_str.split("+")
    
    use_means = use_means[len(all_means[0]):]
    mu = get_mean(all_means.pop(0))(Xtesting=Xtesting,
                                    params=aux_params)
    
    while len(all_kernels):
        op = use_means[0]
        use_means = use_means[len(all_means[0]) + 1:]
        if op == "+":
            mu += get_mean(all_means.pop(0))(Xtesting=Xtesting,
                                             params=aux_params)
        
        elif op == "*":
            mu *= get_mean(all_means.pop(0))(Xtesting=Xtesting,
                                             params=aux_params)
        else:
            raise ValueError("shit happened : %s" % op)
        
    return mu, K
