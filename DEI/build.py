""" Description here

Author: Leonard Berrada
Date: 22 Oct 2015
"""

import numpy as np

from kernels import get_kernel
from means import get_mean

def mu_K(use_kernels=None,
         use_means=None,
         X1=None,
         X2=None,
         Xtesting=None,
         params=None,
         **kwargs):
    
    # ensure params has the right data structure
    params = list(params)
    
    sigma_n = params.pop(0)
    
    # parse string
    temp_str = use_kernels.replace("*", "+")
    all_kernels = temp_str.split("+")
    
    use_kernels.replace(all_kernels[0], '')
    K = get_kernel(all_kernels.pop(0))(X1=X1,
                                       X2=X2,
                                       params=params)
    
    while len(all_kernels):
        op = use_kernels.pop(0)
        use_kernels.replace(all_kernels[0], "")
        if op == "+":
            K += get_kernel(all_kernels.pop(0))(X1=X1,
                                                X2=X2,
                                                params=params)
        
        elif op == "*":
            K *= get_kernel(all_kernels.pop(0))(X1=X1,
                                                X2=X2,
                                                params=params)
        else:
            raise ValueError("shit happened")
        
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
    
    use_means.replace(all_means[0], '')
    mu = get_mean(all_means.pop(0))(Xtesting=Xtesting,
                                    params=params)
    
    while len(all_kernels):
        op = use_means.pop(0)
        use_means.replace(all_means[0], "")
        if op == "+":
            mu += get_mean(all_means.pop(0))(Xtesting=Xtesting,
                                             params=params)
        
        elif op == "*":
            mu *= get_mean(all_means.pop(0))(Xtesting=Xtesting,
                                             params=params)
        else:
            raise ValueError("shit happened")
        
    return mu, K
