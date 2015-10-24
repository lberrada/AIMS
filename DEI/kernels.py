""" Data, Estimation & Inference module

kernels.py : definition of kernels in a dictionary

Author: Leonard Berrada
Date: 19 Oct 2015
"""

import numpy as np

def kernel(use_kernels,
           X1=None,
           X2=None,
           params=None,
           **kwargs):
    
    # ensure params has the right data structure
    params = list(params)
    
    sigma_n = params.pop(0)
    
    # parse string
    temp_str = use_kernels.replace("*", "+")
    all_kernels = temp_str.split("+")
    
    use_kernels.replace(all_kernels[0], '')
    K = get_kernel(all_kernels.pop(0))(X1=None,
                                       X2=None,
                                       params=None,
                                       **kwargs)
    
    while len(len(all_kernels)):
        op = use_kernels.pop(0)
        use_kernels.replace(all_kernels[0], "")
        if op == "+":
            K += get_kernel(all_kernels.pop(0))(X1=None,
                                                X2=None,
                                                params=None,
                                                **kwargs)
        
        elif op == "*":
            K *= get_kernel(all_kernels.pop(0))(X1=None,
                                                X2=None,
                                                params=None,
                                                **kwargs)
        else:
            raise ValueError("shit happened")
        
    if len(K.shape) == 2:
        n = len(K)
        same_x = [np.arange(n), np.arange(n)]
        K[same_x] += sigma_n ** 2
        
    elif not hasattr(K, "__len__"):
        K += sigma_n ** 2
        
    return K
    
  
    
def get_kernel(kernel_name):
    
    if "exponential_quadratic" in kernel_name:
        return exponential_quadratic_kernel
    
    if "rational_quadratic" in kernel_name:
        return rational_quadratic_kernel
            
    if "periodic" in kernel_name:
        return periodic_kernel
    
    if "matern_12" in kernel_name:
        return matern_12_kernel
    
    if "matern_32" in kernel_name:
        return matern_32_kernel
    
    raise ValueError("%s kernel not implemented:" % kernel_name)

def exponential_quadratic_kernel(X1=None,
                                 X2=None,
                                 params=None,
                                 **kwargs):
    
    
    sigma_f = params.pop(0)
    scale = params.pop(0)
    
    D = X1 - X2
    K = sigma_f ** 2 * np.exp(-np.power(D, 2) / (2 * scale ** 2))
    
    return K

def periodic_kernel(X1=None,
                    X2=None,
                    params=None,
                    **kwargs):
    
    sigma_f = params.pop(0)
    period = params.pop(0)
    
    D = X1 - X2
    K = sigma_f ** 2 * np.exp(-2 * np.power(np.sin(2 * np.pi * D / period), 2))
    
    return K

def rational_quadratic_kernel(X1=None,
                              X2=None,
                              params=None,
                              **kwargs):
    
    sigma_f = params.pop(0)
    scale = params.pop(0)
    nu = params.pop(0)
    
    D = X1 - X2
    K = sigma_f ** 2 * np.power(1. + 1. / (2.*nu) * np.power(D / scale, 2), -nu)
    
    return K
    
def matern_12_kernel(X1=None,
                     X2=None,
                     params=None,
                     **kwargs):
    
    sigma_f = params.pop(0)
    scale = params.pop(0)
    
    D = np.abs(X1 - X2)
    
    K = sigma_f ** 2 * np.exp(-D / scale)
    
    return K

def matern_32_kernel(X1=None,
                     X2=None,
                     params=None,
                     **kwargs):
    
    sigma_f = params.pop(0)
    scale = params.pop(0)
    
    D = np.abs(X1 - X2)
    
    K = sigma_f ** 2 * (1. + np.sqrt(3) * D / scale) * np.exp(-np.sqrt(3) * D / scale)
    
    return K
    
    
    
