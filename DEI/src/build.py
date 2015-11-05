""" Description here

Author: Leonard Berrada
Date: 22 Oct 2015
"""

import numpy as np

from kernels import get_kernel
from means import get_mean
import copy

def train_on(self,
             X1=None,
             X2=None,
             XX=None,
             Xtesting=None):
    
    if hasattr(XX, "__len__"):
        X1 = XX[:, None]
        X2 = XX[None, :]
    
    # clean string for combination of means and kernels
    use_means = self.use_means.replace(" ", "")
    use_kernels = self.use_kernels.replace(" ", "")
    
    # ensure aux_params has the right data structure
    aux_params = copy.copy(self.params)
    aux_params = list(aux_params)
    
    #===========================================================================
    # Compute covariance matrix (K)
    #===========================================================================
    
    sigma_n = aux_params.pop(0)
    
    if hasattr(XX, "__len__"):
        d1 = max(XX.shape)
        d2 = d1
        K = np.zeros((d1, d2))
    else:
        d = len(X1)
        K = np.zeros(d)
    
    for to_add_op in use_kernels.split('+'):
        to_add_K = np.ones_like(K)
        for to_mult in to_add_op.split("*"):
            to_add_K *= get_kernel(to_mult)(X1=X1,
                                            X2=X2,
                                            params=aux_params)
        K += to_add_K
            
    if not hasattr(K, "__len__"):
        K += sigma_n ** 2
         
    elif len(K.shape) == 2 and K.shape[0] == K.shape[1]:
        n = len(K)
        same_x = [np.arange(n), np.arange(n)]
        K[same_x] += sigma_n ** 2
    
    mu = 0
    
    # if no testing data, no need to compute mean
    if not hasattr(Xtesting, "__len__") and Xtesting == None:
        return mu, K
        
    #===========================================================================
    # Compute mean (mu)
    #===========================================================================
    
    mu = np.zeros_like(Xtesting, dtype=np.float64)
    
    for to_add_op in use_means.split('+'):
        to_add_mu = np.ones_like(mu, dtype=np.float64)
        for to_mult in to_add_op.split("*"):
            temp = get_mean(to_mult)(Xtesting=Xtesting,
                                           params=aux_params)
            to_add_mu *= temp
        mu += to_add_mu
        
    return mu, K
