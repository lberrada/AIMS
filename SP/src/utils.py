""" Description here

Author: Leonard Berrada
Date: 3 Nov 2015
"""

import numpy as np
import scipy.linalg

def embed_mat_from(X, p):
    
    n = len(X) - p
    res = np.zeros((n, p))
    for k in range(p):
        res[:, k] = X[p- 1 - k:p-1 - k + n]
    return res
    

def pseudo_inverse(X):
    
    aux = X.T.dot(X)
    aux_inv = np.linalg.inv(aux)
    res = aux_inv.dot(X.T)
    
    return res

def weights_auto_corr(X, p):
    
    Xcentered = X - np.mean(X)

    r = np.array([Xcentered[:-(p+1)].T.dot(Xcentered[i: i-(p+1)]) for i in range(p+1)])
    r /= r[0]
    
    res = scipy.linalg.solve_toeplitz(r[:-1], r[1:])
    
    return res

def get_spectral_est(a, e):
    
    step =1e-2
    f_grid = np.arange(step, 1.-step, step)
    
    sigma_e_2 = np.var(e)
    Ts = 1.
    p = len(a)
    
    res = sigma_e_2 * Ts * np.ones_like(f_grid)
    for k in range(len(f_grid)):
        ak_x_exp = [-a[i] * np.exp(-1j * 2.*np.pi * f_grid[k] * i * Ts) for i in range(p)]
        res[k] /= abs(1. + np.sum(ak_x_exp)) ** 2
    
    return res
    
