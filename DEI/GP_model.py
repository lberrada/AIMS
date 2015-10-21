""" Description here

Author: Leonard Berrada
Date: 21 Oct 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kernels import gaussian_kernel
sns.set(color_codes=True)


    

def predict(X=None,
            Y=None,
            xstar=None,
            sigma_f=None,
            sigma_n=None,
            l=None,
            truth=None,
            jitter=1e-6):
    
    print("predicting data...")
    
    K = gaussian_kernel(X1=X[None, :], 
                        X2=X[:, None],
                        sigma_f=sigma_f,
                        sigma_n=sigma_n,
                        scale=l)

    L = np.linalg.cholesky(K)

    def get_y(xxstar):

        Xstar = xxstar * np.ones_like(X)
        Ks = gaussian_kernel(X1=X, 
                            X2=Xstar,
                            sigma_f=sigma_f,
                            sigma_n=sigma_n,
                            scale=l)
        
        Kss = gaussian_kernel(X1=xxstar, 
                            X2=xxstar,
                            sigma_f=sigma_f,
                            sigma_n=sigma_n,
                            scale=l)
            
        aux = np.linalg.solve(L, Ks.T)
        KsxK_inv = np.linalg.solve(L.T, aux).T
        
        yy_mean = np.dot(KsxK_inv, Y)
        yy_var = Kss - np.dot(KsxK_inv, Ks.T)
        
        return yy_mean, yy_var

    if not hasattr(xstar, "__len__"):
        y_mean, y_var = get_y(xstar)

    else:
        y_mean = []
        y_var = []
        for xxstar in xstar:
            
            yy_mean, yy_var = get_y(xxstar)
            y_mean.append(yy_mean)
            y_var.append(yy_var)
            
    print("done")
    print("-"*50)
    
    print("displaying results...")
    
    plt.plot(X,
             Y,
             'go',
             alpha=0.5)
            
    plt.plot(xstar,
             truth,
             'bo',
             alpha=0.5)
    
    plt.errorbar(xstar,
                 y_mean,
                 c='red',
                 yerr=1.96 * np.sqrt(y_var),
                 alpha=0.5)
    plt.show()
    
    print("done")

    return y_mean, y_var