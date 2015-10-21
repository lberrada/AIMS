""" Description here

Author: Leonard Berrada
Date: 21 Oct 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kernels import gaussian_kernel, gaussian_kernel_2, locally_periodic_kernel
sns.set(color_codes=True)


    

def predict(X=None,
            Y=None,
            xstar=None,
            params=None,
            truth=None,
            use_kernel="gaussian"):
    
    print("predicting data...")
    
    if use_kernel =="gaussian":
        kernel = gaussian_kernel
        
    elif use_kernel=="gaussian_2":
        kernel = gaussian_kernel_2
        
    elif use_kernel == "locally_periodic":
        kernel = locally_periodic_kernel
        
    else:
        raise ValueError("%s kernel not implemented:" % use_kernel)
    
    K = kernel(X1=X[None, :], 
               X2=X[:, None],
               params=params)

    L = np.linalg.cholesky(K)

    def get_y(xxstar):

        Xstar = xxstar * np.ones_like(X)
        Ks = kernel(X1=X, 
                    X2=Xstar,
                    params=params)
        
        Kss = kernel(X1=xxstar, 
                     X2=xxstar,
                     params=params)
            
        aux = np.linalg.solve(L, Ks.T)
        KsxK_inv = np.linalg.solve(L.T, aux).T
        
        yy_mean = np.dot(KsxK_inv, Y)
        yy_var = Kss - np.dot(KsxK_inv, Ks.T)
        
        return yy_mean, yy_var

    if not hasattr(xstar, "__len__"):
        y_predicted, y_var = get_y(xstar)

    else:
        y_predicted = []
        y_var = []
        for xxstar in xstar:
            
            yy_mean, yy_var = get_y(xxstar)
            y_predicted.append(yy_mean)
            y_var.append(yy_var)
            
    print("done")
    print("-"*50)
    
    print('computing score...')
    
    ssres = np.sum(np.power(y_predicted - truth, 2))
    sstot = np.sum(np.power(y_predicted - np.mean(y_predicted), 2))
    r2 = 1 - ssres / sstot
    
    print("done")
    print("-"*50)
    
    print("displaying results...")
    plt.suptitle("R^2 : " + str(np.round(r2, 4)))
    
    plt.plot(X,
             Y,
             'go',
             alpha=0.5)
            
    plt.plot(xstar,
             truth,
             'bo',
             alpha=0.5)
    
    plt.errorbar(xstar,
                 y_predicted,
                 c='red',
                 yerr=1.96 * np.sqrt(y_var),
                 alpha=0.5)
    plt.show()
    
    print("done")

    return y_predicted, y_var
