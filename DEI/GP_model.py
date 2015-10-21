""" Description here

Author: Leonard Berrada
Date: 21 Oct 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kernels import gaussian_kernel, gaussian_kernel_2, locally_periodic_kernel
sns.set(color_codes=True)


    

def predict(Xtraining=None,
            Ytraining=None,
            Xtesting=None,
            params=None,
            Ytestingtruth=None,
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
    
    K = kernel(X1=Xtraining[None, :], 
               X2=Xtraining[:, None],
               params=params)

    L = np.linalg.cholesky(K)

    Ypredicted = np.zeros_like(Xtesting)
    Yvar = np.zeros_like(Xtesting)
    for i in range(len(Xtesting)):
        
        xstar = Xtesting[i]
        
        Xstar = xstar * np.ones_like(Xtraining)
        Ks = kernel(X1=Xtraining, 
                    X2=Xstar,
                    params=params)
        
        Kss = kernel(X1=xstar, 
                     X2=xstar,
                     params=params)
            
        aux = np.linalg.solve(L, Ks.T)
        KsxK_inv = np.linalg.solve(L.T, aux).T
        
        Ypredicted[i] = np.dot(KsxK_inv, Ytraining)
        Yvar[i] = Kss - np.dot(KsxK_inv, Ks.T)
        
    print("done")
    print("-"*50)
    
    print('computing score...')
    
    ssres = np.sum(np.power(Ypredicted - Ytestingtruth, 2))
    sstot = np.sum(np.power(Ypredicted - np.mean(Ypredicted), 2))
    r2 = 1 - ssres / sstot

    print("done")
    print("-"*50)
    
    print("displaying results...")
    plt.suptitle("R^2 : " + str(np.round(r2, 4)))
    
    plt.errorbar(Xtesting,
                 Ypredicted,
                 color='red',
                 ecolor='red',
                 yerr=1.96 * np.sqrt(Yvar),
                 alpha=0.5)
    
    plt.plot(Xtraining,
             Ytraining,
             'go',
             alpha=0.5)
            
    plt.plot(Xtesting,
             Ytestingtruth,
             'bo',
             alpha=0.5)
    
    plt.show()
    
    print("done")

