""" Description here

Author: Leonard Berrada
Date: 21 Oct 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kernels import gaussian_kernel, gaussian_kernel_2, locally_periodic_kernel, \
    matern_kernel
import copy
import csv
from means import nearest_neighbour_mean
sns.set(color_codes=True)
    

def predict(Xtraining=None,
            Ytraining=None,
            Xtesting=None,
            params=None,
            Ytestingtruth=None,
            use_kernel="gaussian",
            sequential_mode=False,
            variable=None,
            estimator=None,
            t0=None,
            ymean=None,
            show_plot=True):
    
    print("predicting data...")
    
    if use_kernel == "gaussian":
        kernel = gaussian_kernel
        
    elif use_kernel == "gaussian_2":
        kernel = gaussian_kernel_2
        
    elif use_kernel == "locally_periodic":
        kernel = locally_periodic_kernel
        
    elif use_kernel == "matern":
        kernel = matern_kernel
        
    else:
        raise ValueError("%s kernel not implemented:" % use_kernel)
    
    K = kernel(X1=Xtraining[None, :],
               X2=Xtraining[:, None],
               params=params)

    L = np.linalg.cholesky(K)

    Ypredicted = np.zeros_like(Xtesting)
    Yvar = np.zeros_like(Xtesting)
    index = 0
    
    if sequential_mode:
        training_len = len(Xtraining)
        savedXtraining = copy.copy(Xtraining)
        savedYtraining = copy.copy(Ytraining)
    
    for i in range(1, len(Xtesting)):
        
        xstar = Xtesting[i]
        
        if sequential_mode:
            while index < training_len and savedXtraining[index] < xstar:
                index += 1
            L = np.linalg.cholesky(K[:index, :index])
            Xtraining = savedXtraining[:index]
            Ytraining = savedYtraining[:index]
        
        Xstar = xstar * np.ones_like(Xtraining)
        Ks = kernel(X1=Xtraining,
                    X2=Xstar,
                    params=params)
        
        Kss = kernel(X1=xstar,
                     X2=xstar,
                     params=params)
            
        aux = np.linalg.solve(L, Ks.T)
        KsxK_inv = np.linalg.solve(L.T, aux).T
        
        Ypredicted[i] = np.dot(KsxK_inv, Ytraining) + ymean
        Yvar[i] = Kss - np.dot(KsxK_inv, Ks.T)
        
    if sequential_mode:
        Xtraining = savedXtraining 
        Ytraining = savedYtraining
    
    print("done")
    print("-"*50)
    
    print('computing score...')
    
    ssres = np.sum(np.power(Ypredicted - Ytestingtruth, 2))
    sstot = np.sum(np.power(Ypredicted - np.mean(Ypredicted), 2))
    r2 = 1 - ssres / sstot
    print(r2)
    
    filename = use_kernel + "-" + estimator + "-" + variable + ".csv"
            
    with open(filename, 'a', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter='\t',
                               quoting=csv.QUOTE_MINIMAL)
        my_writer.writerow([round(r2, 3)])

    print("done")
    print("-"*50)
    
    print("creating plot...")
    
    Ttesting = np.array([t0] * len(Xtesting), dtype='datetime64')
    Ttesting += np.array([np.timedelta64(int(x) * 5, 'm') for x in Xtesting], dtype=np.timedelta64)
    
    plt.errorbar(Ttesting,
                 Ypredicted,
                 fmt='o',
                 ms=4,
                 color='red',
                 ecolor='red',
                 yerr=1.96 * np.sqrt(Yvar),
                 alpha=0.3)
    
    plt.plot(Ttesting,
             Ytestingtruth,
             'ko-',
             ms=4,
             alpha=0.7)
    
    plt.plot(Ttesting,
             Ypredicted,
             'ro',
             ms=4)
    
    fig_name = filename.replace("csv", "png")
    plt.savefig(fig_name,
                transparent=False,
                dpi=200,
                bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
        
    print("done")

