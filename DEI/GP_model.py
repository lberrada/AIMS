""" Description here

Author: Leonard Berrada
Date: 21 Oct 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import csv
from build import mu_K
sns.set(color_codes=True)
    

def predict(Xtraining=None,
            Ytraining=None,
            Xtesting=None,
            params=None,
            Ytestingtruth=None,
            use_kernels=None,
            use_means=None,
            sequential_mode=False,
            variable=None,
            estimator=None,
            t0=None,
            show_plot=True):
    
    print("predicting data...")
    
    mu, K = mu_K(use_kernels=use_kernels,
                 use_means=use_means,
                 X1=Xtraining[None, :],
                 X2=Xtraining[:, None],
                 Xtesting=Xtraining,
                 params=params)
    
    L = np.linalg.cholesky(K)

    Ypredicted = np.zeros_like(Xtesting)
    Yvar = np.zeros_like(Xtesting)
    Y_centered = Ytraining - mu
    index = 0
    
    if sequential_mode:
        training_len = len(Xtraining)
        savedXtraining = copy.copy(Xtraining)
        savedYtraining = copy.copy(Y_centered)
    
    for i in range(1, len(Xtesting)):
        
        xstar = Xtesting[i]
        
        if sequential_mode:
            while index < training_len and savedXtraining[index] < xstar:
                index += 1
            L = np.linalg.cholesky(K[:index, :index])
            Xtraining = savedXtraining[:index]
            Y_centered = savedYtraining[:index]
        
        Xstar = xstar * np.ones_like(Xtraining)
        _, Ks = mu_K(use_kernels=use_kernels,
                     use_means=use_means,
                     X1=Xtraining,
                     X2=Xstar,
                     params=params)
        
        mu, Kss = mu_K(use_kernels=use_kernels,
                      use_means=use_means,
                      X1=xstar,
                      X2=xstar,
                      Xtesting=xstar,
                      params=params)
            
        aux = np.linalg.solve(L, Ks.T)
        KsxK_inv = np.linalg.solve(L.T, aux).T
        
        Ypredicted[i] = np.dot(KsxK_inv, Y_centered) + mu
        Yvar[i] = Kss - np.dot(KsxK_inv, Ks.T)
        
    if sequential_mode:
        Xtraining = savedXtraining 
        Ytraining = savedYtraining
    
    print("done")
    print("-"*50)
    
    print('computing score...')
    
    ssres = np.sum(np.power(Ypredicted - Ytestingtruth, 2))
    print(np.sqrt(ssres))
    sstot = np.sum(np.power(Ypredicted - np.mean(Ypredicted), 2))
    r2 = 1 - ssres / sstot
    print(r2)
    
    filename = "./out/" + use_kernels + "-" + use_means + "-" + estimator + "-" + variable + ".csv"
            
    with open(filename, 'a', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter='\t',
                               quoting=csv.QUOTE_MINIMAL)
        my_writer.writerow([round(r2, 3)])

    print("done")
    print("-"*50)
    
    print("creating plot...")
    
    Ttesting = np.array([t0] * len(Xtesting), dtype='datetime64')
    Ttesting += np.array([np.timedelta64(int(x) * 5, 'm') for x in Xtesting], dtype=np.timedelta64)
#     Ttesting = Xtesting
    
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

