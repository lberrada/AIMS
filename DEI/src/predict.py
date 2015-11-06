""" Description here
 
Author: Leonard Berrada
Date: 21 Oct 2015
"""
 
import numpy as np
from build import train_on
 
def predict(self,
            show_plot=True):
     
    print("predicting data...")
     
    mu, K = train_on(self,
                     XX=self.X_training(),
                     Xtesting=self.X_training())
     
    L = np.linalg.cholesky(K)
 
    Ypredicted = np.zeros(self.n_testing)
    Yvar = np.zeros(self.n_testing)
    Y_centered = self.Y_training() - mu
    index = 0 if self.sequential_mode else None
    
     
    for i in range(self.n_testing):
         
        xstar = self.X_testing([i])
         
        if self.sequential_mode:
            if i==0:
                continue
            while index < self.n_training and self.X_training([index]) < xstar:
                index += 1
            L = np.linalg.cholesky(K[:index, :index])
         
        Xstar = xstar * np.ones(len(L))
        _, Ks = train_on(self,
                         X1=self.X_training(stop=index),
                         X2=Xstar)
         
        mu, Kss = train_on(self,
                           X1=xstar,
                           X2=xstar,
                           Xtesting=xstar)
             
        aux = np.linalg.solve(L, Ks.T)
        KsxK_inv = np.linalg.solve(L.T, aux).T
        
        Ypredicted[i] = KsxK_inv.dot(Y_centered[:index]) + float(mu)
        Yvar[i] = Kss - KsxK_inv.dot(Ks.T)
        
    self._testing_df['ymean'] = Ypredicted
    self._testing_df['yvar'] = Yvar
         
    print("done")
     
    
     
