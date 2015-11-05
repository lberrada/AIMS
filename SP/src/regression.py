""" Description here

Author: Leonard Berrada
Date: 4 Nov 2015
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.linalg

class Regresssion:
    
    def __init__(self,
                 xtrain,
                 xtest,
                 ytrain,
                 ytest,
                 **kwargs):
        
        self._training_df = pd.DataFrame()
        self._testing_df = pd.DataFrame()
        
        self.n_training = len(xtrain)
        self._training_df['x'] = xtrain
        self._training_df['y'] = ytrain
        
        self.n_testing = len(xtest)
        self._testing_df['x'] = xtest
        self._testing_df['y'] = ytest
        
        print("done")
        print("-"*50)
        print("showing headers for verification...")
        
        print("Training Data :")
        print(self._testing_df.head())
        print("Testing Data :")
        print(self._training_df.head())
        
    def X_training(self,
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._training_df.x.values[indices]
        else:
            return self._training_df.x.values[start:stop]
    
    def X_testing(self,
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._testing_df.x.values[indices]
        else:
            return self._testing_df.x.values[start:stop]
    
    def Y_training(self,
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._training_df.y.values[indices]
        else:
            return self._training_df.y.values[start:stop]
    
    def Y_testing(self,
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._testing_df.y.values[indices]
        else:
            return self._testing_df.y.values[start:stop]
        
    def Y_pred(self,
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._testing_df.ypred.values[indices]
        else:
            return self._testing_df.ypred.values[start:stop]
        
    def Y_error(self,
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._testing_df.yerr.values[indices]
        else:
            return self._testing_df.yerr.values[start:stop]
        
    def embed_data(self):
        
        n = self.n_training - self.p
        self._emb_matrix = np.zeros((n, self.p))
        
        for k in range(self.p):
            self._emb_matrix[:, k] = self.Y_training(start=self.p - 1 - k,
                                                     stop=self.p - 1 - k + n)
            
            
    def fit(self):
        
        raise NotImplementedError("method should be overwritten")
    
    def predict(self, testing_data):
        
        raise NotImplementedError("method should be overwritten")
    
    def get(self, attr_name):
        
        return getattr(self, attr_name)
    
    def plot_attr(self, attr_name, show=False):
        
        attr_to_plot = getattr(self, attr_name)
        
        plt.plot(attr_to_plot)
        
        if show:
            plt.show()
            
    def plot_var(self, var_name, set_="", show=False):
        
        if 'train' in set_.lower():
            var_to_plot = self._training_df[var_name].values
            
        else:
            var_to_plot = self._testing_df[var_name].values
            
        plt.plot(var_to_plot)
        
        if show:
            plt.show()
        

    
class AutoRegression(Regresssion):
    
    def __init__(self,
                 xtrain,
                 xtest,
                 ytrain,
                 ytest,
                 p):
        
        Regresssion.__init__(self,
                             xtrain,
                             xtest,
                             ytrain,
                             ytest,)
        self.p = p
        
    def fit(self):
        
        self.embed_data()
                                           
        self.pseudo_inv = np.linalg.pinv(self._emb_matrix)
        
    def predict(self):
        
        self._testing_df['ypred'] = self._emb_matrix.dot(self.pseudo_inv.dot(self.Y_testing()))
        self._testing_df['yerr'] = self.Y_testing() - self.Y_pred()
        
class AutoCorrelation(Regresssion):
    
    def __init__(self,
                xtrain,
                 xtest,
                 ytrain,
                 ytest,
                 p):
        
        Regresssion.__init__(self,
                             xtrain,
                             xtest,
                             ytrain,
                             ytest,)
        self.p = p
        
    def fit(self):
        
        self.embed_data()
        
        Xcentered = self.Y_training() - np.mean(self.Y_training())

        r = np.array([Xcentered[:-(self.p + 1)].T.dot(Xcentered[i: i - (self.p + 1)]) for i in range(self.p + 1)])
        r /= r[0]
        
        self.a_hat = scipy.linalg.solve_toeplitz(r[:-1], r[1:])
        
        
    def predict(self):
        
        self._testing_df['ypred'] = self._emb_matrix.dot(self.a_hat)
        self._testing_df['yerr'] = self.Y_testing() - self.Y_pred()
        
    def spectrum(self):
        
        step = 1e-2
        f_grid = np.arange(step, 1. - step, step)
        
        sigma_e_2 = np.var(self.Y_error())
        Ts = 1.
        
        self.spectrum = sigma_e_2 * Ts * np.ones_like(f_grid)
        for k in range(len(f_grid)):
            ak_x_exp = [-self.a_hat[i] * np.exp(-1j * 2.*np.pi * f_grid[k] * i * Ts) for i in range(self.p)]
            self.spectrum[k] /= abs(1. + np.sum(ak_x_exp)) ** 2
        
        
    
