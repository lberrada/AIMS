""" Description here

Author: Leonard Berrada
Date: 5 Nov 2015
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.linalg

class Forecast:
    
    def __init__(self,
                 y,
                 **kwargs):
        
        self._df = pd.DataFrame()
        self._pred_df = pd.DataFrame()
        
        self.n_data = len(y)
        self._df['y'] = y
        
        print("done")
        print("-"*50)
        print("showing headers for verification...")
        
        print("Total Data :")
        print(self._df.head())
        
    def Y(self,
          indices=None,
          start=None,
          stop=None):
        
        if hasattr(indices, "__len__"):
            return self._df.y.values[indices]
        else:
            return self._df.y.values[start:stop]
        
    def Y_pred(self,
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._pred_df.ypred.values[indices]
        else:
            return self._pred_df.ypred.values[start:stop]
        
    def Y_error(self,
                indices=None,
                start=None,
                stop=None):
        
        if hasattr(indices, "__len__"):
            return self._pred_df.yerr.values[indices]
        else:
            return self._pred_df.yerr.values[start:stop]
        
    def embed_data(self):
        
        n = self.n_data - self.p
        self._emb_matrix = np.zeros((n, self.p))
        
        for k in range(self.p):
            self._emb_matrix[:, k] = self.Y(start=self.p - 1 - k,
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
            var_to_plot = self._df[var_name].values
            
        else:
            var_to_plot = self._pred_df[var_name].values
            
        plt.plot(var_to_plot)
        
        if show:
            plt.show()
            
            
class AutoRegression(Forecast):
    
    def __init__(self,
                 y,
                 p=5):
        
        Forecast.__init__(self,
                          y)
        self.p = p
        
    def fit(self):
        
        self.embed_data()
                                           
        self.pseudo_inv = np.linalg.pinv(self._emb_matrix)
        
    def predict(self):
        
        
        self._pred_df['ypred'] = self._emb_matrix.dot(self.pseudo_inv.dot(self.Y(start=self.p)))
        self._pred_df['yerr'] = self.Y(start=self.p) - self.Y_pred()
        
class AutoCorrelation(Forecast):
    
    def __init__(self,
                 y,
                 p=5):
        
        Forecast.__init__(self,
                          y)
        self.p = p
        
    def fit(self):
        
        self.embed_data()
        
        Ycentered = self.Y() - np.mean(self.Y())

        r = np.array([Ycentered[:-(self.p + 1)].T.dot(Ycentered[i: i - (self.p + 1)]) for i in range(self.p + 1)])
        r /= r[0]
        
        self.a_hat = scipy.linalg.solve_toeplitz(r[:-1], r[1:])
        
        
    def predict(self):
        
        self._pred_df['ypred'] = self._emb_matrix.dot(self.a_hat)
        self._pred_df['yerr'] = self.Y(start=self.p) - self.Y_pred()
        
    def spectrum(self, 
                 step=1e-2, 
                 start=0, 
                 stop=1, 
                 Ts=1.):
        
        f_grid = np.arange(start + step, stop - step, step)
        
        sigma_e_2 = np.var(self.Y_error())
        Ts = Ts
        
        self.spectrum = sigma_e_2 * Ts * np.ones_like(f_grid)
        for k in range(len(f_grid)):
            ak_x_exp = [-self.a_hat[i] * np.exp(-1j * 2.*np.pi * f_grid[k] * i * Ts) for i in range(self.p)]
            self.spectrum[k] /= abs(1. + np.sum(ak_x_exp)) ** 2
