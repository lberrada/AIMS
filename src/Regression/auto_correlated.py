""" Description here

Author: Leonard Berrada
Date: 5 Nov 2015
"""

import numpy as np
import pandas as pd
import copy

import matplotlib.pyplot as plt

import scipy.linalg

from regression import RegressionModel

class AutoCorrelation(RegressionModel):

    def __init__(self,
                 y,
                 p=5):

        RegressionModel.__init__(self,
                            y)
        self.p = p

    def fit(self):

        self.embed_data()

        Ycentered = self.Y_training() - np.mean(self.Y_training())

        r = np.array([Ycentered[
                     :-(self.p + 1)].T.dot(Ycentered[i: i - (self.p + 1)]) for i in range(self.p + 1)])
        r /= r[0]

        self._a_hat = scipy.linalg.solve_toeplitz(r[:-1], r[1:])

    def predict(self):

        self._pred_df = pd.DataFrame()
        n_pred = self.n_training + self.n_testing
        self._pred_df['ypred'] = np.zeros(n_pred)
        self._pred_df['yerr'] = np.zeros(n_pred)

        self._pred_df['ypred'][
            :self.n_training - self.p] = self._emb_matrix.dot(self._a_hat)

        y = copy.copy(self.Y_training(start=-self.p))
        for i in range(self.n_training - self.p, n_pred):
            pred = self._a_hat[::-1].dot(y)
            y[:-1] = y[1:]
            y[-1] = pred
            self._pred_df['ypred'][i] = pred

        ground_truth = np.concatenate((self.Y_training(), self.Y_testing()))
        self._pred_df["yerr"] = ground_truth - self.Y_pred()

    def spectrum(self,
                 step=1e-2,
                 start=0,
                 stop=1,
                 Ts=1.):

        self._f_grid = np.arange(start + step, stop - step, step)

        sigma_e_2 = np.var(self.Y_error())
        Ts = Ts

        self.spectrum = sigma_e_2 * Ts * np.ones_like(self._f_grid)
        for k in range(len(self._f_grid)):
            ak_x_exp = [-self._a_hat[i] *
                        np.exp(-1j * 2. * np.pi * self._f_grid[k] * i * Ts) for i in range(self.p)]
            self.spectrum[k] /= abs(1. + np.sum(ak_x_exp)) ** 2
            
    def display(self):
        
        plt.plot(self.X_training(stop=-self.p), 
                 self.Y_training(start=self.p),
                 c='k')
        
        plt.plot(self.X_testing(), 
                 self.Y_testing(),
                 c='b')
        
        plt.plot(self.Y_pred(),
                 c='r')
        
        plt.show()
