""" Description here

Author: Leonard Berrada
Date: 5 Nov 2015
"""

import numpy as np
import pandas as pd
import copy

import matplotlib.pyplot as plt

from regression import RegressionModel


class AutoRegressive(RegressionModel):

    def __init__(self,
                 y,
                 p=5):

        RegressionModel.__init__(self,
                            y)
        self.p = p

    def fit(self):

        self.embed_data()

        self._pseudo_inv = np.linalg.pinv(self._emb_matrix)

        self._a_hat = self._pseudo_inv.dot(self.Y_training(start=self.p))

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
        
    def display(self):
        
        plt.plot(self.X_training(stop=-self.p), 
                 self.Y_training(start=self.p),
                 c='k')
        
        print(self.X_testing())
        
        plt.plot(self.X_testing(), 
                 self.Y_testing(),
                 c='b')
        
        plt.plot(self.Y_pred(),
                 c='r')
        
        plt.show()
        
        
