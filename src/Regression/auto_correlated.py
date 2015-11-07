""" Description here

Author: Leonard Berrada
Date: 5 Nov 2015
"""

import numpy as np
import pandas as pd
import copy

import scipy.linalg

from regression import Regression

class AutoCorrelation(Regression):

    def __init__(self,
                 y,
                 p=5):

        Regression.__init__(self,
                            y)
        self.p = p

    def fit(self):

        self.embed_data()

        Ycentered = self.Y() - np.mean(self.Y())

        r = np.array([Ycentered[
                     :-(self.p + 1)].T.dot(Ycentered[i: i - (self.p + 1)]) for i in range(self.p + 1)])
        r /= r[0]

        self._a_hat = scipy.linalg.solve_toeplitz(r[:-1], r[1:])

    def predict(self, future=None):

        if not hasattr(future, "__len__"):
            self._pred_df['ypred'] = self._emb_matrix.dot(self._a_hat)
            self._pred_df['yerr'] = self.Y(start=self.p) - self.Y_pred()

        else:
            self.predict()
            n_pred = len(future)
            _pred_df = pd.DataFrame()
            _pred_df['ypred'] = np.zeros(n_pred)
            _pred_df['yerr'] = np.zeros(n_pred)
            y = copy.copy(self.Y(start=-self.p))
            for i in range(n_pred):
                pred = self._a_hat[::-1].dot(y)
                y[:-1] = y[1:]
                y[-1] = pred
                _pred_df['ypred'][i] = pred
            _pred_df["yerr"] = future - _pred_df['ypred']

            self._pred_df = self._pred_df.append(_pred_df, ignore_index=True)

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
