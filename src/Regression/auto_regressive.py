""" Description here

Author: Leonard Berrada
Date: 5 Nov 2015
"""

import numpy as np
import pandas as pd
import copy

from regression import Regression


class AutoRegressive(Regression):

    def __init__(self,
                 y,
                 p=5):

        Regression.__init__(self,
                            y)
        self.p = p

    def fit(self):

        self.embed_data()

        self._pseudo_inv = np.linalg.pinv(self._emb_matrix)

        self._a_hat = self._pseudo_inv.dot(self.Y_training(start=self.p))

    def predict(self, future=None):

        if not hasattr(future, "__len__"):
            self._testing_df['ypred'] = self._emb_matrix.dot(self._a_hat)
            self._testing_df['yerr'] = self.Y_training(
                start=self.p) - self.Y_pred()

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