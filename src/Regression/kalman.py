""" Description here

Author: Leonard Berrada
Date: 6 Nov 2015
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regression import RegressionModel


class KalmanFilter(RegressionModel):
    """ simple Linear Gaussian Kalman Filter"""

    def __init__(self,
                 data,
                 p):

        RegressionModel.__init__(self,
                                 data)

        self.p = p

    def apply(self):

        beta = 1.

        self._pred_df = pd.DataFrame()
        self._pred_df['ypred'] = np.zeros(self.n_training - self.p)
        self._pred_df['yerr'] = np.zeros(self.n_training - self.p)

        self.embed_data()

        a_hat = np.array([1.] + [0.] * (self.p - 1)).reshape(self.p, 1)
        P_up = np.eye(self.p)
        Q = 0.01
        R = 0.01

        for i in range(self.n_training - self.p):

            # prediction step
            P_pred = P_up + Q

            # aux variables
            H = self.Y_training(start=i, stop=i + self.p).reshape(1, self.p)
            obs = float(self.Y_training([i]))
            nu = obs - float(H.dot(a_hat))
            S = float(H.dot(P_pred).dot(H.T)) + R
            K = P_pred.dot(H.T) * 1. / S

            # update step
            a_hat += K * nu
            P_up = (np.eye(self.p) - K.dot(H)).dot(P_pred)

            # predict data value
            self._pred_df['ypred'][i] = float(H.dot(a_hat))

        self._pred_df['yerr'] = self.Y_training(start=self.p) - self.Y_pred()

    def display(self):

        plt.plot(self.X_training(stop=-self.p),
                 self.Y_training(start=self.p),
                 c='k')

        plt.plot(self.X_training(stop=-self.p),
                 self.Y_pred(),
                 c='r')

        plt.show()
