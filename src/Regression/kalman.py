""" Description here

Author: Leonard Berrada
Date: 6 Nov 2015
"""

import numpy as np
import pandas as pd
import copy
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

    def fit(self):

        self._pred_df = pd.DataFrame()
        n_pred = self.n_training + self.n_testing - self.p
        self._pred_df['ypred'] = np.zeros(n_pred)
        self._pred_df['yerr'] = np.zeros(n_pred)

        self.embed_data()

        a_hat = np.array([1.] + [0.] * (self.p - 1)).reshape(self.p, 1)
        P_up = np.eye(self.p)
        Q = 0.1
        R = 0.1

        for i in range(self.n_training - self.p):

            # prediction step
            P_pred = P_up + Q

            # aux variables
            H = self.Y_training(start=i, stop=i + self.p).reshape(1, self.p)
            obs = float(self.Y_training([i + self.p]))
            nu = obs - float(H.dot(a_hat))
            S = float(H.dot(P_pred).dot(H.T)) + R
            K = P_pred.dot(H.T) * 1. / S
            
            # predict data value
            self._pred_df['ypred'][i] = float(H.dot(a_hat))

            # update step
            a_hat += K * nu
            P_up = (np.eye(self.p) - K.dot(H)).dot(P_pred)
            
        self._a_hat = a_hat.flatten()
        
        y = copy.copy(self.Y_training(start=-self.p))
        for i in range(self.n_training - self.p, n_pred):
            pred = self._a_hat[::-1].dot(y)
            y[:-1] = y[1:]
            y[-1] = pred
            self._pred_df['ypred'][i] = pred

        ground_truth = np.concatenate((self.Y_training(start=self.p), self.Y_testing()))
        self._pred_df["yerr"] = ground_truth - self.Y_pred()
        

    
