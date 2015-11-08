""" Description here

Author: Leonard Berrada
Date: 6 Nov 2015
"""

import numpy as np
import pandas as pd

from regression import RegressionModel


class KalmanFilter(RegressionModel):

    def __init__(self,
                 data,
                 p):

        RegressionModel.__init__(self,
                                 data)

        self.p = p


    def fit(self):
        
        self._pred_df = pd.DataFrame()
        n_pred = self.n_training + self.n_testing
        self._pred_df['ypred'] = np.zeros(n_pred)
        self._pred_df['yerr'] = np.zeros(n_pred)
        
        Q = np.eye(self.p)

        for i in range(self.n_training):
            pass
            