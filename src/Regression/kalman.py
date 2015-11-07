""" Description here

Author: Leonard Berrada
Date: 6 Nov 2015
"""

import numpy as np

from regression import RegressionModel

class KalmanFilter(RegressionModel):
    
    def __init__(self,
                 data):
    
        RegressionModel.__init__(self, 
                          data)
        
        self.dim = -1
        
        # state transition model
        self.F = np.eye(self.dim)
        
        # control input
        self.B = np.eye(self.dim)
        
        # observation model
        self.H = np.eye(self.dim)
        
    def predict(self, future=None):
        
        if not hasattr(future, "__len__"):
            n_pred = len(self.Y())-1
            self._pred_df['ypred'] = np.zeros(n_pred)
            self._pred_df['yerr'] = np.zeros(n_pred)
            
            old_x = np.random.normal(self.dim)
            for k in range(n_pred):
                pred_x = self.F.dot(old_x) + self.B.dot(self.Y(k))
                pred_cov = self.F.dot()
                
        
        