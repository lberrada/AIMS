""" Description here

Author: Leonard Berrada
Date: 4 Nov 2015
"""

import matplotlib.pyplot as plt

from utils import embed_mat_from, pseudo_inverse, weights_auto_corr

class Regresssion:
    
    def __init__(self, 
                 training_data, 
                 testing_data):
        
        self.training_data = training_data
        self.testing_data = testing_data
        
    def fit(self):
        
        raise NotImplementedError("method should be overwritten")
    
    def predict(self, testing_data):
        
        raise NotImplementedError("method should be overwritten")
    
    def plot_prediction(self, show=False):
        
        plt.plot(self.prediction)
        
        if show:
            plt.show()
            
    def get(self, attr_name):
        
        return getattr(self, attr_name)
        

    
class AutoRegression(Regresssion):
    
    def __init__(self, 
                 training_data, 
                 p):
        
        testing_data = training_data[p:]
        Regresssion.__init__(self, 
                             training_data, 
                             testing_data)
        self.p = p
        
    def fit(self):
        
        self.embedded_mat = embed_mat_from(self.training_data, 
                                           self.p)
        self.pseudo_inv  = pseudo_inverse(self.embedded_mat)
        
    def predict(self):
        
        self.prediction = self.embedded_mat.dot(self.pseudo_inv.dot(self.testing_data))
        self.error = self.testing_data - self.prediction
        
class AutoCorrelation(Regresssion):
    
    def __init__(self,
                 training_data,
                 p):
        
        testing_data = training_data[p:]
        Regresssion.__init__(self, 
                             training_data, 
                             testing_data)
        self.p = p
        
    def fit(self):
        
        self.embedded_mat = embed_mat_from(self.training_data, 
                                           self.p)
        self.a_hat = weights_auto_corr(self.training_data,
                                       self.p)
        
    def predict(self):
        
        self.prediction=self.embedded_mat.dot(self.a_hat)
        self.error = self.testing_data - self.prediction
        
        
    