""" Description here

Author: Leonard Berrada
Date: 4 Nov 2015
"""
from data_processing import process_from_file
from tune import optimize_hyperparameters
from predict import predict


class GaussianProcess:
    
    def __init__(self,
                 filename=None,
                 variable=None,
                 use_kernels=None,
                 use_means=None,
                 estimator=None,
                 sequential_mode=None,
                 params=None):
        
        print(variable, use_means, use_kernels, estimator)
        
        self.filename = filename
        self.variable = variable
        self.use_kernels = use_kernels
        self.use_means = use_means
        self.estimator = estimator
        self.sequential_mode = sequential_mode
        self.params = params
        
        self.process_from_file()

        if not hasattr(self.params, "__len__"):
            print("parameters not given, estimation...")
            self.params = self.tune_hyperparameters()
            
    def process_from_file(self):
        
        (self.Xtraining, 
         self.Ytraining, 
         self.Xtesting, 
         self.Ytestingtruth, 
         self.t0) = process_from_file(self.filename,
                                      variable=self.variable)
            
    def tune_hyperparameters(self):
        
        self.params = optimize_hyperparameters(self.Xtraining,
                                               self.Ytraining,
                                               use_kernels=self.use_kernels,
                                               use_means=self.use_means,
                                               estimator=self.estimator,
                                               variable=self.variable)
        
    def predict(self):
        
        predict(Xtraining=self.Xtraining,
                Ytraining=self.Ytraining,
                Xtesting=self.Xtesting,
                params=self.params,
                Ytestingtruth=self.Ytestingtruth,
                use_kernels=self.use_kernels,
                use_means=self.use_means,
                sequential_mode=self.sequential_mode,
                estimator=self.estimator,
                variable=self.variable,
                t0=self.t0,
                show_plot=True)
        
        