""" Description here

Author: Leonard Berrada
Date: 4 Nov 2015
"""

import numpy as np
import pandas as pd
import copy
import csv
import matplotlib.pyplot as plt

from tune import optimize_hyperparameters
from predict import predict


class GaussianProcess:
    
    def __init__(self,
                 data=None,
                 filename=None,
                 variable=None,
                 use_kernels=None,
                 use_means=None,
                 estimator=None,
                 sequential_mode=None,
                 params=None):
        
        print(variable, use_means, use_kernels, estimator)
        
        self.variable = variable
        self.use_kernels = use_kernels
        self.use_means = use_means
        self.estimator = estimator
        self.sequential_mode = sequential_mode
        self.params = params
        
        if data:
            self._training_df = pd.DataFrame()
            self._testing_df = pd.DataFrame()
            self._training_df['x'] = data['xtrain']
            self._training_df['y'] = data['ytrain']
            self._testing_df['x'] = data['xtest']
            self._testing_df['y'] = data['ytest']
        else:
            self.filename = filename
            self.process_from_file()

        if not hasattr(self.params, "__len__"):
            print("parameters not given, estimation...")
            self.params = self.tune_hyperparameters()
            
    def process_from_file(self):
        
        print("importing data in dataframe...")
        my_dataframe = pd.read_csv("../data/" + self.filename)
        if self.variable == 'tide':
            my_dataframe.rename(columns={'Tide height (m)': 'y',
                                         'True tide height (m)': 'ytruth',
                                         'Reading Date and Time (ISO)': 'x'},
                                inplace=True)
        elif self.variable == 'temperature':
            my_dataframe.rename(columns={'Air temperature (C)': 'y',
                                         'True air temperature (C)': 'ytruth',
                                         'Reading Date and Time (ISO)': 'x'},
                                inplace=True)
        else:
            raise ValueError("Wrong predictor argument (%s), should be 'tide' or 'temperature'" % self.variable)
        
        my_dataframe['x'] = pd.to_datetime(my_dataframe['x'])
        t0 = copy.copy(my_dataframe['x'].ix[0])
        self.t0 = np.datetime64(t0)
        
        my_dataframe['x'] -= t0
        my_dataframe['x'] = my_dataframe['x'].apply(
            lambda x: x / np.timedelta64(5, 'm'))
    
        print("done")
        print('-' * 50)
        
        print("creating training and testing dataframes...")
        
        testing_indices = my_dataframe['y'].index[
            my_dataframe['y'].apply(np.isnan)]
    
        n_rows = len(my_dataframe.index)
        training_indices = [i for i in range(n_rows) if i not in testing_indices]
    
        self._training_df = my_dataframe[['x', 'y', 'ytruth']].ix[training_indices]
        self._testing_df = my_dataframe[['x', 'y', 'ytruth']].ix[testing_indices]
        
        self.n_training = len(training_indices)
        self.n_testing = len(testing_indices)
        
        print("done")
        print('-' * 50)
    
        print("Showing headers for verification:")
        print('\nTraining data :')
        print(self._training_df.head())
        print('\nTesting data :')
        print(self._testing_df.head())
        print('-' * 50)
    
    def X_training(self, 
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._training_df.x.values[indices]
        else:
            return self._training_df.x.values[start:stop]
    
    def X_testing(self, 
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._testing_df.x.values[indices]
        else:
            return self._testing_df.x.values[start:stop]
    
    def Y_training(self, 
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._training_df.y.values[indices]
        else:
            return self._training_df.y.values[start:stop]
    
    def Y_testing(self, 
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._testing_df.y.values[indices]
        else:
            return self._testing_df.y.values[start:stop]
    
    def Y_truth_training(self, 
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._training_df.ytruth.values[indices]
        else:
            return self._training_df.ytruth.values[start:stop]
    
    def Y_truth_testing(self, 
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._testing_df.ytruth.values[indices]
        else:
            return self._testing_df.ytruth.values[start:stop]
    
    def Y_pred_mean(self, 
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._testing_df.ymean.values[indices]
        else:
            return self._testing_df.ymean.values[start:stop]
        
    def Y_pred_var(self, 
                   indices=None,
                   start=None,
                   stop=None):
        
        if hasattr(indices, "__len__"):
            return self._testing_df.yvar.values[indices]
        else:
            return self._testing_df.yvar.values[start:stop]
        
    def tune_hyperparameters(self):
        
        self.params = optimize_hyperparameters(self)
        
    def predict(self):
        
        predict(self,
                show_plot=True)
        
    def compute_score(self,
                      out=None):
        
        print("-"*50)
        print('computing score...')
     
        ssres = np.sum(np.power(self.Y_pred_mean() - self.Y_truth_testing(), 2))
        print("RMS :", np.sqrt(ssres) / self.n_testing)
        sstot = np.sum(np.power(self.Y_pred_mean() - np.mean(self.Y_pred_mean()), 2))
        self.r2 = 1 - ssres / sstot
        print('r2 :', self.r2)
        print("done")
        print("-"*50)
        
        if out:
            try:     
                with open(out, 'a', newline='') as csvfile:
                    my_writer = csv.writer(csvfile, delimiter='\t',
                                           quoting=csv.QUOTE_MINIMAL)
                    my_writer.writerow(['r2', round(self.r2, 3)])
            except:
                print("could not write results in %s, please make sure directory exists" % out)
                    
    def show_prediction(self, 
                        out=None):
        
        print("creating plot...")
     
        Y_std = np.sqrt(self.Y_pred_var())
     
        Ttesting = np.array([self.t0] * self.n_testing, dtype='datetime64')
        Ttesting += np.array([np.timedelta64(int(x) * 5, 'm') for x in self.X_testing()], dtype=np.timedelta64)
         
        plt.fill_between(Ttesting,
                         self.Y_pred_mean() - 1.96 * Y_std,
                         self.Y_pred_mean() + 1.96 * Y_std,
                         color='red',
                         alpha=0.3)
         
        plt.fill_between(Ttesting,
                         self.Y_pred_mean() - Y_std,
                         self.Y_pred_mean() + Y_std,
                         color='red',
                         alpha=0.3)
         
        plt.plot(Ttesting,
                 self.Y_truth_testing(),
                 'ko-',
                 ms=4,
                 alpha=0.7)
         
        plt.plot(Ttesting,
                 self.Y_pred_mean(),
                 'ro',
                 ms=4)
         
        if out:
            try:
                plt.savefig(out,
                            transparent=False,
                            dpi=200,
                            bbox_inches='tight')
            except:
                print("could not save plot in %s, please make sure directory exists" % out)
     
         
        plt.show()
    
             
        print("done")


        
        