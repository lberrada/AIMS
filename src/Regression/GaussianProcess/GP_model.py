""" Description here

Author: Leonard Berrada
Date: 4 Nov 2015
"""

import sys
sys.path.append("../../")

from Regression import RegressionModel

import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(color_codes=True)

import scipy.stats

from utils import timeit
from tune import optimize_hyperparameters
from predict import predict
from params import get_params
from kernels import *
from means import *


class GaussianProcess(RegressionModel):

    def __init__(self,
                 data_dict=None,
                 variable=None,
                 use_kernels=None,
                 use_means=None,
                 estimator=None,
                 sequential_mode=None,
                 params=None):

        print("")
        print("Gaussian Process with :")
        print("- mean : %s" % use_means)
        print("- kernel : %s" % use_kernels)
        print("- estimator : %s \n" % estimator)

        self.variable = variable
        self.use_kernels = use_kernels
        self.use_means = use_means
        self.estimator = estimator
        self.sequential_mode = sequential_mode
        self.params = params

        RegressionModel.__init__(self, data_dict)

        self.process_strings()
        
        self.init_params()

    def init_params(self):
        
        if not hasattr(self.params, "__len__"):
            print("hyper-parameters not given, random initialization...")
            my_params = get_params(self.use_kernels,
                                   self.use_means)

            mean_params = my_params["means"]
            std_params = my_params["stds"]
            use_log = my_params["use_log"]

            self.params = list(np.zeros_like(mean_params))
            for k in range(len(self.params)):
                if use_log[k]:
                    self.params[k] = scipy.stats.lognorm.rvs(1,
                                                             loc=mean_params[
                                                                 k],
                                                             scale=std_params[k])
                else:
                    self.params[k] = scipy.stats.norm.rvs(loc=mean_params[k],
                                                          scale=std_params[k])
                    
            self.update_scales()
                    
            self.tune_hyperparameters()

    def process_strings(self):

        kernels_list = ["cosine", "exponential_quadratic",
                        "rational_quadratic", "matern_12", "matern_32", "periodic"]
        self.kernels_eval = self.use_kernels
        for kernel_name in kernels_list:
            self.kernels_eval = self.kernels_eval.replace(
                kernel_name, kernel_name + "_kernel(X1, X2, params)")

        means_list = ["constant", "linear", "periodic", "quadratic"]
        self.means_eval = self.use_means
        for mean_name in means_list:
            self.means_eval = self.means_eval.replace(
                mean_name, mean_name + "_mean(Xtesting, params)")

    def update_scales(self, nyquist_freq=0.5):

        print(
            "initial parameters updated with uniform drawn within nyquist frequency")

        params = get_params(self.use_kernels,
                            self.use_means)

        for k in range(len(self.params)):
            curr_name = params["names"][k]
            if "period" in curr_name or "scale" in curr_name:
                freq = scipy.stats.uniform.rvs() * nyquist_freq
                self.params[k] = 1. / freq

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
    
    def compute_K(self,
                  X1=None,
                  X2=None,
                  XX=None):

        if hasattr(XX, "__len__"):
            X1 = XX[:, None]
            X2 = XX[None, :]

        params = copy.copy(self.params)
        sigma_n = params.pop(0)

        K = eval(self.kernels_eval)

        if not hasattr(K, "__len__"):
            K += sigma_n ** 2

        elif len(K.shape) == 2 and K.shape[0] == K.shape[1]:
            n = len(K)
            K += sigma_n ** 2 * np.eye(n)

        return K
    
    def compute_mu(self,
                   Xtesting=None):

        params = copy.copy(self.params)

        mu = eval(self.means_eval)

        return mu
    @timeit
    def predict(self):

        predict(self,
                show_plot=True)

    def compute_score(self,
                      out=None):

        print("-" * 50)
        print('computing score...')
        
        if hasattr(self._testing_df, 'ytruth'):
            ground_truth = self.Y_truth_testing()
        else:
            ground_truth = self.Y_testing()

        ssres = np.sum(
            np.power(self.Y_pred_mean() - ground_truth, 2))
        print("RMS :", np.sqrt(ssres) / self.n_testing)
        sstot = np.sum(
            np.power(self.Y_pred_mean() - np.mean(self.Y_pred_mean()), 2))
        self.r2 = 1 - ssres / sstot
        print('r2 :', self.r2)
        print("done")
        print("-" * 50)

        if out:
            try:
                with open(out, 'a', newline='') as csvfile:
                    my_writer = csv.writer(csvfile, delimiter='\t',
                                           quoting=csv.QUOTE_MINIMAL)
                    my_writer.writerow(['r2', round(self.r2, 3)])
            except:
                print(
                    "could not write results in %s, please make sure directory exists" % out)

    def show_prediction(self,
                        out=""):

        print("creating plot...")

        Y_std = np.sqrt(self.Y_pred_var())

        try:
            Ttesting = np.array([self.t0] * self.n_testing, dtype='datetime64')
            Ttesting += np.array([np.timedelta64(int(x) * 5, 'm')
                                  for x in self.X_testing()], dtype=np.timedelta64)
        except:
            Ttesting = self.X_testing()

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

        if hasattr(self._training_df, 'ytruth'):
            plt.plot(Ttesting,
                     self.Y_truth_testing(),
                     'k-',
                     ms=4,
                     alpha=0.7)
        else:
            plt.plot(self.X_training(),
                     self.Y_training(),
                     'k-',
                     ms=4)
            
        plt.plot(self.X_testing(),
                 self.Y_testing(),
                 'b-',
                 ms=4)

        plt.plot(Ttesting,
                 self.Y_pred_mean(),
                 'r-',
                 ms=4)

        if out:
            try:
                plt.savefig(out,
                            transparent=False,
                            dpi=200,
                            bbox_inches='tight')
            except:
                print(
                    "could not save plot in %s, please make sure directory exists" % out)

        plt.show()

        print("done")
