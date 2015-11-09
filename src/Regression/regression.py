""" Description here

Author: Leonard Berrada
Date: 4 Nov 2015
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set(color_codes=True)
except:
    pass


class RegressionModel:

    def __init__(self,
                 data_dict=None,
                 **kwargs):

        assert data_dict.has_key('ytrain')

        self._training_df = pd.DataFrame()

        self.n_training = len(data_dict['ytrain'])
        self._training_df['y'] = data_dict['ytrain']
        if data_dict.has_key('xtrain'):
            self._training_df['x'] = data_dict['xtrain']
        else:
            self._training_df['x'] = np.arange(self.n_training)
        if data_dict.has_key('ytruthtrain'):
            self._training_df['ytruth'] = data_dict['ytruthtrain']

        if data_dict.has_key('ytest'):
            self._testing_df = pd.DataFrame()
            self.n_testing = len(data_dict['ytest'])
            self._testing_df['y'] = data_dict['ytest']
            if data_dict.has_key('xtest'):
                self._testing_df['x'] = data_dict['xtest']
            else:
                self._testing_df['x'] = self.n_training + \
                    np.arange(self.n_testing)
            if data_dict.has_key('ytruthtest'):
                self._testing_df['ytruth'] = data_dict['ytruthtest']
        else:
            self.n_testing = 0

        print("done")
        print("-" * 50)
        print("showing headers for verification...")

        print("Training Data :")
        print(self._training_df.head())

        if self.n_testing:
            print("Testing Data :")
            print(self._testing_df.head())

        self.center_normalize()

        print("NB: data has been centered and normalized")

    def center_normalize(self):

        self.y_mean = np.mean(self.Y_training())
        self.y_std = np.std(self.Y_training())
        
        self._training_df['y'] = (self.Y_training() - self.y_mean) / self.y_std
        
        if self.n_testing:
            self._testing_df['y'] = (
                self.Y_testing() - self.y_mean) / self.y_std

        if hasattr(self._training_df, "ytruth"):
            self._training_df['ytruth'] = (
                self.Y_truth_training() - self.y_mean) / self.y_std
            self._testing_df['ytruth'] = (
                self.Y_truth_testing() - self.y_mean) / self.y_std

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
        
        if not self.n_testing:
            return []

        if hasattr(indices, "__len__"):
            return self._testing_df.y.values[indices]
        else:
            return self._testing_df.y.values[start:stop]

    def Y_pred(self,
               indices=None,
               start=None,
               stop=None):

        if hasattr(indices, "__len__"):
            return self._pred_df.ypred.values[indices]
        else:
            return self._pred_df.ypred.values[start:stop]

    def Y_error(self,
                indices=None,
                start=None,
                stop=None):

        if hasattr(indices, "__len__"):
            return self._pred_df.yerr.values[indices]
        else:
            return self._pred_df.yerr.values[start:stop]

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

    def embed_data(self):

        n = self.n_training - self.p
        self._emb_matrix = np.zeros((n, self.p))

        for k in range(self.p):
            self._emb_matrix[:, k] = self.Y_training(start=self.p - 1 - k,
                                                     stop=self.p - 1 - k + n)

    def fit(self):

        raise NotImplementedError("method should be overwritten")

    def predict(self, testing_data):

        raise NotImplementedError("method should be overwritten")

    def get(self, attr_name):

        return getattr(self, attr_name)

    def plot_attr(self, attr_name, show=False, **kwargs):

        attr_to_plot = getattr(self, attr_name)

        plt.plot(attr_to_plot, **kwargs)

        if show:
            plt.show()

    def plot_var(self, var_name, set_="", lag=None, show=False, **kwargs):

        var_to_plot = getattr(self, var_name)

        plt.plot(var_to_plot(start=lag), **kwargs)

        if show:
            plt.show()
            
    def display(self, out=""):

        plt.plot(self.X_training(stop=-self.p),
                 self.Y_training(start=self.p),
                 c='k',
                 ms=4)
        
        try:
            plt.plot(self.X_testing() - self.p,
                     self.Y_testing(),
                     c='b',
                     ms=4)
        except:
            pass

        plt.plot(self.Y_pred(),
                 c='r',
                 ms=4)
        
        plt.plot(self.Y_error(),
                 c='g',
                 alpha=0.5,
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
