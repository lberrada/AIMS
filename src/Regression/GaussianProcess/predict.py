""" Description here
 
Author: Leonard Berrada
Date: 21 Oct 2015
"""

import numpy as np


def predict(self,
            show_plot=True):

    print("predicting data...")

    mu_training = self.compute_mu(Xtesting=self.X_training())
    K = self.compute_K(XX=self.X_training())

    L = np.linalg.cholesky(K)

    Ypredicted = np.zeros(self.n_testing)
    Yvar = np.zeros(self.n_testing)
    Y_centered = self.Y_training() - mu_training
    index = 0 if self.sequential_mode else None

    mu_testing = self.compute_mu(Xtesting=self.X_testing())

    for i in range(self.n_testing):

        xstar = self.X_testing([i])

        if self.sequential_mode:
            if i == 0:
                continue
            while index < self.n_training and self.X_training([index]) < xstar:
                index += 1
            L = np.linalg.cholesky(K[:index, :index])

        Xstar = xstar * np.ones(len(L))
        Ks = self.compute_K(X1=self.X_training(stop=index),
                            X2=Xstar)

        Kss = self.compute_K(X1=xstar,
                             X2=xstar)

        aux = np.linalg.solve(L, Ks.T)
        KsxK_inv = np.linalg.solve(L.T, aux).T

        Ypredicted[i] = KsxK_inv.dot(Y_centered[:index]) + mu_testing[i]
        Yvar[i] = Kss - KsxK_inv.dot(Ks.T)

    self._testing_df['ymean'] = Ypredicted
    self._testing_df['yvar'] = Yvar

    print("done")
