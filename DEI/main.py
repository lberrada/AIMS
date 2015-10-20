""" Data, Estimation & Inference module

main.py : execution script

Author: Leonard Berrada
Date: 19 Oct 2015
"""

import numpy as np

from data_processing import process_from_file
from kernels import gaussian_kernel, optimize_hp_gaussian

filename = 'sotonmet.txt'

training_df, testing_df = process_from_file(filename,
                                            variable='temperature')

X = training_df.t.values
Y = training_df.y.values
Y -= np.mean(Y)
Xstar = testing_df.t.values
truth = testing_df.ytruth.values
truth -= np.mean(truth)


[sigma_f, sigma_n, l] = optimize_hp_gaussian(X,
                                             Y)

y_mean, y_var = gaussian_kernel(X=X,
                                Y=Y,
                                xstar=Xstar,
                                sigma_f=sigma_f,
                                sigma_n=sigma_n,
                                l=l,
                                truth=truth)






