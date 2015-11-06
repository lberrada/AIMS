""" Description here

Author: Leonard Berrada
Date: 5 Nov 2015
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

sys.path.append("../../DEI/src/")
from process_data import data_from_file

from GP_model import GaussianProcess
from forecast import AutoCorrelation, AutoRegression

file_name = "mg.mat"
model="AC"

args = data_from_file(file_name)

Q = 3
use_kernels = "exponential_quadratic* cosine"
# use_kernels = "rational_quadratic"
for _ in range(Q-1):
    use_kernels += "+ exponential_quadratic * cosine"
use_means = "constant"
estimator = "MLE"

data = dict()


(data['xtrain'], data['xtest'], data['ytrain'], data['ytest']) = args
# plt.plot(data['ytrain'])
# plt.show()

if model.lower() == "ar":
    p = 100
    my_ar = AutoRegression(data['ytrain'], p)
    my_ar.fit()
    future = data["ytest"][len(data["ytrain"]):]
    my_ar.predict(future=future)
    my_ar.plot_var("y", lag=p, set_="train")
    plt.plot(data["ytest"][p:])
    my_ar.plot_var('ypred', show=True, c="red")
    
if model.lower() == "ac":
    p = 20
    my_ac = AutoCorrelation(data['ytrain'], p)
    my_ac.fit()
    future = data["ytest"][len(data["ytrain"]):]
    my_ac.predict(future=future)
#     my_ac.plot_var("y", lag=p, set_="train")
#     plt.plot(data["ytest"][p:])
#     my_ac.plot_var('ypred', show=True, c="red")
    my_ac.spectrum()
    
    plt.plot(my_ac._f_grid, my_ac.spectrum)
    peaks_x = signal.find_peaks_cwt(my_ac.spectrum, np.arange(0.1, 1, 0.1), wavelet=signal.ricker)
    peaks_y = my_ac.spectrum[peaks_x]
    plt.plot(my_ac._f_grid[peaks_x], peaks_y, 'ro')
    
    
    sorted_indices = sorted(np.arange(len(peaks_x)), key=lambda k: peaks_y[k], reverse=True)
    sorted_peaks_x = [peaks_x[ind] for ind in sorted_indices]
    frequencies = my_ac._f_grid[sorted_peaks_x[:Q]]
    periods = list(1. / frequencies)
    
    plt.show()
    model = "GP"
    
if model.lower() == "gp":
    my_gp = GaussianProcess(data=data,
                            use_kernels=use_kernels,
                            use_means=use_means,
                            estimator=estimator,
                            sequential_mode=False)
    my_gp.give_scales(periods)
    my_gp.tune_hyperparameters()
    my_gp.predict()
    my_gp.compute_score()
    my_gp.show_prediction()