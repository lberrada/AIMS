""" Data, Estimation & Inference module

main.py : execution script

Author: Leonard Berrada
Date: 19 Oct 2015
"""

from GP_model import predict
from data_processing import process_from_file
from tune import optimize_hyperparameters
from utils import timeit

@timeit
def run(filename=None,
        variable=None,
        use_kernels=None,
        use_means=None,
        estimator=None,
        sequential_mode=None,
        params=None):
    
    print(variable, use_means, use_kernels, estimator)

    Xtraining, Ytraining, Xtesting, Ytestingtruth, t0 = process_from_file(filename,
                                                                          variable=variable)
    
    if not hasattr(params, "__len__"):
        params = optimize_hyperparameters(Xtraining,
                                          Ytraining,
                                          use_kernels=use_kernels,
                                          use_means=use_means,
                                          estimator=estimator,
                                          variable=variable)
    
    predict(Xtraining=Xtraining,
            Ytraining=Ytraining,
            Xtesting=Xtesting,
            params=params,
            Ytestingtruth=Ytestingtruth,
            use_kernels=use_kernels,
            use_means=use_means,
            sequential_mode=sequential_mode,
            estimator=estimator,
            variable=variable,
            t0=t0,
            show_plot=False)
    








