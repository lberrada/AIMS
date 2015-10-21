""" Data, Estimation & Inference module

main.py : execution script

Author: Leonard Berrada
Date: 19 Oct 2015
"""

from data_processing import process_from_file
from GP_model import predict
from tune import optimize_hyperparameters


def run(filename,
        variable,
        use_kernel,
        estimator,
        sequential_mode):

    Xtraining, Ytraining, Xtesting, Ytestingtruth, t0 = process_from_file(filename,
                                                                      variable=variable)
    
    params = optimize_hyperparameters(Xtraining,
                                      Ytraining,
                                      use_kernel=use_kernel,
                                      estimator=estimator,
                                      variable=variable)

    predict(Xtraining=Xtraining,
            Ytraining=Ytraining,
            Xtesting=Xtesting,
            params=params,
            Ytestingtruth=Ytestingtruth,
            use_kernel=use_kernel,
            sequential_mode=sequential_mode,
            estimator=estimator,
            variable=variable,
            t0=t0,
            show_plot=False)
    

filename = 'sotonmet.txt'

sequential_mode = False

for estimator in ["MLE", "MAP"]:
    for variable in ["tide", "temperature"]:
        for use_kernel in ["gaussian", "gaussian_2", "locally_periodic"]:
            run(filename,
                variable,
                use_kernel,
                estimator,
                sequential_mode)
        






