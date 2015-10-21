""" Data, Estimation & Inference module

main.py : execution script

Author: Leonard Berrada
Date: 19 Oct 2015
"""

from data_processing import process_from_file
from GP_model import predict
from tune import optimize_hyperparameters

filename = 'sotonmet.txt'

variable = 'temperature'
# use_kernel='gaussian'
# use_kernel='gaussian_2'
use_kernel='locally_periodic'

Xtraining, Ytraining, Xtesting, Ytestingtruth = process_from_file(filename,
                                                                  variable=variable)

params = optimize_hyperparameters(Xtraining,
                                  Ytraining,
                                  use_kernel=use_kernel)

predict(Xtraining=Xtraining,
        Ytraining=Ytraining,
        Xtesting=Xtesting,
        params=params,
        Ytestingtruth=Ytestingtruth,
        use_kernel=use_kernel)






