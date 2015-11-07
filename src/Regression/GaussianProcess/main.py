""" Data, Estimation & Inference module

main.py : execution script

Author: Leonard Berrada
Date: 19 Oct 2015
"""

from utils import timeit
from src.GP_model import GaussianProcess

@timeit
def run(filename=None,
        variable=None,
        use_kernels=None,
        use_means=None,
        estimator=None,
        sequential_mode=None,
        params=None):
    

    my_gp = GaussianProcess(filename=filename,
                            variable=variable,
                            use_kernels=use_kernels,
                            use_means=use_means,
                            estimator=estimator,
                            sequential_mode=sequential_mode,
                            params=params)
    
    my_gp.predict()
    my_gp.compute_score()
    my_gp.show_prediction()
    









