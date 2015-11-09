""" Description here

Author: Leonard Berrada
Date: 5 Nov 2015
"""

import sys
sys.path.append("../")

from Regression import AutoRegressive, AutoCorrelation, GaussianProcess, KalmanFilter

from process_data import data_from_file


file_name = "sunspots.mat"

data_dict = data_from_file(file_name)

model = "GP"
# model = "KF"
# model = "AR"
# model = "AC"


if model.lower() == 'kf':
    p = 100
    kf = KalmanFilter(data_dict, p)
    kf.fit()
    kf.display(out="./sun_kf.png")

if model.lower() == "ar":
    p = 50
    my_ar = AutoRegressive(data_dict, p)
    my_ar.fit()
    my_ar.predict()
    my_ar.display(out="./sun_ar.png")

if model.lower() == "ac":
    p = 50
    my_ac = AutoCorrelation(data_dict, p)
    my_ac.fit()
    my_ac.predict()
    my_ac.display(out="./sun_ac.png")
    my_ac.spectrum()


if model.lower() == "gp":

#     Q = 3
#     use_kernels = "exponential_quadratic* cosine"
#     for _ in range(Q - 1):
#         use_kernels += "+ exponential_quadratic * cosine"
        
    use_kernels = "matern_32 + periodic"
    use_means = "constant"
    estimator = "MLE"
    
    params = [0.34, 1., 26.5, 1e-06, 3.18, -2.9]

    my_gp = GaussianProcess(data_dict=data_dict,
                            use_kernels=use_kernels,
                            params=params,
                            use_means=use_means,
                            estimator=estimator,
                            sequential_mode=True)

    my_gp.predict()
    my_gp.compute_score()
    my_gp.show_prediction(out="./sun_gp.png")
