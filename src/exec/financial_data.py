""" Description here

Author: Leonard Berrada
Date: 5 Nov 2015
"""

import sys
sys.path.append("../")

from Regression import AutoRegressive, AutoCorrelation, GaussianProcess, KalmanFilter

from process_data import data_from_file


file_name = "finPredProb.mat"

data_dict = data_from_file(file_name)

model = "GP"
model = "AR"
# model = "AC"
model = "KF"

if model.lower() == 'kf':
    p = 10
    kf = KalmanFilter(data_dict, p)
    kf.fit()
    kf.display()


if model.lower() == "ar":
    p = 200
    my_ar = AutoRegressive(data_dict, p)
    my_ar.fit()
    my_ar.predict()
    my_ar.display()

if model.lower() == "ac":
    p = 50
    my_ac = AutoCorrelation(data_dict, p)
    my_ac.fit()
    my_ac.predict()
    my_ac.display()
    my_ac.spectrum()


if model.lower() == "gp":

    Q = 3
    use_kernels = "exponential_quadratic* cosine"
    for _ in range(Q - 1):
        use_kernels += "+ exponential_quadratic * cosine"
    use_means = "linear + periodic"
    estimator = "MLE"

    my_gp = GaussianProcess(data=data_dict,
                            use_kernels=use_kernels,
                            use_means=use_means,
                            estimator=estimator,
                            sequential_mode=False)

    my_gp.predict()
    my_gp.compute_score()
    my_gp.show_prediction()
