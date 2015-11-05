""" Description here

Author: Leonard Berrada
Date: 5 Nov 2015
"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

sys.path.append("../../DEI/src/")
from process_data import data_from_file

from GP_model import GaussianProcess

file_name = "co2.mat"

ix = 1
p = 5

args = data_from_file(file_name,
                      ix=ix)

Q = 2
use_kernels = "periodic"
# use_kernels = "rational_quadratic"
for _ in range(Q-1):
    use_kernels += "+ exponential_quadratic * periodic"
use_means = "linear + periodic"
estimator = "MLE"

data = dict()


(data['xtrain'], data['xtest'], data['ytrain'], data['ytest']) = args
# plt.plot(data['ytrain'])
# plt.show()

my_gp = GaussianProcess(data=data,
                        use_kernels=use_kernels,
                        use_means=use_means,
                        estimator=estimator,
                        sequential_mode=False)
    
my_gp.predict()
my_gp.compute_score()
my_gp.show_prediction()