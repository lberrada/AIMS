""" Description here

Author: Leonard Berrada
Date: 5 Nov 2015
"""

import sys
import seaborn as sns
sns.set(color_codes=True)

sys.path.append("../../DEI/src/")
from process_data import data_from_file

from GP_model import GaussianProcess

file_name = "mg.mat"

ix = 1
p = 5

args = data_from_file(file_name,
                      ix=ix)

use_kernels = "rational_quadratic"
use_means = "constant_mean"
estimator = "MLE"

my_gp = GaussianProcess(data=args,
                        use_kernels=use_kernels,
                        use_means=use_means,
                        estimator=estimator,
                        sequential_mode=False)
    
my_gp.predict()
my_gp.compute_score()
my_gp.show_prediction()





