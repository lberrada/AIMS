""" Description here

Author: Leonard Berrada
Date: 5 Nov 2015
"""

import sys
import seaborn as sns
sns.set(color_codes=True)

sys.path.append("../../DEI/src/")
from process_data import data_from_file

from forecast import AutoRegression, AutoCorrelation

file_name = "sunspots.mat"

p = 5

args = data_from_file(file_name)

my_ar = AutoRegression(*args, p=p)
my_ar.fit()
my_ar.predict()
my_ar.plot_var('ypred')

my_ac = AutoCorrelation(*args, p=p)
my_ac.fit()
my_ac.predict()
my_ac.plot_var('ypred', show=True)





