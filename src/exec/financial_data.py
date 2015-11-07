""" Description here

Author: Leonard Berrada
Date: 5 Nov 2015
"""

import sys
sys.path.append("../")

import seaborn as sns
sns.set(color_codes=True)

from process_data import data_from_file

from Regression.forecast import AutoRegressive, AutoCorrelation

file_name = "finPredProb.mat"

ix = 1
p = 5

args = data_from_file(file_name,
                      ix=ix)

my_ar = AutoRegressive(*args, p=p)
my_ar.fit()
my_ar.predict()
my_ar.plot_var('ypred')

my_ac = AutoCorrelation(*args, p=p)
my_ac.fit()
my_ac.predict()
my_ac.plot_var('ypred', show=True)



