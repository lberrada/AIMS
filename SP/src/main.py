""" Description here

Author: Leonard Berrada
Date: 2 Nov 2015
"""

import sys
import seaborn as sns
sns.set(color_codes=True)

sys.path.append("../../DEI/src/")
from process_data import process_file

from regression import AutoRegression, AutoCorrelation

# filename = "finPredProb.mat"
# filename = "co2.mat"
# filename = "mg.mat"
file_name = "fXSamples.mat"

ix = 1
p = 5

xtrain, xtest, ytrain, ytest = process_file(file_name, 
                                            p=p, 
                                            ix=ix)

my_ar = AutoRegression(xtrain, xtest, ytrain, ytest, p)
my_ar.fit()
my_ar.predict()
my_ar.plot_var('ypred')

my_ac = AutoCorrelation(xtrain, xtest, ytrain, ytest, p)
my_ac.fit()
my_ac.predict()
my_ac.plot_var('ypred', show=True)

my_ac.spectrum()
my_ac.plot_attr('spectrum', show=True)


