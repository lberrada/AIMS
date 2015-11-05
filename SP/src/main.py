""" Description here

Author: Leonard Berrada
Date: 2 Nov 2015
"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

sys.path.append("../../DEI/src/")

from process_data import import_data
from regression import AutoRegression, AutoCorrelation

# filename = "finPredProb.mat"
# filename = "co2.mat"
# filename = "mg.mat"
filename = "fXSamples.mat"

my_df = import_data(filename)
data_1D = my_df[1].as_matrix()

plt.plot(data_1D)
p = 5


my_ar = AutoRegression(data_1D, p)
my_ar.fit()
my_ar.predict()
my_ar.plot_var('ypred')

my_ac = AutoCorrelation(data_1D, p)
my_ac.fit()
my_ac.predict()
my_ac.plot_var('ypred', show=True)

my_ac.spectrum()
my_ac.plot_attr('spectrum', show=True)


