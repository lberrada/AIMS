""" Description here

Author: Leonard Berrada
Date: 2 Nov 2015
"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

sys.path.append("../../DEI/src/")

from utils import get_spectral_est
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
my_ac = AutoCorrelation(data_1D, p)

my_ar.fit()
my_ar.predict()
my_ar.plot_prediction()

my_ac.fit()
my_ac.predict()
my_ac.plot_prediction(show=True)

a_hat = my_ac.get("a_hat")
err = my_ac.get("error")

spec_pow_density = get_spectral_est(a_hat, err)
plt.plot(spec_pow_density)
plt.show()
