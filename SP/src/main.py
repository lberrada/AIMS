""" Description here

Author: Leonard Berrada
Date: 2 Nov 2015
"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(color_codes=True)

sys.path.append("../../DEI/src/")

from process_data import import_data
from utils import build_lagged_data, get_pseudo_inverse_from, get_weights_from_autocorr, get_spectral_est

# filename = "finPredProb.mat"
# filename = "co2.mat"
# filename = "mg.mat"
filename = "fXSamples.mat"

my_df = import_data(filename)
data_1D = my_df[1].as_matrix()

plt.plot(data_1D)
p = 5
lagged_data = build_lagged_data(data_1D, p)
pseudo_inv = get_pseudo_inverse_from(lagged_data)
prediction = lagged_data.dot(pseudo_inv.dot(data_1D[p:]))
plt.plot(prediction)

a = get_weights_from_autocorr(data_1D, p)

prediction = lagged_data.dot(a)
plt.plot(prediction)

plt.show()

e = data_1D[p:] - prediction

spec_pow_density = get_spectral_est(a, e)
plt.plot(spec_pow_density)
plt.show()
