""" Description here

Author: Leonard Berrada
Date: 20 Oct 2015
"""

import numpy as np

X1 = 10 * np.arange(5)
X2 = 10 * np.arange(3)

D = X1[None, :] - X2[:, None]

print(D)
