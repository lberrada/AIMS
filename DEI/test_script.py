""" Description here

Author: Leonard Berrada
Date: 20 Oct 2015
"""

import numpy as np

X = 10 * np.random.random(5)

D = X[None, :] - X[:, None]

print(D)
sigma_f = 1.
l = 1.
K = np.round(sigma_f ** 2 * np.exp(-np.power(D, 2) / (2 * l ** 2)), 2)
print(K)
print(np.linalg.det(K))
