""" Description here

Author: Leonard Berrada
Date: 22 Oct 2015
"""

import numpy as np


def constant_mean(Xtesting,
                  params,
                  **kwargs):

    alpha = params.pop(0)

    if not hasattr(Xtesting, "__len__"):
        return alpha

    return alpha * np.ones_like(Xtesting)


def linear_mean(Xtesting,
                params,
                **kwargs):

    alpha = params.pop(0)
    beta = params.pop(0)

    return alpha + beta * Xtesting


def periodic_mean(Xtesting,
                  params,
                  **kwargs):

    scale = params.pop(0)
    period = params.pop(0)

    return scale * np.sin(2. * np.pi * Xtesting / period)


def quadratic_mean(Xtesting,
                   params,
                   **kwargs):

    alpha = params.pop(0)
    beta = params.pop(0)
    gamma = params.pop(0)

    return alpha + beta * Xtesting + gamma * np.power(Xtesting, 2)
