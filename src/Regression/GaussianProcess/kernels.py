""" Data, Estimation & Inference module

kernels.py : definition of kernels in a dictionary

Author: Leonard Berrada
Date: 19 Oct 2015
"""

import numpy as np


def exponential_quadratic_kernel(X1=None,
                                 X2=None,
                                 params=None,
                                 **kwargs):

    sigma_f = params.pop(0)
    scale = params.pop(0)

    D = X1 - X2
    K = sigma_f ** 2 * np.exp(-np.power(D, 2) / (2 * scale ** 2))

    return K


def periodic_kernel(X1=None,
                    X2=None,
                    params=None,
                    **kwargs):

    sigma_f = params.pop(0)
    period = params.pop(0)

    D = X1 - X2
    K = sigma_f ** 2 * np.exp(-2 * np.power(np.sin(2 * np.pi * D / period), 2))

    return K


def cosine_kernel(X1=None,
                  X2=None,
                  params=None,
                  **kwargs):

    period = params.pop(0)

    D = X1 - X2
    K = np.cos(2 * np.pi * D / period)

    return K


def rational_quadratic_kernel(X1=None,
                              X2=None,
                              params=None,
                              **kwargs):

    sigma_f = params.pop(0)
    scale = params.pop(0)
    nu = params.pop(0)

    D = X1 - X2
    K = sigma_f ** 2 * \
        np.power(1. + 1. / (2. * nu) * np.power(D / scale, 2), -nu)

    return K


def matern_12_kernel(X1=None,
                     X2=None,
                     params=None,
                     **kwargs):

    sigma_f = params.pop(0)
    scale = params.pop(0)

    D = np.abs(X1 - X2)

    K = sigma_f ** 2 * np.exp(-D / scale)

    return K


def matern_32_kernel(X1=None,
                     X2=None,
                     params=None,
                     **kwargs):

    sigma_f = params.pop(0)
    scale = params.pop(0)

    D = np.abs(X1 - X2)

    K = sigma_f ** 2 * (1. + np.sqrt(3) * D / scale) * \
        np.exp(-np.sqrt(3) * D / scale)

    return K
