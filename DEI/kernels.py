""" Data, Estimation & Inference module

kernels.py : definition of kernels in a dictionary

Author: Leonard Berrada
Date: 19 Oct 2015
"""

kernels_dict = dict()

kernels_dict['gaussian'] = gaussian_kernel


def gaussian_kernel(X=None,
                    Xprime=None,
                    sigma_f=None,
                    sigma_n=None,
                    l=None,
                    epsilon=1e-6):

    assert(X != None and sigma != None and l != None)

    same_x = np.isclose(X, Xprime, rtol=1e-3)

    ones_mat = np.ones_like(X).T
    X_matrix = np.dot(X, ones_mat)
    Xprime_matrix = np.dot(Xprime, ones_mat)

    same_x = np.isclose(X_matrix, Xprime_matrix, rtol=1e-3)

    K = sigma_f * np.exp(- np.power(X_matrix - Xprime_matrix, 2) / (2 * l**2))
    K[same_x] += sigma_n
    
    
