""" Description here

Author: Leonard Berrada
Date: 22 Oct 2015
"""

import numpy as np

def constant_mean(Xtraining,
                  Ytraining,
                  x):
    
    return np.mean(Ytraining)

def nearest_neighbour_mean(Xtraining,
                           Ytraining,
                           x,
                           k=6):
    
    i = k // 2
    stop = i
    while stop < len(Xtraining) and Xtraining[stop - i] < x:
        stop += 1
    
    start = max(0, stop - k)
    ymean = np.mean(Ytraining[start:stop])
    
    return ymean
