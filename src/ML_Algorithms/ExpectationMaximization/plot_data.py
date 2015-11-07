""" converted from Matlab code
source: http://www.robots.ox.ac.uk/~fwood/teaching/AIMS_CDT_ML_2015/homework/HW_2_em/
"""

import sklearn.decomposition
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_data(X, labels=None):
    """% utility function to plot the data based on the first 2 principle
    % components of the data."""

    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(X)
    Y = pca.transform(X)
    
    if hasattr(labels, "__len__"):
        n_labels = max(labels) + 1
        colors = cm.rainbow(np.linspace(0, 1, n_labels))
        colored_labeled = [colors[i] for i in labels]
        plt.scatter(Y[:, 0], Y[:, 1], c=colored_labeled)
    
    else:
        plt.scatter(Y[:, 0], Y[:, 1])
        
    plt.show()
