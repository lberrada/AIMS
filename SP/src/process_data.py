""" Description here

Author: Leonard Berrada
Date: 2 Nov 2015
"""

import scipy.io
import numpy as np

def process_file(file_name, **kwargs):
        
    print("loading %s..." % file_name)
    
    data_dict = scipy.io.loadmat("../data/" + file_name)
    to_remove = []
    
    for key in data_dict:
        if "__" in key:
            to_remove.append(key)
    
    for key_to_rm in to_remove:
        del data_dict[key_to_rm]
        print("\t deleted key: %s" % key_to_rm)
    
    if file_name == "fXSamples.mat":  
        p = kwargs.get('p')
        ix = kwargs.get('ix')
        ytrain = data_dict['x'][:, ix]
        xtrain = np.arange(len(ytrain))
        ytest = data_dict['x'][:, ix][p:]
        xtest = np.arange(len(ytest))
    else:
        raise NotImplementedError
    
    return xtrain, xtest, ytrain, ytest
        
        
    
