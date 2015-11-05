""" Description here

Author: Leonard Berrada
Date: 2 Nov 2015
"""

import scipy.io
import numpy as np

def data_from_file(file_name, **kwargs):
        
    print("loading %s... \n" % file_name)
    
    data_dict = scipy.io.loadmat("../data/" + file_name)
    to_remove = []
    
    for key in data_dict:
        if "__" in key:
            to_remove.append(key)
    
    for key_to_rm in to_remove:
        del data_dict[key_to_rm]
        print("** deleted key: %s **" % key_to_rm)
    
    if file_name == "fXSamples.mat":  
        ix = kwargs.get('ix')
        x = data_dict['x'][:, ix]
        print("NB: forecast model required")
        return (x,)
        
    elif file_name == "finPredProb.mat":
        x = data_dict['ttr'].flatten()
        print("NB: forecast model required")
        return (x,)
    
    elif file_name == "mg.mat":
        ytrain = data_dict['t_tr'].flatten()
        ytest = data_dict['t_te'].flatten()
        xtrain = np.arange(len(ytrain))
        xtest = np.arange(len(ytest))
        print("NB: regression model required")
        return (xtrain, xtest, ytrain, ytest)
    
    elif file_name == "co2.mat":
        x = data_dict['co2'].flatten()
        print("NB: forecast model required")
        return (x,)
    
    elif file_name == "sunspots.mat":
        x = data_dict['activity'].flatten()
        print("NB: forecast model required")
        return (x,)
    
    else:
        raise NotImplementedError
    
        
        
    
