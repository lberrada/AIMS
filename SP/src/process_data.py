""" Description here

Author: Leonard Berrada
Date: 2 Nov 2015
"""

import scipy.io
import numpy as np

def data_from_file(file_name, **kwargs):
        
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
        ix = kwargs.get('ix')
        x = data_dict['x'][:, ix]
        return (x, )
        
    elif file_name == "finPredProb.mat":
        x = data_dict['ttr'].flatten()
        return (x, )
    else:
        raise NotImplementedError
    
        
        
    
