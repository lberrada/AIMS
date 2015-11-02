""" Description here

Author: Leonard Berrada
Date: 2 Nov 2015
"""

import scipy.io
import pandas as pd
import numpy as np

def import_data(filename):
    
    print("loading %s..." % filename)
    
    data_dict = scipy.io.loadmat(filename)
    to_remove = []
    for key in data_dict:
        if "__" in key:
            to_remove.append(key)
    for key_to_rm in to_remove:
        del data_dict[key_to_rm]
        print("\t deleted key: %s" % key_to_rm)
    
    for key in data_dict:
        data_dict[key]= np.array(data_dict[key]).flatten()
        print(key, data_dict[key].shape)
        
    if filename!="mg.mat":
        my_df = pd.DataFrame.from_dict(data_dict)
    else:
        my_df = pd.DataFrame(data_dict["t_tr"])
        testing_df = pd.DataFrame(data_dict["t_te"])
    
    print("done")
    print("-"*50)
    print("showing headers for verification...")
    
    print(my_df.head())
    