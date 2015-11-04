""" Description here

Author: Leonard Berrada
Date: 2 Nov 2015
"""

import scipy.io
import pandas as pd

def import_data(filename):
    
    print("loading %s..." % filename)
    
    data_dict = scipy.io.loadmat("../data/" + filename)
    to_remove = []
    for key in data_dict:
        if "__" in key:
            to_remove.append(key)
    for key_to_rm in to_remove:
        del data_dict[key_to_rm]
        print("\t deleted key: %s" % key_to_rm)
    
    if filename == "fXSamples.mat":  
        my_df = pd.DataFrame(data_dict['x'])
    elif filename != "mg.mat":
        my_df = pd.DataFrame.from_dict(data_dict)
    else:
        my_df = pd.DataFrame(data_dict["t_tr"])
        testing_df = pd.DataFrame(data_dict["t_te"])
    
    print("done")
    print("-"*50)
    print("showing headers for verification...")
    
    print(my_df.head())
    return my_df
    
