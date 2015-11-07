""" Description here

Author: Leonard Berrada
Date: 2 Nov 2015
"""

import scipy.io

def data_from_file(file_name, **kwargs):
        
    print("loading %s... \n" % file_name)
    
    data_dict = scipy.io.loadmat("../../data/" + file_name)
    to_remove = []
    
    for key in data_dict:
        if "__" in key:
            to_remove.append(key)
    
    for key_to_rm in to_remove:
        del data_dict[key_to_rm]
        print("** deleted key: %s **" % key_to_rm)
    
    data = dict()
    
    if file_name == "fXSamples.mat":  
        ix = kwargs.get('ix')
        data['ytest'] = data_dict['x'][:, ix]
        return data
        
    elif file_name == "finPredProb.mat":
        data['ytest'] = data_dict['ttr'].flatten()
        return data
    
    elif file_name == "mg.mat":
        data['ytrain'] = data_dict['t_tr'].flatten()
        data['ytest'] = data_dict['t_te'].flatten()
        return data
    
    elif file_name == "co2.mat":
        all_data = data_dict['co2'].flatten()
        data['ytrain'] = all_data[:500]
        data['ytest'] = all_data
        return data
    
    elif file_name == "sunspots.mat":
        data['ytest'] = data_dict['activity'].flatten()
        return data
    
    else:
        raise NotImplementedError
    
        
        
    
