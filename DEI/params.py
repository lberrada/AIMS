""" Description here

Author: Leonard Berrada
Date: 22 Oct 2015
"""

def get_params(kernel_name, mean_name):
    
    params=dict()
    params["names"] = []
    params["means"] = []
    params["stds"] = []
    params["init"] = []
    
    params["names"].append("sigma_n")
    params["means"].append(1.)
    params["stds"].append(10.)
    params["init"].append(1.)
    
    params["names"].append("sigma_f")
    params["means"].append(1.)
    params["stds"].append(10.)
    params["init"].append(1.)
    
    params["names"].append("scale")
    params["means"].append(1.)
    params["stds"].append(10.)
    params["init"].append(1.)
    
    if kernel_name == "gaussian_2":
        params["names"].append("sigma_f_2")
        params["means"].append(1.)
        params["stds"].append(10.)
        params["init"].append(1.)
        
        params["names"].append("scale_2")
        params["means"].append(1.)
        params["stds"].append(10.)
        params["init"].append(1.)
    
    
    
    