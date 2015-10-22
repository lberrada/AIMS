""" Description here

Author: Leonard Berrada
Date: 22 Oct 2015
"""
import copy

def get_params(use_kernels=None,
               use_means=None):
    
    use_means = use_means.replace(" ", "")
    use_kernels = use_kernels.replace(" ", "")
    
    zero_bound = 1e-6
    
    params = dict()
    params["names"] = ["sigma_n"]
    params["means"] = [0.5]
    params["stds"] = [1.]
    params["bounds"] = [(zero_bound, None)]
    params["use_log"] = [True]
    
    aux_kernel_dict = dict()
    aux_kernel_dict["exponential_quadratic"] = dict()
    aux_kernel_dict["exponential_quadratic"]["names"] = ["eq_sigma_f", "eq_scale"]
    aux_kernel_dict["exponential_quadratic"]["means"] = [1., 20.]
    aux_kernel_dict["exponential_quadratic"]["stds"] = [3, 10]
    aux_kernel_dict["exponential_quadratic"]["bounds"] = [(zero_bound, None), (zero_bound, None)]
    aux_kernel_dict["exponential_quadratic"]["use_log"] = [True, True]
    
    aux_kernel_dict["exponential_quadratic_2"] = dict()
    aux_kernel_dict["exponential_quadratic_2"]["names"] = ["eq_sigma_f", "eq_scale"]
    aux_kernel_dict["exponential_quadratic_2"]["means"] = [0.5, 100.]
    aux_kernel_dict["exponential_quadratic_2"]["stds"] = [1, 40]
    aux_kernel_dict["exponential_quadratic_2"]["bounds"] = [(zero_bound, None), (zero_bound, None)]
    aux_kernel_dict["exponential_quadratic_2"]["use_log"] = [True, True]
    
    aux_kernel_dict["periodic"] = dict()
    aux_kernel_dict["periodic"]["names"] = ["p_period"]
    aux_kernel_dict["periodic"]["means"] = [150.]
    aux_kernel_dict["periodic"]["stds"] = [30.]
    aux_kernel_dict["periodic"]["bounds"] = [(zero_bound, None)]
    aux_kernel_dict["periodic"]["use_log"] = [True]
    
    aux_kernel_dict["rational_quadratic"] = dict()
    aux_kernel_dict["rational_quadratic"]["names"] = ["rq_sigma_f", "rq_scale", "rq_nu"]
    aux_kernel_dict["rational_quadratic"]["means"] = [1., 20., 2.]
    aux_kernel_dict["rational_quadratic"]["stds"] = [3, 10, 2]
    aux_kernel_dict["rational_quadratic"]["bounds"] = [(zero_bound, None), (zero_bound, None), (zero_bound, None)]
    aux_kernel_dict["rational_quadratic"]["use_log"] = [True, True, True]
    
    aux_kernel_dict["matern"] = dict()
    aux_kernel_dict["matern"]["names"] = ["m_sigma_f", "m_scale", "m_nu"]
    aux_kernel_dict["matern"]["means"] = [1., 20., 0.5]
    aux_kernel_dict["matern"]["stds"] = [3, 10, 2]
    aux_kernel_dict["matern"]["bounds"] = [(zero_bound, None), (zero_bound, None), (zero_bound, None)]
    aux_kernel_dict["matern"]["use_log"] = [True, True, True]
    
    
    aux_mean_dict = dict()
    aux_mean_dict["constant"] = dict()
    aux_mean_dict["constant"]["names"] = ["c_alpha"]
    aux_mean_dict["constant"]["means"] = [0.]
    aux_mean_dict["constant"]["stds"] = [5.]
    aux_mean_dict["constant"]["bounds"] = [(None, None)]
    aux_mean_dict["constant"]["use_log"] = [False]
    
    aux_mean_dict["linear"] = dict()
    aux_mean_dict["linear"] ["names"] = ["l_alpha", "l_beta"]
    aux_mean_dict["linear"] ["means"] = [0., 0.]
    aux_mean_dict["linear"] ["stds"] = [5., 10]
    aux_mean_dict["linear"]["bounds"] = [(None, None), (None, None)]
    aux_mean_dict["linear"]["use_log"] = [False, False]
    
    aux_mean_dict["periodic"] = dict()
    aux_mean_dict["periodic"] ["names"] = ["p_scale", "p_period"]
    aux_mean_dict["periodic"] ["means"] = [1., 150.]
    aux_mean_dict["periodic"] ["stds"] = [3., 30]
    aux_mean_dict["periodic"]["bounds"] = [(zero_bound, None), (zero_bound, None)]
    aux_mean_dict["periodic"]["use_log"] = [True, True]
    
    aux_kernels_string = use_kernels.replace("*", "+")
    aux_means_string = use_means.replace("*", "+")
    
    all_kernels = aux_kernels_string.split("+")
    all_means = aux_means_string.split("+")
    
    for k_name in all_kernels:
        params["names"] += aux_kernel_dict[k_name]["names"]
        params["means"] += aux_kernel_dict[k_name]["means"]
        params["stds"] += aux_kernel_dict[k_name]["stds"]
        params["bounds"] += aux_kernel_dict[k_name]["bounds"]
        params["use_log"] += aux_kernel_dict[k_name]["use_log"]
        
    for m_name in all_means:
        params["names"] += aux_mean_dict[m_name]["names"]
        params["means"] += aux_mean_dict[m_name]["means"]
        params["stds"] += aux_mean_dict[m_name]["stds"]
        params["bounds"] += aux_mean_dict[m_name]["bounds"]
        params["use_log"] += aux_mean_dict[m_name]["use_log"]

    
    params["init"] = copy.copy(params["means"])
    
    return params
        
        
        
        
        
        
        
    
    
    
    
    
    
    
