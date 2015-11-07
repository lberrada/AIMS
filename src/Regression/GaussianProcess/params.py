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
    params["stds"] = [0.5]
    params["bounds"] = [(1e-3, None)]
    params["use_log"] = [True]
    
    aux_kernel_dict = dict()
    aux_kernel_dict["exponential_quadratic"] = dict()
    aux_kernel_dict["exponential_quadratic"]["names"] = ["eq_sigma_f", "eq_scale"]
    aux_kernel_dict["exponential_quadratic"]["means"] = [0.5, 5.]
    aux_kernel_dict["exponential_quadratic"]["stds"] = [0.5, 3]
    aux_kernel_dict["exponential_quadratic"]["bounds"] = [(zero_bound, None), (zero_bound, None)]
    aux_kernel_dict["exponential_quadratic"]["use_log"] = [True, True]
    
    aux_kernel_dict["exponential_quadratic_2"] = dict()
    aux_kernel_dict["exponential_quadratic_2"]["names"] = ["eq_sigma_f", "eq_scale"]
    aux_kernel_dict["exponential_quadratic_2"]["means"] = [0.5, 100.]
    aux_kernel_dict["exponential_quadratic_2"]["stds"] = [0.5, 3]
    aux_kernel_dict["exponential_quadratic_2"]["bounds"] = [(zero_bound, None), (zero_bound, None)]
    aux_kernel_dict["exponential_quadratic_2"]["use_log"] = [True, True]
    
    aux_kernel_dict["periodic"] = dict()
    aux_kernel_dict["periodic"]["names"] = ["p_sigma_f", "p_period"]
    aux_kernel_dict["periodic"]["means"] = [0.5, 25.]
    aux_kernel_dict["periodic"]["stds"] = [0.5, 3.]
    aux_kernel_dict["periodic"]["bounds"] = [(zero_bound, None), (zero_bound, None)]
    aux_kernel_dict["periodic"]["use_log"] = [True, True]
    
    aux_kernel_dict["cosine"] = dict()
    aux_kernel_dict["cosine"]["names"] = ["c_period"]
    aux_kernel_dict["cosine"]["means"] = [25.]
    aux_kernel_dict["cosine"]["stds"] = [10.]
    aux_kernel_dict["cosine"]["bounds"] = [(zero_bound, None)]
    aux_kernel_dict["cosine"]["use_log"] = [True]
    
    aux_kernel_dict["rational_quadratic"] = dict()
    aux_kernel_dict["rational_quadratic"]["names"] = ["rq_sigma_f", "rq_scale", "rq_nu"]
    aux_kernel_dict["rational_quadratic"]["means"] = [1., 5., 2.]
    aux_kernel_dict["rational_quadratic"]["stds"] = [0.5, 3, 0.5]
    aux_kernel_dict["rational_quadratic"]["bounds"] = [(zero_bound, None), (zero_bound, None), (zero_bound, None)]
    aux_kernel_dict["rational_quadratic"]["use_log"] = [True, True, True]
    
    aux_kernel_dict["matern_12"] = aux_kernel_dict["exponential_quadratic"]
    aux_kernel_dict["matern_32"] = aux_kernel_dict["exponential_quadratic"]
    
    
    aux_mean_dict = dict()
    aux_mean_dict["constant"] = dict()
    aux_mean_dict["constant"]["names"] = ["c_alpha"]
    aux_mean_dict["constant"]["means"] = [0.]
    aux_mean_dict["constant"]["stds"] = [5.]
    aux_mean_dict["constant"]["bounds"] = [(None, None)]
    aux_mean_dict["constant"]["use_log"] = [False]
    
    aux_mean_dict["linear"] = dict()
    aux_mean_dict["linear"] ["names"] = ["l_alpha", "l_beta"]
    aux_mean_dict["linear"] ["means"] = [0.5, 0.5]
    aux_mean_dict["linear"] ["stds"] = [1., 1]
    aux_mean_dict["linear"]["bounds"] = [(None, None), (None, None)]
    aux_mean_dict["linear"]["use_log"] = [False, False]
    
    aux_mean_dict["quadratic"] = dict()
    aux_mean_dict["quadratic"] ["names"] = ["l_alpha", "l_beta", "l_gamma"]
    aux_mean_dict["quadratic"] ["means"] = [1., 1., 0.5]
    aux_mean_dict["quadratic"] ["stds"] = [5., 5., 5.]
    aux_mean_dict["quadratic"]["bounds"] = [(None, None), (None, None), (None, None)]
    aux_mean_dict["quadratic"]["use_log"] = [False, False, False]
    
    aux_mean_dict["periodic"] = dict()
    aux_mean_dict["periodic"] ["names"] = ["pm_scale", "pm_period"]
    aux_mean_dict["periodic"] ["means"] = [1., 10.]
    aux_mean_dict["periodic"] ["stds"] = [1., 3]
    aux_mean_dict["periodic"]["bounds"] = [(zero_bound, None), (zero_bound, None)]
    aux_mean_dict["periodic"]["use_log"] = [True, True]
    
    for to_add_op in use_kernels.split('+'):
        for to_mult in to_add_op.split("*"):
            k_name = to_mult
            params["names"] += aux_kernel_dict[k_name]["names"]
            params["means"] += aux_kernel_dict[k_name]["means"]
            params["stds"] += aux_kernel_dict[k_name]["stds"]
            params["bounds"] += aux_kernel_dict[k_name]["bounds"]
            params["use_log"] += aux_kernel_dict[k_name]["use_log"]
        
    for to_add_op in use_means.split('+'):
        for to_mult in to_add_op.split("*"):
            m_name = to_mult
            params["names"] += aux_mean_dict[m_name]["names"]
            params["means"] += aux_mean_dict[m_name]["means"]
            params["stds"] += aux_mean_dict[m_name]["stds"]
            params["bounds"] += aux_mean_dict[m_name]["bounds"]
            params["use_log"] += aux_mean_dict[m_name]["use_log"]

    
    return params
        
        
        
        
