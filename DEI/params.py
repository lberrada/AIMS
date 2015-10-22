""" Description here

Author: Leonard Berrada
Date: 22 Oct 2015
"""

def get_params(use_kernels=None,
               use_means=None):
    
    params = dict()
    params["names"] = ["sigma_n"]
    params["means"] = [1.]
    params["stds"] = [10.]
    params["init"] = [1.]
    
    aux_kernel_dict = dict()
    aux_kernel_dict["exponential_quadratic"] = dict()
    aux_kernel_dict["exponential_quadratic"]["names"] = ["eq_sigma_f", "eq_scale"]
    aux_kernel_dict["exponential_quadratic"]["means"] = [1., 1.]
    aux_kernel_dict["exponential_quadratic"]["stds"] = [10, 10]
    aux_kernel_dict["exponential_quadratic"]["init"] = [1, 1]
    
    aux_kernel_dict["periodic"] = dict()
    aux_kernel_dict["periodic"]["names"] = ["p_nu"]
    aux_kernel_dict["periodic"]["means"] = [1.]
    aux_kernel_dict["periodic"]["stds"] = [10]
    aux_kernel_dict["periodic"]["init"] = [1]
    
    aux_kernel_dict["rational_quadratic"] = dict()
    aux_kernel_dict["rational_quadratic"]["names"] = ["rq_sigma_f", "rq_scale", "rq_nu"]
    aux_kernel_dict["rational_quadratic"]["means"] = [1., 1., 1.]
    aux_kernel_dict["rational_quadratic"]["stds"] = [10, 10, 10]
    aux_kernel_dict["rational_quadratic"]["init"] = [1, 1, 1]
    
    aux_kernel_dict["matern"] = dict()
    aux_kernel_dict["matern"]["names"] = ["m_sigma_f", "m_scale", "m_nu"]
    aux_kernel_dict["matern"]["means"] = [1., 1., 1.]
    aux_kernel_dict["matern"]["stds"] = [10, 10, 10]
    aux_kernel_dict["matern"]["init"] = [1, 1, 1]
    
    
    aux_mean_dict = dict()
    aux_mean_dict["constant"] = dict()
    aux_mean_dict["constant"]["names"] = ["c_alpha"]
    aux_mean_dict["constant"]["means"] = [1.]
    aux_mean_dict["constant"]["stds"] = [10]
    aux_mean_dict["constant"]["init"] = [1]
    
    aux_mean_dict["linear"] = dict()
    aux_mean_dict["linear"] ["names"] = ["l_alpha", "l_beta"]
    aux_mean_dict["linear"] ["means"] = [1., 1.]
    aux_mean_dict["linear"] ["stds"] = [10, 10]
    aux_mean_dict["linear"] ["init"] = [1, 1]
    
    aux_mean_dict["periodic"] = dict()
    aux_mean_dict["periodic"] ["names"] = ["p_scale", "p_period"]
    aux_mean_dict["periodic"] ["means"] = [1., 1.]
    aux_mean_dict["periodic"] ["stds"] = [10, 10]
    aux_mean_dict["periodic"] ["init"] = [1, 1]
    
    aux_kernels_string = use_kernels.replace("*", "+")
    aux_means_string = use_means.replace("*", "+")
    
    all_kernels = aux_kernels_string.split("+")
    all_means = aux_means_string.split("+")
    
    for k_name in all_kernels:
        params["names"] += aux_kernel_dict[k_name]["names"]
        params["means"] += aux_kernel_dict[k_name]["names"]
        params["stds"] += aux_kernel_dict[k_name]["names"]
        params["init"] += aux_kernel_dict[k_name]["names"]
        
    for m_name in all_means:
        params["names"] += aux_mean_dict[m_name]["names"]
        params["means"] += aux_mean_dict[m_name]["names"]
        params["stds"] += aux_mean_dict[m_name]["names"]
        params["init"] += aux_mean_dict[m_name]["names"]
        
        
        
        
        
        
        
    
    
    
    
    
    
    
