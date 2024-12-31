# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:33:30 2024

@author: kjychung
"""
import numpy as np
from scipy.optimize import nnls

def logspace_with_bounds(min_k=0.001, max_k=10, num_k=100, prependZero=False):
    
    if num_k == 1:
        return np.zeros(1)
    
    if prependZero:
        num_k -= 1
    
    grid = np.linspace(1, num_k, num=int(num_k), endpoint=True)
    k_vals = min_k * (max_k / min_k) ** ((grid - 1) / (num_k - 1))
    
    if prependZero:
        k_vals = np.insert(k_vals, 0, 0)
        
    return k_vals

def batch_nnls(Q_t, sysmat):
    
    # nbasis, nt, nparams = sysmat.shape
    
    # xs = []
    # idxs = []
    # for q in Q_t:
    #     x_list = []
    #     r_list = []
    #     for ii in range(nbasis):
    #         x, r = nnls(sysmat[ii,:,:], q)
    #         x_list.append(x)
    #         r_list.append(r)
            
    #     idx = np.argmin(r_list)
    #     xs.append(x_list[idx])
    #     idxs.append(idx)
        
    # return xs, idxs
    
    nbasis, nt, nparams = sysmat.shape
    x_list = []
    r_list = []
    for ii in range(nbasis):
        x, r = nnls(sysmat[ii,:,:], Q_t)
        x_list.append(x)
        r_list.append(r)
        
    idx = np.argmin(r_list)
    x = x_list[idx]
        
    return x, idx