# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:54:59 2024

@author: kjychung
"""
import numpy as np

from scipy.integrate import cumtrapz
from scipy.optimize import nnls

from functools import partial
from multiprocessing import Pool

import time
import os

try:
    from . mathtools import logspace_with_bounds, batch_nnls
except:
    from utils.mathtools import logspace_with_bounds, batch_nnls

def convolve_exp(t, f_t, alpha):
    
    """
    f_t is assumed a piecewise step function = Sum_{i=0}^{N-1} {f_t[i] * (H(t-t_i) - H(t-t_{i+1}))} 
    where H(t) is the Heaviside Step function.
    
    This is essentially a convolution of a series of boxcars (scaled by f_t[i]) with an exponential decay function
    """
    
    if alpha == 0:
        return cumtrapz(f_t, x=t, initial=0)
    
    ti_0 = np.expand_dims(t[:-1], axis=-1)
    ti_1 = np.expand_dims(t[1:], axis=-1)
    tt = np.expand_dims(t, axis=0)
    
    hs_ti_0 = np.heaviside(tt-ti_0, 1)
    hs_ti_1 = np.heaviside(tt-ti_1, 1)
    tt_0 = (tt - ti_0)*hs_ti_0
    tt_1 = (tt - ti_1)*hs_ti_1
    
    exp_ti_0 = (1 - np.exp(-alpha*tt_0))*hs_ti_0
    exp_ti_1 = (1 - np.exp(-alpha*tt_1))*hs_ti_1
    
    conv = np.sum((exp_ti_0 - exp_ti_1) * f_t[1:,np.newaxis], axis=0) / alpha
    
    return conv

def akaike_information_criteria(y, yp, nparams):
    """
    y and yp are ndarrays with shape (nt,) or (ncurves,nt). y and yp should 
    have the same size. 
    nparams = number of parameters in the model
    """
    
    yshape = y.shape
    nt = yshape[-1]
    rss = np.sum((y - yp)**2, axis=-1)
    aic = nt*np.log(rss / nt) + 2*nparams + (2*nparams**2 + 2*nparams) / (nt - nparams - 1)
    
    return aic


def aath_tac(t, Cwb, td, Tc, F, K1, k2, Cp=None):
    
    if Cp is None:
        Cp = Cwb
        
    aif_cdf = cumtrapz(Cwb, x=t, initial=0)
    aif_cdf_td = np.interp(t-td, t, aif_cdf)
    aif_cdf_td_Tc = np.interp(t-td-Tc, t, aif_cdf)
    
    aif_conv_exp = convolve_exp(t, Cp, k2)
    aif_conv_exp_td_Tc = np.interp(t-td-Tc, t, aif_conv_exp)
    
    q_iv = F*(aif_cdf_td - aif_cdf_td_Tc)
    q_ev = K1*aif_conv_exp_td_Tc
    
    return q_iv, q_ev


def s1tc_tac(t, Cwb, td, va, K1, k2, Cp=None):
    
    if Cp is None:
        Cp = Cwb
    
    aif_conv_exp = convolve_exp(t, Cp, k2)
    
    q_iv = va*np.interp(t-td, t, Cwb, Cwb[0], Cwb[-1])
    q_ev = (1-va)*K1*np.interp(t-td, t, aif_conv_exp)
    
    return q_iv, q_ev

def create_aath_bfm_matrix(t, Cwb, Cp=None, printlog=False, 
                           td_params=(0,10.25,0.25),
                           Tc_params=(3,10.25,0.25),
                           k2_params=(0.00001,0.05,100),
                           useVa=False
                           ):   
    
    t_str0 = time.time()
    
    if Cp is None:
        Cp = Cwb
        
    nt = len(t)
    
    k2_vals = logspace_with_bounds(*k2_params, prependZero=True)
    nk2 = len(k2_vals)

    td_lut, Tc_lut, k2_lut = np.meshgrid(np.arange(*td_params),
                                        np.arange(*Tc_params),
                                        k2_vals)
    
    td_lut = td_lut.flatten()
    Tc_lut = Tc_lut.flatten()
    k2_lut = k2_lut.flatten()

    ncombos = td_lut.size
    
    Cwb_cdf = cumtrapz(Cwb, x=t, initial=0)

    basis_k2 = np.zeros((nt, nk2))
    for ii in range(nk2):
        basis_k2[:,ii] = convolve_exp(t, Cp, k2_vals[ii])
        
    sysmat = np.zeros((ncombos, nt, 2))
    for ii, (td, Tc, k2) in enumerate(zip(td_lut, Tc_lut, k2_lut)):
        
        idx_k2 = np.where(k2_vals == k2)[0][0]
                    
        Cwb_cdf_t0_shift = np.interp(t-td, t, Cwb_cdf)
        Cwb_cdf_t0w_shift = np.interp(t-td-Tc, t, Cwb_cdf) 
        
        sysmat[ii,:,0] = Cwb_cdf_t0_shift - Cwb_cdf_t0w_shift
        sysmat[ii,:,1] = np.interp(t-td-Tc, t, basis_k2[:,idx_k2])
        
    # Slack constraint such that:
    #   F - K1 - s1 = 0; i.e., F >= K1
    #   using an arbitrary slack value 1e20
    sysmat = np.pad(sysmat, pad_width=((0,0),(0,1),(0,1)), mode='constant', constant_values=0)
    sysmat[:,-1,0] = 1e10
    sysmat[:,-1,1:-1] = -1e10
    sysmat[:,-1,-1] = -1e10
    
    if useVa:
        sysmat = np.append(sysmat, np.zeros_like(sysmat[:,:,0:1]), axis=-1)
        for ii, td in enumerate(td_lut):
            sysmat[ii,:len(t),-1] = np.interp(t-td, t, Cwb, Cwb[0], Cwb[-1])
    
    if printlog: print(f'Precomputing basis functions took {time.time() - t_str0:0.1f} s')
    
    return sysmat, td_lut, Tc_lut, k2_lut

def aath_bfm(t, Cwb, Q_t, Cp=None, multi=False, printlog=False, **kwargs):
    
    t_str0 = time.time()
    
    if Cp is None:
        Cp = Cwb
    
    if Q_t.ndim == 1:
        Q_t = np.expand_dims(Q_t, axis=0)
        
    ncurves, nt = Q_t.shape
    Q_t_fit = np.zeros_like(Q_t)
    
    sysmat, td_lut, Tc_lut, k2_lut = create_aath_bfm_matrix(t, Cwb, Cp=Cp, printlog=printlog, **kwargs)
    
    nbasis, nt_sys, nparams = sysmat.shape
    if nt_sys > nt:
        Q_t = np.copy(np.pad(Q_t, ((0,0),(0, int(nt_sys - nt))), mode='constant', constant_values=0))
    
    if printlog:
        print('ncurves:', ncurves)
        print('sysmat dimensions:', sysmat.shape)
    
    td = np.zeros(ncurves)
    Tc = np.zeros(ncurves)
    F = np.zeros(ncurves)
    K1 = np.zeros(ncurves)
    k2 = np.zeros(ncurves)
    
    if multi:
        f = partial(batch_nnls, sysmat=sysmat)
        batch_size = np.ceil(ncurves / (os.cpu_count())).astype(int)
        batch_size = np.min([batch_size, 2**13])
        
        pool = Pool(processes=os.cpu_count(), maxtasksperchild=1)
        ResultList = pool.map(f, Q_t, chunksize=batch_size)
        pool.close()
        pool.join()
        
        x_list = [res[0] for res in ResultList]
        idx_list = [res[1] for res in ResultList]
        
        td = np.array([td_lut[idx] for idx in idx_list])
        Tc = np.array([Tc_lut[idx] for idx in idx_list])
        k2 = np.array([k2_lut[idx] for idx in idx_list])
        
        F = np.array([x[0] for x in x_list])
        K1 = np.array([x[1] for x in x_list])
        
        for ii in range(ncurves):
            idx = idx_list[ii]
            Q_t_fit[ii,:] = sysmat[idx,:nt,:].dot(x_list[ii])
    else:
        for ii in range(ncurves):
            x_list = []
            r_list = []
            for jj in range(nbasis):
                x, r = nnls(sysmat[jj,:,:], Q_t[ii,:])
                x_list.append(x)
                r_list.append(r)
            
            idx = np.argmin(r_list)
            td[ii] = td_lut[idx]
            Tc[ii] = Tc_lut[idx]
            k2[ii] = k2_lut[idx]
            F[ii] = x_list[idx][0]
            K1[ii] = x_list[idx][1]
            
            Q_t_fit[ii,:] = sysmat[idx,:nt,:].dot(x_list[idx])
        
    E = np.divide(K1, F, where=F>0, out=np.zeros_like(F))
    E = np.clip(E, 0, 0.9999)
    PS = -F*np.log(1-E, where=E<=0.9999, out=np.zeros_like(F))
    vb = F*Tc
    ve = np.divide(K1, k2, where=k2>0, out=np.zeros_like(K1))
    
    aic = akaike_information_criteria(Q_t[:,:len(t)], Q_t_fit, 5) # nparams = 5; F, K1, k2, td, Tc
    
    kparams = {'td' : td,       # [s]
               'Tc' : Tc,       # [s]
               'F' : F*60.0,    # [ml/min/cm3]
               'K1' : K1*60.0,  # [ml/min/cm3]
               'k2' : k2*60.0,  # [min-1]
               'E' : E,         # unitless
               'PS' : PS*60.0,  # [ml/min/cm3]
               'vb' : vb,       # [ml/cm3]
               've' : ve,       # [ml/cm3]
               'aic' : aic
               }

    if printlog:    
        print(f'AATH BFM took {time.time() - t_str0:0.1f} s')
    
    if len(Q_t_fit) == 1:
        Q_t_fit = np.squeeze(Q_t_fit, axis=0)
    
    return Q_t_fit, kparams
    
def s1tc_bfm(t, Cwb, Q_t, Cp=None, multi=False, printlog=False,
             td_params=(0,6.25,0.5),
             k2_params=(0.00001,0.05,100), 
             **kwargs):
    
    t_str0 = time.time()
    
    if Cp is None:
        Cp = Cwb
    
    ncurves, nt = Q_t.shape
    Q_t_fit = np.zeros_like(Q_t)
    
    k2_vals = logspace_with_bounds(*k2_params, prependZero=True)
    nk2 = len(k2_vals)

    td_lut, k2_lut = np.meshgrid(np.arange(*td_params), k2_vals)
    
    td_lut = td_lut.flatten()
    k2_lut = k2_lut.flatten()
    
    # Generate table of basis functions
    nbasis = td_lut.size
    sysmat = np.zeros((nbasis, nt, 2))
    for ii, (td, k2) in enumerate(zip(td_lut, k2_lut)):
        sysmat[ii,:,0] = np.interp(t-td, t, Cwb, Cwb[0], Cwb[-1])
        sysmat[ii,:,1] = np.interp(t-td, t, convolve_exp(t, Cp, k2))
    
    if printlog:
        print('ncurves:', ncurves)
        print('sysmat dimensions:', sysmat.shape)
    
    td = np.zeros(ncurves)
    K1 = np.zeros(ncurves)
    k2 = np.zeros(ncurves)
    vb = np.zeros(ncurves)
    
    if multi:
        f = partial(batch_nnls, sysmat=sysmat)
        batch_size = np.ceil(ncurves / (os.cpu_count())).astype(int)
        batch_size = np.min([batch_size, 2**13])
        
        pool = Pool(processes=os.cpu_count(), maxtasksperchild=1)
        ResultList = pool.map(f, Q_t, chunksize=batch_size)
        pool.close()
        pool.join()
        
        x_list = [res[0] for res in ResultList]
        idx_list = [res[1] for res in ResultList]
        
        td = np.array([td_lut[idx] for idx in idx_list])
        k2 = np.array([k2_lut[idx] for idx in idx_list])
        
        vb = np.array([x[0] for x in x_list])
        K1 = np.array([x[1] for x in x_list])
        
        
        for ii in range(ncurves):
            idx = idx_list[ii]
            Q_t_fit[ii,:] = sysmat[idx,:nt,:].dot(x_list[ii])
    else:
        for ii in range(ncurves):
            # if multi:
            #     f = partial(nnls, b=Q_t[ii,:])
            #     batch_size = np.ceil(nbasis / (os.cpu_count())).astype(int)
                
            #     pool = Pool(maxtasksperchild=1)
            #     ResultList = pool.map(f, sysmat, chunksize=batch_size)
            #     pool.close()
            #     pool.join()
                
            #     x_list = [res[0] for res in ResultList]
            #     r_list = [res[1] for res in ResultList]
            # else:
            x_list = []
            r_list = []
            for jj in range(nbasis):
                x, r = nnls(sysmat[jj,:,:], Q_t[ii,:])
                x_list.append(x)
                r_list.append(r)
            
            idx = np.argmin(r_list)
            td[ii] = td_lut[idx]
            k2[ii] = k2_lut[idx]
            vb[ii] = x_list[idx][0]
            K1[ii] = x_list[idx][1]
            
            Q_t_fit[ii,:] = sysmat[idx,:nt,:].dot(x_list[idx])
        
    # Set AATH-only parameters to 0
    Tc = np.zeros(ncurves)
    F = np.zeros(ncurves)
    E = np.zeros_like(K1)
    PS = np.zeros_like(K1)
    
    ve = np.divide(K1, k2, where=k2>0, out=np.zeros_like(K1))
    
    # aic = np.zeros(ncurves)
    aic = akaike_information_criteria(Q_t, Q_t_fit, 4) # nparams = 4; vb, K1, k2, td
    
    kparams = {'td' : td,       # [s]
               'Tc' : Tc,       # [s]
               'F' : F*60.0,    # [ml/min/cm3]
               'K1' : K1*60.0,  # [ml/min/cm3]
               'k2' : k2*60.0,  # [min-1]
               'E' : E,         # unitless
               'PS' : PS*60.0,  # [ml/min/cm3]
               'vb' : vb,       # [ml/cm3]
               've' : ve,       # [ml/cm3]
               'aic' : aic
               }

    if printlog:
        print(f'S1TC BFM took {time.time() - t_str0:0.1f} s')
    
    return Q_t_fit, kparams