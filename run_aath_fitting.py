# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:08:45 2024

@author: kjychung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import os

from utils import ktools

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
pd.set_option('display.max_columns', None)

t_start = time.time()

# Path to Excel spreadsheet containing time and time-activity curve arrays
path_data = 'data/sampleTACs.xlsx' 

xls = pd.ExcelFile(path_data)
tracer_names = xls.sheet_names # Fluciclovine, FDG, Butanol

fig, axs = plt.subplots(1, 3, dpi=300, figsize=(8,3), layout='tight')
for ii, tracer in enumerate(tracer_names):
    
    # Read the sample TACs
    df = pd.read_excel(xls, sheet_name=tracer)
    t = df.iloc[:,0].to_numpy()     # Time array
    idif = df.iloc[:,1].to_numpy()  # Image-derived input function (IDIF)
    q = df.iloc[:,2].to_numpy()     # Tissue time-activity curve
    
    # Tissue time-activity curve fitting with the AATH model and the basis function algorithm
    # q_fit is the fitted time-activity curve
    # kparams is the Python dictionary containing the fitted model parameters
    q_fit, kparams = ktools.aath_bfm(t, idif, q) 
    
    # Plot the fitted time-activity curve
    axs[ii].plot(t, q / 1000.0, '.k', label='Measured', fillstyle='none')
    axs[ii].plot(t, q_fit / 1000.0, '-c', label='AATH Fitted')
    axs[ii].set_title(tracer, fontweight='bold')
    axs[ii].legend()
    
    # Print the fitted model parameters
    print()
    print(tracer, 'kinetic parameters:')
    for key, value in kparams.items():
        print(f"\t{key} {value[0]:0.3f}")
    print()
    
fig.supxlabel('Time [s]', fontweight='bold')
fig.supylabel('Activity [kBq/mL]', fontweight='bold')
fig.show()

fig.savefig('data/sampleTAC_fitting.png', dpi=300, bbox_inches='tight')

print(f'Process took {time.time() - t_start:0.1f} s')