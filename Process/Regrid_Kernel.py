#!/usr/bin/env python3

print('read in packages')
import numpy as np 
import xarray as xr
import pickle as pk
import natsort as ns 
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import glob 
import sys

kernel_path  = '/home/c/cgf/jgvirgin/Kernels/'
kernels = xr.open_dataset(kernel_path+'CAM3_Kernels_4x5.nc')

f_lat = kernels['lat'].values
f_lon = kernels['lon'].values
fx,fy = np.meshgrid(f_lon,f_lat)

c_lat = np.linspace(-90,90,46)
c_lon = np.linspace(0,355,72)
cx,cy = np.meshgrid(c_lon,c_lat)

print('Reading in Kernels')

knames = ['Alb_TOA','Alb_TOA_CLR','Ts_TOA','Ts_TOA_CLR','Ta_TOA',\
            'Ta_TOA_CLR','WVlw_TOA','WVlw_TOA_CLR','WVsw_TOA','WVsw_TOA_CLR']

kd = {}
for i in range(len(knames)):
    kd[knames[i]] = kernels[knames[i]].values
    kd[knames[i]][kd[knames[i]] > 1e5] = np.nan

print('moving onto cloud kernels')
cld_kernels = xr.open_dataset(kernel_path+'cloud_kernels2.nc')
cld_lw = cld_kernels['LWkernel'].values
cld_sw = cld_kernels['SWkernel'].values

cld_lon = np.arange(1.25, 360, 2.5)
cld_lat = cld_kernels['lat'].values

print('interpolating')
LWkernel_func = interp1d(cld_lat, cld_lw, axis=3, kind='nearest',fill_value="extrapolate")
SWkernel_func = interp1d(cld_lat, cld_sw, axis=3, kind='nearest',fill_value="extrapolate")
LWkernel_interp = LWkernel_func(c_lat)
SWkernel_interp = SWkernel_func(c_lat)

LWkernel_map = np.tile(LWkernel_interp[:,:,:,:,0,None],(1,1,1,1,72))

cld_k_int = {}
cld_k_int['LWkernel'] = LWkernel_map
cld_k_int['SWkernel'] =  SWkernel_interp

print('Saving')
pk.dump(kd,open(kernel_path+'CAM3_Kernels_4x5.pi','wb'))
pk.dump(cld_k_int,open(kernel_path+'cloud_kernels_4x5.pi','wb'))

    







