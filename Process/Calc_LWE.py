#!/usr/bin/env python3

print('read in packages')
import numpy as np 
import xarray as xr
import pickle as pk
import natsort as ns 
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
import glob 
import sys

header = 'f'
header_pre = header.capitalize()

version = 'e12'
case = sys.argv[1]
res = 'f45_f45'
author = 'JGV'
wd = '/scratch/c/cgf/jgvirgin/cesm1_2_2/archive/'

control_name = header+'.'+version+'.'+'F1850CN'+'.'+res+'.'+author+'.'+case
x4_name = header+'.'+version+'.'+'piClim-abrupt-4xCO2'+'.'+res+'.'+author+'.'+case

print('Control Case -',control_name)
print('Perturbed Case -',x4_name)
print('reading in data')

control_files = ns.natsorted(glob.glob(wd+control_name+'/atm/hist/'+control_name+'.cam.h0.*'))[:180]
x4_files = ns.natsorted(glob.glob(wd+x4_name+'/atm/hist/'+x4_name+'.cam.h0.*'))[:180]

control = xr.open_mfdataset(control_files)
x4 = xr.open_mfdataset(x4_files)

lat = control['lat'].values
lon = control['lon'].values

y = lat*np.pi/180
coslat = np.cos(y)
coslat = np.tile(coslat,(lon.size,1)).T

print('- Delta Tas')
control_tas = control['TREFHT']
control_tas_an = control_tas.groupby('time.month').mean('time')
control_tas_vals = control_tas_an.values

x4_tas = x4['TREFHT']
x4_tas_an = x4_tas.groupby('time.month').mean('time')
x4_tas_vals = x4_tas_an.values

dTAS = x4_tas_vals-control_tas_vals

print('Read in CAM3 Surface Temperature Kernel')
kernel_path  = '/home/c/cgf/jgvirgin/Kernels/CAM3_Kernels.nc'
kernels = xr.open_dataset(kernel_path)

biglat = kernels['lat'].values
biglon = kernels['lon'].values

bigx,bigy = np.meshgrid(biglon,biglat)
smallx,smally = np.meshgrid(lon,lat)

print('Interpolate to Coarse horizontal grid')

kTAS = kernels['Ts_TOA'].values

kTAS_interp = np.zeros([12,46,72])
for i in range(12):
    kTAS_interp[i,:,:] = griddata(\
        (bigx.flatten(),bigy.flatten()),kTAS[i,:,:].flatten(),(smallx,smally),method='linear')

print('Calculate Land Surface warming radiative effect')

dLWE = np.nanmean(kTAS_interp*dTAS,axis=0)
indices = ~np.isnan(dLWE)
dLWE_gam = round(np.average(dLWE[indices],weights=coslat[indices]),2)


print('Radiative Adjustment due to the Land Warming Effect - {}'.format(dLWE_gam))

p = Path('piClim-abrupt-4xCO2_LWE.csv')

if p.exists():

    params = pd.read_csv('piClim-abrupt-4xCO2_LWE.csv')
    addon = pd.DataFrame({'Case': [case],'LWE': [dLWE_gam]})
    params = params.append(addon)
    params.to_csv('piClim-abrupt-4xCO2_LWE.csv',index=False)

else:

    params = pd.DataFrame({'Case': [case],'LWE': [dLWE_gam]})
    params.to_csv('piClim-abrupt-4xCO2_LWE.csv',index=False)