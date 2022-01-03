#!/usr/bin/env python3

print('read in packages')
import numpy as np 
import xarray as xr
import pickle as pk
import natsort as ns 
import pandas as pd
from pathlib import Path
import glob 
import sys

def Calc_s0(ERF,alpha):
    ERF_adj = ERF
    s0 = round((ERF_adj/(1-alpha))*4,2)
    return s0

header = 'f'
header_pre = header.capitalize()

version = 'e12'
case = '1'
res = 'f45_f45'
author = 'JGV'
wd = '/scratch/c/cgf/jgvirgin/cesm1_2_2/archive/'

control_name = header+'.'+version+'.'+'F1850CN'+'.'+res+'.'+author+'.'+case
x4_name = header+'.'+version+'.'+'piClim-abrupt-4xCO2'+'.'+res+'.'+author+'.'+case

print('Control Case -',control_name)
print('Perturbed Case -',x4_name)
print('reading in data')

control_files = ns.natsorted(glob.glob(wd+control_name+'/atm/hist/'+control_name+'.cam.h0.*'))
x4_files = ns.natsorted(glob.glob(wd+x4_name+'/atm/hist/'+x4_name+'.cam.h0.*'))

control = xr.open_mfdataset(control_files)
x4 = xr.open_mfdataset(x4_files)

lat = control['lat'].values
lon = control['lon'].values

y = lat*np.pi/180
coslat = np.cos(y)
coslat = np.tile(coslat,(lon.size,1)).T

print('- Control Net TOA flux')
control_fnet = (control['FSNT'].values)-(control['FLNT'].values)
control_fnet_grid = np.nanmean(control_fnet.reshape(30,12,46,72),axis=0)

print('- piClim-abrupt-4xCO2 Net TOA flux')
x4_fnet = (x4['FSNT'].values)-(x4['FLNT'].values)
x4_fnet_grid = np.nanmean(x4_fnet.reshape(30,12,46,72),axis=0)

print('- Effective Radiative Forcing')
erf = x4_fnet_grid-control_fnet_grid
erf_an = np.nanmean(erf,axis=0)
erf_gam = round(np.average(erf_an,weights=coslat),2)

print('- Planetary Albedo')
alpha = (control['FSDTOA']-control['FSNT'])/control['FSDTOA']
alpha_an = alpha.values
alpha_vals = np.nanmean(alpha_an.reshape(30,12,46,72),axis=0)
alpha_gam = round(np.average(np.nanmean(alpha_vals,axis=0),weights=coslat),2)

#print('read in the LWE for solar constant offset reductions')
#LWE = pd.read_csv('piClim-abrupt-4xCO2_LWE.csv')
#LWE_gam = LWE['LWE'][int(case)-1]
#erf_lwe = round(erf_gam-LWE_gam,2)

offset = Calc_s0(erf_gam,alpha_gam)
print('Global mean ERF - {}'.format(erf_gam))
#print('Global mean ERF (LWE corrected) - {}'.format(erf_lwe))
print('Global mean Planetary Albedo - {}'.format(alpha_gam))
print('Required Solar Constant Offset - {}'.format(offset))

new_sc = round(1360.89-offset,2)

p = Path('data/G1_inputs.csv')

if p.exists():

    params = pd.read_csv('data/G1_inputs.csv')
    addon = pd.DataFrame(\
        {'Case': [case],'ERF': [erf_gam],'alpha': [alpha_gam], 's0': [offset],'s': [new_sc]})
    params = params.append(addon)
    params.to_csv('data/G1_inputs.csv',index=False)

else:

    params = pd.DataFrame({'Case': [case],'ERF': [erf_gam],'alpha': [alpha_gam], 's0': [offset],'s': [new_sc]})
    params.to_csv('data/G1_inputs.csv',index=False)
