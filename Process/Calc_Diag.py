#!/usr/bin/env python3

print('\tread in packages')
import numpy as np 
import xarray as xr
import pickle as pk
import natsort as ns 
import glob 
import sys

header ='f'
header_pre = header.capitalize()

compset = sys.argv[1]

case = '1'
if compset == '1850CN' or compset == 'piClim-abrupt-4xCO2':
    case = case
else:
    case = '1b'

if compset == '1850CN':
    compset = header_pre+compset

version = 'e12'
res = 'f45_f45'
author = 'JGV'
full_case = header+'.'+version+'.'+compset+'.'+res+'.'+author+'.'+case

print('\tcase -',full_case)

wd = '/scratch/c/cgf/jgvirgin/cesm1_2_2/archive/'

print('\treading in data')
files = ns.natsorted(glob.glob(wd+full_case+'/atm/hist/'+full_case+'.cam.h0.*'))
data = xr.open_mfdataset(files)

lat = data['lat'].values
lon = data['lon'].values

tau = data['cosp_tau'].values
plevs = data['cosp_prs'].values

print('\taveraging variables')

print('\t\t- surface temp')
tas = data['TS'].values
tas_an = np.mean(tas.reshape(30,12,46,72),axis=0)

print('\t\t- air temp')
ta = data['T'].values
ta_an = np.mean(ta.reshape(30,12,26,46,72),axis=0)

print('\t\t- cloud radiative effects')
cre_lw = data['FLNT']-data['FLNTC']
cre_lw = cre_lw.values
cre_lw_an = np.mean(cre_lw.reshape(30,12,46,72),axis=0)

cre_sw = data['FSNT']-data['FSNTC']
cre_sw = cre_sw.values
cre_sw_an = np.mean(cre_sw.reshape(30,12,46,72),axis=0)

print('\t\t- ISCCP cloud fraction')
cl = data['FISCCP1_COSP']
cl = cl.values
cl_an = np.mean(cl.reshape(30,12,7,7,46,72),axis=0)

print('\t\t- Regular cloud fraction')
cloud = data['CLOUD'].values
cloud_an = np.mean(cloud.reshape(30,12,26,46,72),axis=0)

print('\t\t- Net longwave TOA flux')
flnt = data['FLNT'].values
flnt_an = np.mean(flnt.reshape(30,12,46,72),axis=0)

print('\t\t- Net shortwave TOA flux')
fsnt = data['FSNT'].values
fsnt_an = np.mean(fsnt.reshape(30,12,46,72),axis=0)

print('\t\t- Net TOA flux')
fnet = data['FSNT']-data['FLNT']
fnet = fnet.values
fnet_an = np.mean(fnet.reshape(30,12,46,72),axis=0)

print('\t\t- Clear sky Net TOA flux')
fnetc = data['FSNT']-data['FLNT']
fnetc = fnetc.values
fnetc_an = np.mean(fnetc.reshape(30,12,46,72),axis=0)

print('\t\t- Total Sky Surface Albedo')
salb = (data['FSDS']-data['FSNS'])/data['FSDS']
salb = salb.values
salb_an = np.mean(salb.reshape(30,12,46,72),axis=0)*100

print('\t\t- Clear Sky Surface Albedo')
csalb = (data['FSDSC']-data['FSNSC'])/data['FSDSC']
csalb = csalb.values
csalb_an = np.mean(csalb.reshape(30,12,46,72),axis=0)

print('\t\t- Specific Humidity')
q = data['Q'].values
q_an = np.log(np.mean(q.reshape(30,12,26,46,72),axis=0))

print('\t\t- Surface Pressure')
ps = data['PS'].values
ps = ps/100
ps_an = np.mean(ps.reshape(30,12,46,72),axis=0)

print('\taggregate variables')
output = {}
output['tas'] = tas_an
output['ta'] = ta_an
output['CRElw'] = cre_lw_an
output['CREsw'] = cre_sw_an
output['isccp'] = cl_an
output['cloud'] = cloud_an
output['flnt'] = flnt_an
output['fsnt'] = fsnt_an
output['fnet'] = fnet_an
output['fnetc'] = fnetc_an
output['salb'] = salb_an
output['csalb'] = csalb_an
output['lnQ'] = q_an
output['ps'] = ps_an
output['tau'] = tau
output['ctp'] = plevs
output['lat'] = lat
output['lon'] = lon

print('\tsaving\n')
pk.dump(output,open('F_'+compset+'_diag_grid.pi','wb'))