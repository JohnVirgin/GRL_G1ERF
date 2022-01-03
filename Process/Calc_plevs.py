#!/usr/bin/env python3

print('read in packages')
import numpy as np 
import xarray as xr
import pickle as pk
import natsort as ns 
import glob 
import sys

#!/usr/bin/env python3

print('read in packages')
import numpy as np 
import xarray as xr
import pickle as pk
import natsort as ns 
import glob 
import sys

full_case = 'f.e12.F1850CN.f45_f45.JGV.1'

print('case -',full_case)

wd = '/scratch/c/cgf/jgvirgin/cesm1_2_2/archive/'

print('reading in data')
files = ns.natsorted(glob.glob(wd+full_case+'/atm/hist/'+full_case+'.cam.h0.*'))
data = xr.open_mfdataset(files[:12])

hyam = data['hyam'].values
hybm = data['hybm'].values
ps = data['PS'].values
p0 = data['P0'].values

print('Variable shapes?\n',hyam.shape,hybm.shape,ps.shape,p0.shape)

print('broadcasting')
hyam_t = np.tile(hyam[:,:,None,None],(1,1,46,72))
hybm_t = np.tile(hybm[:,:,None,None],(1,1,46,72))
ps_t = np.tile(ps[:,None,:,:],(1,26,1,1))
p0_t = np.tile(p0[:,None,None,None],(1,26,46,72))

plevs = (hyam_t*p0_t+(hybm_t*ps_t))/100

print(plevs[0,:,0,0])

pk.dump(plevs,open('plevs.pi','wb'))