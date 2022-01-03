#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle as pk
from scipy.interpolate import griddata

print('read in data')

compset = ['F1850CN','piClim-G1','piClim-abrupt-4xCO2','piClim-abrupt-SOLr','piClim-G1_redux']
wd = '/Volumes/eSSD0/Papers/GRL_G1RF/'
data = {}
ta = {}
q = {}
cloud = {}
for i in range(len(compset)):

    data[compset[i]] = pk.load(open(wd+'data/F_'+compset[i]+'_diag_grid.pi','rb'))
    ta[compset[i]] = data[compset[i]]['ta']
    q[compset[i]] = data[compset[i]]['lnQ']
    cloud[compset[i]] = data[compset[i]]['cloud']

print('read in kernel dimensions')

cmip_plevs = np.asarray([10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
cmip_tile = np.tile(cmip_plevs[None,:,None,None],(12,1,46,72))
native_plevs = pk.load(open(wd+'data/plevs.pi','rb'))

print('vertical interpolation')

ta_interp = {}
q_interp = {}
cloud_interp = {}
for comp in data.keys():

    print('\t on compset -',comp)

    ta_interp[comp]= np.zeros([12,17,46,72])
    q_interp[comp] = np.zeros([12,17,46,72])
    cloud_interp[comp] = np.zeros([12,17,46,72])

    for i in range(12):
        for j in range(46):
            for k in range(72):

                t_profile = np.squeeze(ta[comp][i,:,j,k])
                ta_interp[comp][i,:,j,k] = griddata(\
                    np.squeeze(native_plevs[i,:,j,k]),t_profile,cmip_plevs,method="nearest")

                q_profile = np.squeeze(q[comp][i,:,j,k])
                q_interp[comp][i,:,j,k] = griddata(\
                    np.squeeze(native_plevs[i,:,j,k]),q_profile,cmip_plevs,method="nearest")

                cloud_profile = np.squeeze(cloud[comp][i,:,j,k])
                cloud_interp[comp][i,:,j,k] = griddata(\
                    np.squeeze(native_plevs[i,:,j,k]),cloud_profile,cmip_plevs,method="nearest")

print('Done, Saving to new files')
for comp in data.keys():
    data[comp]['ta_int'] = ta_interp[comp]
    data[comp]['lnQ_int'] = q_interp[comp]
    data[comp]['cloud_int'] = cloud_interp[comp]

    pk.dump(data[comp],open(wd+'data/interpolated/F_'+comp+'_diag_grid_int.pi','wb'))


