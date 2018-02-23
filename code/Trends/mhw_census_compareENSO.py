'''

  Software which uses the MHW definition
  of Hobday et al. (2015) applied to 
  select SST time series around the globe

'''

import numpy as np
import scipy.signal as sig
from scipy import linalg
from scipy import stats
from scipy import io
from scipy import interpolate as interp

import ecoliver as ecj

from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm


#
# Load data and make plots
#

# Load iceMean map for making mask
matobj = io.loadmat('/home/ecoliver/Desktop/data/extreme_SSTs/NOAAOISST_iceMean.mat')
ice_mean = matobj['ice_mean']
ice_longestRun = matobj['ice_longestRun']
lat_ice = matobj['lat'][:,0]
lon_ice = matobj['lon'][:,0]
# Make data mask based on land and ice
datamask = np.ones(ice_mean.shape)
#datamask[np.isnan(ice_mean)] = np.nan
#datamask[~np.isnan(ice)] = np.nan # mask where there has been any ice
datamask[ice_longestRun>=6.] = np.nan # mask where ice had runs of 6 days or longer
datamask[lat_ice<=-65.,:] = np.nan
#dl = 10
#datamask_lr = datamask[::dl,::dl]
#datamask_ts = np.swapaxes(np.swapaxes(np.tile(datamask_lr, (35,1,1)), 0, 1), 1, 2)
datamask_ts = np.swapaxes(np.swapaxes(np.tile(datamask, (35,1,1)), 0, 1), 1, 2)


outfile = {}
outfile['wENSO'] = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016.lores'
outfile['wENSO'] = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016'
#outfile['noENSO'] = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016.lores.noENSO.lag0'
#outfile['noENSO'] = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016.lores.noENSO.leadLag.1yr'
#outfile['noENSO'] = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016.lores.noENSO.Hilbert'
##outfile['noENSO'] = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016.lores.noENSO.Hilbert.ssta'
#outfile['noENSO'] = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016.lores.noENSO.lag0.ssta'
#outfile['noENSO'] = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016.lores.noENSO.leadLag.1yr.ssta'
outfile['noENSO'] = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016.lores.noENSO.leadLag.1yr.monthly.ssta'
outfile['noENSO'] = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016.noENSO.leadLag.1yr.monthly.ssta'

#outfile['wENSO'] = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016'
#outfile['noENSO'] = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016.noENSO.leadLag.1yr.monthly.ssta'

SST_mean = {}
MHW_cnt = {}
MHW_dur = {}
MHW_max = {}
MHW_mean = {}
MHW_td = {}
SST_ts = {}
MHW_cnt_ts = {}
MHW_dur_ts = {}
MHW_max_ts = {}
MHW_mean_ts = {}
MHW_td_ts = {}
for key in outfile.keys():
    data = np.load(outfile[key]+'.npz')
    lon_map = data['lon_map']
    lat_map = data['lat_map']
    years = data['years']
    SST_mean[key] = data['SST_mean']*datamask#_lr
    MHW_cnt[key] = data['MHW_cnt']*datamask#_lr
    MHW_dur[key] = data['MHW_dur']*datamask#_lr
    MHW_max[key] = data['MHW_max']*datamask#_lr
    MHW_mean[key] = data['MHW_mean']*datamask#_lr
    MHW_td[key] = data['MHW_td']*datamask#_lr
    SST_ts[key] = data['SST_ts']*datamask_ts
    MHW_cnt_ts[key] = data['MHW_cnt_ts']*datamask_ts
    MHW_dur_ts[key] = data['MHW_dur_ts']*datamask_ts
    MHW_max_ts[key] = data['MHW_max_ts']*datamask_ts
    MHW_mean_ts[key] = data['MHW_mean_ts']*datamask_ts
    MHW_td_ts[key] = data['MHW_td_ts']*datamask_ts

#
# Plots
#

plt.clf()
plt.subplot(2,2,1)
plt.contourf(lon_map, lat_map, SST_mean['wENSO'])
plt.colorbar()
plt.subplot(2,2,2)
plt.contourf(lon_map, lat_map, SST_mean['noENSO'])
plt.colorbar()
plt.subplot(2,2,3)
plt.contourf(lon_map, lat_map, SST_mean['wENSO'] - SST_mean['noENSO'])
plt.colorbar()
plt.subplot(2,2,4)
plt.plot(years, np.nanmean(np.nanmean(SST_ts['wENSO'], axis=0), axis=0), 'k-', linewidth=2)
plt.plot(years, np.nanmean(np.nanmean(SST_ts['noENSO'], axis=0), axis=0), 'b-', linewidth=2)
plt.grid()

plt.clf()
plt.subplot(2,2,1)
plt.contourf(lon_map, lat_map, MHW_cnt['wENSO'], levels=np.arange(0,4+0.5,0.5))
plt.colorbar()
plt.subplot(2,2,2)
plt.contourf(lon_map, lat_map, MHW_cnt['noENSO'], levels=np.arange(0,4+0.5,0.5))
plt.colorbar()
plt.subplot(2,2,3)
plt.contourf(lon_map, lat_map, MHW_cnt['wENSO'] - MHW_cnt['noENSO'], levels=np.arange(-3,3+0.5,0.5), cmap=plt.cm.RdBu_r)
plt.clim(-1, 1)
plt.colorbar()
plt.subplot(2,2,4)
plt.plot(years, np.nanmean(np.nanmean(MHW_cnt_ts['wENSO'], axis=0), axis=0), 'k-', linewidth=2)
plt.plot(years, np.nanmean(np.nanmean(MHW_cnt_ts['noENSO'], axis=0), axis=0), 'b-', linewidth=2)
plt.grid()
plt.legend(['wENSO', 'woENSO'], loc='upper left', fontsize=12)
#plt.savefig('figures/ENSO/MHW_ENSO_cnt.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

plt.clf()
plt.subplot(2,2,1)
plt.contourf(lon_map, lat_map, MHW_dur['wENSO'], levels=np.arange(0,70+0.5,10))
plt.colorbar()
plt.subplot(2,2,2)
plt.contourf(lon_map, lat_map, MHW_dur['noENSO'], levels=np.arange(0,70+0.5,10))
plt.colorbar()
plt.subplot(2,2,3)
plt.contourf(lon_map, lat_map, MHW_dur['wENSO'] - MHW_dur['noENSO'], levels=np.arange(-55,55+1,10), cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.subplot(2,2,4)
plt.plot(years, np.nanmean(np.nanmean(MHW_dur_ts['wENSO'], axis=0), axis=0), 'k-', linewidth=2)
plt.plot(years, np.nanmean(np.nanmean(MHW_dur_ts['noENSO'], axis=0), axis=0), 'b-', linewidth=2)
plt.grid()
plt.legend(['wENSO', 'woENSO'], loc='upper left', fontsize=12)
#plt.savefig('figures/ENSO/MHW_ENSO_dur.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

plt.clf()
plt.subplot(2,2,1)
plt.contourf(lon_map, lat_map, MHW_max['wENSO'], levels=np.arange(0,5+0.5,0.5))
plt.colorbar()
plt.subplot(2,2,2)
plt.contourf(lon_map, lat_map, MHW_max['noENSO'], levels=np.arange(0,5+0.5,0.5))
plt.colorbar()
plt.subplot(2,2,3)
plt.contourf(lon_map, lat_map, MHW_max['wENSO'] - MHW_max['noENSO'], levels=np.arange(-1.2,1.2+0.2,0.2), cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.subplot(2,2,4)
plt.plot(years, np.nanmean(np.nanmean(MHW_max_ts['wENSO'], axis=0), axis=0), 'k-', linewidth=2)
plt.plot(years, np.nanmean(np.nanmean(MHW_max_ts['noENSO'], axis=0), axis=0), 'b-', linewidth=2)
plt.grid()
plt.legend(['wENSO', 'woENSO'], loc='upper left', fontsize=12)
#plt.savefig('figures/ENSO/MHW_ENSO_max.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

plt.clf()
plt.subplot(2,2,1)
plt.contourf(lon_map, lat_map, MHW_mean['wENSO'], levels=np.arange(0,5+0.5,0.5))
plt.colorbar()
plt.subplot(2,2,2)
plt.contourf(lon_map, lat_map, MHW_mean['noENSO'], levels=np.arange(0,5+0.5,0.5))
plt.colorbar()
plt.subplot(2,2,3)
plt.contourf(lon_map, lat_map, MHW_mean['wENSO'] - MHW_mean['noENSO'], levels=np.arange(-1.2,1.2+0.2,0.2), cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.subplot(2,2,4)
plt.plot(years, np.nanmean(np.nanmean(MHW_mean_ts['wENSO'], axis=0), axis=0), 'k-', linewidth=2)
plt.plot(years, np.nanmean(np.nanmean(MHW_mean_ts['noENSO'], axis=0), axis=0), 'b-', linewidth=2)
plt.grid()
plt.legend(['wENSO', 'woENSO'], loc='upper left', fontsize=12)
#plt.savefig('figures/ENSO/MHW_ENSO_mean.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

plt.clf()
plt.subplot(2,2,1)
plt.contourf(lon_map, lat_map, MHW_td['wENSO'], levels=np.arange(0,50+1,5))
plt.colorbar()
plt.subplot(2,2,2)
plt.contourf(lon_map, lat_map, MHW_td['noENSO'], levels=np.arange(0,50+1,5))
plt.colorbar()
plt.subplot(2,2,3)
plt.contourf(lon_map, lat_map, MHW_td['wENSO'] - MHW_td['noENSO'], levels=np.arange(-50,50+1,10), cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.subplot(2,2,4)
plt.plot(years, np.nanmean(np.nanmean(MHW_td_ts['wENSO'], axis=0), axis=0), 'k-', linewidth=2)
plt.plot(years, np.nanmean(np.nanmean(MHW_td_ts['noENSO'], axis=0), axis=0), 'b-', linewidth=2)
plt.grid()
plt.legend(['wENSO', 'woENSO'], loc='upper left', fontsize=12)
#plt.savefig('figures/ENSO/MHW_ENSO_td.png', bbox_inches='tight', pad_inches=0.5, dpi=150)


