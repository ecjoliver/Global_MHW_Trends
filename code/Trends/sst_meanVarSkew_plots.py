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
# Load SST mean, variance, skewness data
#

outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/sst_meanVarSkew'

data = np.load(outfile+'.npz')
lon_map = data['lon_map']
lat_map = data['lat_map']
years = np.unique(data['years'])
SST_mean_ts = data['SST_mean_ts']
SST_var_ts = data['SST_var_ts']
SST_skew_ts = data['SST_skew_ts']

SST_mean_ts[SST_mean_ts==0] = np.nan
SST_var_ts[SST_var_ts==0] = np.nan
SST_skew_ts[SST_skew_ts==0] = np.nan

# Calculate trends
SST_mean_tr = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_var_tr = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_skew_tr = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_mean_dtr = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_var_dtr = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_skew_dtr = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_mean_tr_TS = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_var_tr_TS = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_skew_tr_TS = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_mean_dtr_TS = np.nan*np.zeros((SST_mean_ts.shape[0], SST_mean_ts.shape[1], 2))
SST_var_dtr_TS = np.nan*np.zeros((SST_mean_ts.shape[0], SST_mean_ts.shape[1], 2))
SST_skew_dtr_TS = np.nan*np.zeros((SST_mean_ts.shape[0], SST_mean_ts.shape[1], 2))
for i in range(len(lon_map)):
    print i+1, 'of', len(lon_map)
    for j in range(len(lat_map)):
        if ~np.isnan(SST_mean_ts[j,i,:].sum()):
            SST_mean_tr[j,i], SST_mean_dtr[j,i] = ecj.trend(years, SST_mean_ts[j,i,:])[1:]
            SST_mean_tr_TS[j,i], SST_mean_dtr_TS[j,i,:] = ecj.trend_TheilSen(years, SST_mean_ts[j,i,:])[1:]
        if ~np.isnan(SST_var_ts[j,i,:].sum()):
            SST_var_tr[j,i], SST_var_dtr[j,i] = ecj.trend(years, SST_var_ts[j,i,:])[1:]
            SST_var_tr_TS[j,i], SST_var_dtr_TS[j,i,:] = ecj.trend_TheilSen(years, SST_var_ts[j,i,:])[1:]
        if ~np.isnan(SST_skew_ts[j,i,:].sum()):
            SST_skew_tr[j,i], SST_skew_dtr[j,i] = ecj.trend(years, SST_skew_ts[j,i,:])[1:]
            SST_skew_tr_TS[j,i], SST_skew_dtr_TS[j,i,:] = ecj.trend_TheilSen(years, SST_skew_ts[j,i,:])[1:]

outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/sst_meanVarSkew_tr'
# np.savez(outfile, SST_mean_tr=SST_mean_tr, SST_var_tr=SST_var_tr, SST_skew_tr=SST_skew_tr, SST_mean_dtr=SST_mean_dtr, SST_var_dtr=SST_var_dtr, SST_skew_dtr=SST_skew_dtr, SST_mean_tr_TS=SST_mean_tr_TS, SST_var_tr_TS=SST_var_tr_TS, SST_skew_tr_TS=SST_skew_tr_TS, SST_mean_dtr_TS=SST_mean_dtr_TS, SST_var_dtr_TS=SST_var_dtr_TS, SST_skew_dtr_TS=SST_skew_dtr_TS)
data = np.load(outfile + '.npz')
SST_mean_tr = data['SST_mean_tr']
SST_var_tr = data['SST_var_tr']
SST_skew_tr = data['SST_skew_tr']
SST_mean_dtr = data['SST_mean_dtr']
SST_var_dtr = data['SST_var_dtr']
SST_skew_dtr = data['SST_skew_dtr']
SST_mean_tr_TS = data['SST_mean_tr_TS']
SST_var_tr_TS = data['SST_var_tr_TS']
SST_skew_tr_TS = data['SST_skew_tr_TS']
SST_mean_dtr_TS = data['SST_mean_dtr_TS']
SST_var_dtr_TS = data['SST_var_dtr_TS']
SST_skew_dtr_TS = data['SST_skew_dtr_TS']

#
# Excess trends information
#

outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_census.2016.excessTrends.lores.combine'
data = np.load(outfile+'.npz')
MHW_cnt_tr_lr = data['MHW_cnt_tr']
MHW_max_tr_lr = data['MHW_max_tr']
MHW_dur_tr_lr = data['MHW_dur_tr']
ar1_putrend_cnt = data['ar1_putrend_cnt']
ar1_putrend_max = data['ar1_putrend_max']
ar1_putrend_dur = data['ar1_putrend_dur']
ar1_pltrend_cnt = data['ar1_pltrend_cnt']
ar1_pltrend_max = data['ar1_pltrend_max']
ar1_pltrend_dur = data['ar1_pltrend_dur']
lon_map_lr = data['lon_map']
lat_map_lr = data['lat_map']

#
# Load iceMean map for making mask
#

matobj = io.loadmat('/home/ecoliver/Desktop/data/MHWs/Trends/NOAAOISST_iceMean.mat')
ice_mean = matobj['ice_mean']
ice_longestRun = matobj['ice_longestRun']
# Make data mask based on land and ice
datamask = np.ones(SST_mean_tr.shape)
datamask[np.isnan(SST_mean_tr)] = np.nan
datamask[ice_longestRun>=6.] = np.nan # mask where ice had runs of 6 days or longer
datamask[lat_map<=-65.,:] = np.nan
datamask_ts = np.swapaxes(np.swapaxes(np.tile(datamask, (len(years),1,1)), 0, 1), 1, 2)
datamask_TS = np.swapaxes(np.swapaxes(np.tile(datamask, (2,1,1)), 0, 1), 1, 2)
dl = len(lon_map)/len(lon_map_lr)
datamask_lr = datamask[::dl,::dl]

#
# Shift data 20E, apply data mask
#

# Shift
i_20E = np.where(lon_map>20)[0][0]
lon_map = np.append(lon_map[i_20E:], lon_map[:i_20E]+360)
datamask = np.append(datamask[:,i_20E:], datamask[:,:i_20E], axis=1)
SST_mean_ts = np.append(SST_mean_ts[:,i_20E:,:], SST_mean_ts[:,:i_20E,:], axis=1)
SST_var_ts = np.append(SST_var_ts[:,i_20E:,:], SST_var_ts[:,:i_20E,:], axis=1)
SST_skew_ts = np.append(SST_skew_ts[:,i_20E:,:], SST_skew_ts[:,:i_20E,:], axis=1)
SST_mean_tr = np.append(SST_mean_tr[:,i_20E:], SST_mean_tr[:,:i_20E], axis=1)
SST_var_tr = np.append(SST_var_tr[:,i_20E:], SST_var_tr[:,:i_20E], axis=1)
SST_skew_tr = np.append(SST_skew_tr[:,i_20E:], SST_skew_tr[:,:i_20E], axis=1)
SST_mean_dtr = np.append(SST_mean_dtr[:,i_20E:], SST_mean_dtr[:,:i_20E], axis=1)
SST_var_dtr = np.append(SST_var_dtr[:,i_20E:], SST_var_dtr[:,:i_20E], axis=1)
SST_skew_dtr = np.append(SST_skew_dtr[:,i_20E:], SST_skew_dtr[:,:i_20E], axis=1)
SST_mean_tr_TS = np.append(SST_mean_tr_TS[:,i_20E:], SST_mean_tr_TS[:,:i_20E], axis=1)
SST_var_tr_TS = np.append(SST_var_tr_TS[:,i_20E:], SST_var_tr_TS[:,:i_20E], axis=1)
SST_skew_tr_TS = np.append(SST_skew_tr_TS[:,i_20E:], SST_skew_tr_TS[:,:i_20E], axis=1)
SST_mean_dtr_TS = np.append(SST_mean_dtr_TS[:,i_20E:,:], SST_mean_dtr_TS[:,:i_20E,:], axis=1)
SST_var_dtr_TS = np.append(SST_var_dtr_TS[:,i_20E:,:], SST_var_dtr_TS[:,:i_20E,:], axis=1)
SST_skew_dtr_TS = np.append(SST_skew_dtr_TS[:,i_20E:,:], SST_skew_dtr_TS[:,:i_20E,:], axis=1)

i_20E = np.where(lon_map_lr>20)[0][0]
lon_map_lr = np.append(lon_map_lr[i_20E:], lon_map_lr[:i_20E]+360)
datamask_lr = np.append(datamask_lr[:,i_20E:], datamask_lr[:,:i_20E], axis=1)
ar1_putrend_cnt = np.append(ar1_putrend_cnt[:,i_20E:], ar1_putrend_cnt[:,:i_20E], axis=1)
ar1_putrend_max = np.append(ar1_putrend_max[:,i_20E:], ar1_putrend_max[:,:i_20E], axis=1)
ar1_putrend_dur = np.append(ar1_putrend_dur[:,i_20E:], ar1_putrend_dur[:,:i_20E], axis=1)
ar1_pltrend_cnt = np.append(ar1_pltrend_cnt[:,i_20E:], ar1_pltrend_cnt[:,:i_20E], axis=1)
ar1_pltrend_max = np.append(ar1_pltrend_max[:,i_20E:], ar1_pltrend_max[:,:i_20E], axis=1)
ar1_pltrend_dur = np.append(ar1_pltrend_dur[:,i_20E:], ar1_pltrend_dur[:,:i_20E], axis=1)
MHW_cnt_tr_lr = np.append(MHW_cnt_tr_lr[:,i_20E:], MHW_cnt_tr_lr[:,:i_20E], axis=1)
MHW_max_tr_lr = np.append(MHW_max_tr_lr[:,i_20E:], MHW_max_tr_lr[:,:i_20E], axis=1)
MHW_dur_tr_lr = np.append(MHW_dur_tr_lr[:,i_20E:], MHW_dur_tr_lr[:,:i_20E], axis=1)

# Datamask
SST_mean_ts = datamask_ts*SST_mean_ts
SST_var_ts = datamask_ts*SST_var_ts
SST_skew_ts = datamask_ts*SST_skew_ts
SST_mean_tr = datamask*SST_mean_tr
SST_var_tr = datamask*SST_var_tr
SST_skew_tr = datamask*SST_skew_tr
SST_mean_dtr = datamask*SST_mean_dtr
SST_var_dtr = datamask*SST_var_dtr
SST_skew_dtr = datamask*SST_skew_dtr
SST_mean_tr_TS = datamask*SST_mean_tr_TS
SST_var_tr_TS = datamask*SST_var_tr_TS
SST_skew_tr_TS = datamask*SST_skew_tr_TS
SST_mean_dtr_TS = datamask_TS*SST_mean_dtr_TS
SST_var_dtr_TS = datamask_TS*SST_var_dtr_TS
SST_skew_dtr_TS = datamask_TS*SST_skew_dtr_TS
ar1_putrend_cnt = datamask_lr*ar1_putrend_cnt
ar1_putrend_max = datamask_lr*datamask_lr*ar1_putrend_max
ar1_putrend_dur = datamask_lr*datamask_lr*ar1_putrend_dur
ar1_pltrend_cnt = datamask_lr*ar1_pltrend_cnt
ar1_pltrend_max = datamask_lr*datamask_lr*ar1_pltrend_max
ar1_pltrend_dur = datamask_lr*datamask_lr*ar1_pltrend_dur
MHW_cnt_tr_lr = datamask_lr*MHW_cnt_tr_lr
MHW_max_tr_lr = datamask_lr*datamask_lr*MHW_max_tr_lr
MHW_dur_tr_lr = datamask_lr*datamask_lr*MHW_dur_tr_lr

#
# Calculate Excess Trends, significant trends
#

sign_mean = datamask*(((np.abs(SST_mean_tr)-SST_mean_dtr)>0).astype(float))
sign_var = datamask*(((np.abs(SST_var_tr)-SST_var_dtr)>0).astype(float))
sign_skew = datamask*(((np.abs(SST_skew_tr)-SST_skew_dtr)>0).astype(float))

sign_mean_TS = datamask*((np.sum(np.sign(SST_mean_dtr_TS), axis=2)!=0)).astype(float)
sign_var_TS = datamask*((np.sum(np.sign(SST_var_dtr_TS), axis=2)!=0)).astype(float)
sign_skew_TS = datamask*((np.sum(np.sign(SST_skew_dtr_TS), axis=2)!=0)).astype(float)

excess_cnt = (MHW_cnt_tr_lr > ar1_putrend_cnt) + (MHW_cnt_tr_lr < ar1_pltrend_cnt)
excess_max = (MHW_max_tr_lr > ar1_putrend_max) + (MHW_max_tr_lr < ar1_pltrend_max)
excess_dur = (MHW_dur_tr_lr > ar1_putrend_dur) + (MHW_dur_tr_lr < ar1_pltrend_dur)

#
# Plots
#

domain = [-65, 20, 70, 380]
domain_draw = [-60, 20, 60, 380]
domain_draw = [-60, 60, 60, 380]
dlat = 30
dlon = 60
llon, llat = np.meshgrid(lon_map, lat_map)
llon_lr, llat_lr = np.meshgrid(lon_map_lr, lat_map_lr)
bg_col = '0.6'
cont_col = '1.0'
hatch = '//'
hatch_excess = '\\\\'

plt.figure(figsize=(7,10))
plt.clf()

plt.subplot(3,1,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, SST_mean_tr*10, levels=[-0.9,-0.6,-0.3,-0.1,0.1,0.3,0.6,0.9], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
for i in range(30):
    plt.contourf(lonproj, latproj, sign_mean, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
H.set_label(r'[$^\circ$C / decade]')
plt.title('( ) Mean SST Linear Trend')

plt.subplot(3,1,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, SST_var_tr*10, levels=[-2,-0.2,-0.05,-0.02,0.02,0.05,0.2,2], cmap=plt.cm.RdBu_r)
plt.clim(-0.35,0.35)
H = plt.colorbar()
for i in range(30):
    plt.contourf(lonproj, latproj, sign_var, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
H.set_label(r'[$^\circ$C$^2$ / decade]')
plt.title('(A) SST Variance Linear Trend')

plt.subplot(3,1,3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, SST_skew_tr*10, levels=[-1.5,-0.4,-0.08,-0.04,0.04,0.08,0.4,1.5], cmap=plt.cm.RdBu_r)
plt.clim(-0.75,0.75)
H = plt.colorbar()
for i in range(30):
    plt.contourf(lonproj, latproj, sign_skew, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
H.set_label(r'[1 / decade]')
plt.title('(B) SST Skewness Linear Trend')

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/sst_meanVarSkew_trends_orig.pdf', bbox_inches='tight', pad_inches=0.5)

plt.figure(figsize=(7,10))
plt.clf()
hatch = '////'

plt.subplot(3,1,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[6,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[6,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, SST_mean_tr_TS*10, levels=[-0.9,-0.6,-0.3,-0.1,0.1,0.3,0.6,0.9], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
plt.contourf(lonproj, latproj, sign_mean_TS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
H.set_label(r'[$^\circ$C / decade]')
plt.title('( ) Mean SST Linear Trend')

plt.subplot(3,1,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[6,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[6,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, SST_var_tr_TS*10, levels=[-1.5,-0.5,-0.25,-0.15,-0.05,0.05,0.15,0.25,0.5,1.5], cmap=plt.cm.RdBu_r)
plt.clim(-0.5, 0.5)
H = plt.colorbar()
plt.contourf(lonproj, latproj, sign_var_TS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
H.set_label(r'[$^\circ$C$^2$ / decade]')
plt.title('(A) SST Variance Linear Trend')

plt.subplot(3,1,3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[6,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[6,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, SST_skew_tr_TS*10, levels=[-1.,-0.75,-0.5,-0.1,0.1,0.5,0.75,1.], cmap=plt.cm.RdBu_r)
plt.clim(-0.75,0.75)
H = plt.colorbar()
plt.contourf(lonproj, latproj, sign_skew_TS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
H.set_label(r'[1 / decade]')
plt.title('(B) SST Skewness Linear Trend')

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/sst_meanVarSkew_trends_TS_orig.pdf', bbox_inches='tight', pad_inches=0.5)
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/sst_meanVarSkew_trends_TS_orig.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

#
# Proportion of excess trends explained by variance/skewness trends
#

# OLS

# Excess trends coinciding with variance trends
1.*(excess_cnt * (sign_var[::dl,::dl]==1)).sum() / excess_cnt.sum()
1.*(excess_max * (sign_var[::dl,::dl]==1)).sum() / excess_max.sum()
1.*(excess_dur * (sign_var[::dl,::dl]==1)).sum() / excess_dur.sum()

# Excess trends coinciding with skewness trends
1.*(excess_cnt * (sign_skew[::dl,::dl]==1)).sum() / excess_cnt.sum()
1.*(excess_max * (sign_skew[::dl,::dl]==1)).sum() / excess_max.sum()
1.*(excess_dur * (sign_skew[::dl,::dl]==1)).sum() / excess_dur.sum()

# Excess trends coinciding with both variance and skewness trends
1.*((excess_cnt * (sign_var[::dl,::dl]==1)) + (excess_cnt * (sign_skew[::dl,::dl]==1))).sum() / excess_cnt.sum()
1.*((excess_max * (sign_var[::dl,::dl]==1)) + (excess_max * (sign_skew[::dl,::dl]==1))).sum() / excess_max.sum()
1.*((excess_dur * (sign_var[::dl,::dl]==1)) + (excess_dur * (sign_skew[::dl,::dl]==1))).sum() / excess_dur.sum()

ecj.pattern_correlation(MHW_max_tr_lr, SST_var_tr[::dl,::dl])
ecj.pattern_correlation(MHW_max_tr_lr, SST_skew_tr[::dl,::dl])
ecj.pattern_correlation(MHW_dur_tr_lr, SST_var_tr[::dl,::dl])
ecj.pattern_correlation(MHW_dur_tr_lr, SST_skew_tr[::dl,::dl])

# TS

# Excess trends coinciding with variance trends
1.*(excess_cnt * (sign_var_TS[::dl,::dl]==1)).sum() / excess_cnt.sum()
1.*(excess_max * (sign_var_TS[::dl,::dl]==1)).sum() / excess_max.sum()
1.*(excess_dur * (sign_var_TS[::dl,::dl]==1)).sum() / excess_dur.sum()

# Excess trends coinciding with skewness trends
1.*(excess_cnt * (sign_skew_TS[::dl,::dl]==1)).sum() / excess_cnt.sum()
1.*(excess_max * (sign_skew_TS[::dl,::dl]==1)).sum() / excess_max.sum()
1.*(excess_dur * (sign_skew_TS[::dl,::dl]==1)).sum() / excess_dur.sum()

# Excess trends coinciding with both variance and skewness trends
1.*((excess_cnt * (sign_var_TS[::dl,::dl]==1)) + (excess_cnt * (sign_skew_TS[::dl,::dl]==1))).sum() / excess_cnt.sum()
1.*((excess_max * (sign_var_TS[::dl,::dl]==1)) + (excess_max * (sign_skew_TS[::dl,::dl]==1))).sum() / excess_max.sum()
1.*((excess_dur * (sign_var_TS[::dl,::dl]==1)) + (excess_dur * (sign_skew_TS[::dl,::dl]==1))).sum() / excess_dur.sum()

ecj.pattern_correlation(MHW_max_tr_lr, SST_var_tr_TS[::dl,::dl])





