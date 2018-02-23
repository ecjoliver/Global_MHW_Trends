'''

  Software which uses the MHW definition
  of Hobday et al. (2015) applied to 
  select SST time series around the globe

'''

import numpy as np
import scipy.signal as sig
from scipy import linalg
from scipy import stats

import ecoliver as ecj

from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm


#
# Load data and make plots
#

outfile = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census'

data = np.load(outfile+'.npz')
lon_map = data['lon_map']
lat_map = data['lat_map']
N_ts = data['N_ts']
years = data['years']
SST_ts = data['SST_ts']
MHW_cnt_ts = data['MHW_cnt_ts']
MHW_dur_ts = data['MHW_dur_ts']
MHW_max_ts = data['MHW_max_ts']
MHW_mean_ts = data['MHW_mean_ts']
MHW_cum_ts = data['MHW_cum_ts']
MHW_td_ts = data['MHW_td_ts']
MHW_tc_ts = data['MHW_tc_ts']

SST_ts[SST_ts==0] = np.nan
MHW_dur_ts[MHW_cnt_ts==0] = np.nan
MHW_max_ts[MHW_cnt_ts==0] = np.nan
MHW_mean_ts[MHW_cnt_ts==0] = np.nan
MHW_cum_ts[MHW_cnt_ts==0] = np.nan
MHW_td_ts[MHW_cnt_ts==0] = np.nan
MHW_tc_ts[MHW_cnt_ts==0] = np.nan

#
# Initialize trend variables
#

alpha = 0.05
X = len(lon_map)
Y = len(lat_map)
DIM = (Y,X)
MHW_cnt_OLS_tr = np.NaN*np.zeros(DIM)
MHW_dur_OLS_tr = np.NaN*np.zeros(DIM)
MHW_max_OLS_tr = np.NaN*np.zeros(DIM)
MHW_mean_OLS_tr = np.NaN*np.zeros(DIM)
MHW_cum_OLS_tr = np.NaN*np.zeros(DIM)
MHW_td_OLS_tr = np.NaN*np.zeros(DIM)
MHW_tc_OLS_tr = np.NaN*np.zeros(DIM)
SST_OLS_dtr = np.NaN*np.zeros(DIM)
MHW_cnt_OLS_dtr = np.NaN*np.zeros(DIM)
MHW_dur_OLS_dtr = np.NaN*np.zeros(DIM)
MHW_max_OLS_dtr = np.NaN*np.zeros(DIM)
MHW_mean_OLS_dtr = np.NaN*np.zeros(DIM)
MHW_cum_OLS_dtr = np.NaN*np.zeros(DIM)
MHW_td_OLS_dtr = np.NaN*np.zeros(DIM)
MHW_tc_OLS_dtr = np.NaN*np.zeros(DIM)
MHW_cnt_TS_tr = np.NaN*np.zeros(DIM)
MHW_dur_TS_tr = np.NaN*np.zeros(DIM)
MHW_max_TS_tr = np.NaN*np.zeros(DIM)
MHW_mean_TS_tr = np.NaN*np.zeros(DIM)
MHW_cum_TS_tr = np.NaN*np.zeros(DIM)
MHW_td_TS_tr = np.NaN*np.zeros(DIM)
MHW_tc_TS_tr = np.NaN*np.zeros(DIM)
DIM = (Y,X,2)
SST_TS_dtr = np.NaN*np.zeros(DIM)
MHW_cnt_TS_dtr = np.NaN*np.zeros(DIM)
MHW_dur_TS_dtr = np.NaN*np.zeros(DIM)
MHW_max_TS_dtr = np.NaN*np.zeros(DIM)
MHW_mean_TS_dtr = np.NaN*np.zeros(DIM)
MHW_cum_TS_dtr = np.NaN*np.zeros(DIM)
MHW_td_TS_dtr = np.NaN*np.zeros(DIM)
MHW_tc_TS_dtr = np.NaN*np.zeros(DIM)

#
# Trend functions
#

def meanTrend_OLS(t, y, alpha=0.05):
    # Initialize mean and trend dictionaries
    mean = {}
    trend = {}
    dtrend = {}
#
    # Construct matrix of predictors, first column is all ones to estimate the mean,
    # second column is the time vector, equal to zero at mid-point.
    X = np.array([np.ones(t.shape), t-t.mean()]).T
#
    # Predictand (MHW property of interest)
    valid = ~np.isnan(y) # non-NaN indices
#
    # Perform linear regression over valid indices
    if np.sum(~np.isnan(y)) > 0: # If at least one non-NaN value
        beta = linalg.lstsq(X[valid,:], y[valid])[0]
    else:
        beta = [np.nan, np.nan]
#
    # Insert regression coefficients into mean and trend dictionaries
    mean = beta[0]
    trend = beta[1]
#
    # Confidence limits on trend
    yhat = np.sum(beta*X, axis=1)
    t_stat = stats.t.isf(alpha/2, len(t[valid])-2)
    s = np.sqrt(np.sum((y[valid] - yhat[valid])**2) / (len(t[valid])-2))
    Sxx = np.sum(X[valid,1]**2) - (np.sum(X[valid,1])**2)/len(t[valid]) # np.var(X, axis=1)[1]
    dbeta1 = t_stat * s / np.sqrt(Sxx)
    dtrend = dbeta1
#
    return mean, trend, dtrend

def meanTrend_TS(t, y, alpha=0.05):
    # Initialize mean and trend dictionaries
    mean = {}
    trend = {}
    dtrend = {}
#
    # Construct matrix of predictors, first column is all ones to estimate the mean,
    # second column is the time vector, equal to zero at mid-point.
    X = t-t.mean()
#
    # Predictand (MHW property of interest)
    valid = ~np.isnan(y) # non-NaN indices
#
    # Perform linear regression over valid indices
    if np.sum(~np.isnan(y)) > 0: # If at least one non-NaN value
        slope, y0, beta_lr, beta_up = stats.mstats.theilslopes(y[valid], X[valid], alpha=1-alpha)
        beta = np.array([y0, slope])
    else:
        beta_lr, beta_up = [np.nan, np.nan]
        beta = [np.nan, np.nan]
#
    # Insert regression coefficients into mean and trend dictionaries
    mean = beta[0]
    trend = beta[1]
    dtrend = [beta_lr, beta_up]
#
    return mean, trend, dtrend

#
# Loop over all spatial locations
#

for i in range(X):
    print i+1, 'of', X
    for j in range(Y):
        if np.isnan(MHW_cum_ts[j,i,:]).sum() == len(years):
            continue
        # OLS fit
        tmp, MHW_cnt_OLS_tr[j,i], MHW_cnt_OLS_dtr[j,i] = meanTrend_OLS(years, MHW_cnt_ts[j,i,:], alpha)
        tmp, MHW_dur_OLS_tr[j,i], MHW_dur_OLS_dtr[j,i] = meanTrend_OLS(years, MHW_dur_ts[j,i,:], alpha)
        tmp, MHW_mean_OLS_tr[j,i], MHW_mean_OLS_dtr[j,i] = meanTrend_OLS(years, MHW_mean_ts[j,i,:], alpha)
        tmp, MHW_max_OLS_tr[j,i], MHW_max_OLS_dtr[j,i] = meanTrend_OLS(years, MHW_max_ts[j,i,:], alpha)
        tmp, MHW_cum_OLS_tr[j,i], MHW_cum_OLS_dtr[j,i] = meanTrend_OLS(years, MHW_cum_ts[j,i,:], alpha)
        tmp, MHW_td_OLS_tr[j,i], MHW_td_OLS_dtr[j,i] = meanTrend_OLS(years, MHW_td_ts[j,i,:], alpha)
        tmp, MHW_tc_OLS_tr[j,i], MHW_tc_OLS_dtr[j,i] = meanTrend_OLS(years, MHW_tc_ts[j,i,:], alpha)
        # TS fit
        tmp, MHW_cnt_TS_tr[j,i], MHW_cnt_TS_dtr[j,i,:] = meanTrend_TS(years, MHW_cnt_ts[j,i,:], alpha)
        tmp, MHW_dur_TS_tr[j,i], MHW_dur_TS_dtr[j,i,:] = meanTrend_TS(years, MHW_dur_ts[j,i,:], alpha)
        tmp, MHW_mean_TS_tr[j,i], MHW_mean_TS_dtr[j,i,:] = meanTrend_TS(years, MHW_mean_ts[j,i,:], alpha)
        tmp, MHW_max_TS_tr[j,i], MHW_max_TS_dtr[j,i,:] = meanTrend_TS(years, MHW_max_ts[j,i,:], alpha)
        tmp, MHW_cum_TS_tr[j,i], MHW_cum_TS_dtr[j,i,:] = meanTrend_TS(years, MHW_cum_ts[j,i,:], alpha)
        tmp, MHW_td_TS_tr[j,i], MHW_td_TS_dtr[j,i,:] = meanTrend_TS(years, MHW_td_ts[j,i,:], alpha)
        tmp, MHW_tc_TS_tr[j,i], MHW_tc_TS_dtr[j,i,:] = meanTrend_TS(years, MHW_tc_ts[j,i,:], alpha)

# save data
# outfile = '/home/ecoliver/Desktop/data/extreme_SSTs/compare_OLS_global.npz'
# np.savez(outfile, lon_map=lon_map, lat_map=lat_map, MHW_cnt_OLS_tr=MHW_cnt_OLS_tr, MHW_dur_OLS_tr=MHW_dur_OLS_tr, MHW_max_OLS_tr=MHW_max_OLS_tr, MHW_mean_OLS_tr=MHW_mean_OLS_tr, MHW_cum_OLS_tr=MHW_cum_OLS_tr, MHW_td_OLS_tr=MHW_td_OLS_tr, MHW_tc_OLS_tr=MHW_tc_OLS_tr, MHW_cnt_OLS_dtr=MHW_cnt_OLS_dtr, MHW_dur_OLS_dtr=MHW_dur_OLS_dtr, MHW_max_OLS_dtr=MHW_max_OLS_dtr, MHW_mean_OLS_dtr=MHW_mean_OLS_dtr, MHW_cum_OLS_dtr=MHW_cum_OLS_dtr, MHW_td_OLS_dtr=MHW_td_OLS_dtr, MHW_tc_OLS_dtr=MHW_tc_OLS_dtr, MHW_cnt_TS_tr=MHW_cnt_TS_tr, MHW_dur_TS_tr=MHW_dur_TS_tr, MHW_max_TS_tr=MHW_max_TS_tr, MHW_mean_TS_tr=MHW_mean_TS_tr, MHW_cum_TS_tr=MHW_cum_TS_tr, MHW_td_TS_tr=MHW_td_TS_tr, MHW_tc_TS_tr=MHW_tc_TS_tr, MHW_cnt_TS_dtr=MHW_cnt_TS_dtr, MHW_dur_TS_dtr=MHW_dur_TS_dtr, MHW_max_TS_dtr=MHW_max_TS_dtr, MHW_mean_TS_dtr=MHW_mean_TS_dtr, MHW_cum_TS_dtr=MHW_cum_TS_dtr, MHW_td_TS_dtr=MHW_td_TS_dtr, MHW_tc_TS_dtr=MHW_tc_TS_dtr)

#
# Load data
#

outfile = '/home/ecoliver/Desktop/data/extreme_SSTs/compare_OLS_global'
data = np.load(outfile+'.npz')
MHW_cnt_OLS_tr = data['MHW_cnt_OLS_tr']
MHW_dur_OLS_tr = data['MHW_dur_OLS_tr']
MHW_mean_OLS_tr = data['MHW_mean_OLS_tr']
MHW_max_OLS_tr = data['MHW_max_OLS_tr']
MHW_cum_OLS_tr = data['MHW_cum_OLS_tr']
MHW_td_OLS_tr = data['MHW_td_OLS_tr']
MHW_tc_OLS_tr = data['MHW_tc_OLS_tr']
MHW_cnt_OLS_dtr = data['MHW_cnt_OLS_dtr']
MHW_dur_OLS_dtr = data['MHW_dur_OLS_dtr']
MHW_mean_OLS_dtr = data['MHW_mean_OLS_dtr']
MHW_max_OLS_dtr = data['MHW_max_OLS_dtr']
MHW_cum_OLS_dtr = data['MHW_cum_OLS_dtr']
MHW_td_OLS_dtr = data['MHW_td_OLS_dtr']
MHW_tc_OLS_dtr = data['MHW_tc_OLS_dtr']
MHW_cnt_TS_tr = data['MHW_cnt_TS_tr']
MHW_dur_TS_tr = data['MHW_dur_TS_tr']
MHW_mean_TS_tr = data['MHW_mean_TS_tr']
MHW_max_TS_tr = data['MHW_max_TS_tr']
MHW_cum_TS_tr = data['MHW_cum_TS_tr']
MHW_td_TS_tr = data['MHW_td_TS_tr']
MHW_tc_TS_tr = data['MHW_tc_TS_tr']
MHW_cnt_TS_dtr = data['MHW_cnt_TS_dtr']
MHW_dur_TS_dtr = data['MHW_dur_TS_dtr']
MHW_mean_TS_dtr = data['MHW_mean_TS_dtr']
MHW_max_TS_dtr = data['MHW_max_TS_dtr']
MHW_cum_TS_dtr = data['MHW_cum_TS_dtr']
MHW_td_TS_dtr = data['MHW_td_TS_dtr']
MHW_tc_TS_dtr = data['MHW_tc_TS_dtr']

# Re-map to run 20E to 380E
i_20E = np.where(lon_map>20)[0][0]
lon_map = np.append(lon_map[i_20E:], lon_map[:i_20E]+360)
MHW_cnt_OLS_tr = np.append(MHW_cnt_OLS_tr[:,i_20E:], MHW_cnt_OLS_tr[:,:i_20E], axis=1)
MHW_dur_OLS_tr = np.append(MHW_dur_OLS_tr[:,i_20E:], MHW_dur_OLS_tr[:,:i_20E], axis=1)
MHW_mean_OLS_tr = np.append(MHW_mean_OLS_tr[:,i_20E:], MHW_mean_OLS_tr[:,:i_20E], axis=1)
MHW_max_OLS_tr = np.append(MHW_max_OLS_tr[:,i_20E:], MHW_max_OLS_tr[:,:i_20E], axis=1)
MHW_cum_OLS_tr = np.append(MHW_cum_OLS_tr[:,i_20E:], MHW_cum_OLS_tr[:,:i_20E], axis=1)
MHW_td_OLS_tr = np.append(MHW_td_OLS_tr[:,i_20E:], MHW_td_OLS_tr[:,:i_20E], axis=1)
MHW_tc_OLS_tr = np.append(MHW_tc_OLS_tr[:,i_20E:], MHW_tc_OLS_tr[:,:i_20E], axis=1)
MHW_cnt_OLS_dtr = np.append(MHW_cnt_OLS_dtr[:,i_20E:], MHW_cnt_OLS_dtr[:,:i_20E], axis=1)
MHW_dur_OLS_dtr = np.append(MHW_dur_OLS_dtr[:,i_20E:], MHW_dur_OLS_dtr[:,:i_20E], axis=1)
MHW_mean_OLS_dtr = np.append(MHW_mean_OLS_dtr[:,i_20E:], MHW_mean_OLS_dtr[:,:i_20E], axis=1)
MHW_max_OLS_dtr = np.append(MHW_max_OLS_dtr[:,i_20E:], MHW_max_OLS_dtr[:,:i_20E], axis=1)
MHW_cum_OLS_dtr = np.append(MHW_cum_OLS_dtr[:,i_20E:], MHW_cum_OLS_dtr[:,:i_20E], axis=1)
MHW_td_OLS_dtr = np.append(MHW_td_OLS_dtr[:,i_20E:], MHW_td_OLS_dtr[:,:i_20E], axis=1)
MHW_tc_OLS_dtr = np.append(MHW_tc_OLS_dtr[:,i_20E:], MHW_tc_OLS_dtr[:,:i_20E], axis=1)
MHW_cnt_TS_tr = np.append(MHW_cnt_TS_tr[:,i_20E:], MHW_cnt_TS_tr[:,:i_20E], axis=1)
MHW_dur_TS_tr = np.append(MHW_dur_TS_tr[:,i_20E:], MHW_dur_TS_tr[:,:i_20E], axis=1)
MHW_mean_TS_tr = np.append(MHW_mean_TS_tr[:,i_20E:], MHW_mean_TS_tr[:,:i_20E], axis=1)
MHW_max_TS_tr = np.append(MHW_max_TS_tr[:,i_20E:], MHW_max_TS_tr[:,:i_20E], axis=1)
MHW_cum_TS_tr = np.append(MHW_cum_TS_tr[:,i_20E:], MHW_cum_TS_tr[:,:i_20E], axis=1)
MHW_td_TS_tr = np.append(MHW_td_TS_tr[:,i_20E:], MHW_td_TS_tr[:,:i_20E], axis=1)
MHW_tc_TS_tr = np.append(MHW_tc_TS_tr[:,i_20E:], MHW_tc_TS_tr[:,:i_20E], axis=1)
MHW_cnt_TS_dtr = np.append(MHW_cnt_TS_dtr[:,i_20E:,:], MHW_cnt_TS_dtr[:,:i_20E,:], axis=1)
MHW_dur_TS_dtr = np.append(MHW_dur_TS_dtr[:,i_20E:,:], MHW_dur_TS_dtr[:,:i_20E,:], axis=1)
MHW_mean_TS_dtr = np.append(MHW_mean_TS_dtr[:,i_20E:,:], MHW_mean_TS_dtr[:,:i_20E,:], axis=1)
MHW_max_TS_dtr = np.append(MHW_max_TS_dtr[:,i_20E:,:], MHW_max_TS_dtr[:,:i_20E,:], axis=1)
MHW_cum_TS_dtr = np.append(MHW_cum_TS_dtr[:,i_20E:,:], MHW_cum_TS_dtr[:,:i_20E,:], axis=1)
MHW_td_TS_dtr = np.append(MHW_td_TS_dtr[:,i_20E:,:], MHW_td_TS_dtr[:,:i_20E,:], axis=1)
MHW_tc_TS_dtr = np.append(MHW_tc_TS_dtr[:,i_20E:,:], MHW_tc_TS_dtr[:,:i_20E,:], axis=1)

#
# Plots
#

domain = [-70, 20, 70, 380]
domain_draw = [-60, 20, 60, 380]
dlat = 30
dlon = 60
llon, llat = np.meshgrid(lon_map, lat_map)

hatch = '////'
mask = np.ones(llat.shape)
mask[np.isnan(MHW_dur_OLS_tr)] = np.nan
sign_cnt_OLS = mask*(((np.abs(MHW_cnt_OLS_tr)-MHW_cnt_OLS_dtr)>0).astype(float))
sign_dur_OLS = mask*(((np.abs(MHW_dur_OLS_tr)-MHW_dur_OLS_dtr)>0).astype(float))
sign_mean_OLS = mask*(((np.abs(MHW_mean_OLS_tr)-MHW_mean_OLS_dtr)>0).astype(float))
sign_max_OLS = mask*(((np.abs(MHW_max_OLS_tr)-MHW_max_OLS_dtr)>0).astype(float))
sign_cum_OLS = mask*(((np.abs(MHW_cum_OLS_tr)-MHW_cum_OLS_dtr)>0).astype(float))
sign_td_OLS = mask*(((np.abs(MHW_td_OLS_tr)-MHW_td_OLS_dtr)>0).astype(float))
sign_tc_OLS = mask*(((np.abs(MHW_tc_OLS_tr)-MHW_tc_OLS_dtr)>0).astype(float))
sign_cnt_TS = mask*(( (MHW_cnt_TS_tr<MHW_cnt_TS_dtr[:,:,0]) + (MHW_cnt_TS_tr>MHW_cnt_TS_dtr[:,:,1]) ).astype(float))
sign_dur_TS = mask*(( (MHW_dur_TS_tr<MHW_dur_TS_dtr[:,:,0]) + (MHW_dur_TS_tr>MHW_dur_TS_dtr[:,:,1]) ).astype(float))
sign_mean_TS = mask*(( (MHW_mean_TS_tr<MHW_mean_TS_dtr[:,:,0]) + (MHW_mean_TS_tr>MHW_mean_TS_dtr[:,:,1]) ).astype(float))
sign_max_TS = mask*(( (MHW_max_TS_tr<MHW_max_TS_dtr[:,:,0]) + (MHW_max_TS_tr>MHW_max_TS_dtr[:,:,1]) ).astype(float))
sign_cum_TS = mask*(( (MHW_cum_TS_tr<MHW_cum_TS_dtr[:,:,0]) + (MHW_cum_TS_tr>MHW_cum_TS_dtr[:,:,1]) ).astype(float))
sign_td_TS = mask*(( (MHW_td_TS_tr<MHW_td_TS_dtr[:,:,0]) + (MHW_td_TS_tr>MHW_td_TS_dtr[:,:,1]) ).astype(float))
sign_tc_TS = mask*(( (MHW_tc_TS_tr<MHW_tc_TS_dtr[:,:,0]) + (MHW_tc_TS_tr>MHW_tc_TS_dtr[:,:,1]) ).astype(float))

fig = plt.figure(figsize=(20,3))

plt.clf()
AX = plt.subplot(1,3,1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_cnt_OLS_tr*10, levels=[-3.5,-2,-1,-0.5,0.5,1,2,3.5], cmap=plt.cm.RdBu_r)
H = plt.contourf(lonproj, latproj, sign_cnt_OLS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Frequency OLS trend')
AX = plt.subplot(1,3,2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_cnt_TS_tr*10, levels=[-3.5,-2,-1,-0.5,0.5,1,2,3.5], cmap=plt.cm.RdBu_r)
H = plt.contourf(lonproj, latproj, sign_cnt_TS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Frequency TS trend')
AX = plt.subplot(1,3,3)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, (MHW_cnt_OLS_tr-MHW_cnt_TS_tr)*10, levels=[-3.5,-2,-1,-0.5,0.5,1,2,3.5], cmap=plt.cm.RdBu_r)
plt.title('Difference (OLS-TS)')
AXPOS = AX.get_position()
CAX = fig.add_axes([AXPOS.x1+0.015, AXPOS.y0, 0.01, AXPOS.y1-AXPOS.y0])
HB = plt.colorbar(H, CAX, orientation='vertical')
HB.set_label(r'[count / decade]')
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/OLS_test/MHW_OLS_test_cnt.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

plt.clf()
AX = plt.subplot(1,3,1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_dur_OLS_tr*10, levels=[-50,-20,-10,-2.5,2.5,10,20,50], cmap=plt.cm.RdBu_r)
plt.clim(-18,18)
H = plt.contourf(lonproj, latproj, sign_dur_OLS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Duration OLS trend')
AX = plt.subplot(1,3,2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_dur_TS_tr*10, levels=[-50,-20,-10,-2.5,2.5,10,20,50], cmap=plt.cm.RdBu_r)
plt.clim(-18,18)
H = plt.contourf(lonproj, latproj, sign_dur_TS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Duration TS trend')
AX = plt.subplot(1,3,3)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, (MHW_dur_OLS_tr-MHW_dur_TS_tr)*10, levels=[-50,-20,-10,-2.5,2.5,10,20,50], cmap=plt.cm.RdBu_r)
plt.clim(-18,18)
plt.title('Difference (OLS-TS)')
AXPOS = AX.get_position()
CAX = fig.add_axes([AXPOS.x1+0.015, AXPOS.y0, 0.01, AXPOS.y1-AXPOS.y0])
HB = plt.colorbar(H, CAX, orientation='vertical')
HB.set_label(r'[days / decade]')
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/OLS_test/MHW_OLS_test_dur.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

plt.clf()
AX = plt.subplot(1,3,1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_mean_OLS_tr*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.clim(-0.5,0.5)
H = plt.contourf(lonproj, latproj, sign_mean_OLS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Mean intensity OLS trend')
AX = plt.subplot(1,3,2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_mean_TS_tr*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.clim(-0.5,0.5)
H = plt.contourf(lonproj, latproj, sign_mean_TS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Mean intensity TS trend')
AX = plt.subplot(1,3,3)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, (MHW_mean_OLS_tr-MHW_mean_TS_tr)*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.clim(-0.5,0.5)
plt.title('Difference (OLS-TS)')
AXPOS = AX.get_position()
CAX = fig.add_axes([AXPOS.x1+0.015, AXPOS.y0, 0.01, AXPOS.y1-AXPOS.y0])
HB = plt.colorbar(H, CAX, orientation='vertical')
HB.set_label(r'[$^\circ$C / decade]')
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/OLS_test/MHW_OLS_test_mean.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

plt.clf()
AX = plt.subplot(1,3,1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_max_OLS_tr*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.clim(-0.5,0.5)
H = plt.contourf(lonproj, latproj, sign_max_OLS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Max intensity OLS trend')
AX = plt.subplot(1,3,2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_max_TS_tr*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.clim(-0.5,0.5)
H = plt.contourf(lonproj, latproj, sign_max_TS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Max intensity TS trend')
AX = plt.subplot(1,3,3)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, (MHW_max_OLS_tr-MHW_max_TS_tr)*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.clim(-0.5,0.5)
plt.title('Difference (OLS-TS)')
AXPOS = AX.get_position()
CAX = fig.add_axes([AXPOS.x1+0.015, AXPOS.y0, 0.01, AXPOS.y1-AXPOS.y0])
HB = plt.colorbar(H, CAX, orientation='vertical')
HB.set_label(r'[$^\circ$C / decade]')
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/OLS_test/MHW_OLS_test_max.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

plt.clf()
AX = plt.subplot(1,3,1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_cum_OLS_tr*10, levels=[-100,-20,-10,-2.5,2.5,10,20,100], cmap=plt.cm.RdBu_r)
plt.clim(-22,22)
H = plt.contourf(lonproj, latproj, sign_cum_OLS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Cumulative intensity OLS trend')
AX = plt.subplot(1,3,2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_cum_TS_tr*10, levels=[-100,-20,-10,-2.5,2.5,10,20,100], cmap=plt.cm.RdBu_r)
plt.clim(-22,22)
H = plt.contourf(lonproj, latproj, sign_cum_TS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Cumulative intensity TS trend')
AX = plt.subplot(1,3,3)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, (MHW_cum_OLS_tr-MHW_cum_TS_tr)*10, levels=[-100,-20,-10,-2.5,2.5,10,20,100], cmap=plt.cm.RdBu_r)
plt.clim(-22,22)
plt.title('Difference (OLS-TS)')
AXPOS = AX.get_position()
CAX = fig.add_axes([AXPOS.x1+0.015, AXPOS.y0, 0.01, AXPOS.y1-AXPOS.y0])
HB = plt.colorbar(H, CAX, orientation='vertical')
HB.set_label(r'[$^\circ$C$\times$days / decade]')
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/OLS_test/MHW_OLS_test_cum.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

plt.clf()
AX = plt.subplot(1,3,1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_td_OLS_tr*10, levels=[-200,-30,-20,-10,-5,5,10,20,30,200], cmap=plt.cm.RdBu_r)
plt.clim(-30,30)
H = plt.contourf(lonproj, latproj, sign_td_OLS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Total MHW days OLS trend')
AX = plt.subplot(1,3,2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_td_TS_tr*10, levels=[-200,-30,-20,-10,-5,5,10,20,30,200], cmap=plt.cm.RdBu_r)
plt.clim(-30,30)
H = plt.contourf(lonproj, latproj, sign_td_TS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Total MHW days TS trend')
AX = plt.subplot(1,3,3)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, (MHW_td_OLS_tr-MHW_td_TS_tr)*10, levels=[-200,-30,-20,-10,-5,5,10,20,30,200], cmap=plt.cm.RdBu_r)
plt.clim(-30,30)
plt.title('Difference (OLS-TS)')
AXPOS = AX.get_position()
CAX = fig.add_axes([AXPOS.x1+0.015, AXPOS.y0, 0.01, AXPOS.y1-AXPOS.y0])
HB = plt.colorbar(H, CAX, orientation='vertical')
HB.set_label(r'[days / decade]')
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/OLS_test/MHW_OLS_test_td.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

plt.clf()
AX = plt.subplot(1,3,1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_tc_OLS_tr*10, levels=[-200,-40,-20,-10,-5,5,10,20,40,200], cmap=plt.cm.RdBu_r)
plt.clim(-45,45)
H = plt.contourf(lonproj, latproj, sign_tc_OLS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Total cum. intens. trend')
AX = plt.subplot(1,3,2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_tc_TS_tr*10, levels=[-200,-40,-20,-10,-5,5,10,20,40,200], cmap=plt.cm.RdBu_r)
plt.clim(-45,45)
H = plt.contourf(lonproj, latproj, sign_tc_TS, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.title('Total cum. intens. TS trend')
AX = plt.subplot(1,3,3)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, (MHW_tc_OLS_tr-MHW_tc_TS_tr)*10, levels=[-200,-40,-20,-10,-5,5,10,20,40,200], cmap=plt.cm.RdBu_r)
plt.clim(-45,45)
plt.title('Difference (OLS-TS)')
AXPOS = AX.get_position()
CAX = fig.add_axes([AXPOS.x1+0.015, AXPOS.y0, 0.01, AXPOS.y1-AXPOS.y0])
HB = plt.colorbar(H, CAX, orientation='vertical')
HB.set_label(r'[$^\circ$C$\times$days / decade]')
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/OLS_test/MHW_OLS_test_tc.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

# Figure for paper

fig = plt.figure(figsize=(17,9))
plt.clf()
# Frequency
AX = plt.subplot(3,3,1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_cnt_OLS_tr*10, levels=[-3.5,-2,-1,-0.5,0.5,1,2,3.5], cmap=plt.cm.RdBu_r)
plt.title('(A) Frequency OLS trend')
AX = plt.subplot(3,3,2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_cnt_TS_tr*10, levels=[-3.5,-2,-1,-0.5,0.5,1,2,3.5], cmap=plt.cm.RdBu_r)
plt.title('(B) Frequency TS trend')
AX = plt.subplot(3,3,3)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, (MHW_cnt_OLS_tr-MHW_cnt_TS_tr)*10, levels=[-3.5,-2,-1,-0.5,0.5,1,2,3.5], cmap=plt.cm.RdBu_r)
plt.title('(C) Difference (OLS-TS)')
AXPOS = AX.get_position()
CAX = fig.add_axes([AXPOS.x1+0.015, AXPOS.y0, 0.01, AXPOS.y1-AXPOS.y0])
HB = plt.colorbar(H, CAX, orientation='vertical')
HB.set_label(r'[count / decade]')
# Intensity
AX = plt.subplot(3,3,4)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_mean_OLS_tr*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.clim(-0.5,0.5)
plt.title('(D) Mean intensity OLS trend')
AX = plt.subplot(3,3,5)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_mean_TS_tr*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.clim(-0.5,0.5)
plt.title('(E) Mean intensity TS trend')
AX = plt.subplot(3,3,6)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, (MHW_mean_OLS_tr-MHW_mean_TS_tr)*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.clim(-0.5,0.5)
plt.title('(F) Difference (OLS-TS)')
AXPOS = AX.get_position()
CAX = fig.add_axes([AXPOS.x1+0.015, AXPOS.y0, 0.01, AXPOS.y1-AXPOS.y0])
HB = plt.colorbar(H, CAX, orientation='vertical')
HB.set_label(r'[$^\circ$C / decade]')
# Duration
AX = plt.subplot(3,3,7)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_dur_OLS_tr*10, levels=[-50,-20,-10,-2.5,2.5,10,20,50], cmap=plt.cm.RdBu_r)
plt.clim(-18,18)
plt.title('(G) Duration OLS trend')
AX = plt.subplot(3,3,8)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, MHW_dur_TS_tr*10, levels=[-50,-20,-10,-2.5,2.5,10,20,50], cmap=plt.cm.RdBu_r)
plt.clim(-18,18)
plt.title('(H) Duration TS trend')
AX = plt.subplot(3,3,9)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.drawcoastlines(linewidth=0.5, color=(.3,.3,.3))
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
H = plt.contourf(lonproj, latproj, (MHW_dur_OLS_tr-MHW_dur_TS_tr)*10, levels=[-50,-20,-10,-2.5,2.5,10,20,50], cmap=plt.cm.RdBu_r)
plt.clim(-18,18)
plt.title('(I) Difference (OLS-TS)')
AXPOS = AX.get_position()
CAX = fig.add_axes([AXPOS.x1+0.015, AXPOS.y0, 0.01, AXPOS.y1-AXPOS.y0])
HB = plt.colorbar(H, CAX, orientation='vertical')
HB.set_label(r'[days / decade]')
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/compare_OLS_global_orig.png', bbox_inches='tight', pad_inches=0.5, dpi=300)

#
# Time series
#

SST_ts_glob = np.zeros(MHW_cnt_ts.shape[2])
MHW_cnt_ts_glob = np.zeros(MHW_cnt_ts.shape[2])
MHW_dur_ts_glob = np.zeros(MHW_dur_ts.shape[2])
MHW_max_ts_glob = np.zeros(MHW_max_ts.shape[2])
MHW_mean_ts_glob = np.zeros(MHW_mean_ts.shape[2])
MHW_cum_ts_glob = np.zeros(MHW_cum_ts.shape[2])
MHW_td_ts_glob = np.zeros(MHW_cum_ts.shape[2])
MHW_tc_ts_glob = np.zeros(MHW_cum_ts.shape[2])
SST_ts_notrop = np.zeros(MHW_cnt_ts.shape[2])
MHW_cnt_ts_notrop = np.zeros(MHW_cnt_ts.shape[2])
MHW_dur_ts_notrop = np.zeros(MHW_dur_ts.shape[2])
MHW_max_ts_notrop = np.zeros(MHW_max_ts.shape[2])
MHW_mean_ts_notrop = np.zeros(MHW_mean_ts.shape[2])
MHW_cum_ts_notrop = np.zeros(MHW_cum_ts.shape[2])
MHW_td_ts_notrop = np.zeros(MHW_cum_ts.shape[2])
MHW_tc_ts_notrop = np.zeros(MHW_cum_ts.shape[2])

# Cosine scaling
scaling = np.cos(llat*np.pi/180)
# Sum / average over globe
for tt in range(MHW_cnt_ts.shape[2]):
    # Create mask
    mask = np.ones(llat.shape)
    mask[np.isnan(SST_ts[:,:,tt])] = np.nan
    mask_notrop = 1.*mask
    mask_notrop[np.abs(llat)<=20] = np.nan
    # SST
    SST_ts_glob[tt] = np.average(SST_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    SST_ts_notrop[tt] = np.average(SST_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Create mask
    mask = np.ones(llat.shape)
    mask[np.isnan(MHW_dur_ts[:,:,tt])] = np.nan
    mask_notrop = 1.*mask
    mask_notrop[np.abs(llat)<=20] = np.nan
    # Count
    MHW_cnt_ts_glob[tt] = np.average(MHW_cnt_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    MHW_cnt_ts_notrop[tt] = np.average(MHW_cnt_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Duration
    MHW_dur_ts_glob[tt] = np.average(MHW_dur_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    MHW_dur_ts_notrop[tt] = np.average(MHW_dur_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Maximum intensity
    MHW_max_ts_glob[tt] = np.average(MHW_max_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    MHW_max_ts_notrop[tt] = np.average(MHW_max_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Mean intensity
    MHW_mean_ts_glob[tt] = np.average(MHW_mean_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    MHW_mean_ts_notrop[tt] = np.average(MHW_mean_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Cumulative intensity
    MHW_cum_ts_glob[tt] = np.average(MHW_cum_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    MHW_cum_ts_notrop[tt] = np.average(MHW_cum_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Total MHW days
    MHW_td_ts_glob[tt] = np.average(MHW_td_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    MHW_td_ts_notrop[tt] = np.average(MHW_td_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Total MHW cumulative intensity
    MHW_tc_ts_glob[tt] = np.average(MHW_tc_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    MHW_tc_ts_notrop[tt] = np.average(MHW_tc_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])

# Trends and significance
alpha = 0.01
print 'Frequency global'
mean, slope, dslope = meanTrend_OLS(years, SST_ts_glob, alpha)
print slope-dslope, slope, slope+dslope
mean, slope, dslope = meanTrend_TS(years, SST_ts_glob, alpha)
print dslope[0], slope, dslope[1]
print 'Frequency non-trop'
mean, slope, dslope = meanTrend_OLS(years, SST_ts_notrop, alpha)
print slope-dslope, slope, slope+dslope
mean, slope, dslope = meanTrend_TS(years, SST_ts_notrop, alpha)
print dslope[0], slope, dslope[1]
print 'Frequency global'
mean, slope, dslope = meanTrend_OLS(years, MHW_cnt_ts_glob, alpha)
print slope-dslope, slope, slope+dslope
mean, slope, dslope = meanTrend_TS(years, MHW_cnt_ts_glob, alpha)
print dslope[0], slope, dslope[1]
print 'Frequency non-trop'
mean, slope, dslope = meanTrend_OLS(years, MHW_cnt_ts_notrop, alpha)
print slope-dslope, slope, slope+dslope
mean, slope, dslope = meanTrend_TS(years, MHW_cnt_ts_notrop, alpha)
print dslope[0], slope, dslope[1]
print 'Duration global'
mean, slope, dslope = meanTrend_OLS(years, MHW_dur_ts_glob, alpha)
print slope-dslope, slope, slope+dslope
mean, slope, dslope = meanTrend_TS(years, MHW_dur_ts_glob, alpha)
print dslope[0], slope, dslope[1]
print 'Duration non-trop'
mean, slope, dslope = meanTrend_OLS(years, MHW_dur_ts_notrop, alpha)
print slope-dslope, slope, slope+dslope
mean, slope, dslope = meanTrend_TS(years, MHW_dur_ts_notrop, alpha)
print dslope[0], slope, dslope[1]
print 'Mean intensity global'
mean, slope, dslope = meanTrend_OLS(years, MHW_mean_ts_glob, alpha)
print slope-dslope, slope, slope+dslope
mean, slope, dslope = meanTrend_TS(years, MHW_mean_ts_glob, alpha)
print dslope[0], slope, dslope[1]
print 'Mean intensity non-trop'
mean, slope, dslope = meanTrend_OLS(years, MHW_mean_ts_notrop, alpha)
print slope-dslope, slope, slope+dslope
mean, slope, dslope = meanTrend_TS(years, MHW_mean_ts_notrop, alpha)
print dslope[0], slope, dslope[1]
print 'Max intensity global'
mean, slope, dslope = meanTrend_OLS(years, MHW_max_ts_glob, alpha)
print slope-dslope, slope, slope+dslope
mean, slope, dslope = meanTrend_TS(years, MHW_max_ts_glob, alpha)
print dslope[0], slope, dslope[1]
print 'Max intensity non-trop'
mean, slope, dslope = meanTrend_OLS(years, MHW_max_ts_notrop, alpha)
print slope-dslope, slope, slope+dslope
mean, slope, dslope = meanTrend_TS(years, MHW_max_ts_notrop, alpha)
print dslope[0], slope, dslope[1]
print 'Total days global'
mean, slope, dslope = meanTrend_OLS(years, MHW_td_ts_glob, alpha)
print slope-dslope, slope, slope+dslope
mean, slope, dslope = meanTrend_TS(years, MHW_td_ts_glob, alpha)
print dslope[0], slope, dslope[1]
print 'Total days non-trop'
mean, slope, dslope = meanTrend_OLS(years, MHW_td_ts_notrop, alpha)
print slope-dslope, slope, slope+dslope
mean, slope, dslope = meanTrend_TS(years, MHW_td_ts_notrop, alpha)
print dslope[0], slope, dslope[1]



