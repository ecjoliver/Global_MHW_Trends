'''

  Software which uses the MHW definition
  of Hobday et al. (2015) applied to 
  select SST time series around the globe

'''

import numpy as np
import scipy as sp
import scipy.signal as sig
from scipy import linalg
from scipy import stats
from scipy import interpolate

import matplotlib as mpl
from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm

import ecoliver as ecj

#
# Load data and make plots
#

pathroot = '/home/ecoliver/Desktop/'
#pathroot = '/media/ecoliver/DataOne/mokami_move_backup/'

dataSets = ['HadISST', 'HadSST3']
yearStart = 1871
dataStart = {'HadISST': 1871, 'HadSST3': 1871}
dataEnd = {'HadISST': 2016, 'HadSST3': 2016}
years_data = np.arange(yearStart, 2016+1)
outfile = {}
datamask = {}
llon = {}
llat = {}
MHW_f_m = {}
MHW_d_m = {}
MHW_i_m = {}
MHW_td_m = {}
MHW_f_ts = {}
MHW_d_ts = {}
MHW_i_ts = {}
MHW_td_ts = {}
MHW_ts_glob = {}
MHW_d_ts_glob = {}
MHW_i_ts_glob = {}
MHW_td_ts_glob = {}
validCells_f_ts_glob = {}
validCells_d_ts_glob = {}
validCells_i_ts_glob = {}
validCells_td_ts_glob = {}
mask_f_ts = {}
mask_d_ts = {}
mask_i_ts = {}
mask_td_ts = {}
p_KS_f = {}
p_KS_d = {}
p_KS_td = {}
sign_p_f = {}
sign_p_d = {}
sign_p_td = {}
MHW_f_dtt = {}
MHW_d_dtt = {}
MHW_td_dtt = {}

# AVHRR
data_ts_AVHRR = np.load('/home/ecoliver/Desktop/data/MHWs/Trends/mhw_census.2016_ts.npz')
years = data_ts_AVHRR['years']
MHW_ts_glob['AVHRR']  = data_ts_AVHRR['MHW_cnt_ts_glob']
MHW_d_ts_glob['AVHRR']  = data_ts_AVHRR['MHW_dur_ts_glob']
MHW_td_ts_glob['AVHRR'] = data_ts_AVHRR['MHW_td_ts_glob']
cnt_mean = np.mean(MHW_ts_glob['AVHRR'])
dur_mean = np.mean(MHW_d_ts_glob['AVHRR'])
td_mean = np.mean(MHW_td_ts_glob['AVHRR'])

# time slice periods
tt1 = (years_data>=1925) * (years_data<=1954)
tt2 = (years_data>=1987) * (years_data<=2016)

# Other data
for dataSet in dataSets:
    print dataSet
    outfile[dataSet] = pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1871.2016.' + dataSet + '.npz'
    data = np.load(outfile[dataSet])
    #if (dataSet == 'HadISST') + (dataSet == 'HadSST3'):
    #   lon_data = data['lon_had']
    #   lat_data = data['lat_had']
    #else:
    lon_data = data['lon_data']
    lat_data = data['lat_data']
    MHW_m = data['MHW_m'].item()
    MHW_ts = data['MHW_ts'].item()
    datamask[dataSet] = data['datamask']
    datamask[dataSet][datamask[dataSet]==0] = np.nan
    datamask_ts = np.swapaxes(np.swapaxes(np.tile(datamask[dataSet], (len(years_data),1,1)), 0, 1), 1, 2)
    # Extract mean and time series, simultaneously re-map to run 20E to 380E
    i_20E = np.where(lon_data>20)[0][0]
    lon_data = np.append(lon_data[i_20E:], lon_data[:i_20E]+360)
    datamask[dataSet] = np.append(datamask[dataSet][:,i_20E:], datamask[dataSet][:,:i_20E], axis=1)
    datamask_ts = np.append(datamask_ts[:,i_20E:,:], datamask_ts[:,:i_20E,:], axis=1)
    MHW_f_m[dataSet] = np.append(MHW_m['count']['threshCount'][:,i_20E:], MHW_m['count']['threshCount'][:,:i_20E], axis=1)
    MHW_d_m[dataSet] = np.append(MHW_m['duration']['maxAnom'][:,i_20E:], MHW_m['duration']['maxAnom'][:,:i_20E], axis=1)
    MHW_i_m[dataSet] = np.append(MHW_m['intensity_mean']['threshAnom'][:,i_20E:], MHW_m['intensity_mean']['threshAnom'][:,:i_20E], axis=1)
    MHW_td_m[dataSet] = np.append(MHW_m['total_days']['threshAnom'][:,i_20E:], MHW_m['total_days']['threshAnom'][:,:i_20E], axis=1) # pkey doesn't matter, all the same
    MHW_f_ts[dataSet] = np.append(MHW_ts['count']['threshCount'][:,i_20E:,:], MHW_ts['count']['threshCount'][:,:i_20E,:], axis=1)
    MHW_d_ts[dataSet] = np.append(MHW_ts['duration']['maxAnom'][:,i_20E:,:], MHW_ts['duration']['maxAnom'][:,:i_20E,:], axis=1)
    MHW_i_ts[dataSet] = np.append(MHW_ts['intensity_mean']['threshAnom'][:,i_20E:,:], MHW_ts['intensity_mean']['threshAnom'][:,:i_20E,:], axis=1)
    MHW_td_ts[dataSet] = np.append(MHW_ts['total_days']['threshAnom'][:,i_20E:,:], MHW_ts['total_days']['threshAnom'][:,:i_20E,:], axis=1) # pkey doesn't matter, all the same
    # Hack out some fixes for ridiculous durations
    MHW_td_ts[dataSet][MHW_td_ts[dataSet]>=1000] = np.nan
    MHW_d_ts[dataSet][MHW_d_ts[dataSet]>=1000] = np.nan
    #
    del(MHW_m)
    del(MHW_ts)
    llon[dataSet], llat[dataSet] = np.meshgrid(lon_data, lat_data)
    # Sum / average over globe
    scaling = np.cos(llat[dataSet]*np.pi/180)
    MHW_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_d_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_i_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_td_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    validCells_f_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    validCells_d_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    validCells_i_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    validCells_td_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    for tt in np.where((years_data >= dataStart[dataSet]) * (years_data <= dataEnd[dataSet]))[0].tolist():
        # Count - Create mask
        mask = np.ones(llat[dataSet].shape)
        mask[np.isnan(MHW_f_ts[dataSet][:,:,tt])] = np.nan
        validCells_f_ts_glob[dataSet][tt] = np.sum(~np.isnan(mask))
        # Count
        MHW_ts_glob[dataSet][tt] = np.average(MHW_f_ts[dataSet][:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
        # Duration - Create mask
        mask = np.ones(llat[dataSet].shape)
        mask[np.isnan(MHW_d_ts[dataSet][:,:,tt])] = np.nan
        validCells_d_ts_glob[dataSet][tt] = np.sum(~np.isnan(mask))
        # Duration
        MHW_d_ts_glob[dataSet][tt] = np.average(MHW_d_ts[dataSet][:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
        # Intensity - Create mask
        #mask = np.ones(llat[dataSet].shape)
        #mask[np.isnan(MHW_i_ts[dataSet][:,:,tt])] = np.nan
        #validCells_i_ts_glob[dataSet][tt] = np.sum(~np.isnan(mask))
        # Mean intensity
        #MHW_i_ts_glob[dataSet][tt] = np.average(MHW_i_ts[dataSet][:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
        # Total MHW days - Create mask
        mask = np.ones(llat[dataSet].shape)
        mask[np.isnan(MHW_td_ts[dataSet][:,:,tt])] = np.nan
        validCells_td_ts_glob[dataSet][tt] = np.sum(~np.isnan(mask))
        # Total MHW days
        MHW_td_ts_glob[dataSet][tt] = np.average(MHW_td_ts[dataSet][:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    # Create masks for coverage < 50%
    mask_f_ts[dataSet] = np.ones(len(validCells_f_ts_glob[dataSet]))
    mask_f_ts[dataSet][validCells_f_ts_glob[dataSet]<0.5*validCells_f_ts_glob[dataSet].max()] = np.nan
    mask_d_ts[dataSet] = np.ones(len(validCells_d_ts_glob[dataSet]))
    mask_d_ts[dataSet][validCells_d_ts_glob[dataSet]<0.5*validCells_d_ts_glob[dataSet].max()] = np.nan
    mask_i_ts[dataSet] = np.ones(len(validCells_i_ts_glob[dataSet]))
    mask_i_ts[dataSet][validCells_i_ts_glob[dataSet]<0.5*validCells_i_ts_glob[dataSet].max()] = np.nan
    mask_td_ts[dataSet] = np.ones(len(validCells_td_ts_glob[dataSet]))
    mask_td_ts[dataSet][validCells_td_ts_glob[dataSet]<0.5*validCells_td_ts_glob[dataSet].max()] = np.nan
    # Shift proxy-based time series to have same mean as AVHRR-based time series
    avhrr = np.in1d(years_data, years)
    MHW_ts_glob[dataSet] = MHW_ts_glob[dataSet] + (cnt_mean - np.nanmean((MHW_ts_glob[dataSet]*mask_f_ts[dataSet])[avhrr]))
    MHW_d_ts_glob[dataSet] = MHW_d_ts_glob[dataSet] + (dur_mean - np.nanmean((MHW_d_ts_glob[dataSet]*mask_d_ts[dataSet])[avhrr]))
    MHW_td_ts_glob[dataSet] = MHW_td_ts_glob[dataSet] + (td_mean - np.nanmean((MHW_td_ts_glob[dataSet]*mask_td_ts[dataSet])[avhrr]))
    #
    # Maps of mean difference
    #
    p_KS_f[dataSet] = np.nan*np.zeros((MHW_f_ts[dataSet].shape[0], MHW_f_ts[dataSet].shape[1]))
    p_KS_d[dataSet] = np.nan*np.zeros((MHW_f_ts[dataSet].shape[0], MHW_f_ts[dataSet].shape[1]))
    p_KS_td[dataSet] = np.nan*np.zeros((MHW_f_ts[dataSet].shape[0], MHW_f_ts[dataSet].shape[1]))
    for j in range(p_KS_f[dataSet].shape[0]):
        print j
        for i in range(p_KS_f[dataSet].shape[1]):
            #if (MHW_cnt_ts[j,i,tt1].sum() == 0) + (MHW_cnt_ts[j,i,tt2].sum() == 0):
            #    continue
            t, p_KS_f[dataSet][j,i] = stats.ks_2samp(MHW_f_ts[dataSet][j,i,tt2], MHW_f_ts[dataSet][j,i,tt1])
            t, p_KS_d[dataSet][j,i] = stats.ks_2samp(MHW_d_ts[dataSet][j,i,tt2], MHW_d_ts[dataSet][j,i,tt1])
            t, p_KS_td[dataSet][j,i] = stats.ks_2samp(MHW_td_ts[dataSet][j,i,tt2], MHW_td_ts[dataSet][j,i,tt1])
    sign_p_f[dataSet] = datamask[dataSet]*(p_KS_f[dataSet]<=0.05).astype(float)
    sign_p_d[dataSet] = datamask[dataSet]*(p_KS_d[dataSet]<=0.05).astype(float)
    sign_p_td[dataSet] = datamask[dataSet]*(p_KS_td[dataSet]<=0.05).astype(float)
    #
    MHW_f_dtt[dataSet] = np.ma.masked_invalid(np.nanmean(MHW_f_ts[dataSet][:,:,tt2], axis=2) - np.nanmean(MHW_f_ts[dataSet][:,:,tt1], axis=2))
    MHW_d_dtt[dataSet] = np.ma.masked_invalid(np.nanmean(MHW_d_ts[dataSet][:,:,tt2], axis=2) - np.nanmean(MHW_d_ts[dataSet][:,:,tt1], axis=2))
    MHW_td_dtt[dataSet] = np.ma.masked_invalid(np.nanmean(MHW_td_ts[dataSet][:,:,tt2], axis=2) - np.nanmean(MHW_td_ts[dataSet][:,:,tt1], axis=2))

# Aggregate time series from different data sets
MHW_ts_glob_agg = np.zeros((len(MHW_ts_glob[dataSet]),len(dataSets)))
MHW_d_ts_glob_agg = np.zeros((len(MHW_d_ts_glob[dataSet]),len(dataSets)))
MHW_td_ts_glob_agg = np.zeros((len(MHW_td_ts_glob[dataSet]),len(dataSets)))
cnt = 0
for dataSet in dataSets:
    MHW_ts_glob_agg[:,cnt] = MHW_ts_glob[dataSet]*mask_f_ts[dataSet]
    MHW_d_ts_glob_agg[:,cnt] = MHW_d_ts_glob[dataSet]*mask_d_ts[dataSet]
    MHW_td_ts_glob_agg[:,cnt] = MHW_td_ts_glob[dataSet]*mask_td_ts[dataSet]
    cnt += 1


# Compare time series

plt.figure() #figsize=(19,5))
plt.clf()
# Time series
plt.subplot(3,1,1)
plt.plot(years_data, MHW_ts_glob['HadISST']*mask_f_ts['HadISST'], 'k-', linewidth=2)
plt.plot(years_data, MHW_ts_glob['HadSST3']*mask_f_ts['HadSST3'], 'b-', linewidth=2)
plt.plot(years_data, MHW_ts_glob['HadISST']*mask_f_ts['HadISST'], 'k.', linewidth=2)
plt.plot(years_data, MHW_ts_glob['HadSST3']*mask_f_ts['HadSST3'], 'b.', linewidth=2)
plt.grid()
plt.ylabel('Frequency [count]')
plt.legend(dataSets, loc='upper left', fontsize=12)
plt.title('Global-mean time series')
# Time series
plt.subplot(3,1,2)
plt.plot(years_data, MHW_d_ts_glob['HadISST']*mask_d_ts['HadISST'], 'k-', linewidth=2)
plt.plot(years_data, MHW_d_ts_glob['HadSST3']*mask_d_ts['HadSST3'], 'b-', linewidth=2)
plt.plot(years_data, MHW_d_ts_glob['HadISST']*mask_d_ts['HadISST'], 'k.', linewidth=2)
plt.plot(years_data, MHW_d_ts_glob['HadSST3']*mask_d_ts['HadSST3'], 'b.', linewidth=2)
plt.grid()
plt.ylabel('Duration [days]')
# Time series
plt.subplot(3,1,3)
plt.plot(years_data, MHW_td_ts_glob['HadISST']*mask_td_ts['HadISST'], 'k-', linewidth=2)
plt.plot(years_data, MHW_td_ts_glob['HadSST3']*mask_td_ts['HadSST3'], 'b-', linewidth=2)
plt.plot(years_data, MHW_td_ts_glob['HadISST']*mask_td_ts['HadISST'], 'k.', linewidth=2)
plt.plot(years_data, MHW_td_ts_glob['HadSST3']*mask_td_ts['HadSST3'], 'b.', linewidth=2)
plt.grid()
plt.ylabel('Total MHW days [days]')

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_proxies_1871_ts.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

