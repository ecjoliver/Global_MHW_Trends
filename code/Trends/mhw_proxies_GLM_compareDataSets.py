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
pathroot = '/home/oliver/'
#pathroot = '/media/ecoliver/DataOne/mokami_move_backup/'

#dataSets = ['HadISST', 'HadSST3', 'ERSST', 'Kaplan', 'COBE', 'CERA20C'] #, 'SODA']
dataSets = ['HadISST', 'HadSST3', 'ERSST', 'Kaplan', 'COBE', 'CERA20C', 'SODA']
dataSets = ['HadISST',            'ERSST',           'COBE', 'CERA20C', 'SODA']
yearStart = 1900
dataStart = {'HadISST': 1900, 'HadSST3': 1900, 'ERSST': 1900, 'Kaplan': 1900, 'COBE': 1900, 'CERA20C': 1901, 'SODA': 1900}
dataEnd = {'HadISST': 2016, 'HadSST3': 2016, 'ERSST': 2016, 'Kaplan': 2016, 'COBE': 2016, 'CERA20C': 2010, 'SODA': 2013}
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
MHW_f_CI = {}
MHW_d_CI = {}
MHW_i_CI = {}
MHW_td_CI = {}
rho_f = {}
rho_d = {}
rho_i = {}
p_f = {}
p_d = {}
p_i = {}
MHW_f_ts_glob = {}
MHW_d_ts_glob = {}
MHW_i_ts_glob = {}
MHW_td_ts_glob = {}
MHW_f_sigMod_glob = {}
MHW_d_sigMod_glob = {}
MHW_i_sigMod_glob = {}
MHW_td_sigMod_glob = {}
MHW_f_N_glob = {}
MHW_d_N_glob = {}
MHW_i_N_glob = {}
MHW_td_N_glob = {}
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
data_ts_AVHRR = np.load(pathroot + 'data/MHWs/Trends/mhw_census.2016_ts.npz')
years = data_ts_AVHRR['years']
MHW_f_ts_glob['AVHRR']  = data_ts_AVHRR['MHW_cnt_ts_glob']
MHW_d_ts_glob['AVHRR']  = data_ts_AVHRR['MHW_dur_ts_glob']
MHW_td_ts_glob['AVHRR'] = data_ts_AVHRR['MHW_td_ts_glob']
cnt_mean = np.mean(MHW_f_ts_glob['AVHRR'])
dur_mean = np.mean(MHW_d_ts_glob['AVHRR'])
td_mean = np.mean(MHW_td_ts_glob['AVHRR'])

# time slice periods
tt1 = (years_data>=1925) * (years_data<=1954)
tt2 = (years_data>=1987) * (years_data<=2016)

# Other data
for dataSet in dataSets:
    print dataSet
    outfile[dataSet] = pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1900.2016.' + dataSet + '.npz'
    #outfile[dataSet] = pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1900.2016.' + dataSet + '.no_MEI_PDO_AMO.npz'
    data = np.load(outfile[dataSet])
    lon_data = data['lon_data']
    lat_data = data['lat_data']
    MHW_m = data['MHW_m'].item()
    MHW_ts = data['MHW_ts'].item()
    MHW_CI = data['MHW_CI'].item()
    RHO = data['rho'].item()
    P = data['p'].item()
    if dataSet == 'COBE':
        outfile_dm = pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1900.2016.' + dataSet + '.datamask.npz'
        data_dm = np.load(outfile_dm)
        datamask[dataSet] = data_dm['datamask']
    else:
        datamask[dataSet] = data['datamask']
    datamask[dataSet][datamask[dataSet]==0] = np.nan
    datamask_ts = np.swapaxes(np.swapaxes(np.tile(datamask[dataSet], (len(years_data),1,1)), 0, 1), 1, 2)
    datamask_CI = datamask_CI = np.swapaxes(np.swapaxes(np.swapaxes(np.tile(datamask_ts, (2,1,1,1)), 0, 1), 1, 2), 2, 3)
    # Extract mean and time series, simultaneously re-map to run 20E to 380E, and apply datamasks
    i_20E = np.where(lon_data>20)[0][0]
    lon_data = np.append(lon_data[i_20E:], lon_data[:i_20E]+360)
    datamask[dataSet] = np.append(datamask[dataSet][:,i_20E:], datamask[dataSet][:,:i_20E], axis=1)
    datamask_ts = np.append(datamask_ts[:,i_20E:,:], datamask_ts[:,:i_20E,:], axis=1)
    datamask_CI = np.append(datamask_CI[:,i_20E:,:,:], datamask_CI[:,:i_20E,:,:], axis=1)
    MHW_f_m[dataSet] = datamask[dataSet]*np.append(MHW_m['count']['threshCount'][:,i_20E:], MHW_m['count']['threshCount'][:,:i_20E], axis=1)
    MHW_d_m[dataSet] = datamask[dataSet]*np.append(MHW_m['duration']['maxAnom'][:,i_20E:], MHW_m['duration']['maxAnom'][:,:i_20E], axis=1)
    MHW_i_m[dataSet] = datamask[dataSet]*np.append(MHW_m['intensity_mean']['threshAnom'][:,i_20E:], MHW_m['intensity_mean']['threshAnom'][:,:i_20E], axis=1)
    MHW_td_m[dataSet] = datamask[dataSet]*np.append(MHW_m['total_days']['threshAnom'][:,i_20E:], MHW_m['total_days']['threshAnom'][:,:i_20E], axis=1) # pkey doesn't matter, all the same
    MHW_f_ts[dataSet] = datamask_ts*np.append(MHW_ts['count']['threshCount'][:,i_20E:,:], MHW_ts['count']['threshCount'][:,:i_20E,:], axis=1)
    MHW_d_ts[dataSet] = datamask_ts*np.append(MHW_ts['duration']['maxAnom'][:,i_20E:,:], MHW_ts['duration']['maxAnom'][:,:i_20E,:], axis=1)
    MHW_i_ts[dataSet] = datamask_ts*np.append(MHW_ts['intensity_mean']['threshAnom'][:,i_20E:,:], MHW_ts['intensity_mean']['threshAnom'][:,:i_20E,:], axis=1)
    MHW_td_ts[dataSet] = datamask_ts*np.append(MHW_ts['total_days']['threshAnom'][:,i_20E:,:], MHW_ts['total_days']['threshAnom'][:,:i_20E,:], axis=1) # pkey doesn't matter, all the same
    MHW_f_CI[dataSet] = datamask_CI*np.append(MHW_CI['count']['threshCount'][:,i_20E:,:,:], MHW_CI['count']['threshCount'][:,:i_20E,:], axis=1)
    MHW_d_CI[dataSet] = datamask_CI*np.append(MHW_CI['duration']['maxAnom'][:,i_20E:,:,:], MHW_CI['duration']['maxAnom'][:,:i_20E,:], axis=1)
    MHW_i_CI[dataSet] = datamask_CI*np.append(MHW_CI['intensity_mean']['threshAnom'][:,i_20E:,:,:], MHW_CI['intensity_mean']['threshAnom'][:,:i_20E,:], axis=1)
    MHW_td_CI[dataSet] = datamask_CI*np.append(MHW_CI['total_days']['threshAnom'][:,i_20E:,:,:], MHW_CI['total_days']['threshAnom'][:,:i_20E,:], axis=1) # pkey doesn't matter, all the same
    rho_f[dataSet] = datamask[dataSet]*np.append(RHO['count']['threshCount'][:,i_20E:], RHO['count']['threshCount'][:,:i_20E], axis=1)
    rho_d[dataSet] = datamask[dataSet]*np.append(RHO['duration']['maxAnom'][:,i_20E:], RHO['duration']['maxAnom'][:,:i_20E], axis=1)
    rho_i[dataSet] = datamask[dataSet]*np.append(RHO['intensity_mean']['threshAnom'][:,i_20E:], RHO['intensity_mean']['threshAnom'][:,:i_20E], axis=1)
    p_f[dataSet] = datamask[dataSet]*np.append(P['count']['threshCount'][:,i_20E:], P['count']['threshCount'][:,:i_20E], axis=1)
    p_d[dataSet] = datamask[dataSet]*np.append(P['duration']['maxAnom'][:,i_20E:], P['duration']['maxAnom'][:,:i_20E], axis=1)
    p_i[dataSet] = datamask[dataSet]*np.append(P['intensity_mean']['threshAnom'][:,i_20E:], P['intensity_mean']['threshAnom'][:,:i_20E], axis=1)
    # Hack out some fixes for ridiculous durations
    MHW_td_ts[dataSet][MHW_td_ts[dataSet]>=365.] = 365.
    MHW_d_ts[dataSet][MHW_d_ts[dataSet]>=365.] = 365.
    MHW_d_CI[dataSet][MHW_d_CI[dataSet]>=365.] = 365.
    #
    del(MHW_m)
    del(MHW_ts)
    del(MHW_CI)
    del(RHO)
    del(P)
    llon[dataSet], llat[dataSet] = np.meshgrid(lon_data, lat_data)
    # Sum / average over globe
    scaling = np.cos(llat[dataSet]*np.pi/180)
    L = 5
    Nens = 100
    MHW_f_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_d_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_i_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_td_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_f_sigMod_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_d_sigMod_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_i_sigMod_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_td_sigMod_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_f_N_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_d_N_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_i_N_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_td_N_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    validCells_f_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    validCells_d_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    validCells_i_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    validCells_td_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    for tt in np.where((years_data >= dataStart[dataSet]) * (years_data <= dataEnd[dataSet]))[0].tolist():
        # Count - Create mask
        mask = np.ones(llat[dataSet].shape)
        mask[np.isnan(MHW_f_ts[dataSet][:,:,tt])] = np.nan
        validCells_f_ts_glob[dataSet][tt] = np.sum(~np.isnan(mask))
        MHW_f_ts_glob[dataSet][tt] = np.average(MHW_f_ts[dataSet][:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
        sigMod_f = np.mean(np.abs(MHW_f_CI[dataSet][::L,::L,tt,:] - np.swapaxes(np.swapaxes(np.tile(MHW_f_ts[dataSet][::L,::L,tt], (2,1,1)), 0, 1), 1, 2)), axis=2)/1.96
        MHW_f_sigMod_glob[dataSet][tt] = (sigMod_f**2)[~np.isnan(mask[::L,::L])].sum()
        MHW_f_N_glob[dataSet][tt] = np.isnan(mask[::L,::L]).sum()
        # Duration - Create mask
        mask = np.ones(llat[dataSet].shape)
        mask[np.isnan(MHW_d_ts[dataSet][:,:,tt])] = np.nan
        validCells_d_ts_glob[dataSet][tt] = np.sum(~np.isnan(mask))
        MHW_d_ts_glob[dataSet][tt] = np.average(MHW_d_ts[dataSet][:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
        sigMod_d = np.mean(np.abs(MHW_d_CI[dataSet][::L,::L,tt,:] - np.swapaxes(np.swapaxes(np.tile(MHW_d_ts[dataSet][::L,::L,tt], (2,1,1)), 0, 1), 1, 2)), axis=2)/1.96
        MHW_d_sigMod_glob[dataSet][tt] = (sigMod_d**2)[~np.isnan(mask[::L,::L])].sum()
        MHW_d_N_glob[dataSet][tt] = np.isnan(mask[::L,::L]).sum()
        # Total MHW days - Create mask
        mask = np.ones(llat[dataSet].shape)
        mask[np.isnan(MHW_td_ts[dataSet][:,:,tt])] = np.nan
        validCells_td_ts_glob[dataSet][tt] = np.sum(~np.isnan(mask))
        MHW_td_ts_glob[dataSet][tt] = np.average(MHW_td_ts[dataSet][:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
        MHW_td_ts_glob_ens = np.zeros((len(lat_data[::L]),len(lon_data[::L]), Nens))
        for i in range(Nens):
            MHW_td_ts_glob_ens[:,:,i] = ((MHW_f_ts[dataSet][::L,::L,tt] + sigMod_f*np.random.randn(len(lat_data[::L]),len(lon_data[::L])))*(MHW_d_ts[dataSet][::L,::L,tt] + sigMod_d*np.random.randn(len(lat_data[::L]),len(lon_data[::L]))))
        sigMod_td = np.std(MHW_td_ts_glob_ens, axis=2)
        MHW_td_sigMod_glob[dataSet][tt] = (sigMod_td**2)[~np.isnan(mask[::L,::L])].sum()
        MHW_td_N_glob[dataSet][tt] = np.isnan(mask[::L,::L]).sum()
    # Create masks for coverage < 50%
    mask_f_ts[dataSet] = np.ones(len(validCells_f_ts_glob[dataSet]))
    mask_f_ts[dataSet][validCells_f_ts_glob[dataSet]<0.5*validCells_f_ts_glob[dataSet].max()] = np.nan
    mask_d_ts[dataSet] = np.ones(len(validCells_d_ts_glob[dataSet]))
    mask_d_ts[dataSet][validCells_d_ts_glob[dataSet]<0.5*validCells_d_ts_glob[dataSet].max()] = np.nan
    mask_i_ts[dataSet] = np.ones(len(validCells_i_ts_glob[dataSet]))
    mask_i_ts[dataSet][validCells_i_ts_glob[dataSet]<0.5*validCells_i_ts_glob[dataSet].max()] = np.nan
    mask_td_ts[dataSet] = np.ones(len(validCells_td_ts_glob[dataSet]))
    mask_td_ts[dataSet][validCells_td_ts_glob[dataSet]<0.5*validCells_td_ts_glob[dataSet].max()] = np.nan
    # Calculate covariance in order to fix sigma
    # Count
    tmp_ts = np.reshape(MHW_f_ts[dataSet][::L,::L,:], (len(lat_data[::L])*len(lon_data[::L]), len(years_data)), order='C')
    cov_glob = 0
    for i1 in range(tmp_ts.shape[0]):
        print dataSet, 'Count', i1+1, tmp_ts.shape[0]
        for i2 in range(i1, tmp_ts.shape[0]):
            x1 = tmp_ts[i1,:]
            x2 = tmp_ts[i2,:]
            if (np.isnan(x1).sum()==len(x1)) + (np.isnan(x2).sum()==len(x2)):
                continue
            else:
                cov_glob += 2.*np.cov(ecj.pad(x1), ecj.pad(x2))[0,1]
    MHW_f_sigMod_glob[dataSet] = np.sqrt((MHW_f_sigMod_glob[dataSet] + cov_glob) / MHW_f_N_glob[dataSet]**2)
    # Duration
    tmp_ts = np.reshape(MHW_d_ts[dataSet][::L,::L,:], (len(lat_data[::L])*len(lon_data[::L]), len(years_data)), order='C')
    cov_glob = 0
    for i1 in range(tmp_ts.shape[0]):
        print dataSet, 'Duration', i1+1, tmp_ts.shape[0]
        for i2 in range(i1, tmp_ts.shape[0]):
            x1 = tmp_ts[i1,:]
            x2 = tmp_ts[i2,:]
            if (np.isnan(x1).sum()==len(x1)) + (np.isnan(x2).sum()==len(x2)):
                continue
            else:
                cov_glob += 2.*np.cov(ecj.pad(x1), ecj.pad(x2))[0,1]
    MHW_d_sigMod_glob[dataSet] = np.sqrt((MHW_d_sigMod_glob[dataSet] + cov_glob) / MHW_d_N_glob[dataSet]**2)
    # Total days
    tmp_ts = np.reshape(MHW_td_ts[dataSet][::L,::L,:], (len(lat_data[::L])*len(lon_data[::L]), len(years_data)), order='C')
    cov_glob = 0
    for i1 in range(tmp_ts.shape[0]):
        print dataSet, 'Total Days', i1+1, tmp_ts.shape[0]
        for i2 in range(i1, tmp_ts.shape[0]):
            x1 = tmp_ts[i1,:]
            x2 = tmp_ts[i2,:]
            if (np.isnan(x1).sum()==len(x1)) + (np.isnan(x2).sum()==len(x2)):
                continue
            else:
                cov_glob += 2.*np.cov(ecj.pad(x1), ecj.pad(x2))[0,1]
    MHW_td_sigMod_glob[dataSet] = np.sqrt((MHW_td_sigMod_glob[dataSet] + cov_glob) / MHW_td_N_glob[dataSet]**2)
    # Shift proxy-based time series to have same mean as AVHRR-based time series
    avhrr = np.in1d(years_data, years)
    MHW_f_ts_glob[dataSet] = MHW_f_ts_glob[dataSet] + (cnt_mean - np.nanmean((MHW_f_ts_glob[dataSet]*mask_f_ts[dataSet])[avhrr]))
    MHW_d_ts_glob[dataSet] = MHW_d_ts_glob[dataSet] + (dur_mean - np.nanmean((MHW_d_ts_glob[dataSet]*mask_d_ts[dataSet])[avhrr]))
    MHW_td_ts_glob[dataSet] = MHW_td_ts_glob[dataSet] + (td_mean - np.nanmean((MHW_td_ts_glob[dataSet]*mask_td_ts[dataSet])[avhrr]))
    #
    # Maps of mean difference
    #
    p_KS_f[dataSet] = np.nan*np.zeros((MHW_f_ts[dataSet].shape[0], MHW_f_ts[dataSet].shape[1]))
    p_KS_d[dataSet] = np.nan*np.zeros((MHW_f_ts[dataSet].shape[0], MHW_f_ts[dataSet].shape[1]))
    p_KS_td[dataSet] = np.nan*np.zeros((MHW_f_ts[dataSet].shape[0], MHW_f_ts[dataSet].shape[1]))
    for j in range(p_KS_f[dataSet].shape[0]):
        print dataSet, j
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
MHW_f_ts_glob_agg = np.zeros((len(MHW_f_ts_glob[dataSet]),len(dataSets)))
MHW_d_ts_glob_agg = np.zeros((len(MHW_d_ts_glob[dataSet]),len(dataSets)))
MHW_td_ts_glob_agg = np.zeros((len(MHW_td_ts_glob[dataSet]),len(dataSets)))
cnt = 0
for dataSet in dataSets:
    MHW_f_ts_glob_agg[:,cnt] = MHW_f_ts_glob[dataSet]*mask_f_ts[dataSet]
    MHW_d_ts_glob_agg[:,cnt] = MHW_d_ts_glob[dataSet]*mask_d_ts[dataSet]
    MHW_td_ts_glob_agg[:,cnt] = MHW_td_ts_glob[dataSet]*mask_td_ts[dataSet]
    cnt += 1

MHW_f_ts_glob_aggAll = np.nanmean(MHW_f_ts_glob_agg, axis=1)
MHW_d_ts_glob_aggAll = np.nanmean(MHW_d_ts_glob_agg, axis=1)
MHW_td_ts_glob_aggAll = np.nanmean(MHW_td_ts_glob_agg, axis=1)

# Aggregate global time series model errors
MHW_f_sigMod_agg = np.zeros(MHW_f_sigMod_glob[dataSet].shape)
MHW_d_sigMod_agg = np.zeros(MHW_d_sigMod_glob[dataSet].shape)
MHW_td_sigMod_agg = np.zeros(MHW_td_sigMod_glob[dataSet].shape)
cnt_f = np.zeros(MHW_td_sigMod_glob[dataSet].shape)
cnt_d = np.zeros(MHW_td_sigMod_glob[dataSet].shape)
cnt_td = np.zeros(MHW_td_sigMod_glob[dataSet].shape)
for dataSet in dataSets:
    valid = ~np.isnan(MHW_f_sigMod_glob[dataSet])*np.isfinite(MHW_f_sigMod_glob[dataSet])
    MHW_f_sigMod_agg[valid] += MHW_f_sigMod_glob[dataSet][valid]**2
    cnt_f[valid] += 1
    valid = ~np.isnan(MHW_d_sigMod_glob[dataSet])*np.isfinite(MHW_d_sigMod_glob[dataSet])
    MHW_d_sigMod_agg[valid] += MHW_d_sigMod_glob[dataSet][valid]**2
    cnt_d[valid] += 1
    valid = ~np.isnan(MHW_td_sigMod_glob[dataSet])*np.isfinite(MHW_td_sigMod_glob[dataSet])
    MHW_td_sigMod_agg[valid] += MHW_td_sigMod_glob[dataSet][valid]**2
    cnt_td[valid] += 1

valid_f = ~np.isnan(np.sum(MHW_f_ts_glob_agg, axis=1))
valid_d = ~np.isnan(np.sum(MHW_d_ts_glob_agg, axis=1))
valid_td = ~np.isnan(np.sum(MHW_td_ts_glob_agg, axis=1))
twoCov_f = 2*np.triu(np.cov(MHW_f_ts_glob_agg[valid_f,:].T), k=1).sum()
twoCov_d = 2*np.triu(np.cov(MHW_d_ts_glob_agg[valid_d,:].T), k=1).sum()
twoCov_td = 2*np.triu(np.cov(MHW_td_ts_glob_agg[valid_td,:].T), k=1).sum()

MHW_f_sigMod_agg = np.sqrt(MHW_f_sigMod_agg + twoCov_f)/cnt_f
MHW_d_sigMod_agg = np.sqrt(MHW_d_sigMod_agg + twoCov_d)/cnt_d
MHW_td_sigMod_agg = np.sqrt(MHW_td_sigMod_agg + twoCov_td)/cnt_td

# Maps of agreement, time slice difference, time mean - on shared grid
dataSetsAgr = dataSets[:]
dl = 5.
dataSetsAgr = ['HadISST', 'ERSST', 'COBE', 'CERA20C', 'SODA']
dl = 2.
lon_agr = np.arange(20, 380, dl)
lat_agr = np.arange(90, -90, -dl) 
llon_agr, llat_agr = np.meshgrid(lon_agr, lat_agr)
MHW_f_m_agr = np.nan*np.zeros((len(lat_agr), len(lon_agr), len(dataSetsAgr)))
MHW_d_m_agr = np.nan*np.zeros((len(lat_agr), len(lon_agr), len(dataSetsAgr)))
MHW_td_m_agr = np.nan*np.zeros((len(lat_agr), len(lon_agr), len(dataSetsAgr)))
MHW_f_dtt_agr = np.nan*np.zeros((len(lat_agr), len(lon_agr), len(dataSetsAgr)))
MHW_d_dtt_agr = np.nan*np.zeros((len(lat_agr), len(lon_agr), len(dataSetsAgr)))
MHW_td_dtt_agr = np.nan*np.zeros((len(lat_agr), len(lon_agr), len(dataSetsAgr)))
cnt = 0
for dataSet in dataSetsAgr:
    for i in range(len(lon_agr)):
        print dataSet, i+1, len(lon_agr)
        for j in range(len(lat_agr)):
            ii = (llon[dataSet][0,:] > lon_agr[i]) * (llon[dataSet][0,:] <= (lon_agr[i]+dl))
            jj = (llat[dataSet][:,0] < lat_agr[j]) * (llat[dataSet][:,0] >= (lat_agr[j]-dl))
            MHW_f_m_agr[j,i,cnt] = np.nanmean(MHW_f_m[dataSet][jj,:][:,ii].flatten())
            MHW_d_m_agr[j,i,cnt] = np.nanmean(MHW_d_m[dataSet][jj,:][:,ii].flatten())
            MHW_td_m_agr[j,i,cnt] = np.nanmean(MHW_td_m[dataSet][jj,:][:,ii].flatten())
            MHW_f_dtt_agr[j,i,cnt] = np.nanmean(MHW_f_dtt[dataSet][jj,:][:,ii].data.flatten())
            MHW_d_dtt_agr[j,i,cnt] = np.nanmean(MHW_d_dtt[dataSet][jj,:][:,ii].data.flatten())
            MHW_td_dtt_agr[j,i,cnt] = np.nanmean(MHW_td_dtt[dataSet][jj,:][:,ii].data.flatten())
    cnt += 1

mask_agr = 1. - (np.sum(np.isnan(MHW_f_dtt_agr), axis=2)==len(dataSetsAgr)).astype(int)
mask_agr[mask_agr==0] = np.nan

# Count of agreement
MHW_f_dtt_agrCnt = np.zeros((len(lat_agr), len(lon_agr)))
MHW_d_dtt_agrCnt = np.zeros((len(lat_agr), len(lon_agr)))
MHW_td_dtt_agrCnt = np.zeros((len(lat_agr), len(lon_agr)))
cnt = 0
for dataSet in dataSetsAgr:
    MHW_f_dtt_agrCnt += np.sign(MHW_f_dtt_agr[:,:,cnt]) == np.sign(np.mean(MHW_f_dtt_agr, axis=2))
    MHW_d_dtt_agrCnt += np.sign(MHW_d_dtt_agr[:,:,cnt]) == np.sign(np.mean(MHW_d_dtt_agr, axis=2))
    MHW_td_dtt_agrCnt += np.sign(MHW_td_dtt_agr[:,:,cnt]) == np.sign(np.mean(MHW_td_dtt_agr, axis=2))
    cnt += 1

# Save some time series
# outfile = pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1900.2016.ALL_ts.no_MEI_PDO_AMO.npz'
# outfile = pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1900.2016.ALL_ts.npz'
# np.savez(outfile, years=years, years_data=years_data, MHW_f_ts_glob=MHW_f_ts_glob, MHW_d_ts_glob=MHW_d_ts_glob, MHW_td_ts_glob=MHW_td_ts_glob, mask_f_ts=mask_f_ts, mask_d_ts=mask_d_ts, mask_td_ts=mask_td_ts, MHW_f_ts_glob_aggAll=MHW_f_ts_glob_aggAll, MHW_d_ts_glob_aggAll=MHW_d_ts_glob_aggAll, MHW_td_ts_glob_aggAll=MHW_td_ts_glob_aggAll, MHW_f_sigMod_agg=MHW_f_sigMod_agg, MHW_d_sigMod_agg=MHW_d_sigMod_agg, MHW_td_sigMod_agg=MHW_td_sigMod_agg)

# Compare time series

plt.figure() #figsize=(19,5))
plt.clf()
dataSetsAll = dataSets[:]
dataSetsAll.append('Dataset mean')
# Time series
plt.subplot(3,1,1)
for dataSet in dataSets:
    plt.plot(years_data, MHW_f_ts_glob[dataSet]*mask_f_ts[dataSet], '-', linewidth=1)
    #plt.plot(years_data, MHW_f_ts_glob[dataSet]*mask_f_ts[dataSet], '.', linewidth=2)
plt.plot(years_data, np.nanmedian(MHW_f_ts_glob_aggAll, axis=1), 'k-', linewidth=2)
plt.grid()
plt.ylabel('Frequency [count]')
plt.legend(dataSetsAll, loc='upper left', fontsize=12)
plt.title('Global-mean time series')
# Time series
plt.subplot(3,1,2)
for dataSet in dataSets:
    plt.plot(years_data, MHW_d_ts_glob[dataSet]*mask_d_ts[dataSet], '-', linewidth=1)
    #plt.plot(years_data, MHW_f_ts_glob[dataSet]*mask_f_ts[dataSet], '.', linewidth=2)
plt.plot(years_data, np.nanmedian(MHW_d_ts_glob_aggAll, axis=1), 'k-', linewidth=2)
plt.grid()
plt.ylabel('Duration [days]')
# Time series
plt.subplot(3,1,3)
for dataSet in dataSets:
    plt.plot(years_data, MHW_td_ts_glob[dataSet]*mask_d_ts[dataSet], '-', linewidth=1)
    #plt.plot(years_data, MHW_f_ts_glob[dataSet]*mask_f_ts[dataSet], '.', linewidth=2)
plt.plot(years_data, np.nanmedian(MHW_td_ts_glob_aggAll, axis=1), 'k-', linewidth=2)
plt.grid()
plt.ylabel('Total MHW days [days]')
plt.title('Global-mean time series')

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_proxies_compareDataSets_ts.png', bbox_inches='tight', pad_inches=0.25, dpi=300)

# Full set of difference maps
domain = [-65, 20, 70, 380]
domain_draw = [-60, 20, 60, 380]
domain_draw = [-60, 60, 60, 380]
dlat = 30
dlon = 60
hatch = '////'
bg_col = '0.6'
cont_col = '0.0'
cmap1 = plt.get_cmap('YlOrRd')
cmap1 = mpl.colors.ListedColormap(cmap1(range(0,256,42)))
cmap1.set_bad(color = 'k', alpha = 0.)
cmap2 = plt.get_cmap('RdBu_r')
cmap2 = mpl.colors.ListedColormap(cmap2(range(0,256,40)))
cmap2.set_bad(color = 'k', alpha = 0.)

cmap3 = plt.get_cmap('YlOrRd')
cmap3 = mpl.colors.ListedColormap(cmap3(np.floor(np.linspace(0, 256, len(dataSets)+1)).astype(int).tolist()))
cmap3.set_bad(color = 'k', alpha = 0.)

Nagr = len(dataSetsAgr) #- 1

plt.figure()
plt.clf()
cnt = 0
for dataSet in dataSets:
    proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
    lonproj, latproj = proj(llon[dataSet], llat[dataSet])
    # Frequency
    plt.subplot(len(dataSets), 3, 3*cnt+1, axisbg=bg_col)
    proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
    proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
    if dataSet == dataSets[-1]:
        proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
    else:
        proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
    plt.pcolor(lonproj, latproj, MHW_f_dtt[dataSet], vmin=-2.5, vmax=2.5, cmap=cmap2)
    if dataSet == dataSets[0]:
        plt.title('Frequency')
    H = plt.colorbar()
    plt.clim(-2.1, 2.1)
    H.set_ticks(np.linspace(-2.1, 2.1, 8))
    H.set_label('[count]')
    plt.contourf(lonproj, latproj, sign_p_d[dataSet], hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
    # Duration
    plt.subplot(len(dataSets), 3, 3*cnt+2, axisbg=bg_col)
    proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
    proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False], dashes=[3,900])
    if dataSet == dataSets[-1]:
        proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
    else:
        proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
    plt.pcolor(lonproj, latproj, MHW_d_dtt[dataSet], vmin=-20, vmax=20, cmap=cmap2)
    if dataSet == dataSets[0]:
        plt.title('Duration')
    H = plt.colorbar()
    plt.clim(-14, 14)
    H.set_ticks(np.linspace(-14, 14, 8))
    H.set_label('[days]')
    plt.contourf(lonproj, latproj, sign_p_d[dataSet], hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
    # Total days
    plt.subplot(len(dataSets), 3, 3*cnt+3, axisbg=bg_col)
    proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
    proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False], dashes=[3,900])
    if dataSet == dataSets[-1]:
        proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
    else:
        proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
    plt.pcolor(lonproj, latproj, MHW_td_dtt[dataSet], vmin=-60, vmax=60, cmap=cmap2)
    if dataSet == dataSets[0]:
        plt.title('Total Days')
    H = plt.colorbar()
    plt.clim(-56, 56) # should be multipele of 7
    H.set_ticks(np.linspace(-56, 56, 8))
    H.set_label('[days]')
    plt.contourf(lonproj, latproj, sign_p_f[dataSet], hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
    #
    cnt += 1

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_proxies_compareDataSets_allDataSets_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
# plt.savefig('MHW_proxies_compareDataSets_allDataSets_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)

plt.figure()
plt.clf()
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
lonproj, latproj = proj(llon_agr, llat_agr)
# Frequency
plt.subplot(3,1,1, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(np.sum(MHW_f_dtt_agr>0, axis=2)*mask_agr), vmin=0, vmax=len(dataSetsAgr)+1, cmap=cmap3)
H = plt.colorbar()
plt.clim(-0.5, len(dataSetsAgr)+0.5)
H.set_ticks(np.linspace(0, len(dataSetsAgr), len(dataSetsAgr)+1))
# Duration
plt.subplot(3,1,2, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(np.sum(MHW_d_dtt_agr>0, axis=2)*mask_agr), vmin=0, vmax=len(dataSetsAgr)+1, cmap=cmap3)
H = plt.colorbar()
plt.clim(-0.5, len(dataSetsAgr)+0.5)
H.set_ticks(np.linspace(0, len(dataSetsAgr), len(dataSetsAgr)+1))
# Total days
plt.subplot(3,1,3, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(np.sum(MHW_td_dtt_agr>0, axis=2)*mask_agr), vmin=0, vmax=len(dataSetsAgr)+1, cmap=cmap3)
H = plt.colorbar()
plt.clim(-0.5, len(dataSetsAgr)+0.5)
H.set_ticks(np.linspace(0, len(dataSetsAgr), len(dataSetsAgr)+1))

plt.figure()
plt.clf()
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
lonproj, latproj = proj(llon_agr, llat_agr)
lonproj_d, latproj_d = proj(llon_agr+0.5*dl, llat_agr-0.5*dl)
# Frequency
plt.subplot(3,1,1, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(np.mean(MHW_f_dtt_agr, axis=2)), vmin=-2.5, vmax=2.5, cmap=cmap2)
H = plt.colorbar()
plt.clim(-2.1, 2.1)
H.set_ticks(np.linspace(-2.1, 2.1, 8))
H.set_label('[count]')
plt.contourf(lonproj_d, latproj_d, (MHW_f_dtt_agrCnt>=Nagr).astype(float)*mask_agr, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
# Duration
plt.subplot(3,1,2, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(np.mean(MHW_d_dtt_agr, axis=2)), vmin=-20, vmax=20, cmap=cmap2)
H = plt.colorbar()
plt.clim(-14, 14)
H.set_ticks(np.linspace(-14, 14, 8))
H.set_label('[days]')
plt.contourf(lonproj_d, latproj_d, (MHW_d_dtt_agrCnt>=Nagr).astype(float)*mask_agr, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
# Total days
plt.subplot(3,1,3, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(np.mean(MHW_td_dtt_agr, axis=2)), vmin=60, vmax=60, cmap=cmap2)
H = plt.colorbar()
plt.clim(-56, 56)
H.set_ticks(np.linspace(-56, 56, 8))
H.set_label('[days]')
plt.contourf(lonproj_d, latproj_d, (MHW_td_dtt_agrCnt>=Nagr).astype(float)*mask_agr, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')


# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_proxies_compareDataSets_map.png', bbox_inches='tight', pad_inches=0.25, dpi=300)


# Figure 4 for manuscript

dataSetsAllNOAA = dataSets[:]
dataSetsAllNOAA.append('NOAA OI SST')
dataSetsAllNOAA.append('Dataset mean')

plt.figure(figsize=(13,8))
plt.clf()
hatch = '////'
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
lonproj, latproj = proj(llon_agr, llat_agr)
lonproj_d, latproj_d = proj(llon_agr+0.5*dl, llat_agr-0.5*dl)
# MHW Frequency
plt.subplot(3,2,1, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(np.mean(MHW_f_dtt_agr, axis=2)), vmin=-2.5, vmax=2.5, cmap=cmap2)
H = plt.colorbar()
plt.clim(-2.1, 2.1)
H.set_ticks(np.linspace(-2.1, 2.1, 8))
H.set_label('[count]')
plt.contourf(lonproj_d, latproj_d, (MHW_f_dtt_agrCnt>=Nagr).astype(float)*mask_agr, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
ax = plt.subplot2grid((7,2), (0,1), rowspan=2)
for dataSet in dataSets:
    plt.plot(years_data, MHW_f_ts_glob[dataSet]*mask_f_ts[dataSet], '-', linewidth=1)
plt.plot(years, MHW_f_ts_glob['AVHRR'], 'r-', linewidth=2)
plt.plot(years_data, MHW_f_ts_glob_aggAll, 'k-', linewidth=2)
plt.fill_between(years_data, (MHW_f_ts_glob_aggAll - MHW_f_sigMod_agg*1.96), (MHW_f_ts_glob_aggAll + MHW_f_sigMod_agg*1.96), color='0.8')
#plt.fill_between(years_data, (MHW_f_ts_glob_aggAll - MHW_f_sigMod_agg), (MHW_f_ts_glob_aggAll + MHW_f_sigMod_agg), color='0.6')
plt.xlim(1900, 2020)
plt.ylim(1., 6.0)
#plt.grid()
ax.set_xticklabels([])
# MHW Duration
plt.subplot(3,2,3, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(np.mean(MHW_d_dtt_agr, axis=2)), vmin=-20, vmax=20, cmap=cmap2)
H = plt.colorbar()
plt.clim(-14, 14)
H.set_ticks(np.linspace(-14, 14, 8))
H.set_label('[days]')
plt.contourf(lonproj_d, latproj_d, (MHW_d_dtt_agrCnt>=Nagr).astype(float)*mask_agr, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
ax = plt.subplot2grid((7,2), (2,1), rowspan=3)
for dataSet in dataSets:
    plt.plot(years_data, MHW_d_ts_glob[dataSet]*mask_d_ts[dataSet], '-', linewidth=1)
plt.plot(years, MHW_d_ts_glob['AVHRR'], 'r-', linewidth=2)
plt.plot(years_data, MHW_d_ts_glob_aggAll, 'k-', linewidth=2)
plt.legend(dataSetsAllNOAA, loc='upper left', fontsize=10, ncol=2)
plt.fill_between(years_data, (MHW_d_ts_glob_aggAll - MHW_d_sigMod_agg*1.96), (MHW_d_ts_glob_aggAll + MHW_d_sigMod_agg*1.96), color='0.8')
#plt.fill_between(years_data, (MHW_d_ts_glob_aggAll - MHW_d_sigMod_agg), (MHW_d_ts_glob_aggAll + MHW_d_sigMod_agg), color='0.6')
plt.xlim(1900, 2020)
plt.ylim(7, 27)
#plt.grid()
ax.set_xticklabels([])
# MHW Total Days
plt.subplot(3,2,5, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(np.mean(MHW_td_dtt_agr, axis=2)), vmin=60, vmax=60, cmap=cmap2)
H = plt.colorbar()
plt.clim(-56, 56)
H.set_ticks(np.linspace(-56, 56, 8))
H.set_label('[days]')
plt.contourf(lonproj_d, latproj_d, (MHW_td_dtt_agrCnt>=Nagr).astype(float)*mask_agr, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.subplot2grid((7,2), (5,1), rowspan=2)
for dataSet in dataSets:
    plt.plot(years_data, MHW_td_ts_glob[dataSet]*mask_d_ts[dataSet], '-', linewidth=1)
plt.plot(years, MHW_td_ts_glob['AVHRR'], 'r-', linewidth=2)
plt.plot(years_data, MHW_td_ts_glob_aggAll, 'k-', linewidth=2)
plt.fill_between(years_data, (MHW_td_ts_glob_aggAll - MHW_td_sigMod_agg*1.96), (MHW_td_ts_glob_aggAll + MHW_td_sigMod_agg*1.96), color='0.8')
#plt.fill_between(years_data, (MHW_td_ts_glob_aggAll - MHW_td_sigMod_agg), (MHW_td_ts_glob_aggAll + MHW_td_sigMod_agg), color='0.6')
plt.xlim(1900, 2020)
plt.ylim(5, 105)
#plt.grid()

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_proxies_compareDataSets_ts_orig.png', bbox_inches='tight', pad_inches=0.25, dpi=300)
# plt.savefig('MHW_proxies_compareDataSets_ts_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)

# Positive change
1.*np.sum(np.mean(MHW_f_dtt_agr, axis=2)>0) / np.sum(~np.isnan(np.mean(MHW_f_dtt_agr, axis=2)))
1.*np.sum(np.mean(MHW_d_dtt_agr, axis=2)>0) / np.sum(~np.isnan(np.mean(MHW_d_dtt_agr, axis=2)))
1.*np.sum(np.mean(MHW_td_dtt_agr, axis=2)>0) / np.sum(~np.isnan(np.mean(MHW_td_dtt_agr, axis=2)))

# Global-avg time-slice means
np.nanmean(MHW_f_ts_glob_aggAll[tt2]) - np.nanmean(MHW_f_ts_glob_aggAll[tt1])
stats.ks_2samp(MHW_f_ts_glob_aggAll[tt2], MHW_f_ts_glob_aggAll[tt1])
np.nanmean(MHW_d_ts_glob_aggAll[tt2]) - np.nanmean(MHW_d_ts_glob_aggAll[tt1])
stats.ks_2samp(MHW_d_ts_glob_aggAll[tt2], MHW_d_ts_glob_aggAll[tt1])
np.nanmean(MHW_td_ts_glob_aggAll[tt2]) - np.nanmean(MHW_td_ts_glob_aggAll[tt1])
stats.ks_2samp(MHW_td_ts_glob_aggAll[tt2], MHW_td_ts_glob_aggAll[tt1])
np.nanmean(MHW_td_ts_glob_aggAll[tt1])

# Proportion of change over tt1
(np.nanmean(MHW_f_ts_glob_aggAll[tt2]) - np.nanmean(MHW_f_ts_glob_aggAll[tt1]))/np.nanmean(MHW_f_ts_glob_aggAll[tt1])
(np.nanmean(MHW_d_ts_glob_aggAll[tt2]) - np.nanmean(MHW_d_ts_glob_aggAll[tt1]))/np.nanmean(MHW_d_ts_glob_aggAll[tt1])
(np.nanmean(MHW_td_ts_glob_aggAll[tt2]) - np.nanmean(MHW_td_ts_glob_aggAll[tt1]))/np.nanmean(MHW_td_ts_glob_aggAll[tt1])

# Correlation between proxy time series and NOAA OI SST
for dataSet in dataSets:
    print dataSet, ecj.pattern_correlation(MHW_f_ts_glob['AVHRR'], (MHW_f_ts_glob[dataSet]*mask_f_ts[dataSet])[years_data>=years[0]])

print 'dataSet-mean', ecj.pattern_correlation(MHW_f_ts_glob['AVHRR'], MHW_f_ts_glob_aggAll[years_data>=years[0]])

for dataSet in dataSets:
    print dataSet, ecj.pattern_correlation(MHW_d_ts_glob['AVHRR'], (MHW_d_ts_glob[dataSet]*mask_d_ts[dataSet])[years_data>=years[0]])

print 'dataSet-mean', ecj.pattern_correlation(MHW_d_ts_glob['AVHRR'], MHW_d_ts_glob_aggAll[years_data>=years[0]])

# Figure S9 for manuscript

plt.figure(figsize=(13,8))
plt.clf()
hatch = '////'
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
lonproj, latproj = proj(llon_agr, llat_agr)
lonproj_d, latproj_d = proj(llon_agr+0.5*dl, llat_agr-0.5*dl)
# MHW Frequency
plt.subplot(3,2,1, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(np.mean(MHW_f_m_agr, axis=2)), vmin=0.5, vmax=4, cmap=cmap1)
plt.title('(A) Mean MHW frequency')
H = plt.colorbar()
plt.clim(0.8,3.6)
H.set_ticks(np.linspace(0.8, 3.6, 8))
H.set_label('Annual number [count]')
# MHW Duration
plt.subplot(3,2,3, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(np.mean(MHW_d_m_agr, axis=2)), vmin=5, vmax=25, cmap=cmap1)
plt.title('(B) Mean MHW duration')
H = plt.colorbar()
plt.clim(7, 21)
H.set_ticks(np.linspace(7, 21, 8))
H.set_label('[days]')
# MHW Total Days
plt.subplot(3,2,5, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(np.mean(MHW_td_m_agr, axis=2)), vmin=10, vmax=70, cmap=cmap1)
plt.title('(C) Mean Total MHW days')
H = plt.colorbar()
plt.clim(6, 62)
H.set_ticks(np.linspace(6, 62, 8))
H.set_label('[days]')

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_proxies_compareDataSets_mean.png', bbox_inches='tight', pad_inches=0.25, dpi=300)
# plt.savefig('MHW_proxies_compareDataSets_mean.png', bbox_inches='tight', pad_inches=0.05, dpi=300)

# Figure S8 for manuscript (correlations)

plt.figure()
plt.clf()
cnt = 0
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
lonproj, latproj = proj(llon_agr, llat_agr)
lonproj_had, latproj_had = proj(llon['HadISST'], llat['HadISST'])
for dataSet in dataSets:
    proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
    lonproj, latproj = proj(llon[dataSet], llat[dataSet])
    # Frequency
    plt.subplot(len(dataSets), 4, 4*cnt+1, axisbg=bg_col)
    proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
    proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
    if dataSet == dataSets[-1]:
        proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
    else:
        proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
    if dataSet == 'CERA20C':
        #plt.pcolor(lonproj_had, latproj_had, rho_f[dataSet][:-1,:], vmin=0, vmax=1, cmap=plt.cm.YlOrRd)
        plt.contourf(lonproj_had, latproj_had, rho_f[dataSet][:-1,:], levels=[0,0.2,0.4,0.6,0.8,1.0], cmap=plt.cm.YlOrRd)
    else:
        #plt.pcolor(lonproj, latproj, rho_f[dataSet], vmin=0, vmax=1, cmap=plt.cm.YlOrRd)
        plt.contourf(lonproj, latproj, rho_f[dataSet], levels=[0,0.2,0.4,0.6,0.8,1.0], cmap=plt.cm.YlOrRd)
    if dataSet == dataSets[0]:
        plt.title('Frequency')
    H = plt.colorbar()
    plt.clim(0, 1)
    H.set_label('Correlation')
    plt.contourf(lonproj, latproj, (p_f[dataSet]<=0.05).astype(float), hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
    # Duration
    plt.subplot(len(dataSets), 4, 4*cnt+2, axisbg=bg_col)
    proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
    proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False], dashes=[3,900])
    if dataSet == dataSets[-1]:
        proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
    else:
        proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
    if dataSet == 'CERA20C':
        #plt.pcolor(lonproj_had, latproj_had, rho_i[dataSet][:-1,:], vmin=0, vmax=1, cmap=plt.cm.YlOrRd)
        plt.contourf(lonproj_had, latproj_had, rho_i[dataSet][:-1,:], levels=[0,0.2,0.4,0.6,0.8,1.0], cmap=plt.cm.YlOrRd)
    else:
        #plt.pcolor(lonproj, latproj, rho_i[dataSet], vmin=0, vmax=1, cmap=plt.cm.YlOrRd)
        plt.contourf(lonproj, latproj, rho_i[dataSet], levels=[0,0.2,0.4,0.6,0.8,1.0], cmap=plt.cm.YlOrRd)
    if dataSet == dataSets[0]:
        plt.title('Intensity')
    H = plt.colorbar()
    plt.clim(0, 1)
    H.set_label('Correlation')
    plt.contourf(lonproj, latproj, (p_i[dataSet]<=0.05).astype(float), hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
    # Total days
    plt.subplot(len(dataSets), 4, 4*cnt+3, axisbg=bg_col)
    proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
    proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False], dashes=[3,900])
    if dataSet == dataSets[-1]:
        proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
    else:
        proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
    if dataSet == 'CERA20C':
        #plt.pcolor(lonproj_had, latproj_had, rho_d[dataSet][:-1,:], vmin=0, vmax=1, cmap=plt.cm.YlOrRd)
        plt.contourf(lonproj_had, latproj_had, rho_d[dataSet][:-1,:], levels=[0,0.2,0.4,0.6,0.8,1.0], cmap=plt.cm.YlOrRd)
    else:
        #plt.pcolor(lonproj, latproj, rho_d[dataSet], vmin=0, vmax=1, cmap=plt.cm.YlOrRd)
        plt.contourf(lonproj, latproj, rho_d[dataSet], levels=[0,0.2,0.4,0.6,0.8,1.0], cmap=plt.cm.YlOrRd)
    if dataSet == dataSets[0]:
        plt.title('Duration')
    H = plt.colorbar()
    plt.clim(0, 1)
    H.set_label('Correlation')
    plt.contourf(lonproj, latproj, (p_d[dataSet]<=0.05).astype(float), hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
    #
    H = plt.subplot(len(dataSets), 4, 4*cnt+4)
    x = np.arange(0,1.0,0.01)
    pdf = sp.stats.gaussian_kde(ecj.nonans(rho_f[dataSet].flatten())).evaluate(x)
    plt.plot(x, pdf, 'k-')
    pdf = sp.stats.gaussian_kde(ecj.nonans(rho_i[dataSet].flatten())).evaluate(x)
    plt.plot(x, pdf, 'r-')
    pdf = sp.stats.gaussian_kde(ecj.nonans(rho_d[dataSet].flatten())).evaluate(x)
    plt.plot(x, pdf, 'b-')
    plt.ylim(0,4)
    plt.xlim(0,1)
    #plt.grid()
    if dataSet == dataSets[-1]:
        plt.xlabel('Correlation coefficient')
    else:
        H.set_xticklabels([])
    H.yaxis.tick_right()
    H.yaxis.set_label_position('right')
    H.set_ylabel('Probability density')
    plt.legend(['Frequency', 'Intensity', 'Duration'], loc='upper left', fontsize=10, ncol=3)
    #
    cnt += 1

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_proxies_compareDataSets_allRho_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
# plt.savefig('MHW_proxies_compareDataSets_allRho_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)

for dataSet in dataSets:
    scaling = np.cos(llat[dataSet]*np.pi/180)
    print ((p_f[dataSet]<=0.05).astype(float)*scaling).sum() / ((~np.isnan(p_f[dataSet])).astype(float)*scaling).sum()

for dataSet in dataSets:
    scaling = np.cos(llat[dataSet]*np.pi/180)
    print ((p_i[dataSet]<=0.05).astype(float)*scaling).sum() / ((~np.isnan(p_i[dataSet])).astype(float)*scaling).sum()

for dataSet in dataSets:
    scaling = np.cos(llat[dataSet]*np.pi/180)
    print ((p_d[dataSet]<=0.05).astype(float)*scaling).sum() / ((~np.isnan(p_d[dataSet])).astype(float)*scaling).sum()

#
# Save output for Dan
#

from netCDF4 import Dataset

Y, X = llon_agr.shape
T = len(years_data)

nc = Dataset(pathroot + '/data/MHWs/Trends/mhw_proxies_GLM_dataFrDan.nc', 'w') #, format='NETCDF3_64BIT')

nc.createDimension('lon', X)
nc.createDimension('lat', Y)
nc.createDimension('time', T)

nclon = nc.createVariable('lon', 'f4', ('lon',))
nclat = nc.createVariable('lat', 'f4', ('lat',))
ncyears  = nc.createVariable('years', 'f4', ('time',))
ncts = nc.createVariable('MHW_totalDays_ts', 'f4', ('time',))
ncmean = nc.createVariable('MHW_totalDays_map_mean', 'f4', ('lat','lon'))
nctr = nc.createVariable('MHW_totalDays_map_change', 'f4', ('lat','lon'))

nclon[:] = lon_agr
nclat[:] = lat_agr
ncyears[:] = years_data
ncts[:] = MHW_td_ts_glob_aggAll
tmp = np.mean(MHW_td_m_agr, axis=2)
tmp[tmp>=100.] = 100.
ncmean[:,:] = tmp
nctr[:,:] = np.mean(MHW_td_dtt_agr, axis=2)

nc.close()



