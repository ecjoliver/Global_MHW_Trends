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

dataSets = ['HadSST3']
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
MHW_f_sig = {}
MHW_d_sig = {}
MHW_i_sig = {}
MHW_td_sig = {}
MHW_f_ts_glob = {}
MHW_d_ts_glob = {}
MHW_i_ts_glob = {}
MHW_td_ts_glob = {}
MHW_f_sigMod_glob = {}
MHW_d_sigMod_glob = {}
MHW_i_sigMod_glob = {}
MHW_td_sigMod_glob = {}
MHW_f_sigObs_glob = {}
MHW_d_sigObs_glob = {}
MHW_i_sigObs_glob = {}
MHW_td_sigObs_glob = {}
MHW_f_sigTot_glob = {}
MHW_d_sigTot_glob = {}
MHW_i_sigTot_glob = {}
MHW_td_sigTot_glob = {}
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
    outfile[dataSet] = pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1900.2016.' + dataSet + '.sig.npz'
    #outfile[dataSet] = pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1900.2016.' + dataSet + '.no_MEI_PDO_AMO.npz'
    data = np.load(outfile[dataSet])
    lon_data = data['lon_data']
    lat_data = data['lat_data']
    MHW_m = data['MHW_m'].item()
    MHW_ts = data['MHW_ts'].item()
    MHW_CI = data['MHW_CI'].item()
    MHW_sig = data['MHW_sig'].item()
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
    MHW_f_sig[dataSet] = datamask_CI*np.append(MHW_sig['count']['threshCount'][:,i_20E:,:,:], MHW_sig['count']['threshCount'][:,:i_20E,:], axis=1)
    MHW_d_sig[dataSet] = datamask_CI*np.append(MHW_sig['duration']['maxAnom'][:,i_20E:,:,:], MHW_sig['duration']['maxAnom'][:,:i_20E,:], axis=1)
    MHW_i_sig[dataSet] = datamask_CI*np.append(MHW_sig['intensity_mean']['threshAnom'][:,i_20E:,:,:], MHW_sig['intensity_mean']['threshAnom'][:,:i_20E,:], axis=1)
    MHW_td_sig[dataSet] = datamask_CI*np.append(MHW_sig['total_days']['threshAnom'][:,i_20E:,:,:], MHW_sig['total_days']['threshAnom'][:,:i_20E,:], axis=1) # pkey doesn't matter, all the same
    # Hack out some fixes for ridiculous durations
    MHW_td_ts[dataSet][MHW_td_ts[dataSet]>=365.] = 365.
    MHW_d_ts[dataSet][MHW_d_ts[dataSet]>=365.] = 365.
    MHW_d_CI[dataSet][MHW_d_CI[dataSet]>=365.] = 365.
    MHW_d_sig[dataSet][MHW_d_sig[dataSet]>=365.] = 365.
    #
    del(MHW_m)
    del(MHW_ts)
    del(MHW_CI)
    del(MHW_sig)
    llon[dataSet], llat[dataSet] = np.meshgrid(lon_data, lat_data)
    # Sum / average over globe
    scaling = np.cos(llat[dataSet]*np.pi/180)
    L = 1
    Nens = 100
    MHW_f_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_d_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_i_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_td_ts_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_f_sigMod_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_d_sigMod_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_i_sigMod_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_td_sigMod_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_f_sigObs_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_d_sigObs_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_i_sigObs_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
    MHW_td_sigObs_glob[dataSet] = np.zeros(MHW_f_ts[dataSet].shape[2])
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
        sigObs_f = np.mean(np.abs(MHW_f_sig[dataSet][::L,::L,tt,:] - np.swapaxes(np.swapaxes(np.tile(MHW_f_ts[dataSet][::L,::L,tt], (2,1,1)), 0, 1), 1, 2)), axis=2)/1.96
        MHW_f_sigMod_glob[dataSet][tt] = (sigMod_f**2)[~np.isnan(mask[::L,::L])].sum()
        MHW_f_sigObs_glob[dataSet][tt] = (sigObs_f**2)[~np.isnan(mask[::L,::L])].sum()
        MHW_f_N_glob[dataSet][tt] = np.isnan(mask[::L,::L]).sum()
        # Duration - Create mask
        mask = np.ones(llat[dataSet].shape)
        mask[np.isnan(MHW_d_ts[dataSet][:,:,tt])] = np.nan
        validCells_d_ts_glob[dataSet][tt] = np.sum(~np.isnan(mask))
        MHW_d_ts_glob[dataSet][tt] = np.average(MHW_d_ts[dataSet][:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
        sigMod_d = np.mean(np.abs(MHW_d_CI[dataSet][::L,::L,tt,:] - np.swapaxes(np.swapaxes(np.tile(MHW_d_ts[dataSet][::L,::L,tt], (2,1,1)), 0, 1), 1, 2)), axis=2)/1.96
        sigObs_d = np.mean(np.abs(MHW_d_sig[dataSet][::L,::L,tt,:] - np.swapaxes(np.swapaxes(np.tile(MHW_d_ts[dataSet][::L,::L,tt], (2,1,1)), 0, 1), 1, 2)), axis=2)/1.96
        fixNan = np.isnan(sigObs_d) * ~np.isnan(sigMod_d)
        sigObs_d[fixNan] = 0.
        MHW_d_sigMod_glob[dataSet][tt] = (sigMod_d**2)[~np.isnan(mask[::L,::L])].sum()
        MHW_d_sigObs_glob[dataSet][tt] = (sigObs_d**2)[~np.isnan(mask[::L,::L])].sum()
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
        MHW_td_ts_glob_ens = np.zeros((len(lat_data[::L]),len(lon_data[::L]), Nens))
        for i in range(Nens):
            MHW_td_ts_glob_ens[:,:,i] = ((MHW_f_ts[dataSet][::L,::L,tt] + sigObs_f*np.random.randn(len(lat_data[::L]),len(lon_data[::L])))*(MHW_d_ts[dataSet][::L,::L,tt] + sigObs_d*np.random.randn(len(lat_data[::L]),len(lon_data[::L]))))
        sigObs_td = np.std(MHW_td_ts_glob_ens, axis=2)
        MHW_td_sigObs_glob[dataSet][tt] = (sigObs_td**2)[~np.isnan(mask[::L,::L])].sum()
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
    MHW_f_sigTot_glob[dataSet] = np.sqrt((MHW_f_sigMod_glob[dataSet] + MHW_f_sigObs_glob[dataSet] + cov_glob) / MHW_f_N_glob[dataSet]**2)
    MHW_f_sigMod_glob[dataSet] = np.sqrt((MHW_f_sigMod_glob[dataSet] + cov_glob) / MHW_f_N_glob[dataSet]**2)
    MHW_f_sigObs_glob[dataSet] = np.sqrt((MHW_f_sigObs_glob[dataSet] + cov_glob) / MHW_f_N_glob[dataSet]**2)
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
    MHW_d_sigTot_glob[dataSet] = np.sqrt((MHW_d_sigMod_glob[dataSet] + MHW_d_sigObs_glob[dataSet] + cov_glob) / MHW_d_N_glob[dataSet]**2)
    MHW_d_sigMod_glob[dataSet] = np.sqrt((MHW_d_sigMod_glob[dataSet] + cov_glob) / MHW_d_N_glob[dataSet]**2)
    MHW_d_sigObs_glob[dataSet] = np.sqrt((MHW_d_sigObs_glob[dataSet] + cov_glob) / MHW_d_N_glob[dataSet]**2)
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
    MHW_td_sigTot_glob[dataSet] = np.sqrt((MHW_td_sigMod_glob[dataSet] + MHW_td_sigObs_glob[dataSet] + cov_glob) / MHW_td_N_glob[dataSet]**2)
    MHW_td_sigMod_glob[dataSet] = np.sqrt((MHW_td_sigMod_glob[dataSet] + cov_glob) / MHW_td_N_glob[dataSet]**2)
    MHW_td_sigObs_glob[dataSet] = np.sqrt((MHW_td_sigObs_glob[dataSet] + cov_glob) / MHW_td_N_glob[dataSet]**2)
    # Shift proxy-based time series to have same mean as AVHRR-based time series
    avhrr = np.in1d(years_data, years)
    MHW_f_ts_glob[dataSet] = MHW_f_ts_glob[dataSet] + (cnt_mean - np.nanmean((MHW_f_ts_glob[dataSet]*mask_f_ts[dataSet])[avhrr]))
    MHW_d_ts_glob[dataSet] = MHW_d_ts_glob[dataSet] + (dur_mean - np.nanmean((MHW_d_ts_glob[dataSet]*mask_d_ts[dataSet])[avhrr]))
    MHW_td_ts_glob[dataSet] = MHW_td_ts_glob[dataSet] + (td_mean - np.nanmean((MHW_td_ts_glob[dataSet]*mask_td_ts[dataSet])[avhrr]))

# Load in other datasets
infile = pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1900.2016.ALL_ts.npz'
data = np.load(infile)
#MHW_f_ts_glob[which] = data['MHW_f_ts_glob'].item()
#MHW_d_ts_glob[which] = data['MHW_d_ts_glob'].item()
#MHW_td_ts_glob[which] = data['MHW_td_ts_glob'].item()
#mask_f_ts[which] = data['mask_f_ts'].item()
#mask_d_ts[which] = data['mask_d_ts'].item()
#mask_td_ts[which] = data['mask_td_ts'].item()
MHW_f_ts_glob_aggAll = data['MHW_f_ts_glob_aggAll']
MHW_d_ts_glob_aggAll = data['MHW_d_ts_glob_aggAll']
#MHW_td_ts_glob_aggAll = data['MHW_td_ts_glob_aggAll']
#MHW_f_sigMod_agg[which] = data['MHW_f_sigMod_agg']
#MHW_d_sigMod_agg[which] = data['MHW_d_sigMod_agg']
#MHW_td_sigMod_agg[which] = data['MHW_td_sigMod_agg']

#
# Plot HadSST3 errors
#

plt.figure(figsize=(16,7))
plt.clf()
yearStart = 1920
AX = plt.subplot(2,3,1)
plt.plot(years_data, MHW_f_ts_glob_aggAll, 'b-', linewidth=1)
plt.plot(years_data, MHW_f_ts_glob[dataSet]*mask_f_ts[dataSet], 'k-', linewidth=2)
plt.legend(['Dataset mean (as in Fig. 5)', 'HadSST3'])
plt.fill_between(years_data, mask_f_ts[dataSet]*(MHW_f_ts_glob[dataSet] - MHW_f_sigMod_glob[dataSet]*1.96), mask_f_ts[dataSet]*(MHW_f_ts_glob[dataSet] + MHW_f_sigMod_glob[dataSet]*1.96), color='0.8')
plt.ylabel('Frequency [count]')
plt.xlim(yearStart, 2020)
AX.set_xticklabels([])
plt.title('(A) Model error')
AX = plt.subplot(2,3,2)
plt.plot(years_data, MHW_f_ts_glob_aggAll, 'b-', linewidth=1)
plt.plot(years_data, MHW_f_ts_glob[dataSet]*mask_f_ts[dataSet], 'k-', linewidth=2)
plt.fill_between(years_data, mask_f_ts[dataSet]*(MHW_f_ts_glob[dataSet] - MHW_f_sigObs_glob[dataSet]*1.96), mask_f_ts[dataSet]*(MHW_f_ts_glob[dataSet] + MHW_f_sigObs_glob[dataSet]*1.96), color='0.8')
plt.xlim(yearStart, 2020)
AX.set_xticklabels([])
plt.title('(B) Observational error')
AX = plt.subplot(2,3,3)
plt.plot(years_data, MHW_f_ts_glob_aggAll, 'b-', linewidth=1)
plt.plot(years_data, MHW_f_ts_glob[dataSet]*mask_f_ts[dataSet], 'k-', linewidth=2)
plt.fill_between(years_data, mask_f_ts[dataSet]*(MHW_f_ts_glob[dataSet] - MHW_f_sigTot_glob[dataSet]*1.96), mask_f_ts[dataSet]*(MHW_f_ts_glob[dataSet] + MHW_f_sigTot_glob[dataSet]*1.96), color='0.8')
plt.xlim(yearStart, 2020)
AX.set_xticklabels([])
plt.title('(C) Total error')
AX = plt.subplot(2,3,4)
plt.plot(years_data, MHW_d_ts_glob_aggAll, 'b-', linewidth=1)
plt.plot(years_data, MHW_d_ts_glob[dataSet]*mask_d_ts[dataSet], 'k-', linewidth=2)
plt.fill_between(years_data, mask_d_ts[dataSet]*(MHW_d_ts_glob[dataSet] - MHW_d_sigMod_glob[dataSet]*1.96), mask_d_ts[dataSet]*(MHW_d_ts_glob[dataSet] + MHW_d_sigMod_glob[dataSet]*1.96), color='0.8')
plt.ylabel('Duration [days]')
plt.ylim(5, 30)
plt.xlim(yearStart, 2020)
plt.title('(D) Model error')
AX = plt.subplot(2,3,5)
plt.plot(years_data, MHW_d_ts_glob_aggAll, 'b-', linewidth=1)
plt.plot(years_data, MHW_d_ts_glob[dataSet]*mask_d_ts[dataSet], 'k-', linewidth=2)
plt.fill_between(years_data, mask_d_ts[dataSet]*(MHW_d_ts_glob[dataSet] - MHW_d_sigObs_glob[dataSet]*1.96), mask_d_ts[dataSet]*(MHW_d_ts_glob[dataSet] + MHW_d_sigObs_glob[dataSet]*1.96), color='0.8')
plt.ylim(5, 30)
plt.xlim(yearStart, 2020)
plt.title('(E) Observational error')
AX = plt.subplot(2,3,6)
plt.plot(years_data, MHW_d_ts_glob_aggAll, 'b-', linewidth=1)
plt.plot(years_data, MHW_d_ts_glob[dataSet]*mask_d_ts[dataSet], 'k-', linewidth=2)
plt.fill_between(years_data, mask_d_ts[dataSet]*(MHW_d_ts_glob[dataSet] - MHW_d_sigTot_glob[dataSet]*1.96), mask_d_ts[dataSet]*(MHW_d_ts_glob[dataSet] + MHW_d_sigTot_glob[dataSet]*1.96), color='0.8')
plt.ylim(5, 30)
plt.xlim(yearStart, 2020)
plt.title('(F) Total error')
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_proxies_errors_' + HadID + '_orig.pdf', bbox_inches='tight', pad_inches=0.5)
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_proxies_errors_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
# plt.savefig('MHW_proxies_errors_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)


