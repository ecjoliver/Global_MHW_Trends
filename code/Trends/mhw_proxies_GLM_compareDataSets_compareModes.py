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
modeFooter = {'Original': '', 'No MEI': '.no_MEI', 'No PDO': '.no_PDO', 'No AMO': '.no_AMO'}
modes = modeFooter.keys()
years_data = np.arange(yearStart, 2016+1)
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
MHW_f_ts_glob = {}
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
for mode in modes:
    MHW_f_m[mode] = {}
    MHW_d_m[mode] = {}
    MHW_i_m[mode] = {}
    MHW_td_m[mode] = {}
    MHW_f_ts[mode] = {}
    MHW_d_ts[mode] = {}
    MHW_i_ts[mode] = {}
    MHW_td_ts[mode] = {}
    MHW_f_ts_glob[mode] = {}
    MHW_d_ts_glob[mode] = {}
    MHW_i_ts_glob[mode] = {}
    MHW_td_ts_glob[mode] = {}
    validCells_f_ts_glob[mode] = {}
    validCells_d_ts_glob[mode] = {}
    validCells_i_ts_glob[mode] = {}
    validCells_td_ts_glob[mode] = {}
    mask_f_ts[mode] = {}
    mask_d_ts[mode] = {}
    mask_i_ts[mode] = {}
    mask_td_ts[mode] = {}
    for dataSet in dataSets:
        print mode, 'of', modes, '\n', dataSet, 'of', dataSets
        outfile = pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1900.2016.' + dataSet + modeFooter[mode] + '.npz'
        data = np.load(outfile)
        lon_data = data['lon_data']
        lat_data = data['lat_data']
        MHW_m = data['MHW_m'].item()
        MHW_ts = data['MHW_ts'].item()
        if dataSet == 'COBE':
            outfile_dm = pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1900.2016.' + dataSet + '.datamask.npz'
            data_dm = np.load(outfile_dm)
            datamask[dataSet] = data_dm['datamask']
        else:
            datamask[dataSet] = data['datamask']
        datamask[dataSet][datamask[dataSet]==0] = np.nan
        datamask_ts = np.swapaxes(np.swapaxes(np.tile(datamask[dataSet], (len(years_data),1,1)), 0, 1), 1, 2)
        # Extract mean and time series, simultaneously re-map to run 20E to 380E, and apply datamasks
        i_20E = np.where(lon_data>20)[0][0]
        lon_data = np.append(lon_data[i_20E:], lon_data[:i_20E]+360)
        datamask[dataSet] = np.append(datamask[dataSet][:,i_20E:], datamask[dataSet][:,:i_20E], axis=1)
        datamask_ts = np.append(datamask_ts[:,i_20E:,:], datamask_ts[:,:i_20E,:], axis=1)
        MHW_f_m[mode][dataSet] = datamask[dataSet]*np.append(MHW_m['count']['threshCount'][:,i_20E:], MHW_m['count']['threshCount'][:,:i_20E], axis=1)
        MHW_d_m[mode][dataSet] = datamask[dataSet]*np.append(MHW_m['duration']['maxAnom'][:,i_20E:], MHW_m['duration']['maxAnom'][:,:i_20E], axis=1)
        MHW_i_m[mode][dataSet] = datamask[dataSet]*np.append(MHW_m['intensity_mean']['threshAnom'][:,i_20E:], MHW_m['intensity_mean']['threshAnom'][:,:i_20E], axis=1)
        MHW_td_m[mode][dataSet] = datamask[dataSet]*np.append(MHW_m['total_days']['threshAnom'][:,i_20E:], MHW_m['total_days']['threshAnom'][:,:i_20E], axis=1) # pkey doesn't matter, all the same
        MHW_f_ts[mode][dataSet] = datamask_ts*np.append(MHW_ts['count']['threshCount'][:,i_20E:,:], MHW_ts['count']['threshCount'][:,:i_20E,:], axis=1)
        MHW_d_ts[mode][dataSet] = datamask_ts*np.append(MHW_ts['duration']['maxAnom'][:,i_20E:,:], MHW_ts['duration']['maxAnom'][:,:i_20E,:], axis=1)
        MHW_i_ts[mode][dataSet] = datamask_ts*np.append(MHW_ts['intensity_mean']['threshAnom'][:,i_20E:,:], MHW_ts['intensity_mean']['threshAnom'][:,:i_20E,:], axis=1)
        MHW_td_ts[mode][dataSet] = datamask_ts*np.append(MHW_ts['total_days']['threshAnom'][:,i_20E:,:], MHW_ts['total_days']['threshAnom'][:,:i_20E,:], axis=1) # pkey doesn't matter, all the same
        # Hack out some fixes for ridiculous durations
        MHW_td_ts[mode][dataSet][MHW_td_ts[mode][dataSet]>=365.] = 365.
        MHW_d_ts[mode][dataSet][MHW_d_ts[mode][dataSet]>=365.] = 365.
        #
        del(MHW_m)
        del(MHW_ts)
        llon[dataSet], llat[dataSet] = np.meshgrid(lon_data, lat_data)
        # Sum / average over globe
        scaling = np.cos(llat[dataSet]*np.pi/180)
        L = 5
        Nens = 100
        MHW_f_ts_glob[mode][dataSet] = np.zeros(MHW_f_ts[mode][dataSet].shape[2])
        MHW_d_ts_glob[mode][dataSet] = np.zeros(MHW_f_ts[mode][dataSet].shape[2])
        MHW_i_ts_glob[mode][dataSet] = np.zeros(MHW_f_ts[mode][dataSet].shape[2])
        MHW_td_ts_glob[mode][dataSet] = np.zeros(MHW_f_ts[mode][dataSet].shape[2])
        validCells_f_ts_glob[mode][dataSet] = np.zeros(MHW_f_ts[mode][dataSet].shape[2])
        validCells_d_ts_glob[mode][dataSet] = np.zeros(MHW_f_ts[mode][dataSet].shape[2])
        validCells_i_ts_glob[mode][dataSet] = np.zeros(MHW_f_ts[mode][dataSet].shape[2])
        validCells_td_ts_glob[mode][dataSet] = np.zeros(MHW_f_ts[mode][dataSet].shape[2])
        for tt in np.where((years_data >= dataStart[dataSet]) * (years_data <= dataEnd[dataSet]))[0].tolist():
            # Count - Create mask
            mask = np.ones(llat[dataSet].shape)
            mask[np.isnan(MHW_f_ts[mode][dataSet][:,:,tt])] = np.nan
            validCells_f_ts_glob[mode][dataSet][tt] = np.sum(~np.isnan(mask))
            MHW_f_ts_glob[mode][dataSet][tt] = np.average(MHW_f_ts[mode][dataSet][:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
            # Duration - Create mask
            mask = np.ones(llat[dataSet].shape)
            mask[np.isnan(MHW_d_ts[mode][dataSet][:,:,tt])] = np.nan
            validCells_d_ts_glob[mode][dataSet][tt] = np.sum(~np.isnan(mask))
            MHW_d_ts_glob[mode][dataSet][tt] = np.average(MHW_d_ts[mode][dataSet][:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
            # Total MHW days - Create mask
            mask = np.ones(llat[dataSet].shape)
            mask[np.isnan(MHW_td_ts[mode][dataSet][:,:,tt])] = np.nan
            validCells_td_ts_glob[mode][dataSet][tt] = np.sum(~np.isnan(mask))
            MHW_td_ts_glob[mode][dataSet][tt] = np.average(MHW_td_ts[mode][dataSet][:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
        # Create masks for coverage < 50%
        mask_f_ts[mode][dataSet] = np.ones(len(validCells_f_ts_glob[mode][dataSet]))
        mask_f_ts[mode][dataSet][validCells_f_ts_glob[mode][dataSet]<0.5*validCells_f_ts_glob[mode][dataSet].max()] = np.nan
        mask_d_ts[mode][dataSet] = np.ones(len(validCells_d_ts_glob[mode][dataSet]))
        mask_d_ts[mode][dataSet][validCells_d_ts_glob[mode][dataSet]<0.5*validCells_d_ts_glob[mode][dataSet].max()] = np.nan
        mask_i_ts[mode][dataSet] = np.ones(len(validCells_i_ts_glob[mode][dataSet]))
        mask_i_ts[mode][dataSet][validCells_i_ts_glob[mode][dataSet]<0.5*validCells_i_ts_glob[mode][dataSet].max()] = np.nan
        mask_td_ts[mode][dataSet] = np.ones(len(validCells_td_ts_glob[mode][dataSet]))
        mask_td_ts[mode][dataSet][validCells_td_ts_glob[mode][dataSet]<0.5*validCells_td_ts_glob[mode][dataSet].max()] = np.nan

# Aggregate time series from different data sets
MHW_f_ts_glob_aggAll = {}
MHW_d_ts_glob_aggAll = {}
for mode in modes:
    MHW_f_ts_glob_agg = np.zeros((len(MHW_f_ts_glob[mode][dataSet]),len(dataSets)))
    MHW_d_ts_glob_agg = np.zeros((len(MHW_d_ts_glob[mode][dataSet]),len(dataSets)))
    cnt = 0
    for dataSet in dataSets:
        MHW_f_ts_glob_agg[:,cnt] = MHW_f_ts_glob[mode][dataSet]*mask_f_ts[mode][dataSet]
        MHW_d_ts_glob_agg[:,cnt] = MHW_d_ts_glob[mode][dataSet]*mask_d_ts[mode][dataSet]
        cnt += 1
    MHW_f_ts_glob_aggAll[mode] = np.nanmean(MHW_f_ts_glob_agg, axis=1)
    MHW_d_ts_glob_aggAll[mode] = np.nanmean(MHW_d_ts_glob_agg, axis=1)

# Maps of agreement, time slice difference, time mean - on shared grid
dataSetsAgr = ['HadISST', 'ERSST', 'COBE', 'CERA20C', 'SODA']
dl = 2.
lon_agr = np.arange(20, 380, dl)
lat_agr = np.arange(90, -90, -dl) 
llon_agr, llat_agr = np.meshgrid(lon_agr, lat_agr)
MHW_f_md_agg = {}
MHW_d_md_agg = {}
MHW_f_sd_agg = {}
MHW_d_sd_agg = {}
for mode in modes:
    MHW_f_md_agg[mode] = np.nan*np.zeros((len(lat_agr), len(lon_agr), len(dataSetsAgr)))
    MHW_d_md_agg[mode] = np.nan*np.zeros((len(lat_agr), len(lon_agr), len(dataSetsAgr)))
    MHW_f_sd_agg[mode] = np.nan*np.zeros((len(lat_agr), len(lon_agr), len(dataSetsAgr)))
    MHW_d_sd_agg[mode] = np.nan*np.zeros((len(lat_agr), len(lon_agr), len(dataSetsAgr)))
    cnt = 0
    for dataSet in dataSetsAgr:
        for i in range(len(lon_agr)):
            print mode, dataSet, i+1, len(lon_agr)
            for j in range(len(lat_agr)):
                ii = (llon[dataSet][0,:] > lon_agr[i]) * (llon[dataSet][0,:] <= (lon_agr[i]+dl))
                jj = (llat[dataSet][:,0] < lat_agr[j]) * (llat[dataSet][:,0] >= (lat_agr[j]-dl))
                MHW_f_md_agg[mode][j,i,cnt] = np.nanmean(MHW_f_m['Original'][dataSet][jj,:][:,ii].flatten()) - np.nanmean(MHW_f_m[mode][dataSet][jj,:][:,ii].flatten())
                MHW_d_md_agg[mode][j,i,cnt] = np.nanmean(MHW_d_m['Original'][dataSet][jj,:][:,ii].flatten()) - np.nanmean(MHW_d_m[mode][dataSet][jj,:][:,ii].flatten())
                MHW_f_sd_agg[mode][j,i,cnt] = np.nanstd(np.nanmean(MHW_f_ts['Original'][dataSet][jj,:,:][:,ii,:].reshape(-1, MHW_f_ts['Original'][dataSet].shape[-1]), axis=0) - np.nanmean(MHW_f_ts[mode][dataSet][jj,:,:][:,ii,:].reshape(-1, MHW_f_ts[mode][dataSet].shape[-1]), axis=0))
                MHW_d_sd_agg[mode][j,i,cnt] = np.nanstd(np.nanmean(MHW_d_ts['Original'][dataSet][jj,:,:][:,ii,:].reshape(-1, MHW_d_ts['Original'][dataSet].shape[-1]), axis=0) - np.nanmean(MHW_d_ts[mode][dataSet][jj,:,:][:,ii,:].reshape(-1, MHW_d_ts[mode][dataSet].shape[-1]), axis=0))
        cnt += 1

for mode in modes:
    MHW_f_md_agg[mode] = np.nanmean(MHW_f_md_agg[mode], axis=2)
    MHW_d_md_agg[mode] = np.nanmean(MHW_d_md_agg[mode], axis=2)
    MHW_f_sd_agg[mode] = np.sqrt(np.nanmean(MHW_f_sd_agg[mode]**2, axis=2))
    MHW_d_sd_agg[mode] = np.sqrt(np.nanmean(MHW_d_sd_agg[mode]**2, axis=2))

# Figures for MS

domain = [-65, 20, 70, 380]
domain_draw = [-60, 20, 60, 380]
domain_draw = [-60, 60, 60, 380]
dlat = 30
dlon = 60
bg_col = '0.6'
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
lonproj, latproj = proj(llon_agr, llat_agr)
levels_m_f = {}
levels_m_d = {}
levels_v_f = {}
levels_v_d = {}
levels_m_f['No MEI'] = [-1.5,-1,-0.5,-0.25,-0.1,0.1,0.25,0.5,1,1.5]
levels_m_f['No PDO'] = [-1,-0.5,-0.35,-0.2,-0.1,0.1,0.2,0.25,0.5,1]
levels_m_f['No AMO'] = [-1,-0.5,-0.35,-0.2,-0.1,0.1,0.2,0.25,0.5,1]
levels_m_d['No MEI'] = [-15,-7.5,-5,-2.5,-1,1,2.5,5,7.5,15]
levels_m_d['No PDO'] = [-8,-6,-4,-2,-1,1,2,4,6,8]
levels_m_d['No AMO'] = [-8,-6,-4,-2,-1,1,2,4,6,8]
levels_v_f['No MEI'] = np.arange(0.,3.+0.5,0.5) #np.arange(0.,3.5+0.5,0.5)
levels_v_f['No PDO'] = np.arange(0.,3.+0.5,0.5)
levels_v_f['No AMO'] = np.arange(0.,3.+0.5,0.5)
levels_v_d['No MEI'] = [0,2.5,5,7.5,10,15,25,50]
levels_v_d['No PDO'] = [0,2.5,5,7.5,10,15,25,50]
levels_v_d['No AMO'] = [0,2.5,5,7.5,10,15,25,50]

for mode in ['No MEI', 'No PDO', 'No AMO']:
    plt.figure(figsize=(19,5))
    plt.clf()
    # MHW Frequency (mean)
    plt.subplot(2,3,1, axisbg=bg_col)
    proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
    proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
    proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
    plt.contourf(lonproj, latproj, MHW_f_md_agg[mode], levels=levels_m_f[mode], cmap=plt.cm.RdBu_r)
    H = plt.colorbar()
    H.set_label(r'Annual number [count]')
    plt.clim(levels_m_f[mode][1], levels_m_f[mode][-2]) #plt.clim(-2.2,2.2)
    plt.title('Mean of difference')
    # MHW Frequency (std)
    plt.subplot(2,3,2, axisbg=bg_col)
    proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
    proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
    proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
    plt.contourf(lonproj, latproj, MHW_f_sd_agg[mode], levels=levels_v_f[mode], cmap=plt.cm.rainbow)
    H = plt.colorbar()
    H.set_label(r'Annual number [count]')
    plt.clim(0, levels_v_f[mode][-2]) #plt.clim(0, 3.5)
    plt.title('Std. dev. of difference')
    # Time series
    plt.subplot(2,3,3)
    plt.plot(years_data, MHW_f_ts_glob_aggAll['Original'], 'k-', linewidth=2)
    plt.plot(years_data, MHW_f_ts_glob_aggAll[mode], 'r-', linewidth=2)
    #plt.grid()
    plt.xlim(1900, 2020)
    plt.ylabel('[count]')
    plt.legend(['Original', mode], loc='upper left')
    plt.title('Global-mean time series')
    # MHW Duration (mean)
    plt.subplot(2,3,4, axisbg=bg_col)
    proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
    proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
    proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
    plt.contourf(lonproj, latproj, MHW_d_md_agg[mode], levels=levels_m_d[mode], cmap=plt.cm.RdBu_r)
    H = plt.colorbar()
    H.set_label(r'[days]')
    plt.clim(levels_m_d[mode][1], levels_m_d[mode][-2]) #plt.clim(-15,15)
    # MHW Duration (std)
    plt.subplot(2,3,5, axisbg=bg_col)
    proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
    proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
    proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
    #plt.contourf(lonproj, latproj, np.log10(datamask*np.nanstd(MHW_dur_ts - MHW_dur_ts_noENSO, axis=2)))#, levels=np.arange(0.5,3.5+0.5,0.5), cmap=plt.cm.rainbow)
    plt.contourf(lonproj, latproj, MHW_d_sd_agg[mode], levels=levels_v_d[mode], cmap=plt.cm.rainbow)
    H = plt.colorbar()
    H.set_label(r'[days]')
    plt.clim(0, levels_v_d[mode][-2]) #plt.clim(0,100)
    # Time series
    plt.subplot(2,3,6)
    plt.plot(years_data, MHW_d_ts_glob_aggAll['Original'], 'k-', linewidth=2)
    plt.plot(years_data, MHW_d_ts_glob_aggAll[mode], 'r-', linewidth=2)
    #plt.grid()
    plt.xlim(1900, 2020)
    plt.ylabel('[days]')
    plt.legend(['Original', mode], loc='upper left')
    #
    #plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_noModes_HadISST_' + mode + '.png', bbox_inches='tight', dpi=300)
    plt.savefig('MHW_noModes_' + mode + '.png', bbox_inches='tight', dpi=300)


