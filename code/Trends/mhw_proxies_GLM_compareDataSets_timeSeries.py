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

dataSets = ['HadISST',            'ERSST',           'COBE', 'CERA20C', 'SODA']
infiles = {'wModes': pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1900.2016.ALL_ts.npz', 'noModes': pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1900.2016.ALL_ts.no_MEI_PDO_AMO.npz'}

MHW_f_ts_glob = {}
MHW_d_ts_glob = {}
MHW_td_ts_glob = {}
mask_f_ts = {}
mask_d_ts = {}
mask_td_ts = {}
MHW_f_ts_glob_aggAll = {}
MHW_d_ts_glob_aggAll = {}
MHW_td_ts_glob_aggAll = {}
MHW_f_sigMod_agg = {}
MHW_d_sigMod_agg = {}
MHW_td_sigMod_agg = {}
for which in infiles.keys():
    data = np.load(infiles[which])
    years = data['years']
    years_data = data['years_data']
    MHW_f_ts_glob[which] = data['MHW_f_ts_glob'].item()
    MHW_d_ts_glob[which] = data['MHW_d_ts_glob'].item()
    MHW_td_ts_glob[which] = data['MHW_td_ts_glob'].item()
    mask_f_ts[which] = data['mask_f_ts'].item()
    mask_d_ts[which] = data['mask_d_ts'].item()
    mask_td_ts[which] = data['mask_td_ts'].item()
    MHW_f_ts_glob_aggAll[which] = data['MHW_f_ts_glob_aggAll']
    MHW_d_ts_glob_aggAll[which] = data['MHW_d_ts_glob_aggAll']
    MHW_td_ts_glob_aggAll[which] = data['MHW_td_ts_glob_aggAll']
    MHW_f_sigMod_agg[which] = data['MHW_f_sigMod_agg']
    MHW_d_sigMod_agg[which] = data['MHW_d_sigMod_agg']
    MHW_td_sigMod_agg[which] = data['MHW_td_sigMod_agg']

# Figure 5 for manuscript

dataSetsAllNOAA = dataSets[:]
dataSetsAllNOAA.append('Dataset mean (original)')
dataSetsAllNOAA.append('Dataset mean (modes removed)')

plt.figure(figsize=(13,8))
plt.clf()
# MHW Frequency
ax = plt.subplot2grid((7,2), (0,1), rowspan=2)
for dataSet in dataSets:
    plt.plot(years_data, MHW_f_ts_glob['noModes'][dataSet]*mask_f_ts['noModes'][dataSet], '-', linewidth=1)
plt.plot(years_data, MHW_f_ts_glob_aggAll['wModes'], 'b-', linewidth=1) #, color=(0.3,0.3,0.9))
plt.plot(years_data, MHW_f_ts_glob_aggAll['noModes'], 'k-', linewidth=2)
plt.fill_between(years_data, (MHW_f_ts_glob_aggAll['noModes'] - MHW_f_sigMod_agg['noModes']*1.96), (MHW_f_ts_glob_aggAll['noModes'] + MHW_f_sigMod_agg['noModes']*1.96), color='0.8')
plt.xlim(1900, 2020)
plt.ylim(1.5, 5.5)
ax.set_xticklabels([])
plt.ylabel('[count]')
# MHW Duration
ax = plt.subplot2grid((7,2), (2,1), rowspan=3)
for dataSet in dataSets:
    plt.plot(years_data, MHW_d_ts_glob['noModes'][dataSet]*mask_d_ts['noModes'][dataSet], '-', linewidth=1)
plt.plot(years_data, MHW_d_ts_glob_aggAll['wModes'], 'b-', linewidth=1) #, color=(0.3,0.3,0.9))
plt.plot(years_data, MHW_d_ts_glob_aggAll['noModes'], 'k-', linewidth=2)
plt.legend(dataSetsAllNOAA, loc='upper left', fontsize=10, ncol=2)
plt.fill_between(years_data, (MHW_d_ts_glob_aggAll['noModes'] - MHW_d_sigMod_agg['noModes']*1.96), (MHW_d_ts_glob_aggAll['noModes'] + MHW_d_sigMod_agg['noModes']*1.96), color='0.8')
plt.xlim(1900, 2020)
plt.ylim(7.5, 21.5)
ax.set_xticklabels([])
plt.ylabel('[days]')
# MHW Total Days
plt.subplot2grid((7,2), (5,1), rowspan=2)
for dataSet in dataSets:
    plt.plot(years_data, MHW_td_ts_glob['noModes'][dataSet]*mask_d_ts['noModes'][dataSet], '-', linewidth=1)
plt.plot(years_data, MHW_td_ts_glob_aggAll['wModes'], 'b-', linewidth=1) #, color=(0.3,0.3,0.9))
plt.plot(years_data, MHW_td_ts_glob_aggAll['noModes'], 'k-', linewidth=2)
plt.fill_between(years_data, (MHW_td_ts_glob_aggAll['noModes'] - MHW_td_sigMod_agg['noModes']*1.96), (MHW_td_ts_glob_aggAll['noModes'] + MHW_td_sigMod_agg['noModes']*1.96), color='0.8')
plt.xlim(1900, 2020)
plt.ylim(10, 100)
plt.ylabel('[days]')

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_proxies_compareDataSets_noModes_orig.png', bbox_inches='tight', pad_inches=0.25, dpi=300)
# plt.savefig('MHW_proxies_compareDataSets_noModes_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)


# Std of modes-only ts
print np.nanstd(MHW_f_ts_glob_aggAll['wModes'] - MHW_f_ts_glob_aggAll['noModes'])
print np.nanstd(MHW_d_ts_glob_aggAll['wModes'] - MHW_d_ts_glob_aggAll['noModes'])
print np.nanstd(MHW_td_ts_glob_aggAll['wModes'] - MHW_td_ts_glob_aggAll['noModes'])

# Prop of variance accounted for by 'noModes's-only ts
print np.nanvar(MHW_f_ts_glob_aggAll['wModes'] - MHW_f_ts_glob_aggAll['noModes'])/np.var(MHW_f_ts_glob_aggAll['wModes'])
print np.nanvar(MHW_d_ts_glob_aggAll['wModes'] - MHW_d_ts_glob_aggAll['noModes'])/np.nanvar(MHW_d_ts_glob_aggAll['wModes'])
print np.nanvar(MHW_td_ts_glob_aggAll['wModes'] - MHW_td_ts_glob_aggAll['noModes'])/np.nanvar(MHW_td_ts_glob_aggAll['wModes'])

# Compare difference in global mean change
tt1 = (years_data>=1925) * (years_data<=1954)
tt2 = (years_data>=1987) * (years_data<=2016)

dtt_f_wModes  = np.nanmean(MHW_f_ts_glob_aggAll['wModes'][tt2]) - np.nanmean(MHW_f_ts_glob_aggAll['wModes'][tt1])
dtt_f_noModes = np.nanmean(MHW_f_ts_glob_aggAll['noModes'][tt2]) - np.nanmean(MHW_f_ts_glob_aggAll['noModes'][tt1])
print (dtt_f_wModes - dtt_f_noModes) / dtt_f_wModes
dtt_d_wModes  = np.nanmean(MHW_d_ts_glob_aggAll['wModes'][tt2]) - np.nanmean(MHW_d_ts_glob_aggAll['wModes'][tt1])
dtt_d_noModes = np.nanmean(MHW_d_ts_glob_aggAll['noModes'][tt2]) - np.nanmean(MHW_d_ts_glob_aggAll['noModes'][tt1])
print (dtt_d_wModes - dtt_d_noModes) / dtt_d_wModes
dtt_td_wModes  = np.nanmean(MHW_td_ts_glob_aggAll['wModes'][tt2]) - np.nanmean(MHW_td_ts_glob_aggAll['wModes'][tt1])
dtt_td_noModes = np.nanmean(MHW_td_ts_glob_aggAll['noModes'][tt2]) - np.nanmean(MHW_td_ts_glob_aggAll['noModes'][tt1])

print dtt_f_noModes, dtt_d_noModes, dtt_td_noModes
