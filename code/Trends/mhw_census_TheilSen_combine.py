'''

  Combine two data files

'''

# Load required modules

import numpy as np
from scipy import io
from scipy import linalg
from scipy import stats
from datetime import date

from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm

import marineHeatWaves as mhw
import trendSimAR1

#
# Load data files and combine
#

file1 = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016.excessTrends.lores'
file2 = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016.excessTrends.lores.neg'

#di = 4
#i = 900/di
#i = 360/2
i1 = 335

data1 = np.load(file1+'.npz')
data2 = np.load(file2+'.npz')

MHW_cnt_tr = np.append(data1['MHW_cnt_tr'][:,:i1], np.fliplr(data2['MHW_cnt_tr'])[:,i1:], axis=1)
MHW_mean_tr = np.append(data1['MHW_mean_tr'][:,:i1], np.fliplr(data2['MHW_mean_tr'])[:,i1:], axis=1)
MHW_max_tr = np.append(data1['MHW_max_tr'][:,:i1], np.fliplr(data2['MHW_max_tr'])[:,i1:], axis=1)
MHW_dur_tr = np.append(data1['MHW_dur_tr'][:,:i1], np.fliplr(data2['MHW_dur_tr'])[:,i1:], axis=1)

ar1_tau = np.append(data1['ar1_tau'][:,:i1], np.fliplr(data2['ar1_tau'])[:,i1:], axis=1)
ar1_sig_eps = np.append(data1['ar1_sig_eps'][:,:i1], np.fliplr(data2['ar1_sig_eps'])[:,i1:], axis=1)
ar1_putrend_cnt = np.append(data1['ar1_putrend_cnt'][:,:i1], np.fliplr(data2['ar1_putrend_cnt'])[:,i1:], axis=1)
ar1_putrend_mean = np.append(data1['ar1_putrend_mean'][:,:i1], np.fliplr(data2['ar1_putrend_mean'])[:,i1:], axis=1)
ar1_putrend_max = np.append(data1['ar1_putrend_max'][:,:i1], np.fliplr(data2['ar1_putrend_max'])[:,i1:], axis=1)
ar1_putrend_dur = np.append(data1['ar1_putrend_dur'][:,:i1], np.fliplr(data2['ar1_putrend_dur'])[:,i1:], axis=1)
ar1_pltrend_cnt = np.append(data1['ar1_pltrend_cnt'][:,:i1], np.fliplr(data2['ar1_pltrend_cnt'])[:,i1:], axis=1)
ar1_pltrend_mean = np.append(data1['ar1_pltrend_mean'][:,:i1], np.fliplr(data2['ar1_pltrend_mean'])[:,i1:], axis=1)
ar1_pltrend_max = np.append(data1['ar1_pltrend_max'][:,:i1], np.fliplr(data2['ar1_pltrend_max'])[:,i1:], axis=1)
ar1_pltrend_dur = np.append(data1['ar1_pltrend_dur'][:,:i1], np.fliplr(data2['ar1_pltrend_dur'])[:,i1:], axis=1)
ar1_mean_cnt = np.append(data1['ar1_mean_cnt'][:,:i1], np.fliplr(data2['ar1_mean_cnt'])[:,i1:], axis=1)
ar1_mean_mean = np.append(data1['ar1_mean_mean'][:,:i1], np.fliplr(data2['ar1_mean_mean'])[:,i1:], axis=1)
ar1_mean_max = np.append(data1['ar1_mean_max'][:,:i1], np.fliplr(data2['ar1_mean_max'])[:,i1:], axis=1)
ar1_mean_dur = np.append(data1['ar1_mean_dur'][:,:i1], np.fliplr(data2['ar1_mean_dur'])[:,i1:], axis=1)

lon_map = np.append(data1['lon_map'][:i1], np.flipud(data2['lon_map'])[i1:])
lat_map = data1['lat_map']

#outfile = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census_TS.2016.combine'
outfile = '/home/ecoliver/Desktop/data/extreme_SSTs/mhw_census.2016.excessTrends.lores.combine'
np.savez(outfile, lon_map=lon_map, lat_map=lat_map, MHW_cnt_tr=MHW_cnt_tr, MHW_dur_tr=MHW_dur_tr, MHW_mean_tr=MHW_mean_tr, MHW_max_tr=MHW_max_tr, ar1_tau=ar1_tau, ar1_sig_eps=ar1_sig_eps, ar1_putrend_cnt=ar1_putrend_cnt, ar1_putrend_mean=ar1_putrend_mean, ar1_putrend_max=ar1_putrend_max, ar1_putrend_dur=ar1_putrend_dur, ar1_pltrend_cnt=ar1_pltrend_cnt, ar1_pltrend_mean=ar1_pltrend_mean, ar1_pltrend_max=ar1_pltrend_max, ar1_pltrend_dur=ar1_pltrend_dur, ar1_mean_cnt=ar1_mean_cnt, ar1_mean_mean=ar1_mean_mean, ar1_mean_max=ar1_mean_max, ar1_mean_dur=ar1_mean_dur)

