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

file1 = '/home/ecoliver/Desktop/data/MHWs/Trends/sst_meanVarSkew_L'
file2 = '/home/ecoliver/Desktop/data/MHWs/Trends/sst_meanVarSkew_R'

#di = 4
#i = 900/di
#i = 360/2
i1 = 1110

data1 = np.load(file1+'.npz')
data2 = np.load(file2+'.npz')

years = data1['years']
lat_map = data1['lat_map']
lon_map = np.append(data1['lon_map'][:i1], data2['lon_map'][i1:])

SST_mean_ts = np.append(data1['SST_mean_ts'][:,:i1,:], data2['SST_mean_ts'][:,i1:,:], axis=1)
SST_var_ts = np.append(data1['SST_var_ts'][:,:i1,:], data2['SST_var_ts'][:,i1:,:], axis=1)
SST_skew_ts = np.append(data1['SST_skew_ts'][:,:i1,:], data2['SST_skew_ts'][:,i1:,:], axis=1)

outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/sst_meanVarSkew'
np.savez(outfile, lon_map=lon_map, lat_map=lat_map, SST_mean_ts=SST_mean_ts, SST_var_ts=SST_var_ts, SST_skew_ts=SST_skew_ts, years=years)


