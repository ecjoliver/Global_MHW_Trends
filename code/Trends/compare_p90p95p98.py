'''

  Compare global mean MHW time series calculated
  using 90th, 95th and 98th percentiles.

'''

import numpy as np
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

header = '/home/ecoliver/Desktop/data/MHWs/Trends/'
tsfiles = {'p90': 'mhw_census.2016_ts', 'p95': 'mhw_census.2016_ts.p95', 'p98': 'mhw_census.2016_ts.p98'}
ps = ['p90', 'p95', 'p98']

MHW_cnt_ts_glob = {}
MHW_dur_ts_glob = {}
MHW_max_ts_glob = {}
MHW_td_ts_glob = {}
for p in ps:
    filename = header + tsfiles[p] + '.npz'
    data = np.load(filename)
    years = data['years']
    MHW_cnt_ts_glob[p] = data['MHW_cnt_ts_glob']
    MHW_dur_ts_glob[p] = data['MHW_dur_ts_glob']
    MHW_max_ts_glob[p] = data['MHW_max_ts_glob']
    MHW_td_ts_glob[p] = data['MHW_td_ts_glob']


# Plot w/ trends
alpha = 0.05 # 0.01, 0.10
plt.clf()
# Frequency
plt.subplot(2,2,1)
for p in ps:
    mean, slope, dslope = ecj.trend_TheilSen(years, MHW_cnt_ts_glob[p], alpha)
    print 'cnt', p, dslope[0], slope, dslope[1] # slope*len(years) # linear increase over record
    if (dslope[0] > 0) | (dslope[1] < 0):
        plt.plot(years, MHW_cnt_ts_glob[p], '-', linewidth=2)
    else:
        plt.plot(years, MHW_cnt_ts_glob[p], '-')
plt.xlim(years.min()-1, years.max()+1)
plt.grid()
plt.ylabel('Frequency [count]')
plt.legend(ps, loc='upper left', fontsize=12)
# Duration
plt.subplot(2,2,2)
for p in ps:
    mean, slope, dslope = ecj.trend_TheilSen(years, MHW_dur_ts_glob[p], alpha)
    print 'dur', p, dslope[0], slope, dslope[1] # slope*len(years) # linear increase over record
    if (dslope[0] > 0) | (dslope[1] < 0):
        plt.plot(years, MHW_dur_ts_glob[p], '-', linewidth=2)
    else:
        plt.plot(years, MHW_dur_ts_glob[p], '-')
plt.xlim(years.min()-1, years.max()+1)
plt.grid()
plt.ylabel('Duration [days]')
# Max intensity
plt.subplot(2,2,3)
for p in ps:
    mean, slope, dslope = ecj.trend_TheilSen(years, MHW_max_ts_glob[p], alpha)
    print 'max', p, dslope[0], slope, dslope[1] # slope*len(years) # linear increase over record
    if (dslope[0] > 0) | (dslope[1] < 0):
        plt.plot(years, MHW_max_ts_glob[p], '-', linewidth=2)
    else:
        plt.plot(years, MHW_max_ts_glob[p], '-')
plt.xlim(years.min()-1, years.max()+1)
plt.grid()
plt.ylabel('Maximum intensity [$^\circ$C]')
# Total days
plt.subplot(2,2,4)
for p in ps:
    mean, slope, dslope = ecj.trend_TheilSen(years, MHW_td_ts_glob[p], alpha)
    print 'td', p, dslope[0], slope, dslope[1] # slope*len(years) # linear increase over record
    if (dslope[0] > 0) | (dslope[1] < 0):
        plt.plot(years, MHW_td_ts_glob[p], '-', linewidth=2)
    else:
        plt.plot(years, MHW_td_ts_glob[p], '-')
plt.xlim(years.min()-1, years.max()+1)
plt.xlim(years.min()-1, years.max()+1)
plt.grid()
plt.ylabel('Annual MHW days')








