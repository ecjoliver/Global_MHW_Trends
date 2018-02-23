'''

  Software which uses the MHW definition
  of Hobday et al. (2015) applied to 
  select SST time series around the globe

'''

# Load required modules

import numpy as np
from scipy import io
from scipy import linalg
from scipy import stats
from datetime import date
from netCDF4 import Dataset

from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm

import marineHeatWaves as mhw

# Load some select time series

#
# observations
#

pathroot = '/mnt/erebor/'
#pathroot = '/home/ecoliver/Desktop/'
header = pathroot+'data/sst/noaa_oi_v2/avhrr/'
file0 = header + '1982/avhrr-only-v2.19820101.nc'
t = np.arange(date(1982,1,1).toordinal(),date(2016,12,31).toordinal()+1)
T = len(t)
year = np.zeros((T))
month = np.zeros((T))
day = np.zeros((T))
for i in range(T):
    year[i] = date.fromordinal(t[i]).year
    month[i] = date.fromordinal(t[i]).month
    day[i] = date.fromordinal(t[i]).day

#
# lat and lons of obs
#

fileobj = Dataset(file0, 'r')
lon = fileobj.variables['lon'][:].astype(float)
lat = fileobj.variables['lat'][:].astype(float)
fill_value = fileobj.variables['sst']._FillValue.astype(float)
scale = fileobj.variables['sst'].scale_factor.astype(float)
offset = fileobj.variables['sst'].add_offset.astype(float)
fileobj.close()

#
# Size of mhwBlock variable
#

matobj = io.loadmat(header + 'timeseries/avhrr-only-v2.ts.' + str(300).zfill(4) + '.mat')
mhws, clim = mhw.detect(t, matobj['sst_ts'][300,:])
mhwBlock = mhw.blockAverage(t, mhws)
years = mhwBlock['years_centre']
NB = len(years)

#
# initialize some variables
#

X = len(lon)
Y = len(lat)
i_which = range(0,X) #,10)
j_which = range(0,Y) #,10)
#i_which = range(0,X,4)
#j_which = range(0,Y,4)
DIM = (len(j_which), len(i_which))
N_ts = np.zeros((len(j_which), len(i_which), NB))
SST_mean_ts = np.zeros((len(j_which), len(i_which), NB))
SST_var_ts = np.zeros((len(j_which), len(i_which), NB))
SST_skew_ts = np.zeros((len(j_which), len(i_which), NB))
lon_map =  np.NaN*np.zeros(len(i_which))
lat_map =  np.NaN*np.zeros(len(j_which))

#
# loop through locations
#

icnt = 0
for i in i_which:
    print i, 'of', len(lon)-1
#   load SST
    matobj = io.loadmat(header + 'timeseries/avhrr-only-v2.ts.' + str(i+1).zfill(4) + '.mat')
    sst_ts = matobj['sst_ts']
    lon_map[icnt] = lon[i]
#   loop over j
    jcnt = 0
    for j in j_which:
        lat_map[jcnt] = lat[j]
        if np.logical_not(np.isfinite(sst_ts[j,:].sum())) + ((sst_ts[j,:]<-1).sum()>0): # check for land, ice
            jcnt += 1
            continue
#   Count number of MHWs of each length
        mhws, clim = mhw.detect(t, sst_ts[j,:])
        # SST mean, var, skew
        SST_mean_ij = np.zeros(years.shape)
        SST_var_ij = np.zeros(years.shape)
        SST_skew_ij = np.zeros(years.shape)
        for yr in range(len(years)):
            SST_mean_ij[yr] = np.mean(sst_ts[j,year==years[yr]])
            SST_var_ij[yr] = np.var((sst_ts[j,:] - clim['seas'])[year==years[yr]])
            SST_skew_ij[yr] = stats.skew((sst_ts[j,:] - clim['seas'])[year==years[yr]])
        #
        SST_mean_ts[jcnt,icnt,:] = SST_mean_ij
        SST_var_ts[jcnt,icnt,:] = SST_var_ij
        SST_skew_ts[jcnt,icnt,:] = SST_skew_ij
        # Up counts
        jcnt += 1
    icnt += 1
    # Save data so far
    outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/sst_meanVarSkew'
    np.savez(outfile, lon_map=lon_map, lat_map=lat_map, SST_mean_ts=SST_mean_ts, SST_var_ts=SST_var_ts, SST_skew_ts=SST_skew_ts, years=year)



