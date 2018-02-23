'''

  Software which uses monthly proxies from HadISST
  to estimate long-term trends in marine heatwaves

'''

# Load required modules

import numpy as np
from scipy import io
from scipy import linalg
from scipy import stats
from scipy import signal
from scipy import ndimage
from datetime import date
from netCDF4 import Dataset
import statsmodels.api as sm

from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm

import marineHeatWaves as mhw

#
# Generate proxies
#

# Years to analyse
#yearStart = 1871
yearStart = 1900
yearEnd = 2016

HadID = 'HadISST'
header_had = '/mnt/erebor/data/sst/HadSST/'
if HadID == 'HadISST':
    file_had = header_had + 'HadISST1/HadISST_sst.nc'
    year_had0 = 1870
elif HadID == 'HadSST3':
    file_had = header_had + 'HadSST3/HadSST.3.1.1.0.median.nc'
    year_had0 = 1850

fileobj = Dataset(file_had, mode='r')
sst_had = fileobj.variables['sst'][:].data
fillValue = fileobj.variables['sst']._FillValue
lon_had = fileobj.variables['longitude'][:]
lat_had = fileobj.variables['latitude'][:]
t_had = np.floor(date(year_had0,1,1).toordinal() + fileobj.variables['time'][:]).astype(int)
fileobj.close()

llon_had, llat_had = np.meshgrid(lon_had, lat_had)
scaling = np.cos(llat_had*np.pi/180)

sst_had[sst_had==fillValue] = np.nan
sst_had[sst_had<=-2] = np.nan
sst_had[sst_had>=35] = np.nan

T_had = len(t_had)
LON = len(lon_had)
LAT = len(lat_had)

year_had = np.zeros(T_had)
month_had = np.zeros(T_had)
for mth in range(T_had):
    year_had[mth] = date.fromordinal(t_had[mth].astype(int)).year
    month_had[mth] = date.fromordinal(t_had[mth].astype(int)).month

# Limit data to post-1900
years_proxies = range(yearStart, yearEnd+1)
tt = (year_had >= yearStart) * (year_had <= yearEnd)
sst_had = sst_had[tt]
year_had = year_had[tt]
month_had = month_had[tt]

# lat and lons of obs
lon_had_pos = lon_had.copy()
lon_had_pos[lon_had_pos<0] +=360.
res_had = np.diff(lon_had)[0]
years_had = np.array(years_proxies).copy()
avhrr = (years_had >= 1982) * (years_had <= yearEnd)

# Create landmask based on land
datamask = 1.-(np.sum(np.isnan(sst_had), axis=0) == sst_had.shape[0]).astype(int) # landmask
# Create landmask based on repeat seasonal cycle
repSeas = np.zeros((len(lat_had), len(lon_had)))
for i in range(12):
    repSeas += np.sum(np.diff(sst_had[i::12,:,:], axis=0)==0, axis=0)
# Combine the two landmasks
datamask[repSeas>10] = 0.

#
# AVHRR observations (daily)
#

pathroot = '/mnt/erebor/'
#pathroot = '/home/ecoliver/Desktop/'
#pathroot = '/bs/projects/geology/Oliver/'
header = pathroot+'data/sst/noaa_oi_v2/avhrr/'
file0 = header + '1982/avhrr-only-v2.19820101.nc'
t = np.arange(date(1982,1,1).toordinal(),date(yearEnd,12,31).toordinal()+1)
t_HadSST = np.arange(date(yearStart,1,1).toordinal(),date(yearEnd,12,31).toordinal()+1)
T = len(t)
year = np.nan*np.zeros((T))
month = np.nan*np.zeros((T))
day = np.nan*np.zeros((T))
for i in range(T):
    year[i] = date.fromordinal(t[i]).year
    month[i] = date.fromordinal(t[i]).month
    day[i] = date.fromordinal(t[i]).day

# lat and lons of obs
fileobj = Dataset(file0, mode='r')
lon = fileobj.variables['lon'][:]
lat = fileobj.variables['lat'][:]
res = np.diff(lon)[0]
fill_value = fileobj.variables['sst']._FillValue
scale = fileobj.variables['sst'].scale_factor
offset = fileobj.variables['sst'].add_offset
fileobj.close()

#
# initialize some variables
#

# Variable to store daily NOAA data on Hadley SST grid

LON = len(lon_had)
LAT = len(lat_had)
sst_noaa = np.nan*np.zeros((T, LAT, LON))

#
# loop through locations
#

for i in range(LON):
    print i+1, 'of', LON
#   load SST
    i1 = np.where(lon > lon_had_pos[i] - res_had/2.)[0][0]
    i2 = np.where(lon > lon_had_pos[i] + res_had/2. - res)[0][0]
    sst_ts = np.nan*np.zeros((len(lat), T, i2-i1+1))
    cnt = 0
    for i0 in range(i1, i2+1):
      #if i0 == 1338:
      #    continue
      matobj = io.loadmat(header + 'timeseries/avhrr-only-v2.ts.' + str(i0+1).zfill(4) + '.mat')
      sst_ts[:,:,cnt] = matobj['sst_ts']
      cnt += 1
#   loop over j
    j = 0
    for j in range(LAT):
        if lat_had[j] - res_had/2. < lat.min():
            j1 = 0
        else:
            j1 = np.where(lat > lat_had[j] - res_had/2.)[0][0]
        if lat_had[j] + res_had/2. > lat.max():
            j2 = len(lat)-1
        else:
            j2 = np.where(lat > lat_had[j] + res_had/2.)[0][0]
        sst_noaa[:,j,i] = np.nanmean(np.nanmean(sst_ts[j1:j2+1,:,:], axis=2), axis=0)

# Save data
outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_proxies_NOAA.' + HadID
np.savez(outfile, lon_had=lon_had, lat_had=lat_had, sst_noaa=sst_noaa)




