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
from Scientific.IO import NetCDF

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import mpl_toolkits.basemap as bm

import deseason as ds
import ecoliver as ecj

import marineHeatWaves as mhw

# Some basic parameters

coldSpells = False # Detect coldspells instead of heatwaves
seasonalMeans = True
col_clim = '0.25'
col_thresh = 'g-'
if coldSpells:
    mhwname = 'MCS'
    mhwfullname = 'coldspell'
    col_evMax = (0, 102./255, 204./255)
    col_ev = (153./255, 204./255, 1)
    col_bar = (0.5, 0.5, 1)
    cmap_i = plt.cm.YlGnBu_r
else:
    mhwname = 'MHW'
    mhwfullname = 'heatwave'
    col_evMax = 'r'
    col_ev = (1, 0.6, 0.5)
    col_bar = (1, 0.5, 0.5)
    cmap_i = plt.cm.hot_r

#
# observations
#

pathroot = '/mnt/erebor/'
#pathroot = '/home/ecoliver/Desktop/'
#pathroot = '/media/ecoliver/DataOne/'
header = pathroot+'data/sst/noaa_oi_v2/avhrr/'
file0 = header + '1982/avhrr-only-v2.19820101.nc'

#
# lat and lons of obs
#

fileobj = NetCDF.NetCDFFile(file0, mode='r')
lon = fileobj.variables['lon'].getValue().astype(float)
lat = fileobj.variables['lat'].getValue().astype(float)
fill_value = fileobj.variables['sst']._FillValue.astype(float)
scale = fileobj.variables['sst'].scale_factor.astype(float)
offset = fileobj.variables['sst'].add_offset.astype(float)
fileobj.close()

#
# Load data at locations of interest
#

locations = {}
# Plymouth Sound (for Dan)
locations['lon'] = [360 - 4 - 13./16]
locations['lat'] = [50 + 15./60   - 0.25]
locations['lon_map'] = [360 - 4 - 13./16]
locations['lat_map'] = [50 + 15./60]
locations['name'] = ['Plym']
sw_box = 0
# SE Tas (for Craig Mundy)
locations['lon'] = [147.21]
locations['lat'] = [-43.69]
locations['lon_map'] = [147.21]
locations['lat_map'] = [-43.69]
locations['name'] = ['SETas']
sw_box = 0
# SA (for Craig Mundy)
locations['lon'] = [139.49]
locations['lat'] = [-37.03]
locations['lon_map'] = [locations['lon'][0]*1.]
locations['lat_map'] = [locations['lat'][0]*1. - 1.]
locations['name'] = ['SA']
sw_box = 0
# WA
locations['lon'] = [112.5]
locations['lat'] = [-29.5]
locations['lon_map'] = [107.5]
locations['lat_map'] = [-29]
locations['name'] = ['WA']
sw_box = 0
# NW Atl
locations['lon'] = [360-60.]
locations['lat'] = [40.]
locations['lon'] = [360-63.5]
locations['lat'] = [42.]
locations['lon_map'] = [360-67.]
locations['lat_map'] = [39.9]
locations['name'] = ['NW_Atl']
sw_box = 0
# Med
locations['lon'] = [9.1144]
locations['lat'] = [43.6271]
locations['lon_map'] = [locations['lon'][0]*1.]
locations['lat_map'] = [locations['lat'][0]*1. - 2.]
locations['name'] = ['Med']
sw_box = 0
# S. Africa
#locations['lon'] = [31.50788]
#locations['lat'] = [-30.012958]
#locations['lon_map'] = [locations['lon'][0]*1. + 2]
#locations['lat_map'] = [locations['lat'][0]*1. - 2.]
#locations['name'] = ['S_Afr']
#sw_box = 0
# Cockburn Sound WA
#locations['lon'] = [115.+(36.+23.86/60)/60, 115.+(45.+38.38/60)/60]
#locations['lat'] = [-(32.+(25.+26.09)/60), -(32.+(7.+18.89/60)/60)]
#locations['lon_map'] = [np.mean(locations['lon'][0]) - 2]
#locations['lat_map'] = [np.mean(locations['lat'][0])]
#locations['name'] = ['Cockburn Snd WA']
#sw_box = 1
# Santa Barbara CA
#locations['lon'] = [360-(120.+(34+8.13/60)/60), 360-(119.+(13+17.0/60)/60)]
#locations['lat'] = [33.+(44+24.55/60)/60, 34.+(23+22.14/60)/60]
#locations['lon_map'] = [np.mean(locations['lon'][0]) - 2]
#locations['lat_map'] = [np.mean(locations['lat'][0])]
#locations['name'] = ['Santa Barbara Basin CA']
#sw_box = 1
# Caribbean
locations['lon'] = [360-(82.+(49+11.30/60)/60), 360-(61.+(53+35.64/60)/60)]
locations['lat'] = [10+(13+4.58/60)/60, 21+(33+52.80/60)/60]
locations['lon_map'] = [np.mean(locations['lon'][0])]
locations['lat_map'] = [np.mean(locations['lat'][0])]
locations['name'] = ['Caribbean']
sw_box = 1
# California
locations['lon'] = [360-(117.+(11+39.82/60)/60), 360-(116.+(21+2.67/60)/60)]
locations['lat'] = [29.+(11+50.88/60)/60, 31.+( 3+54.46/60)/60]
locations['lon_map'] = [np.mean(locations['lon'][0]) - 2]
locations['lat_map'] = [np.mean(locations['lat'][0])]
locations['name'] = ['California']
sw_box = 1
# Lundy Island
locations['lon'] = [360-(4.+(51.+37.23/60)/60), 360-(4.+(25.+46.02/60)/60)]
locations['lat'] = [51+(4.+49.07/60)/60, 51+(20 + 57.51/60)/60]
locations['lon_map'] = [np.mean(locations['lon'][0])]
locations['lat_map'] = [np.mean(locations['lat'][0])]
locations['name'] = ['Lundy Island']
sw_box = 1

# Time and date vectors
t, dates, T, year, month, day, doy = ecj.timevector([1982,1,1], [2014,12,31])
sst = np.zeros((T))

# Load data
if sw_box:
    i1 = np.where(lon > locations['lon'][0])[0][0] - 1
    i2 = np.where(lon > locations['lon'][1])[0][0]
    j1 = np.where(lat > locations['lat'][0])[0][0] - 1
    j2 = np.where(lat > locations['lat'][1])[0][0]
    cnt = 0
    for i in range(i1,i2+1):
        matobj = io.loadmat(header + 'timeseries/avhrr-only-v2.ts.' + str(i+1).zfill(4) + '.mat')
        for j in range(j1,j2+1):
            if ~np.isnan(matobj['sst_ts'][j,:].sum()):
                cnt += 1
                sst += matobj['sst_ts'][j,:]
    sst /= cnt
else:
    i = np.where(lon > locations['lon'])[0][0]
    j = np.where(lat > locations['lat'])[0][0]
    matobj = io.loadmat(header + 'timeseries/avhrr-only-v2.ts.' + str(i+1).zfill(4) + '.mat')
    sst = matobj['sst_ts'][j,:]

# For up-to-date point loc recent data
#t, dates, T, year, month, day, do = ecj.timevector([1982,1,1], [2015,6,7])
#matobj = io.loadmat(header + 'timeseries/sst_ts_NWAtl.mat')
# t, dates, T, year, month, day, do = ecj.timevector([1982,1,1], [2015,7,22])
# matobj = io.loadmat(header + 'timeseries/sst_ts_Med.mat')
# sst = np.NaN*np.zeros((N_locs,T))
# sst[0,:] = matobj['sst_ts'][0,:]

#
# Apply Marine Heat Wave definition
#

n = 0
mhws, clim = mhw.detect(t, sst, coldSpells=coldSpells)
mhwBlock = mhw.blockAverage(t, mhws, temp=sst)

def meanTrend_OLS(mhwBlock, alpha=0.05):
    # Initialize mean and trend dictionaries
    mean = {}
    trend = {}
    dtrend = {}
#
    # Construct matrix of predictors, first column is all ones to estimate the mean,
    # second column is the time vector, equal to zero at mid-point.
    t = mhwBlock['years_centre']
    X = np.array([np.ones(t.shape), t-t.mean()]).T
#
    # Loop over all keys in mhwBlock
    for key in mhwBlock.keys():
        # Skip time-vector keys of mhwBlock
        if (key == 'years_centre') + (key == 'years_end') + (key == 'years_start'):
            continue
#
        # Predictand (MHW property of interest)
        y = mhwBlock[key]
        valid = ~np.isnan(y) # non-NaN indices
#
        # Perform linear regression over valid indices
        if np.sum(~np.isnan(y)) > 0: # If at least one non-NaN value
            beta = linalg.lstsq(X[valid,:], y[valid])[0]
        else:
            beta = [np.nan, np.nan]
#
        # Insert regression coefficients into mean and trend dictionaries
        mean[key] = beta[0]
        trend[key] = beta[1]
#
        # Confidence limits on trend
        yhat = np.sum(beta*X, axis=1)
        t_stat = stats.t.isf(alpha/2, len(t[valid])-2)
        s = np.sqrt(np.sum((y[valid] - yhat[valid])**2) / (len(t[valid])-2))
        Sxx = np.sum(X[valid,1]**2) - (np.sum(X[valid,1])**2)/len(t[valid]) # np.var(X, axis=1)[1]
        dbeta1 = t_stat * s / np.sqrt(Sxx)
        dtrend[key] = dbeta1
#
    return mean, trend, dtrend

def meanTrend_TS(mhwBlock, alpha=0.05):
    # Initialize mean and trend dictionaries
    mean = {}
    trend = {}
    dtrend = {}
#
    # Construct matrix of predictors, first column is all ones to estimate the mean,
    # second column is the time vector, equal to zero at mid-point.
    t = mhwBlock['years_centre']
    X = t-t.mean()
#
    # Loop over all keys in mhwBlock
    for key in mhwBlock.keys():
        # Skip time-vector keys of mhwBlock
        if (key == 'years_centre') + (key == 'years_end') + (key == 'years_start'):
            continue
#
        # Predictand (MHW property of interest)
        y = mhwBlock[key]
        valid = ~np.isnan(y) # non-NaN indices
#
        # Perform linear regression over valid indices
        if np.sum(~np.isnan(y)) > 0: # If at least one non-NaN value
            slope, y0, beta_lr, beta_up = stats.mstats.theilslopes(y[valid], X[valid], alpha=1-alpha)
            beta = np.array([y0, slope])
        else:
            beta_lr, beta_up = [np.nan, np.nan]
            beta = [np.nan, np.nan]
#
        # Insert regression coefficients into mean and trend dictionaries
        mean[key] = beta[0]
        trend[key] = beta[1]
#
        dtrend[key] = [beta_lr, beta_up]
#
    return mean, trend, dtrend

mean_OLS, trend_OLS, dtrend_OLS = meanTrend_OLS(mhwBlock)
mean_TS, trend_TS, dtrend_TS = meanTrend_TS(mhwBlock)

#
# Plot annual averages
#

years = mhwBlock['years_centre']
plt.figure(figsize=(13,7))
plt.clf()
plt.subplot(2,2,2)
plt.plot(years, mhwBlock['count'], 'k-')
plt.plot(years, mhwBlock['count'], 'ko')
plt.plot(years, mean_OLS['count'] + trend_OLS['count']*(years-years.mean()), 'c-')
plt.plot(1980.5, mean_OLS['count'], 'co')
plt.plot(years, mean_TS['count'] + trend_TS['count']*(years-years.mean()), 'g-')
plt.plot(1980.5, mean_TS['count'], 'go')
plt.title('Frequency (trend = ' + '{:.2}'.format(10*trend_OLS['count']) + ', ' + '{:.2}'.format(10*trend_TS['count']) + ' per decade)')
if np.abs(trend_OLS['count']) - dtrend_OLS['count'] < 0:
     plt.plot(1981.5, mean_OLS['count'], 'kx')
if (trend_TS['count'] > dtrend_TS['count'][0]) * (trend_TS['count'] < dtrend_TS['count'][1]):
     plt.plot(1981.5, mean_TS['count'], 'kx')
plt.ylabel('[count per year]')
plt.grid()
plt.subplot(2,2,1)
plt.plot(years, mhwBlock['duration'], 'k-')
plt.plot(years, mhwBlock['duration'], 'ko')
plt.plot(years, mean_OLS['duration'] + trend_OLS['duration']*(years-years.mean()), 'c-')
plt.plot(1980.5, mean_OLS['duration'], 'co')
plt.plot(years, mean_TS['duration'] + trend_TS['duration']*(years-years.mean()), 'g-')
plt.plot(1980.5, mean_TS['duration'], 'go')
plt.title('Duration (trend = ' + '{:.2}'.format(10*trend_OLS['duration']) + ', ' + '{:.2}'.format(10*trend_TS['duration']) + ' per decade)')
if np.abs(trend_OLS['duration']) - dtrend_OLS['duration'] < 0:
     plt.plot(1981.5, mean_OLS['duration'], 'kx')
if (trend_TS['duration'] > dtrend_TS['duration'][0]) * (trend_TS['duration'] < dtrend_TS['duration'][1]):
     plt.plot(1981.5, mean_TS['duration'], 'kx')
plt.ylabel('[days]')
plt.grid()
plt.subplot(2,2,4)
plt.plot(years, mhwBlock['intensity_mean'], 'k-')
plt.plot(years, mhwBlock['intensity_mean'], 'ko')
plt.plot(years, mean_OLS['intensity_mean'] + trend_OLS['intensity_mean']*(years-years.mean()), 'c-')
plt.plot(1980.5, mean_OLS['intensity_mean'], 'co')
plt.plot(years, mean_TS['intensity_mean'] + trend_TS['intensity_mean']*(years-years.mean()), 'g-')
plt.plot(1980.5, mean_TS['intensity_mean'], 'go')
plt.title('Intensity (trend = ' + '{:.2}'.format(10*trend_OLS['intensity_mean']) + ', ' + '{:.2}'.format(10*trend_TS['intensity_mean']) + ' per decade)')
if np.abs(trend_OLS['intensity_mean']) - dtrend_OLS['intensity_mean'] < 0:
     plt.plot(1981.5, mean_OLS['intensity_mean'], 'kx')
if (trend_TS['intensity_mean'] > dtrend_TS['intensity_mean'][0]) * (trend_TS['intensity_mean'] < dtrend_TS['intensity_mean'][1]):
     plt.plot(1981.5, mean_TS['intensity_mean'], 'kx')
plt.ylabel(r'[$^\circ$C]')
plt.grid()
plt.subplot(2,2,3)
plt.plot(years, mhwBlock['intensity_cumulative'], 'k-')
plt.plot(years, mhwBlock['intensity_cumulative'], 'ko')
plt.plot(years, mean_OLS['intensity_cumulative'] + trend_OLS['intensity_cumulative']*(years-years.mean()), 'c-')
plt.plot(1980.5, mean_OLS['intensity_cumulative'], 'co')
plt.plot(years, mean_TS['intensity_cumulative'] + trend_TS['intensity_cumulative']*(years-years.mean()), 'g-')
plt.plot(1980.5, mean_TS['intensity_cumulative'], 'go')
plt.title('Cum. Intens. (trend = ' + '{:.2}'.format(10*trend_OLS['intensity_cumulative']) + ', ' + '{:.2}'.format(10*trend_TS['intensity_cumulative']) + ' per decade)')
if np.abs(trend_OLS['intensity_cumulative']) - dtrend_OLS['intensity_cumulative'] < 0:
     plt.plot(1981.5, mean_OLS['intensity_cumulative'], 'kx')
if (trend_TS['intensity_cumulative'] > dtrend_TS['intensity_cumulative'][0]) * (trend_TS['intensity_cumulative'] < dtrend_TS['intensity_cumulative'][1]):
     plt.plot(1981.5, mean_TS['intensity_cumulative'], 'kx')
plt.ylabel(r'[$^\circ$C$\times$days]')
plt.grid()

plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/OLS_test/' + mhwname + '_annualAverages_meanTrend_' + locations['name'][0] + '.png', bbox_inches='tight', pad_inches=0.5, dpi=150)
