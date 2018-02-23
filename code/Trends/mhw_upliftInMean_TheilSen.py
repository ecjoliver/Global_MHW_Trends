'''

  Software which uses the MHW definition
  of Hobday et al. (2015) applied to
  AR1 statistical simulated SST

'''

# Load required modules

import numpy as np
import scipy as sp
from scipy import io
from scipy import signal
from scipy import linalg
from scipy import stats
from datetime import date
from netCDF4 import Dataset

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import mpl_toolkits.basemap as bm

import deseason as ds
import ecoliver as ecj

import marineHeatWaves as mhw
import trendSimAR1

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


# Load SST and fit AR1 model

pathroot = '/mnt/erebor/'
header = pathroot+'data/sst/noaa_oi_v2/avhrr/'
file0 = header + '1982/avhrr-only-v2.19820101.nc'
fileobj = Dataset(file0, mode='r')
lon = fileobj.variables['lon'][:]
lat = fileobj.variables['lat'][:]
fileobj.close()

locations = {}
locations['lon'] = [112.5, 9, 360-67, 148.5, 360-144]
locations['lat'] = [-29.5, 43.5, 43, -42.5, 44]
locations['lon_map'] = [107.5, 12.3, 360-67, np.Inf, np.Inf]
locations['lat_map'] = [-29., 39.9, 43, np.Inf, np.Inf]
locations['name'] = ['WA', 'Med', 'NW_Atl', 'E_Tas', 'Blob']
N_locs = len(locations['lon'])

sst = {}
for n in range(N_locs):
    site = locations['name'][n]
    #sst[site] = np.NaN*np.zeros((T,))
    print locations['name'][n]
    i = np.where(lon > locations['lon'][n])[0][0]
    j = np.where(lat > locations['lat'][n])[0][0]
    matobj = io.loadmat(header + 'timeseries/avhrr-only-v2.ts.' + str(i+1).zfill(4) + '.mat')
    sst[site] = matobj['sst_ts'][j,:]


#site = 'E_Tas'
site = 'WA'
site = 'Med'
site = 'NW_Atl'
site = 'Blob'

t_obs, dates_obs, T_obs, year_obs, month_obs, day_obs, doy_obs = ecj.timevector([1982,1,1], [2016,12,31])
sst_obs = sst[site] #np.loadtxt('data/sst_' + site + '.csv', delimiter=',')
mhws_obs, clim_obs = mhw.detect(t_obs, sst_obs)
mhwBlock_obs = mhw.blockAverage(t_obs, mhws_obs)
mean_obs, trend_obs, dtrend_obs = meanTrend_TS(mhwBlock_obs) #mhw.meanTrend(mhwBlock_obs)

# SST trend

years = mhwBlock_obs['years_centre']
SST_block = np.zeros(years.shape)
for yr in range(len(years)):
    SST_block[yr] = np.mean(sst_obs[year_obs==years[yr]])
X = np.array([np.ones(years.shape), years-years.mean()]).T
beta = linalg.lstsq(X, SST_block)[0]
sst_trend_obs = beta[1]*10

# Loop over SST trend values and simulate MHW property trends

N_ens = 1000 # Number of ensembles, per trend value
N_tr = 10 # Number of trend values to consider
trend_max = 1. # Maximum trend (1 degree per decade)
trend_range = np.linspace(0., 1., N_tr)*trend_max

keys = ['count', 'intensity_max_max', 'duration']
units = ['events', '$^\circ$C', 'days']
N_keys = len(keys)
trends = {}
for key in keys:
    trends[key] = np.zeros((N_ens,N_tr))

for i_tr in range(N_tr):
    print i_tr+1, 'of', N_tr
    tau, sig_eps, trends0, means = trendSimAR1.simulate(t_obs, sst_obs, clim_obs['seas'], trend_range[i_tr], N_ens)
    for key in keys:
        trends[key][:,i_tr] = trends0[key]

# Analyse results

plt.figure(figsize=(18,5))

i = 0
for key in keys:
    plt.subplot(1, N_keys, i+1)
    plt.plot(trend_range, 10*trends[key].T, '-', color=(0.5,0.5,0.5))
    plt.plot(trend_range, 10*np.mean(trends[key], axis=0), 'k-o', linewidth=2)
    plt.plot(trend_range, 10*np.percentile(trends[key], 2.5, axis=0), 'b-o', linewidth=2)
    plt.plot(trend_range, 10*np.percentile(trends[key], 97.5, axis=0), 'r-o', linewidth=2)
    plt.plot(sst_trend_obs, 10*trend_obs[key], 'wo', markersize=10, markeredgewidth=2)
    plt.grid()
    plt.xlabel(r'SST trend [$^\circ$C decade$^{-1}$]')
    plt.ylabel(r'MHW ' + key + ' trend [' + units[i] + ' decade$^{-1}$]')
    plt.title(site)
    i += 1

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/trend_simulation_TheilSen_' + site + '_orig.pdf', bbox_inches='tight', pad_inches=0.5)

# Check for excess trend

print 'Results for ' + site
print 'AR1 Time-scale: ' + str(tau)
print 'AR1 noise std. dev.: ' + str(sig_eps)
for key in keys:
    trend_pred = np.interp(sst_trend_obs, trend_range, 10*np.mean(trends[key], axis=0))
    trend_pred_95 = np.interp(sst_trend_obs, trend_range, 10*np.percentile(trends[key], 97.5, axis=0))
    trend_pred_5 = np.interp(sst_trend_obs, trend_range, 10*np.percentile(trends[key], 2.5, axis=0))
    trend_actual  =  trend_obs[key]*10
    print key, trend_actual, trend_pred, trend_pred_5, trend_pred_95
    print 'Excess trend in ' + key + ': ' + str(trend_actual > trend_pred) + ' (mean)'
    print 'Excess trend in ' + key + ': ' + str((trend_actual > trend_pred_95) + (trend_actual < trend_pred_5)) + ' (5% sign)'

