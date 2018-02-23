'''

  Software which uses the MHW definition
  of Hobday et al. (2015) applied to 
  select sst time series around the globe

'''

# Load required modules

import numpy as np
from scipy import io
from scipy import signal
from scipy import linalg
from scipy import stats
from scipy import ndimage
from datetime import date
import statsmodels.api as sm

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import mpl_toolkits.basemap as bm

import ecoliver as ecj

import marineHeatWaves as mhw

# Stations
locations = {}
locations['Pacific_Grove'] = [-1.*(121 + 54.2/60), 36 + 37.3/60]
locations['Scripps_Pier'] = [-1.*(117 + 15.5/60), 32 + 52.0/60]
locations['Newport_Beach'] = [-1.*(117 + 56.0/60), 33 + 36.0/60]
locations['Arendal'] = [8 + 47.0/60, 58 + 29.0/60]
locations['Port_Erin'] = [-1.*(4 + 46.1/60), 54 + 5.1/60]
locations['Race_Rocks'] = [-123.533, 48.298]
stations = locations.keys()

mhws = {}
years = {}
for station in stations:
    outfile = 'data/longTimeseries/mhwData_' + station + '.npz'
    data = np.load(outfile)
    years[station] = data['years']
    mhws[station] = data['mhwBlock'].item()

# Proxies
pathroot = '/home/oliver/'
dataSets = ['HadISST',            'ERSST',           'COBE', 'CERA20C', 'SODA']
Proxy = {}
lon_data = {}
lat_data = {}
llon_data = {}
llat_data = {}
landmask = {}
for dataSet in dataSets:
    print dataSet
    proxyfile = pathroot + 'data/MHWs/Trends/mhw_proxies_GLM.1900.2016.' + dataSet
    data = np.load(proxyfile+'.npz')
    lon_data[dataSet] = data['lon_data']
    lat_data[dataSet] = np.flipud(data['lat_data'])
    years_data = np.arange(1900, 2016+1)
    Proxy[dataSet] = {}
    Proxy[dataSet]['count'] = np.flipud(data['MHW_ts'].item()['count']['threshCount'])
    Proxy[dataSet]['duration'] = np.flipud(data['MHW_ts'].item()['duration']['maxAnom'])
    landmask[dataSet] = ~np.isnan(np.nanmean(Proxy[dataSet]['duration'], axis=2))
    llon_data[dataSet], llat_data[dataSet] = np.meshgrid(lon_data[dataSet], lat_data[dataSet])
    #ii, jj = np.meshgrid(np.arange(len(lon_data)), np.arange(len(lat_data)))

# Correlate station against proxies
mhwCorr = {}
mhwCorr_detrend = {}
for station in stations:
    mhwCorr[station] = {}
    mhwCorr_detrend[station] = {}
    # Temporal overlap
    tt = np.in1d(years_data, years[station])
    # Correlate
    for metric in ['count', 'duration']:
        mhwCorr[station][metric] = {}
        mhwCorr_detrend[station][metric] = {}
        for dataSet in dataSets:
            mhwCorr[station][metric][dataSet] = np.zeros(landmask[dataSet].shape)
            mhwCorr_detrend[station][metric][dataSet] = np.zeros(landmask[dataSet].shape)
            for i in range(len(lon_data[dataSet])):
                print station, metric, dataSet, i+1, len(lon_data[dataSet])
                for j in range(len(lat_data[dataSet])):
                    valid = ~np.isnan(mhws[station][metric]) * ~np.isnan(Proxy[dataSet][metric][j,i,tt])
                    if valid.sum() >= 2:
                        mhwCorr[station][metric][dataSet][j,i] = np.corrcoef(mhws[station][metric][valid], Proxy[dataSet][metric][j,i,tt][valid])[0,1]
                        mhwCorr_detrend[station][metric][dataSet][j,i] = np.corrcoef(signal.detrend(mhws[station][metric][valid]), signal.detrend(Proxy[dataSet][metric][j,i,tt][valid]))[0,1]
                    else:
                        mhwCorr[station][metric][dataSet][j,i] = np.nan
                        mhwCorr_detrend[station][metric][dataSet][j,i] = np.nan

# Average correlations across datasets, regridding to shared grid
# Average correlations using Fisher z-transformation" Corey, D. M., Dunlap, W. P., & Burke, M. J. (1998). Averaging correlations: Expected values and bias in combined Pearson rs and Fisher's z transformations. The Journal of general psychology, 125(3), 245-261.
dl = 2.
lon_agr = np.arange(-180, 180, dl)
lat_agr = np.arange(90, -90, -dl)
llon_agr, llat_agr = np.meshgrid(lon_agr, lat_agr)
mhwCorr_avg = {}
mhwCorr_avg_detrend = {}
for station in stations:
    mhwCorr_avg[station] = {}
    mhwCorr_avg_detrend[station] = {}
    for metric in ['count', 'duration']:
        mhwCorr_avg[station][metric] = np.zeros((len(lat_agr), len(lon_agr)))
        mhwCorr_avg_detrend[station][metric] = np.zeros((len(lat_agr), len(lon_agr)))
        for dataSet in dataSets:
            for i in range(len(lon_agr)):
                print station, metric, dataSet, i+1, len(lon_agr)
                for j in range(len(lat_agr)):
                    ii = (llon_data[dataSet][0,:] > lon_agr[i]) * (llon_data[dataSet][0,:] <= (lon_agr[i]+dl)) + (llon_data[dataSet][0,:]+360 > lon_agr[i]) * (llon_data[dataSet][0,:]+360 <= (lon_agr[i]+dl)) + (llon_data[dataSet][0,:]-360 > lon_agr[i]) * (llon_data[dataSet][0,:]-360 <= (lon_agr[i]+dl))
                    jj = (llat_data[dataSet][:,0] < lat_agr[j]) * (llat_data[dataSet][:,0] >= (lat_agr[j]-dl))
                    r = mhwCorr[station][metric][dataSet][jj,:][:,ii].flatten()
                    z = 0.5*np.log((1 + r)/(1 - r))
                    mhwCorr_avg[station][metric][j,i] += np.nanmean(z)
                    r = mhwCorr_detrend[station][metric][dataSet][jj,:][:,ii].flatten()
                    z = 0.5*np.log((1 + r)/(1 - r))
                    mhwCorr_avg_detrend[station][metric][j,i] += np.nanmean(z)
        # Transform back from z to r, averaging along the way
        mhwCorr_avg[station][metric] = (np.exp(2*mhwCorr_avg[station][metric]/len(dataSets)) - 1)/(np.exp(2*mhwCorr_avg[station][metric]/len(dataSets)) +1)
        mhwCorr_avg_detrend[station][metric] = (np.exp(2*mhwCorr_avg_detrend[station][metric]/len(dataSets)) - 1)/(np.exp(2*mhwCorr_avg_detrend[station][metric]/len(dataSets)) +1)

# Plots
sites = ['Pacific_Grove', 'Scripps_Pier', 'Newport_Beach', 'Arendal', 'Port_Erin', 'Race_Rocks']

bg_col = '0.6'
cont_col = '0.0'
domain = [-65, -180, 70, 180]
domain_draw = [-60, -180, 60, 180]
dlat = 30
dlon = 90
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
lonproj, latproj = proj(llon_agr, llat_agr)

cmap1 = plt.get_cmap('rainbow')
cmap1 = mpl.colors.ListedColormap(cmap1(np.floor(np.linspace(0, 255, 10)).astype(int)))
#cmap1 = mpl.colors.ListedColormap(cmap1(np.floor(np.linspace(0, 255, 5)).astype(int)))
cmap1.set_bad(color = 'k', alpha = 0.)

fig = plt.figure(figsize=(8,13))
plt.clf()
cnt = 0
AX = {}
for station in sites:
    AX[station] = {}
    for metric in ['count', 'duration']:
        cnt += 1
        AX[station][metric] = plt.subplot(len(sites), 2, cnt, axisbg=bg_col)
        proj.fillcontinents(color=cont_col, lake_color=cont_col, ax=None, zorder=None, alpha=None)
        if metric == 'count':
            proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[6,900])
        else:
            proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False], dashes=[6,900])
        if station == sites[-1]:
            proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[6,900])
        else:
            proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[6,900])
        H = plt.pcolor(lonproj, latproj, np.ma.masked_invalid(mhwCorr_avg_detrend[station][metric]), vmin=0, vmax=1, cmap=cmap1)
        plt.clim(0, 1)
        lonprojSite, latprojSite = proj(locations[station][0], locations[station][1])
        plt.plot(lonprojSite, latprojSite, 'ro', markersize=3, markeredgewidth=0)
        plt.title(station + ' ' + metric)

AXPOS = AX['Newport_Beach']['duration'].get_position()
CAX = fig.add_axes([AXPOS.x1+0.015, AXPOS.y0, 0.01, AXPOS.y1-AXPOS.y0])
HB = plt.colorbar(H, CAX, orientation='vertical')
HB.set_label('Correlation')

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/long_timeseries_correlations.png', bbox_inches='tight', dpi=300)
# plt.savefig('long_timeseries_correlations.png', bbox_inches='tight', dpi=300)








