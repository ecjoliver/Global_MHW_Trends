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
import ecoliver as ecj

# Switch to perform uncertainty sampling using HadSST3 uncertainties
sw_sig = False
Nsig = 200
# Switch to remove effects of climate modes
sw_nomodes = False
which_modes = ['MEI', 'PDO', 'AMO']

# Switch to choose which Hadley Centre SST data to use
#sw_data = 'HadSST3'
#sw_data = 'HadISST'
#sw_data = 'COBE'
#sw_data = 'Kaplan'
#sw_data = 'ERSST'
#sw_data = 'CERA20C'
sw_data = 'SODA'
print sw_data

if sw_data != 'HadSST3':
    sw_sig = False # Force this case...
# Switch to perform AR1 calculations
#sw_AR1 = False

# Years to analyse
#yearStart = 1871
yearStart = 1900
yearEnd = 2016

#
# Generate proxies
#

header_data = '/mnt/erebor/data/sst/'
header_data = '/home/oliver/data/sst/'

header_out = '/home/ecoliver/Desktop/data/MHWs/Trends/'
header_out = '/home/oliver/data/MHW/Trends/'

# Some meta data
if sw_data == 'HadISST':
    file_data = header_data + 'HadSST/HadISST1/HadISST_sst.nc'
    year_data0 = 1870 # ordinal for time vector
    dataEnd = yearEnd
elif sw_data == 'HadSST3':
    file_data = header_data + 'HadSST/HadSST3/HadSST.3.1.1.0.median.nc'
    year_data0 = 1850 # ordinal for time vector
    dataEnd = yearEnd
    if sw_sig:
        file_sig = header_data + 'HadSST/HadSST3/HadSST.3.1.1.0.measurement_and_sampling_uncertainty.nc'
elif sw_data == 'COBE':
    file_data = header_data + 'COBE/COBE2/sst.mon.mean.nc'
    year_data0 = 1891 # ordinal for time vector
    dataEnd = yearEnd
elif sw_data == 'Kaplan':
    file_data = header_data + 'Kaplan/Kaplan_Extended_SST_v2/sst.mon.anom.nc'
    year_data0 = 1800 # ordinal for time vector
    dataEnd = yearEnd
elif sw_data == 'ERSST':
    file_data = header_data + 'ERSST/v5/ersst.v5.'
    dataEnd = yearEnd
elif sw_data == 'CERA20C':
    file_data = header_data + '../ERA/CERA-20C/CERA20C_SST_'
    dataStart = 1901
    dataEnd = 2010
elif sw_data == 'SODA':
    file_data = header_data + '../soda/SODAsi.3_'
    dataEnd = 2013

# Load in actual data
if (sw_data == 'HadISST') + (sw_data == 'HadSST3'):
    fileobj = Dataset(file_data, mode='r')
    sst_data = fileobj.variables['sst'][:].data # time x lat x lon
    fillValue = fileobj.variables['sst']._FillValue
    lon_data = fileobj.variables['longitude'][:]
    lat_data = fileobj.variables['latitude'][:] # from +90 -> -90
    t_data = np.floor(date(year_data0,1,1).toordinal() + fileobj.variables['time'][:]).astype(int)
    fileobj.close()
    if sw_sig:
        fileobj = Dataset(file_sig, mode='r')
        sig_data = fileobj.variables['uncertainty'][:].data
        fillValue_sig = fileobj.variables['uncertainty']._FillValue
        fileobj.close()
        # flip sig to be in line with sst, and only keep values over same time range as sst
        sig_data = sig_data[:len(t_data),::-1,:]
    sst_data[sst_data==fillValue] = np.nan
    sst_data[sst_data<=-2] = np.nan
    sst_data[sst_data>=35] = np.nan
    if sw_sig:
        sig_data[sig_data==fillValue] = np.nan
elif sw_data == 'COBE':
    fileobj = Dataset(file_data, mode='r')
    sst_data = fileobj.variables['sst'][:].data # time x lat x lon
    fillValue = sst_data.max()
    lon_data = fileobj.variables['lon'][:]
    lat_data = fileobj.variables['lat'][:] # from +90 -> -90
    t_data = np.floor(date(year_data0,1,1).toordinal() + fileobj.variables['time'][:]).astype(int)
    fileobj.close()
    sst_data[sst_data==fillValue] = np.nan
    sst_data[sst_data<=-2] = np.nan
    sst_data[sst_data>=35] = np.nan
elif sw_data == 'Kaplan':
    fileobj = Dataset(file_data, mode='r')
    sst_data = fileobj.variables['sst'][:].data # time x lat x lon
    fillValue = fileobj.variables['sst'].missing_value
    lon_data = fileobj.variables['lon'][:]
    lat_data = fileobj.variables['lat'][:] # from -90 -> +90
    t_data = np.floor(date(year_data0,1,1).toordinal() + fileobj.variables['time'][:]).astype(int)
    fileobj.close()
    sst_data[sst_data==fillValue] = np.nan
    sst_data[sst_data<=-2] = np.nan
    sst_data[sst_data>=35] = np.nan
    lat_data = np.flipud(lat_data) # flip order of lat coords
    sst_data = sst_data[:,::-1,:]
elif sw_data == 'ERSST':
    fileobj = Dataset(file_data + '190001.nc', mode='r')
    lon_data = fileobj.variables['lon'][:]
    lat_data = fileobj.variables['lat'][:] # from -90 -> +90
    fillValue = fileobj.variables['sst']._FillValue
    fileobj.close()
    sst_data = np.nan*np.zeros(((yearEnd-yearStart+1)*12, len(lat_data), len(lon_data))) # time x lat x lon
    t_data = np.zeros(((yearEnd-yearStart+1)*12,))
    dates_data = []
    cnt = 0
    for yr in range(yearStart, yearEnd+1):
        for mth in range(1, 12+1):
            fileobj = Dataset(file_data + str(yr) + str(mth).zfill(2) + '.nc', mode='r')
            sst_data[cnt,:,:] = fileobj.variables['sst'][0,0,:,:].data
            fileobj.close()
            dates_data.append(date(yr,mth,1))
            t_data[cnt] = dates_data[cnt].toordinal()
            cnt += 1
    sst_data[sst_data==fillValue] = np.nan
    sst_data[sst_data<=-2] = np.nan
    sst_data[sst_data>=35] = np.nan
    lat_data = np.flipud(lat_data) # flip order of lat coords
    sst_data = sst_data[:,::-1,:]
elif sw_data == 'CERA20C':
    fileobj = Dataset(file_data + '1901.nc', mode='r')
    lon_data = fileobj.variables['longitude'][:]
    lat_data = fileobj.variables['latitude'][:] # from +90 -> -90
    fillValue = fileobj.variables['sst']._FillValue
    fileobj.close()
    # landsea mask
    fileobj = Dataset(file_data + 'LandSeaMask.nc', mode='r')
    lsm = fileobj.variables['lsm'][0,0,:,:]
    fileobj.close()
    mask = 1. - lsm
    mask[mask == 0.] = np.nan
    mask = signal.convolve2d(mask, np.ones((3,3)), mode='same')
    mask[~np.isnan(mask)] = 1.
    sst_data = np.nan*np.zeros(((yearEnd-yearStart+1)*12, len(lat_data), len(lon_data))) # time x lat x lon
    t_data = np.zeros(((yearEnd-yearStart+1)*12,))
    dates_data = []
    cnt = 0
    for yr in range(yearStart, yearEnd+1):
        if (yr < dataStart) + (yr > dataEnd):
            for mth in range(1, 12+1):
                dates_data.append(date(yr,mth,1))
                t_data[cnt] = dates_data[cnt].toordinal()
                cnt += 1
        else:
            fileobj = Dataset(file_data + str(yr) + '.nc', mode='r')
            sst_data0 = np.mean(fileobj.variables['sst'][:], axis=1) - 273.15
            for mth in range(1, 12+1):
                dates_data.append(date(yr,mth,1))
                t_data[cnt] = dates_data[cnt].toordinal()
                sst_data[cnt,:,:] = sst_data0[mth-1,:,:]*mask
                cnt += 1
            fileobj.close()
    sst_data[sst_data==fillValue] = np.nan
    sst_data[sst_data<=-2] = np.nan
    sst_data[sst_data>=35] = np.nan
elif sw_data == 'SODA':
    fileobj = Dataset(file_data + '190001.cdf', mode='r')
    lon_data = fileobj.variables['LON'][:]
    lat_data = fileobj.variables['LAT'][:] # from -90 -> +90
    fillValue = fileobj.variables['TEMP']._FillValue
    fileobj.close()
    sst_data = np.nan*np.zeros(((yearEnd-yearStart+1)*12, len(lat_data), len(lon_data))) # time x lat x lon
    t_data = np.zeros(((yearEnd-yearStart+1)*12,))
    dates_data = []
    cnt = 0
    for yr in range(yearStart, yearEnd+1):
        for mth in range(1, 12+1):
            if yr <= dataEnd:
                fileobj = Dataset(file_data + str(yr) + str(mth).zfill(2) + '.cdf', mode='r')
                sst_data[cnt,:,:] = fileobj.variables['TEMP'][0,:,:].data
                fileobj.close()
            dates_data.append(date(yr,mth,1))
            t_data[cnt] = dates_data[cnt].toordinal()
            cnt += 1
    sst_data[sst_data==fillValue] = np.nan
    sst_data[sst_data<=-2] = np.nan
    sst_data[sst_data>=35] = np.nan
    lat_data = np.flipud(lat_data) # flip order of lat coords
    sst_data = sst_data[:,::-1,:]


T_data = len(t_data)
LON = len(lon_data)
LAT = len(lat_data)

year_data = np.zeros(T_data)
month_data = np.zeros(T_data)
for mth in range(T_data):
    year_data[mth] = date.fromordinal(t_data[mth].astype(int)).year
    month_data[mth] = date.fromordinal(t_data[mth].astype(int)).month

# Limit data to post-yearStart
years_proxies = range(yearStart, yearEnd+1)
tt = (year_data >= yearStart) * (year_data <= yearEnd)
sst_data = sst_data[tt]
if sw_sig:
    sig_data = sig_data[tt]
year_data = year_data[tt]
month_data = month_data[tt]

# Initialize empty dictionary of proxies
T_proxies = len(years_proxies)
missing = np.nan*np.zeros((T_proxies, LAT, LON))
proxy = {}
proxy_mode = {}
proxy_sig = {}
for pkey in ['mean', 'max', 'meanAnom', 'maxAnom', 'threshAnom', 'threshMaxAnom', 'posCount', 'threshCount', 'duration']:
    proxy[pkey] = np.nan*np.zeros((T_proxies, LAT, LON))
    if sw_nomodes:
        proxy_mode[pkey] = np.nan*np.zeros((T_proxies, LAT, LON))
    if sw_sig:
        proxy_sig[pkey] = np.nan*np.zeros((T_proxies, LAT, LON, Nsig))

# Climatology period (1983-2012)
#tt1 = np.where(year_data == 1979)[0][0]
tt1 = np.where(year_data == 1983)[0][0]
tt2 = np.where(year_data == 2012)[0][-1]
nYears = 2012 - 1983 + 1

# Generate predictor matrix for climate modes
if sw_nomodes:
    #
    # ENSO (MEI index)
    #
    MEI = np.genfromtxt('../../data/modes/mei.combined.1871_2016')
    t_MEI, tmp, tmp, year_MEI, month_MEI, day_MEI, tmp = ecj.timevector([1871, 1, 1], [2016, 12, 31])
    year_MEI = year_MEI[day_MEI==15]
    MEI = MEI[year_MEI>=yearStart]
    MEI = MEI - MEI.mean()
    MEI = signal.detrend(MEI)
    # Predictor matrix with lags
    Nl = 12 # Number of lead-lags
    which = []
    X = np.zeros((len(MEI),3+2*Nl+2))
    X[:,0] += 1 # Mean
    X[:,1] = np.arange(len(MEI)) - np.arange(len(MEI)).mean() # Linear trend
    which.append('Mean')
    which.append('Trend')
    # Mode leads SST
    cnt = 1
    for k in range(1,Nl+1):
        cnt += 1
        X[:-k,cnt] = MEI[k:]
        which.append('MEI')
    # 0-lag
    cnt += 1
    X[:,cnt] = MEI
    which.append('MEI')
    # Mode lags SST
    for k in range(1,Nl+1):
        cnt += 1
        X[k:,cnt] = MEI[:-k]
        which.append('MEI')
    #
    # PDO
    #
    PDO = np.genfromtxt('../../data/modes/pdo.ncdcnoaa.csv', delimiter=',', skip_header=2)[:,1]
    t_PDO, tmp, tmp, year_PDO, month_PDO, day_PDO, tmp = ecj.timevector([1854, 1, 1], [2016, 12, 31])
    year_PDO = year_PDO[day_PDO==15]
    PDO = PDO[year_PDO>=yearStart]
    XX = np.ones((len(MEI),3)) # Remove influence of MEI
    XX[:,1] = np.arange(len(MEI)) - np.arange(len(MEI)).mean() # Linear trend
    XX[:,2] = MEI
    beta = linalg.lstsq(XX, PDO)[0]
    PDO_MEI = beta[2]*XX[:,2]
    PDO -= PDO_MEI
    Ny = 5 # Number of years over which to smooth index
    PDO = ecj.runavg(PDO , 12*Ny+1)
    PDO = PDO - PDO.mean()
    PDO = signal.detrend(PDO)
    cnt += 1
    X[:,cnt] = PDO
    which.append('PDO')
    #
    # AMO
    #
    AMO = np.genfromtxt('../../data/modes/amon.us.long.data', skip_header=1, skip_footer=4)[:-1,1:].flatten()
    t_AMO, tmp, tmp, year_AMO, month_AMO, day_AMO, tmp = ecj.timevector([1856, 1, 1], [2016, 12, 31])
    year_AMO = year_AMO[day_AMO==15]
    Ny = 5 # Number of years over which to smooth index
    AMO = ecj.runavg(AMO, 12*Ny+1)
    AMO = AMO[year_AMO>=yearStart]
    AMO = AMO - AMO.mean()
    AMO = signal.detrend(AMO)
    cnt += 1
    X[:,cnt] = AMO
    which.append('AMO')
    #
    # Select which modes to use
    #
    X = X[:,np.in1d(np.array(which), np.append(np.array(['Mean', 'Trend']), which_modes))]

# Loop over (lat,lon) and calculate proxies 
# j = 30
# i = 20
# E. Trop. Pac. HadSST3: i = 18; j = 17;
pThresh = 90
for i in range(LON):
    print i+1, LON
    for j in range(LAT):
        sst = sst_data[:,j,i]
        sst_climatology = np.tile(np.nanmean(np.reshape(sst[tt1:tt2+1], (nYears, 12)).T, axis=1), len(years_proxies)) # Monthly seasonal climatology
        sst_thresh = np.tile(np.nanpercentile(np.reshape(sst[tt1:tt2+1], (nYears, 12)).T, pThresh, axis=1), len(years_proxies)) # Monthly seasonal percentile threshold
        sst_anom = sst - sst_climatology # Monthly anomalies from climatology
        sst_anomCens = 1.*sst_anom
        sst_anomCens[sst_anomCens<=0] = np.nan # Censor anomalies < 0
        sst_threshAnom = sst - sst_thresh # Monthly anomalies from threshold
        exceed_bool = 1.*sst_threshAnom
        exceed_bool[exceed_bool<=0] = False
        exceed_bool[exceed_bool>0] = True
        sst_threshAnom[sst_threshAnom<=0] = np.nan # Censor anomalies < 0
        # Annual aggregate statistics
        missing[:,j,i] = np.sum(np.isnan(np.reshape(sst, (len(years_proxies), 12)).T), axis=0)
        proxy['mean'][:,j,i] = np.mean(np.reshape(sst, (len(years_proxies), 12)).T, axis=0) # Annual mean
        proxy['maxAnom'][:,j,i] = np.max(np.reshape(sst_anom, (len(years_proxies), 12)).T, axis=0) # Annual maximum anomalies
        proxy['meanAnom'][:,j,i] = np.nanmean(np.reshape(sst_anomCens, (len(years_proxies), 12)).T, axis=0) # Annual mean of positive anomalies
        proxy['posCount'][:,j,i] = np.sum(np.reshape(sst_anom, (len(years_proxies), 12)).T > 0, axis=0) # Annual count above climatology
        proxy['threshCount'][:,j,i] = np.sum(np.reshape(sst_threshAnom, (len(years_proxies), 12)).T > 0, axis=0) # Annual count above threshold
        proxy['threshAnom'][:,j,i] = np.nanmean(np.reshape(sst_threshAnom, (len(years_proxies), 12)).T, axis=0) # Anual average anomaly above threshold
        proxy['threshMaxAnom'][:,j,i] = np.nanmax(np.reshape(sst_threshAnom, (len(years_proxies), 12)).T, axis=0) # Anual average anomaly above threshold
        events, n_events = ndimage.label(exceed_bool)
        durations = np.nan*np.zeros(len(sst))
        for ev in range(1,n_events+1):
            durations[np.floor(np.mean(np.where(events == ev))).astype(int)] = np.sum(events == ev)
        proxy['duration'][:,j,i] = np.nanmean(np.reshape(durations, (len(years_proxies), 12)).T, axis=0) # Annual average duration of threshold exceedances

# Mode-less proxies
if sw_nomodes:
    for i in range(LON):
        print i+1, LON
        for j in range(LAT):
            sst = sst_data[:,j,i]
            sst_climatology = np.tile(np.nanmean(np.reshape(sst[tt1:tt2+1], (nYears, 12)).T, axis=1), len(years_proxies)) # Monthly seasonal climatology
            sst_thresh = np.tile(np.nanpercentile(np.reshape(sst[tt1:tt2+1], (nYears, 12)).T, pThresh, axis=1), len(years_proxies)) # Monthly seasonal percentile threshold
            sst_anom = sst - sst_climatology # Monthly anomalies from climatology
            # Fit modes to SST
            valid = ~np.isnan(sst_anom)
            if valid.sum() > 1:
                beta = linalg.lstsq(X[valid,:], sst_anom[valid])[0][2:]
                sst_anom -= np.dot(np.array([beta,]), X[:,2:].T)[0,:]
                sst = sst_climatology + sst_anom
            #
            sst_anomCens = 1.*sst_anom
            sst_anomCens[sst_anomCens<=0] = np.nan # Censor anomalies < 0
            sst_threshAnom = sst - sst_thresh # Monthly anomalies from threshold
            exceed_bool = 1.*sst_threshAnom
            exceed_bool[exceed_bool<=0] = False
            exceed_bool[exceed_bool>0] = True
            sst_threshAnom[sst_threshAnom<=0] = np.nan # Censor anomalies < 0
            # Annual aggregate statistics
            missing[:,j,i] = np.sum(np.isnan(np.reshape(sst, (len(years_proxies), 12)).T), axis=0)
            proxy_mode['mean'][:,j,i] = np.mean(np.reshape(sst, (len(years_proxies), 12)).T, axis=0) # Annual mean
            proxy_mode['maxAnom'][:,j,i] = np.max(np.reshape(sst_anom, (len(years_proxies), 12)).T, axis=0) # Annual maximum anomalies
            proxy_mode['meanAnom'][:,j,i] = np.nanmean(np.reshape(sst_anomCens, (len(years_proxies), 12)).T, axis=0) # Annual mean of positive anomalies
            proxy_mode['posCount'][:,j,i] = np.sum(np.reshape(sst_anom, (len(years_proxies), 12)).T > 0, axis=0) # Annual count above climatology
            proxy_mode['threshCount'][:,j,i] = np.sum(np.reshape(sst_threshAnom, (len(years_proxies), 12)).T > 0, axis=0) # Annual count above threshold
            proxy_mode['threshAnom'][:,j,i] = np.nanmean(np.reshape(sst_threshAnom, (len(years_proxies), 12)).T, axis=0) # Anual average anomaly above threshold
            proxy_mode['threshMaxAnom'][:,j,i] = np.nanmax(np.reshape(sst_threshAnom, (len(years_proxies), 12)).T, axis=0) # Anual average anomaly above threshold
            events, n_events = ndimage.label(exceed_bool)
            durations = np.nan*np.zeros(len(sst))
            for ev in range(1,n_events+1):
                durations[np.floor(np.mean(np.where(events == ev))).astype(int)] = np.sum(events == ev)
            proxy_mode['duration'][:,j,i] = np.nanmean(np.reshape(durations, (len(years_proxies), 12)).T, axis=0) # Annual average duration of threshold exceedances

if sw_sig:
    for i in range(LON):
        print i+1, LON
        for j in range(LAT):
            for isig in range(Nsig):
                sst = sst_data[:,j,i] + sig_data[:,j,i]*np.random.randn(len(year_data))
                sst_climatology = np.tile(np.nanmean(np.reshape(sst[tt1:tt2+1], (nYears, 12)).T, axis=1), len(years_proxies)) # Monthly seasonal climatology
                sst_thresh = np.tile(np.nanpercentile(np.reshape(sst[tt1:tt2+1], (nYears, 12)).T, pThresh, axis=1), len(years_proxies)) # Monthly seasonal percentile threshold
                sst_anom = sst - sst_climatology # Monthly anomalies from climatology
                # Fit modes to SST
                if sw_nomodes:
                    valid = ~np.isnan(sst_anom)
                    if valid.sum() > 1:
                        beta = linalg.lstsq(X[valid,:], sst_anom[valid])[0][2:]
                        sst_anom -= np.dot(np.array([beta,]), X[:,2:].T)[0,:]
                        sst = sst_climatology + sst_anom
                #
                sst_anomCens = 1.*sst_anom
                sst_anomCens[sst_anomCens<=0] = np.nan # Censor anomalies < 0
                sst_threshAnom = sst - sst_thresh # Monthly anomalies from threshold
                exceed_bool = 1.*sst_threshAnom
                exceed_bool[exceed_bool<=0] = False
                exceed_bool[exceed_bool>0] = True
                sst_threshAnom[sst_threshAnom<=0] = np.nan # Censor anomalies < 0
                # Annual aggregate statistics
                proxy_sig['mean'][:,j,i,isig] = np.mean(np.reshape(sst, (len(years_proxies), 12)).T, axis=0) # Annual mean
                proxy_sig['maxAnom'][:,j,i,isig] = np.max(np.reshape(sst_anom, (len(years_proxies), 12)).T, axis=0) # Annual maximum anomalies
                proxy_sig['meanAnom'][:,j,i,isig] = np.nanmean(np.reshape(sst_anomCens, (len(years_proxies), 12)).T, axis=0) # Annual mean of positive anomalies
                proxy_sig['posCount'][:,j,i,isig] = np.sum(np.reshape(sst_anom, (len(years_proxies), 12)).T > 0, axis=0) # Annual count above climatology
                proxy_sig['threshCount'][:,j,i,isig] = np.sum(np.reshape(sst_threshAnom, (len(years_proxies), 12)).T > 0, axis=0) # Annual count above threshold
                proxy_sig['threshAnom'][:,j,i,isig] = np.nanmean(np.reshape(sst_threshAnom, (len(years_proxies), 12)).T, axis=0) # Anual average anomaly above threshold
                proxy_sig['threshMaxAnom'][:,j,i,isig] = np.nanmax(np.reshape(sst_threshAnom, (len(years_proxies), 12)).T, axis=0) # Anual average anomaly above threshold
                events, n_events = ndimage.label(exceed_bool)
                durations = np.nan*np.zeros(len(sst))
                for ev in range(1,n_events+1):
                    durations[np.floor(np.mean(np.where(events == ev))).astype(int)] = np.sum(events == ev)
                proxy_sig['duration'][:,j,i,isig] = np.nanmean(np.reshape(durations, (len(years_proxies), 12)).T, axis=0) # Annual average duration of threshold exceedances

# Make meanAnom, maxAnom (threshAnom, threshMaxAnom) as NaN when posCount (threshCount) = 0
proxy['maxAnom'][proxy['posCount']==0] = np.nan
proxy['meanAnom'][proxy['posCount']==0] = np.nan
proxy['threshAnom'][proxy['threshCount']==0] = np.nan
proxy['threshMaxAnom'][proxy['threshCount']==0] = np.nan
if sw_sig:
    proxy_sig['maxAnom'][proxy_sig['posCount']==0] = np.nan
    proxy_sig['meanAnom'][proxy_sig['posCount']==0] = np.nan
    proxy_sig['threshAnom'][proxy_sig['threshCount']==0] = np.nan
    proxy_sig['threshMaxAnom'][proxy_sig['threshCount']==0] = np.nan

# Mask any cell for which at least one month was missing
n_missing = 0 # Number of missing values allowed
for key in proxy.keys():
    proxy[key][missing>n_missing] = np.nan
    if sw_sig:
        for isig in range(Nsig):
            tmp = proxy_sig[key][:,:,:,isig].copy()
            tmp[missing>n_missing] = np.nan
            proxy_sig[key][:,:,:,isig] = tmp.copy()

# lat and lons of obs
lon_data_pos = lon_data.copy()
lon_data_pos[lon_data_pos<0] +=360.
res_data = np.diff(lon_data)[0]
years_data = np.array(years_proxies).copy()
avhrr = (years_data >= 1982) * (years_data <= dataEnd)

# Create landmask based on land
datamask = 1.-(np.sum(np.isnan(sst_data), axis=0) == sst_data.shape[0]).astype(int) # landmask
#np.savez(header_out + 'mhw_proxies_GLM.' + str(yearStart) + '.' + str(yearEnd) + '.COBE.datamask', datamask=datamask)
# Create landmask based on repeat seasonal cycle
repSeas = np.zeros((len(lat_data), len(lon_data)))
for i in range(12):
    repSeas += np.sum(np.diff(sst_data[i::12,:,:], axis=0)==0, axis=0)
# Combine the two landmasks
datamask[repSeas>10] = 0.

#
# AVHRR observations (daily)
#

#pathroot = '/mnt/erebor/'
#pathroot = '/home/ecoliver/Desktop/'
#pathroot = '/bs/projects/geology/Oliver/'
header = header_data + 'noaa_oi_v2/avhrr/'
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

avhrr_match = year <= dataEnd

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

MHW_keys = ['count', 'duration', 'intensity_mean', 'total_days', 'total_icum']

alpha = 0.05
Nens = 100
LON = len(lon_data)
LAT = len(lat_data)
DIM = (LAT, LON)
DIM_ts = (LAT, LON, len(years_data))
DIM_CI = (LAT, LON, len(years_data), 2)
rho_f = np.nan*np.zeros(DIM)
p_f = np.nan*np.zeros(DIM)
MHW_m = {}
MHW_tr = {}
MHW_dtr = {}
MHW_ts = {}
MHW_CI = {}
MHW_sig = {}
rho = {}
p = {}
for key in MHW_keys:
    MHW_m[key] = {}
    MHW_tr[key] = {}
    MHW_dtr[key] = {}
    MHW_ts[key] = {}
    MHW_CI[key] = {}
    MHW_sig[key] = {}
    rho[key] = {}
    p[key] = {}
    for pkey in proxy.keys():
        MHW_m[key][pkey] = np.nan*np.zeros(DIM)
        MHW_tr[key][pkey] = np.nan*np.zeros(DIM)
        MHW_dtr[key][pkey] = np.nan*np.zeros(DIM)
        MHW_ts[key][pkey] = np.nan*np.zeros(DIM_ts)
        MHW_CI[key][pkey] = np.nan*np.zeros(DIM_CI)
        MHW_sig[key][pkey] = np.nan*np.zeros(DIM_CI)
        rho[key][pkey] = np.nan*np.zeros(DIM)
        p[key][pkey] = np.nan*np.zeros(DIM)

SST_mean = np.nan*np.zeros(DIM)
SST_tr = np.nan*np.zeros(DIM)
ar1_tau = np.nan*np.zeros(DIM)
ar1_sig_eps = np.nan*np.zeros(DIM)
ar1_pltrend_cnt = np.nan*np.zeros(DIM)
ar1_pltrend_mean = np.nan*np.zeros(DIM)
ar1_pltrend_dur = np.nan*np.zeros(DIM)
ar1_putrend_cnt = np.nan*np.zeros(DIM)
ar1_putrend_mean = np.nan*np.zeros(DIM)
ar1_putrend_dur = np.nan*np.zeros(DIM)
ar1_mean_cnt = np.nan*np.zeros(DIM)
ar1_mean_mean = np.nan*np.zeros(DIM)
ar1_mean_dur = np.nan*np.zeros(DIM)

#
# loop through locations
#

# Distributions and link functions to use for each predictand
proxyScale = {}
proxyScale['mean'] = 'norm'
proxyScale['max'] = 'norm'
proxyScale['meanAnom'] = 'norm'
proxyScale['maxAnom'] = 'norm'
proxyScale['threshAnom'] = 'norm'
proxyScale['threshMaxAnom'] = 'norm'
proxyScale['posCount'] = 'log'
proxyScale['threshCount'] = 'log'
proxyScale['duration'] = 'log'

dist = {}
for pkey in proxy.keys():
    dist[pkey] = {}
    if proxyScale[pkey] == 'norm': # for mean, maxAnom, meanAnom, threshAnom, threshMeanAnom
        dist[pkey]['count'] = sm.families.Poisson()
        dist[pkey]['intensity_mean'] = sm.families.Gaussian()
        dist[pkey]['duration'] = sm.families.Poisson()
    elif proxyScale[pkey] == 'log': # for posCount, threshCount, duration
        dist[pkey]['count'] = sm.families.Poisson(sm.families.links.identity)
        dist[pkey]['intensity_mean'] = sm.families.Gaussian(sm.families.links.log)
        dist[pkey]['duration'] = sm.families.Poisson(sm.families.links.identity)

offset = {} # Offset to bring the lower limit of 'duration' down to zero
offset['count'] = 0.
offset['intensity_mean'] = 0.
offset['duration'] = 5.0

def nonans(array):
    '''
    Return input array [1D numpy array] with
    all nan values removed
    '''
    return array[~np.isnan(array)]

def proxy_fit(key, pkey, years, years_data, avhrr, dist, offset, sw_ens, sw_nomodes=False):
    '''
    Get correlation, etc of HadSST-based proxy fit to
    AVHRR annual aggregated MHW properties
    '''
    valid = ~np.isnan(mhwBlock[key]) * ~np.isnan(proxy[pkey][avhrr,j,i])
    if (valid.sum() <= 2) + (sw_data=='SODA')*(valid.sum() <= 3):
        return np.nan, np.nan, np.nan, np.nan
    else:
        # Confidence interval based on model error
        if sw_ens == 'model':
            X = np.array([np.ones(years.shape), proxy[pkey][avhrr,j,i]])
            glm = sm.GLM(mhwBlock[key][valid] - offset[key], X[:,valid].T, family=dist[pkey][key])
            res = glm.fit()
            if sw_nomodes:
                X = np.array([np.ones(years_data.shape), proxy_mode[pkey][:,j,i]])
            else:
                X = np.array([np.ones(years_data.shape), proxy[pkey][:,j,i]])
            pred = res.predict(X.T) + offset[key]
            rho, p = stats.pearsonr(mhwBlock[key][valid], pred[avhrr][valid])
            # CI calculation
            params_mu = res.params
            params_sig = np.mean(np.abs(np.tile(params_mu, [2, 1]).T - res.conf_int()), axis=1)/1.96
            pred_ens = np.zeros((len(pred), Nens))
            for iens in range(Nens):
                pred_ens[:,iens] = glm.predict(params=params_mu + params_sig*np.random.randn(len(params_mu)), exog=X.T) + offset[key]
            pred_CI = np.zeros((len(pred), 2))
            pred_CI[:,0] = np.percentile(pred_ens, 2.5, axis=1)
            pred_CI[:,1] = np.percentile(pred_ens, 97.5, axis=1)
            return rho, p, pred, pred_CI
        # Confidence interval based on observational uncertainty
        elif sw_ens == 'obs':
            X = np.array([np.ones(years.shape), proxy[pkey][avhrr,j,i]])
            glm = sm.GLM(mhwBlock[key][valid] - offset[key], X[:,valid].T, family=dist[pkey][key])
            res = glm.fit()
            pred_ens = np.zeros((len(years_data), Nsig))
            for isig in range(Nsig):
                X = np.array([np.ones(years_data.shape), proxy_sig[pkey][:,j,i,isig]])
                pred_ens[:,isig] = res.predict(X.T) + offset[key]
            pred_CI = np.zeros((len(years_data), 2))
            for tt in range(len(years_data)):
                if np.isnan(pred_ens[tt,:]).sum() == Nsig:
                    pred_CI[tt,:] = np.nan
                else:
                    pred_CI[tt,0] = np.percentile(nonans(pred_ens[tt,:]), 2.5)
                    pred_CI[tt,1] = np.percentile(nonans(pred_ens[tt,:]), 97.5)
            return np.nan, np.nan, np.nan, pred_CI

def proxy_meanTrend(pred, years_data):
    '''
    Calculate mean and slope of proxy MHW time series
    '''
    valid = ~np.isnan(pred)
    X_lin = np.array([np.ones(years_data.shape), years_data-years_data.mean()])
    beta = linalg.lstsq(X_lin[:,valid].T, pred[valid])[0]
    m, tr = beta[0], beta[1]
    yhat = np.sum(beta*X_lin.T, axis=1)
    t_stat = stats.t.isf(alpha/2, len(years_data[valid])-2)
    s = np.sqrt(np.sum((pred[valid] - yhat[valid])**2) / (len(years_data[valid])-2))
    Sxx = np.sum(X_lin[1,valid]**2) - (np.sum(X_lin[1,valid])**2)/len(years_data[valid])
    dtr = t_stat * s / np.sqrt(Sxx)
    return m, tr, dtr

for i in range(LON):
    print i+1, 'of', LON
#   load SST
    i1 = np.where(lon > lon_data_pos[i] - res_data/2.)[0][0]
    i2 = np.where(lon > lon_data_pos[i] + res_data/2. - res)[0][0]
    sst_ts = np.nan*np.zeros((len(lat), avhrr_match.sum(), i2-i1+1))
    cnt = 0
    for i0 in range(i1, i2+1):
      #if i0 == 1338:
      #    continue
      matobj = io.loadmat(header + 'timeseries/avhrr-only-v2.ts.' + str(i0+1).zfill(4) + '.mat')
      sst_ts[:,:,cnt] = matobj['sst_ts'][:,avhrr_match]
      cnt += 1
#   loop over j
    j = 0
    for j in range(LAT):
        if lat_data[j] - res_data/2. < lat.min():
            j1 = 0
        else:
            j1 = np.where(lat > lat_data[j] - res_data/2.)[0][0]
        if lat_data[j] + res_data/2. > lat.max():
            j2 = len(lat)-1
        else:
            j2 = np.where(lat > lat_data[j] + res_data/2.)[0][0]
        sst = np.nanmean(np.nanmean(sst_ts[j1:j2+1,:,:], axis=2), axis=0)
        if np.logical_not(np.isfinite(sst.sum())) + ((sst<-1).sum()>0): # check for land, ice
            continue
#   Count number of MHWs of each length
        mhws, clim = mhw.detect(t[avhrr_match], sst, climatologyPeriod=[1983, np.min([dataEnd, 2012])])
        mhwBlock = mhw.blockAverage(t[avhrr_match], mhws)
        years = mhwBlock['years_centre']
#   Skip proxy_fit calculation if not enough data
        if ((proxy['threshCount'][avhrr,j,i]>0).sum() <= 5) + ((~np.isnan(proxy['maxAnom'][avhrr,j,i])).sum() <= 5) + ((~np.isnan(proxy['threshAnom'][avhrr,j,i])).sum() <= 5):
            continue
# Loop over all combinations of MHW properties and proxy keys, to build and test all models
        for key in MHW_keys:
            if (key == 'total_days') + (key == 'total_icum'):
                continue
            else:
                for pkey in proxy.keys():
                    rho[key][pkey][j,i], p[key][pkey][j,i], MHW_ts[key][pkey][j,i,:], MHW_CI[key][pkey][j,i,:,:] = proxy_fit(key, pkey, years, years_data, avhrr, dist, offset, 'model', sw_nomodes=sw_nomodes)
                    if sw_sig:
                        tmp, tmp, tmp, MHW_sig[key][pkey][j,i,:,:] = proxy_fit(key, pkey, years, years_data, avhrr, dist, offset, 'obs')
                    if np.isnan(MHW_ts[key][pkey][j,i,:]).sum() < len(years_data):
                        MHW_m[key][pkey][j,i], MHW_tr[key][pkey][j,i], MHW_dtr[key][pkey][j,i] = proxy_meanTrend(MHW_ts[key][pkey][j,i,:], years_data)
        # Special case for total_days and total_icum
        for pkey in proxy.keys():
            # total days
            MHW_ts['total_days'][pkey][j,i,:] = MHW_ts['count']['threshCount'][j,i,:]*MHW_ts['duration']['maxAnom'][j,i,:]
            if np.isnan(MHW_ts['total_days'][pkey][j,i,:]).sum() < len(years_data):
                MHW_m['total_days'][pkey][j,i], MHW_tr['total_days'][pkey][j,i], MHW_dtr['total_days'][pkey][j,i] = proxy_meanTrend(MHW_ts['total_days'][pkey][j,i,:], years_data)
            # total icum
            MHW_ts['total_icum'][pkey][j,i,:] = MHW_ts['count']['threshCount'][j,i,:]*MHW_ts['duration']['maxAnom'][j,i,:]*MHW_ts['intensity_mean']['threshAnom'][j,i,:]
            if np.isnan(MHW_ts['total_icum'][pkey][j,i,:]).sum() < len(years_data):
                MHW_m['total_icum'][pkey][j,i], MHW_tr['total_icum'][pkey][j,i], MHW_dtr['total_icum'][pkey][j,i] = proxy_meanTrend(MHW_ts['total_icum'][pkey][j,i,:], years_data)
        # Get mean and linear trend in annual mean SST
        if np.isnan(proxy['mean'][:,j,i]).sum() == len(proxy['mean'][:,j,i]):
            continue
        else:
            valid = np.where(~np.isnan(proxy['mean'][:,j,i]))[0]
            X = np.array([np.ones(years_data.shape), years_data-years_data.mean()]).T
            beta = linalg.lstsq(X[valid,:], proxy['mean'][valid,j,i])[0]
            SST_mean[j,i] = beta[0]
            SST_tr[j,i] = beta[1]
        # AR1 Model fit to simulate MHW property trends
        #if sw_AR1:
        #    a, tmp, sig_eps = trendSimAR1.ar1fit(sst)
        #    a, tmp, sig_eps = trendSimAR1.ar1fit(signal.detrend(sst - clim['seas']))
        #    ar1_tau[j,i], ar1_sig_eps[j,i], trends, means = trendSimAR1.simulate(t_HadSST, sst, clim['seas'], SST_tr[j,i]*10, 100, params=[a, sig_eps])
        #    ar1_pltrend_cnt[j,i] = np.percentile(trends['count'], 2.5)
        #    ar1_pltrend_mean[j,i] = np.percentile(trends['intensity_mean'], 2.5)
        #    ar1_pltrend_dur[j,i] = np.percentile(trends['duration'], 2.5)
        #    ar1_putrend_cnt[j,i] = np.percentile(trends['count'], 97.5)
        #    ar1_putrend_mean[j,i] = np.percentile(trends['intensity_mean'], 97.5)
        #    ar1_putrend_dur[j,i] = np.percentile(trends['duration'], 97.5)
        #    ar1_mean_cnt[j,i] = means['count'].mean()
        #    ar1_mean_mean[j,i] = means['intensity_mean'].mean()
        #    ar1_mean_dur[j,i] = means['duration'].mean()

# Save data so far
if sw_nomodes:
    outfile = header_out + 'mhw_proxies_GLM.' + str(yearStart) + '.' + str(yearEnd) + '.' + sw_data + '.no_' + '_'.join(which_modes)
else:
    outfile = header_out + 'mhw_proxies_GLM.' + str(yearStart) + '.' + str(yearEnd) + '.' + sw_data

if sw_sig:
    np.savez(outfile + '.sig', lon_data=lon_data, lat_data=lat_data, MHW_m=MHW_m, MHW_tr=MHW_tr, MHW_dtr=MHW_dtr, MHW_ts=MHW_ts, MHW_CI=MHW_CI, MHW_sig=MHW_sig, rho=rho, p=p, SST_mean=SST_mean, SST_tr=SST_tr, datamask=datamask)
else:
    np.savez(outfile, lon_data=lon_data, lat_data=lat_data, MHW_m=MHW_m, MHW_tr=MHW_tr, MHW_dtr=MHW_dtr, MHW_ts=MHW_ts, MHW_CI=MHW_CI, rho=rho, p=p, SST_mean=SST_mean, SST_tr=SST_tr, datamask=datamask)




#np.savez(outfile, lon_data=lon_data, lat_data=lat_data, MHW_m=MHW_m, MHW_tr=MHW_tr, MHW_dtr=MHW_dtr, MHW_ts=MHW_ts, rho=rho, p=p, SST_mean=SST_mean, SST_tr=SST_tr, ar1_tau=ar1_tau, ar1_sig_eps=ar1_sig_eps, ar1_putrend_cnt=ar1_putrend_cnt, ar1_putrend_mean=ar1_putrend_mean, ar1_putrend_dur=ar1_putrend_dur, ar1_pltrend_cnt=ar1_pltrend_cnt, ar1_pltrend_mean=ar1_pltrend_mean, ar1_pltrend_dur=ar1_pltrend_dur, ar1_mean_cnt=ar1_mean_cnt, ar1_mean_mean=ar1_mean_mean, ar1_mean_dur=ar1_mean_dur)
