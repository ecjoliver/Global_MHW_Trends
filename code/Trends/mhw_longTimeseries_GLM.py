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

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import mpl_toolkits.basemap as bm

import ecoliver as ecj

import marineHeatWaves as mhw

sw_write = 0
sw_saveData = 0

pThresh = 90
yearSplit_validTrain = 1982
climPeriodFull = 0
maxPadLength = 5
alphaPred = 0.01
alpha1 = 0.05
alpha2 = 0.10
Nens = 100
col_obs = '0.5'
col_CI = (1,0.6,0.6)
sw_fixScripps = True # -0.45 degC offset to Scipps Pier after 1988

# Write table data to file
if sw_write:
    outfile_train = open('data/longTimeseries/mhw_longTimeseries_GLM.train.csv', 'w')
    outfile_valid = open('data/longTimeseries/mhw_longTimeseries_GLM.valid.csv', 'w')
    outfile_train.write('Location, Time-span, R_F_mean R_I_mean R_D_mean, R_F_maxAnom R_I_maxAnom R_D_maxAnom, R_F_meanAnom R_I_meanAnom R_D_meanAnom, R_F_posCount R_I_posCount R_D_posCount, R_F_threshCount R_I_threshCount R_D_threshCount, R_F_dur R_I_dur R_D_dur, R_F_threshMaxAnom R_I_threshMaxAnom R_D_threshMaxAnom, R_F_threshAnom R_I_threshAnom R_D_threshAnom\n')
    outfile_valid.write('Location, Time-span, R_F_mean R_I_mean R_D_mean, R_F_maxAnom R_I_maxAnom R_D_maxAnom, R_F_meanAnom R_I_meanAnom R_D_meanAnom, R_F_posCount R_I_posCount R_D_posCount, R_F_threshCount R_I_threshCount R_D_threshCount, R_F_dur R_I_dur R_D_dur, R_F_threshMaxAnom R_I_threshMaxAnom R_D_threshMaxAnom, R_F_threshAnom R_I_threshAnom R_D_threshAnom\n')

# Loop over stations
n = 0
Nst = 6
plt.figure(101, figsize=(26,8))
plt.figure(102, figsize=(26,8))
for i in range(Nst):
    # Load some select time series
    if i == 0:
        station = 'Pacific_Grove'
        if sw_write:
            outfile_train.write('Pacific Grove USA, ')
            outfile_valid.write('Pacific Grove USA, ')
        header = '../../data/long_time_series/'
        file = header + 'PG_1919-201502.txt'
        sst = np.genfromtxt(file, skip_header=29)[:,5]
        t = np.arange(date(1919,1,20).toordinal(),date(2015,2,28).toordinal()+1)
        # Whole-year range (1920-2014)
        sst = sst[346:35044+1]
        t = t[346:35044+1]
        if climPeriodFull:
            climPeriod = [1921,2014]
            #climPeriod = [1921,2003]
        else:
            climPeriod = [1983,2012]
            #climPeriod = [1979,2003]
    if i == 1:
        station = 'Scripps_Pier'
        if sw_write:
            outfile_train.write('Scripps Pier USA, ')
            outfile_valid.write('Scripps Pier USA, ')
        header = '../../data/long_time_series/'
        file = header + 'SIO_TEMP_1916-201510.txt'
        sst = np.genfromtxt(file, skip_header=26, delimiter='\t')[:,5]
        t = np.arange(date(1916,8,22).toordinal(),date(2015,10,31).toordinal()+1)
        # Whole-year range (1917-2014)
        sst = sst[132:35925+1]
        t = t[132:35925+1]
        if climPeriodFull:
            climPeriod = [1918,2013]
        else:
            climPeriod = [1983,2012]
    if i == 2:
        station = 'Newport_Beach'
        if sw_write:
            outfile_train.write('Newport Beach USA, ')
            outfile_valid.write('Newport Beach USA, ')
        header = '../../data/long_time_series/'
        file = header + 'NB_TEMP_1924-201402.txt'
        sst = np.genfromtxt(file, skip_header=29, delimiter='\t')[0:35711+1,5]
        t = np.arange(date(1924,11,1).toordinal(),date(2014,2,28).toordinal()+1)
        # Whole-year range (1925-2013)
        sst = sst[61:32567+1]
        t = t[61:32567+1]
        if climPeriodFull:
            climPeriod = [1926,2012]
        else:
            climPeriod = [1983,2012]
    if i == 3:
        station = 'Arendal'
        if sw_write:
            outfile_train.write('Arendal Norway, ')
            outfile_valid.write('Arendal Norway, ')
        header = '../../data/long_time_series/'
        # Load in 1918-2008 data (daily)
        file = header + 'flode_daily_1918-2008.asc'
        dat = np.genfromtxt(file, delimiter=',')
        sst1 = dat[:,3]
        ymd1 = dat[:,0:3].astype(int)
        t1 = [date(ymd1[tt][0], ymd1[tt][1], ymd1[tt][2]).toordinal() for tt in range(len(sst1))]
        sst1[sst1==-99] = np.nan
        # Load in 2009-2016 data (hourly)
        file = header + 'flode_hourly_2009-today.asc'
        dat = np.genfromtxt(file, delimiter=',')
        sst2 = dat[:,4]
        ymd2 = dat[:,0:3].astype(int)
        t2 = [date(ymd2[tt][0], ymd2[tt][1], ymd2[tt][2]).toordinal() for tt in range(len(sst2))]
        sst2[sst2==-99] = np.nan
        # Flesh out full array based on dates (restrict to 1924-2016)
        t, dates, T, year, month, day, doy = ecj.timevector([1924,1,1], [2016,12,31])
        sst = np.zeros(t.shape)
        count = np.zeros(sst.shape)
        # Insert sst1
        for tt in range(len(sst1)):
            tmp = np.where(t == t1[tt])[0]
            if len(tmp) > 0:
                ttt = tmp[0]
                sst[ttt] += sst1[tt]
                count[ttt] += 1
        # Insert sst2
        for tt in range(len(sst2)):
            tmp = np.where(t == t2[tt])[0]
            if len(tmp) > 0:
                ttt = tmp[0]
                sst[ttt] += sst2[tt]
                count[ttt] += 1
        # Average the 2-daily values, and remove missing (0's)
        # NOTE: any missing hour leads to a missing day value!
        sst[count>0] /= count[count>0]
        sst[sst==0.] = np.nan
        if climPeriodFull:
            climPeriod = [1925,2015]
        else:
            climPeriod = [1983,2012]
    if i == 4:
        station = 'Port_Erin'
        if sw_write:
            outfile_train.write('Port Erin UK, ')
            outfile_valid.write('Port Erin UK, ')
        header = '../../data/long_time_series/'
        # Load in 1904-2012 data
        file = header + 'Port_Erin_UK_daily_temps_1904_to_2012.daily.csv'
        sst1 = np.genfromtxt(file, skip_header=1, delimiter=',')[:,3]
        t = np.arange(date(1904,1,1).toordinal(),date(2012,10,28).toordinal()+1)
        t1, dates1, T1, year1, month1, day1, doy1 = ecj.timevector([1904,1,1], [2012,10,28])
        # Load in 2013-2014 data
        file = header + 'CefasSWTdata20170707source01_PortErin.csv'
        sst20 = np.genfromtxt(file, skip_header=1, delimiter=',')[:,5]
        datestring = np.genfromtxt(file, skip_header=1, delimiter=',', dtype='S')[:,1]
        # Flesh out full array based on dates
        t, dates, T, year, month, day, doy = ecj.timevector([1904,1,1], [2014,12,31])
        sst = np.zeros(t.shape)
        # Insert sst1 (1904-2012)
        sst[np.in1d(t, t1)] = sst1
        # Insert sst2 (2013-2014)
        count = np.zeros(sst.shape)
        for tt in range(len(sst20)):
            y = np.int(datestring[tt][6:10])
            m = np.int(datestring[tt][3:5])
            d = np.int(datestring[tt][0:2])
            sst[np.where(t == date(y,m,d).toordinal())[0][0]] += sst20[tt]
            count[np.where(t == date(y,m,d).toordinal())[0][0]] += 1
        # Average the 2-daily values, and remove missing (0's)
        sst[count>0] /= count[count>0]
        sst[sst==0.] = np.nan
        if climPeriodFull:
            climPeriod = [1905,2013]
        else:
            climPeriod = [1983,2012]
    if i == 5:
        station = 'Race_Rocks'
        if sw_write:
            outfile_train.write('Race Rocks Canada, ')
            outfile_valid.write('Race Rocks Canada, ')
        header = '../../data/long_time_series/'
        file = header + 'BC_lighthouses/RaceRocksDailySalTemp.csv'
        sst0 = np.genfromtxt(file, delimiter=',', skip_header=1)[:,4]
        sst0[sst0==99.9] = np.nan
        datestring = np.genfromtxt(file, delimiter=',', skip_header=1, dtype='S')[:,0]
        yearStart = np.int(datestring[0][0:4])
        yearEnd   = np.int(datestring[-1][0:4])
        # Flesh out full array based on dates
        t, dates, T, year, month, day, doy = ecj.timevector([yearStart,1,1], [yearEnd,12,31])
        sst = np.nan*np.zeros(t.shape)
        for tt in range(len(sst0)):
            y = np.int(datestring[tt][0:4])
            m = np.int(datestring[tt][5:7])
            d = np.int(datestring[tt][8:10])
            sst[np.where(t == date(y,m,d).toordinal())[0][0]] = sst0[tt]
        if climPeriodFull:
            climPeriod = [yearStart+1,yearEnd-1]
        else:
            climPeriod = [1983,2012]
    # Generate time vector(s)
    T = len(t)
    dates = [date.fromordinal(tt.astype(int)) for tt in t]
    year = np.zeros((T))
    month = np.zeros((T))
    day = np.zeros((T))
    for tt in range(T):
        year[tt] = date.fromordinal(t[tt]).year
        month[tt] = date.fromordinal(t[tt]).month
        day[tt] = date.fromordinal(t[tt]).day
    if sw_write:
        outfile_train.write(str(year.min().astype(int)) + '-' + str(year.max().astype(int)) + ', ')
        outfile_valid.write(str(year.min().astype(int)) + '-' + str(year.max().astype(int)) + ', ')
    # Fix Scripps Pier data
    if sw_fixScripps * (i == 1):
        sst[year>=1988] -= 0.45
    # Apply Marine Heat Wave definition
    print '\n' + station + '\n'
    print 'Proportion of valid values: ' + str(100. - 100.*np.isnan(sst).sum()/len(sst))
    mhws, clim = mhw.detect(t, sst, climatologyPeriod=climPeriod, maxPadLength=maxPadLength)
    mhwBlock = mhw.blockAverage(t, mhws, clim, removeMissing=True, temp=sst)
    mean, trend, dtrend = mhw.meanTrend(mhwBlock)
    # Number of years with no MHW events pre and post 1950
    #count_pre50 = mhwBlock['count'][(mhwBlock['years_centre']<=1950)*(~np.isnan(mhwBlock['count']))]
    #count_post50 = mhwBlock['count'][(mhwBlock['years_centre']>1950)*(~np.isnan(mhwBlock['count']))]
    #print 'Pre-1950 0-count: ' + str(1.*np.sum(count_pre50==0)/len(count_pre50)) + ' post-1950 0-count: ' + str(1.*np.sum(count_post50==0)/len(count_post50)) + ' Difference: ' + str(1.*np.sum(count_post50==0)/len(count_post50) - 1.*np.sum(count_pre50==0)/len(count_pre50))
    years = mhwBlock['years_centre']
    #Ny = len(mhwBlock['years_centre'])/2
    #postYears = [date.fromordinal(t[-1]).year - 30 + 1, date.fromordinal(t[-1]).year]
    postYears = [1984, 2013]
    ttPre  = (years >= 1925) * (years <= 1954)
    ttPost = (years >= postYears[0]) * (years <= postYears[1])
    count_preMid = ecj.nonans(mhwBlock['count'][ttPre])
    count_postMid = ecj.nonans(mhwBlock['count'][ttPost])
    print 'Pre-midpoint 0-count: ' + str(1.*np.sum(count_preMid==0)/len(count_preMid)) + ' post-midpoint 0-count: ' + str(1.*np.sum(count_postMid==0)/len(count_postMid)) + ' Difference: ' + str(1.*np.sum(count_postMid==0)/len(count_postMid) - 1.*np.sum(count_preMid==0)/len(count_preMid))
    # Mean properties before/after 1950
    meanDiffMeans = {}
    meanDiff = {}
    pDiff = {}
    for key in ['count', 'intensity_max_max', 'duration', 'temp_mean']:
        meanDiffMeans[key] = [np.nanmean(mhwBlock[key][ttPre]), np.nanmean(mhwBlock[key][ttPost])]
        meanDiff[key] = np.nanmean(mhwBlock[key][ttPost]) - np.nanmean(mhwBlock[key][ttPre])
        tmp, pDiff[key] = ecj.ttest_serialcorr(ecj.nonans(mhwBlock[key][ttPost]), ecj.nonans(mhwBlock[key][ttPre]))
    print 'Mean temperature change: ' + str(meanDiff['temp_mean']) + ' degC (p = ' + str(pDiff['temp_mean']) + ')'
    # Monthly-average sst
    sst = ecj.pad(sst) # Pad for purposes of Annual averages, missing years will be excluded later
    sst_monthly = np.zeros(len(years)*12)
    year_monthly = np.zeros(len(years)*12)
    month_monthly = np.zeros(len(years)*12)
    cnt = 0
    for yr in range(len(years)):
        # Make monthly series for year of interest
        for mth in range(12):
            sst_monthly[cnt] = np.mean(sst[(year==years[yr]) * (month==mth+1)])
            year_monthly[cnt] = years[yr]
            month_monthly[cnt] = mth+1
            cnt += 1
    # Calculation seasonal cycle mean and threshold climatology
    # NOTE: Assumes whole years (start on Jan 1, ends on Dec 31)
    tt1 = np.where(year_monthly == climPeriod[0])[0][0]
    tt2 = np.where(year_monthly == climPeriod[1])[0][-1]
    nYears = climPeriod[1] - climPeriod[0] + 1
    sst_climatology = np.tile(np.mean(np.reshape(sst_monthly[tt1:tt2+1], (nYears, 12)).T, axis=1), len(years)) # Monthly seasonal climatology
    sst_thresh = np.tile(np.percentile(np.reshape(sst_monthly[tt1:tt2+1], (nYears, 12)).T, pThresh, axis=1), len(years)) # Monthly seasonal percentile threshold
    sst_monthly_anom = sst_monthly - sst_climatology # Monthly anomalies from climatology
    sst_monthly_anomCens = 1.*sst_monthly_anom
    sst_monthly_anomCens[sst_monthly_anomCens<=0] = np.nan # Censor anomalies < 0
    sst_monthly_threshAnom = sst_monthly - sst_thresh # Monthly anomalies from threshold
    exceed_bool = 1.*sst_monthly_threshAnom
    exceed_bool[exceed_bool<=0] = False
    exceed_bool[exceed_bool>0] = True
    sst_monthly_threshAnom[sst_monthly_threshAnom<=0] = np.nan # Censor anomalies < 0
    # Annual aggregate statistics
    sst_annual_mean = np.mean(np.reshape(sst_monthly, (len(years), 12)).T, axis=0) # Annual mean
    sst_annual_maxAnom = np.max(np.reshape(sst_monthly_anom, (len(years), 12)).T, axis=0) # Annual maximum anomalies
    sst_annual_meanAnom = np.nanmean(np.reshape(sst_monthly_anomCens, (len(years), 12)).T, axis=0) # Annual mean of positive anomalies
    sst_annual_posCount = np.sum(np.reshape(sst_monthly_anom, (len(years), 12)).T > 0, axis=0) # Annual count above climatology
    sst_annual_threshCount = np.sum(np.reshape(sst_monthly_threshAnom, (len(years), 12)).T > 0, axis=0) # Annual count above threshold
    sst_annual_threshAnom = np.nanmean(np.reshape(sst_monthly_threshAnom, (len(years), 12)).T, axis=0) # Anual average anomaly above threshold
    sst_annual_threshMaxAnom = np.nanmax(np.reshape(sst_monthly_threshAnom, (len(years), 12)).T, axis=0) # Anual average anomaly above threshold
    events, n_events = ndimage.label(exceed_bool)
    durations = np.nan*np.zeros(len(sst_monthly))
    for ev in range(1,n_events+1):
        durations[np.floor(np.mean(np.where(events == ev))).astype(int)] = np.sum(events == ev)
    sst_annual_duration = np.nanmean(np.reshape(durations, (len(years), 12)).T, axis=0) # Annual average duration of threshold exceedances
    # Calculate percentiles
    pctiles = range(50, 95+1, 5)
    sst_pctiles = np.zeros((len(years), len(pctiles)))
    sst_pctiles_trend = np.zeros((len(pctiles)))
    for yr in range(len(years)):
        sst_pctiles[yr,:] = np.percentile(sst[year==years[yr].astype(int)], pctiles)
    for pctile in range(len(pctiles)):
        tmp1, sst_pctiles_trend[pctile], tmp2 = ecj.trend(years, sst_pctiles[:,pctile])
    # Set up time range for training/validation of regression
    training = years >= yearSplit_validTrain
    validation = years <  yearSplit_validTrain
    keys = ['count', 'intensity_max_max', 'duration']
    # Distributions and link functions to use for each predictand
    distG = {} # for mean, maxAnom, meanAnom, threshAnom, threshMeanAnom
    distG['count'] = sm.families.Poisson()
    distG['intensity_max_max'] = sm.families.Gaussian()
    distG['duration'] = sm.families.Poisson()
    distL = {} # for posCount, threshCount, duration
    distL['count'] = sm.families.Poisson(sm.families.links.identity)
    distL['intensity_max_max'] = sm.families.Gaussian(sm.families.links.log)
    distL['duration'] = sm.families.Poisson(sm.families.links.identity)
    offset = {} # Offset to bring the lower limit of 'duration' down to zero
    offset['count'] = 0.
    offset['intensity_max_max'] = 0.
    offset['duration'] = 5.
    # Correlations between MHW metrics and annual mean sst
    for key in keys:
        valid = ~np.isnan(mhwBlock[key])
        X = np.array([np.ones(years.shape), sst_annual_mean])
        glm = sm.GLM(mhwBlock[key][valid*training] - offset[key], X[:,valid*training].T, family=distG[key])
        res = glm.fit()
        pred = res.predict(X.T) + offset[key]
        rho, p = stats.pearsonr(mhwBlock[key][valid*training], pred[valid*training])
        if sw_write:
            outfile_train.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
        rho, p = stats.pearsonr(mhwBlock[key][valid*validation], pred[valid*validation])
        if sw_write:
            outfile_valid.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
    if sw_write:
        outfile_train.write(', ')
        outfile_valid.write(', ')
    # Correlations between MHW metrics and annual max sst
    for key in keys:
        valid = ~np.isnan(mhwBlock[key])
        X = np.array([np.ones(years.shape), sst_annual_maxAnom])
        glm = sm.GLM(mhwBlock[key][valid*training] - offset[key], X[:,valid*training].T, family=distG[key])
        res = glm.fit()
        pred = res.predict(X.T) + offset[key]
        rho, p = stats.pearsonr(mhwBlock[key][valid*training], pred[valid*training])
        if sw_write:
            outfile_train.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
        rho, p = stats.pearsonr(mhwBlock[key][valid*validation], pred[valid*validation])
        if sw_write:
            outfile_valid.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
        if key == 'duration':
            pred_duration = pred
            params_mu = res.params
            params_sig = np.mean(np.abs(np.tile(params_mu, [2, 1]).T - res.conf_int()), axis=1)/1.96
            pred_duration_ens = np.zeros((len(pred_duration), Nens))
            for iens in range(Nens):
                pred_duration_ens[:,iens] = glm.predict(params=params_mu + params_sig*np.random.randn(len(params_mu)), exog=X.T) +  + offset[key]
            pred_duration_CI = np.zeros((len(pred_duration), 2))
            pred_duration_CI[:,0] = np.percentile(pred_duration_ens, 100*(alphaPred/2), axis=1)
            pred_duration_CI[:,1] = np.percentile(pred_duration_ens, 100*(1 - alphaPred/2), axis=1)
            X = np.array([np.ones(years.shape), years-years.mean()])
            pred_duration_slope = linalg.lstsq(X[:,valid].T, pred_duration[valid])[0][1]
            duration_intercept, duration_slope = linalg.lstsq(X[:,valid].T, mhwBlock[key][valid])[0]
            duration_line = duration_intercept + duration_slope*X[1,:]
            yhat = np.sum(duration_slope*X[:,valid].T, axis=1)
            t_stat = stats.t.isf(alpha1/2, len(years)-2)
            s = np.sqrt(np.sum((mhwBlock[key][valid] - yhat)**2) / (len(years)-2))
            Sxx = np.sum(X[1,:].T**2) - (np.sum(X[1,:].T)**2)/len(years) # np.var(X, axis=1)[1]
            duration_dslope1 = t_stat * s / np.sqrt(Sxx)
            t_stat = stats.t.isf(alpha2/2, len(years)-2)
            s = np.sqrt(np.sum((mhwBlock[key][valid] - yhat)**2) / (len(years)-2))
            Sxx = np.sum(X[1,:].T**2) - (np.sum(X[1,:].T)**2)/len(years) # np.var(X, axis=1)[1]
            duration_dslope2 = t_stat * s / np.sqrt(Sxx)
    if sw_write:
        outfile_train.write(', ')
        outfile_valid.write(', ')
    # Correlations between MHW metrics and annual mean positive sst anomalies
    for key in keys:
        valid = ~np.isnan(mhwBlock[key]) * ~np.isnan(sst_annual_meanAnom)
        X = np.array([np.ones(years.shape), sst_annual_meanAnom])
        glm = sm.GLM(mhwBlock[key][valid*training] - offset[key], X[:,valid*training].T, family=distG[key])
        res = glm.fit()
        pred = res.predict(X.T) + offset[key]
        rho, p = stats.pearsonr(mhwBlock[key][valid*training], pred[valid*training])
        if sw_write:
            outfile_train.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
        rho, p = stats.pearsonr(mhwBlock[key][valid*validation], pred[valid*validation])
        if sw_write:
            outfile_valid.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
    if sw_write:
        outfile_train.write(', ')
        outfile_valid.write(', ')
    # Correlations between MHW metrics and annual count of positive-anomaly-months
    for key in keys:
        valid = ~np.isnan(mhwBlock[key])
        X = np.array([np.ones(years.shape), sst_annual_posCount])
        glm = sm.GLM(mhwBlock[key][valid*training] - offset[key], X[:,valid*training].T, family=distL[key])
        res = glm.fit()
        pred = res.predict(X.T) + offset[key]
        rho, p = stats.pearsonr(mhwBlock[key][valid*training], pred[valid*training])
        if sw_write:
            outfile_train.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
        rho, p = stats.pearsonr(mhwBlock[key][valid*validation], pred[valid*validation])
        if sw_write:
            outfile_valid.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
    if sw_write:
        outfile_train.write(', ')
        outfile_valid.write(', ')
    # Correlations between MHW metrics and annual count of months above threshold
    for key in keys:
        valid = ~np.isnan(mhwBlock[key])
        X = np.array([np.ones(years.shape), sst_annual_threshCount])
        glm = sm.GLM(mhwBlock[key][valid*training] - offset[key], X[:,valid*training].T, family=distL[key])
        res = glm.fit()
        pred = res.predict(X.T) + offset[key]
        rho, p = stats.pearsonr(mhwBlock[key][valid*training], pred[valid*training])
        if sw_write:
            outfile_train.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
        rho, p = stats.pearsonr(mhwBlock[key][valid*validation], pred[valid*validation])
        if sw_write:
            outfile_valid.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
        if key == 'count':
            pred_frequency = pred
            params_mu = res.params
            params_sig = np.mean(np.abs(np.tile(params_mu, [2, 1]).T - res.conf_int()), axis=1)/1.96
            pred_frequency_ens = np.zeros((len(pred_frequency), Nens))
            for iens in range(Nens):
                pred_frequency_ens[:,iens] = glm.predict(params=params_mu + params_sig*np.random.randn(len(params_mu)), exog=X.T) +  + offset[key]
            pred_frequency_CI = np.zeros((len(pred_frequency), 2))
            pred_frequency_CI[:,0] = np.percentile(pred_frequency_ens, 100*(alphaPred/2), axis=1)
            pred_frequency_CI[:,1] = np.percentile(pred_frequency_ens, 100*(1 - alphaPred/2), axis=1)
            X = np.array([np.ones(years.shape), years-years.mean()])
            pred_frequency_slope = linalg.lstsq(X[:,valid].T, pred_frequency[valid])[0][1]
            frequency_intercept, frequency_slope = linalg.lstsq(X[:,valid].T, mhwBlock[key][valid])[0]
            frequency_line = frequency_intercept + frequency_slope*X[1,:]
            yhat = np.sum(frequency_slope*X[:,valid].T, axis=1)
            t_stat = stats.t.isf(alpha1/2, len(years)-2)
            s = np.sqrt(np.sum((mhwBlock[key][valid] - yhat)**2) / (len(years)-2))
            Sxx = np.sum(X[1,:].T**2) - (np.sum(X[1,:].T)**2)/len(years) # np.var(X, axis=1)[1]
            frequency_dslope1 = t_stat * s / np.sqrt(Sxx)
            t_stat = stats.t.isf(alpha2/2, len(years)-2)
            s = np.sqrt(np.sum((mhwBlock[key][valid] - yhat)**2) / (len(years)-2))
            Sxx = np.sum(X[1,:].T**2) - (np.sum(X[1,:].T)**2)/len(years) # np.var(X, axis=1)[1]
            frequency_dslope2 = t_stat * s / np.sqrt(Sxx)
    if sw_write:
        outfile_train.write(', ')
        outfile_valid.write(', ')
    # Correlations between MHW metrics and average annual duration of monthly threshold exceedances
    for key in keys:
        valid = ~np.isnan(mhwBlock[key]) * ~np.isnan(sst_annual_duration)
        X = np.array([np.ones(years.shape), sst_annual_duration])
        glm = sm.GLM(mhwBlock[key][valid*training] - offset[key], X[:,valid*training].T, family=distL[key])
        res = glm.fit()
        pred = res.predict(X.T) + offset[key]
        rho, p = stats.pearsonr(mhwBlock[key][valid*training], pred[valid*training])
        if sw_write:
            outfile_train.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
        rho, p = stats.pearsonr(mhwBlock[key][valid*validation], pred[valid*validation])
        if sw_write:
            outfile_valid.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
    if sw_write:
        outfile_train.write(', ')
        outfile_valid.write(', ')
    # Correlations between MHW metrics and max annual anomaly above threshold
    for key in keys:
        valid = ~np.isnan(mhwBlock[key]) * ~np.isnan(sst_annual_threshMaxAnom)
        X = np.array([np.ones(years.shape), sst_annual_threshMaxAnom])
        glm = sm.GLM(mhwBlock[key][valid*training] - offset[key], X[:,valid*training].T, family=distG[key])
        res = glm.fit()
        pred = res.predict(X.T) + offset[key]
        rho, p = stats.pearsonr(mhwBlock[key][valid*training], pred[valid*training])
        if sw_write:
            outfile_train.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
        rho, p = stats.pearsonr(mhwBlock[key][valid*validation], pred[valid*validation])
        if sw_write:
            outfile_valid.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
        if key == 'intensity_max_max':
            pred_intensity = pred
            params_mu = res.params
            params_sig = np.mean(np.abs(np.tile(params_mu, [2, 1]).T - res.conf_int()), axis=1)/1.96
            pred_intensity_ens = np.zeros((len(pred_intensity), Nens))
            for iens in range(Nens):
                pred_intensity_ens[:,iens] = glm.predict(params=params_mu + params_sig*np.random.randn(len(params_mu)), exog=X.T) +  + offset[key]
            pred_intensity_CI = np.zeros((len(pred_intensity), 2))
            pred_intensity_CI[:,0] = np.percentile(pred_intensity_ens, 100*(alphaPred/2), axis=1)
            pred_intensity_CI[:,1] = np.percentile(pred_intensity_ens, 100*(1 - alphaPred/2), axis=1)
            X = np.array([np.ones(years.shape), years-years.mean()])
            pred_intensity_slope = linalg.lstsq(X[:,~np.isnan(sst_annual_threshAnom)].T, pred_intensity[~np.isnan(sst_annual_threshAnom)])[0][1]
            intensity_intercept, intensity_slope = linalg.lstsq(X[:,~np.isnan(mhwBlock[key])].T, mhwBlock[key][~np.isnan(mhwBlock[key])])[0]
            intensity_line = intensity_intercept + intensity_slope*X[1,:]
            yhat = np.sum(intensity_slope*X[:,~np.isnan(mhwBlock[key])].T, axis=1)
            t_stat = stats.t.isf(alpha1/2, len(years)-2)
            s = np.sqrt(np.sum((mhwBlock[key][~np.isnan(mhwBlock[key])] - yhat)**2) / (len(years)-2))
            Sxx = np.sum(X[1,:].T**2) - (np.sum(X[1,:].T)**2)/len(years) # np.var(X, axis=1)[1]
            intensity_dslope1 = t_stat * s / np.sqrt(Sxx)
            t_stat = stats.t.isf(alpha2/2, len(years)-2)
            s = np.sqrt(np.sum((mhwBlock[key][~np.isnan(mhwBlock[key])] - yhat)**2) / (len(years)-2))
            Sxx = np.sum(X[1,:].T**2) - (np.sum(X[1,:].T)**2)/len(years) # np.var(X, axis=1)[1]
            intensity_dslope2 = t_stat * s / np.sqrt(Sxx)
    if sw_write:
        outfile_train.write(', ')
        outfile_valid.write(', ')
    # Correlations between MHW metrics and average annual anomaly above threshold
    for key in keys:
        valid = ~np.isnan(mhwBlock[key]) * ~np.isnan(sst_annual_threshAnom)
        X = np.array([np.ones(years.shape), sst_annual_threshAnom])
        glm = sm.GLM(mhwBlock[key][valid*training] - offset[key], X[:,valid*training].T, family=distG[key])
        res = glm.fit()
        pred = res.predict(X.T) + offset[key]
        rho, p = stats.pearsonr(mhwBlock[key][valid*training], pred[valid*training])
        if sw_write:
            outfile_train.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
        rho, p = stats.pearsonr(mhwBlock[key][valid*validation], pred[valid*validation])
        if sw_write:
            outfile_valid.write(str(np.round(rho*100)/100) + ' (' + str(np.round(p*100)/100) + ') ')
    if sw_write:
        outfile_train.write('\n')
        outfile_valid.write('\n')
    # Comparison of annual MHW stats and prediction from annual SST stats
    plt.figure(101)
    ax = plt.subplot(3,Nst,i+1)
    plt.plot(years, mhwBlock['count'], '-', color=col_obs)
    plt.plot(years, mhwBlock['count'], 'ko', markersize=4)
    plt.plot(years, pred_frequency, 'r-')
    plt.fill_between(years, pred_frequency_CI[:,0], pred_frequency_CI[:,1], color=col_CI)
    plt.ylim(-1, 9)
    plt.xlim(1900, 2020)
    ax.set_xticklabels([])
    plt.grid()
    if i==0:
        plt.ylabel('[count]')
    else:
        ax.set_yticklabels([])
    plt.title('Frequency, ' + '{:.4}'.format(10*frequency_slope) + ', ' + '{:.4}'.format(10*pred_frequency_slope))
    ax = plt.subplot(3,Nst,i+1+Nst)
    plt.plot(years, mhwBlock['intensity_max_max'], '-', color=col_obs)
    plt.plot(years, mhwBlock['intensity_max_max'], 'ko', markersize=4)
    plt.plot(years, pred_intensity, 'r-')
    plt.plot(years, pred_intensity, 'ro', markersize=4)
    plt.fill_between(years, pred_intensity_CI[:,0], pred_intensity_CI[:,1], color=col_CI)
    plt.ylim(0.8, 8.2)
    plt.xlim(1900, 2020)
    ax.set_xticklabels([])
    plt.grid()
    if i==0:
        plt.ylabel(r'[$^\circ$C]')
    else:
        ax.set_yticklabels([])
    plt.title('Intensity, ' + '{:.4}'.format(10*intensity_slope) + ', ' + '{:.4}'.format(10*pred_intensity_slope))
    ax = plt.subplot(3,Nst,i+1+2*Nst)
    plt.plot(years, mhwBlock['duration'], '-', color=col_obs)
    plt.plot(years, mhwBlock['duration'], 'ko', markersize=4)
    plt.plot(years, pred_duration, 'r-')
    plt.plot(years, pred_duration, 'ro', markersize=4)
    plt.fill_between(years, pred_duration_CI[:,0], pred_duration_CI[:,1], color=col_CI)
    plt.ylim(0, 50)
    plt.xlim(1900, 2020)
    plt.grid()
    if i==0:
        plt.ylabel('[days]')
    else:
        ax.set_yticklabels([])
    plt.title('Duration, ' + '{:.4}'.format(10*duration_slope) + ', ' + '{:.4}'.format(10*pred_duration_slope))
    plt.show()
    #
    plt.figure(102)
    ax = plt.subplot(3,Nst,i+1)
    #plt.plot(years, frequency_line, 'r-')
    plt.plot(years, mhwBlock['count'], '-', color=col_obs)
    plt.plot(years, mhwBlock['count'], 'ko', markersize=4)
    plt.plot(years[ttPre], meanDiffMeans['count'][0]*np.ones(len(years[ttPre])), 'r-')
    plt.plot(years[ttPost], meanDiffMeans['count'][1]*np.ones(len(years[ttPost])), 'r-')
    plt.ylim(-1, 9)
    plt.xlim(1900, 2020)
    ax.set_xticklabels([])
    plt.grid()
    if i==0:
        plt.ylabel('[count]')
    else:
        ax.set_yticklabels([])
    #if np.abs(frequency_slope) > frequency_dslope1:
    #    plt.title('Frequency, ' + '{:.4}'.format(10*frequency_slope) + ' **')
    #elif np.abs(frequency_slope) > frequency_dslope2:
    #    plt.title('Frequency, ' + '{:.4}'.format(10*frequency_slope) + ' *')
    #else:
    #    plt.title('Frequency, ' + '{:.4}'.format(10*frequency_slope))
    plt.title('Frequency, ' + '{:.4}'.format(meanDiff['count']) + ', {:.4}'.format(pDiff['count']))
    ax = plt.subplot(3,Nst,i+1+Nst)
    #plt.plot(years, intensity_line, 'r-')
    plt.plot(years, mhwBlock['intensity_max_max'], '-', color=col_obs)
    plt.plot(years, mhwBlock['intensity_max_max'], 'ko', markersize=4)
    plt.plot(years[ttPre], meanDiffMeans['intensity_max_max'][0]*np.ones(len(years[ttPre])), 'r-')
    plt.plot(years[ttPost], meanDiffMeans['intensity_max_max'][1]*np.ones(len(years[ttPost])), 'r-')
    plt.ylim(0.8, 8.2)
    plt.xlim(1900, 2020)
    ax.set_xticklabels([])
    plt.grid()
    if i==0:
        plt.ylabel(r'[$^\circ$C]')
    else:
        ax.set_yticklabels([])
    #if np.abs(intensity_slope) > intensity_dslope1:
    #    plt.title('Intensity, ' + '{:.4}'.format(10*intensity_slope) + ' **')
    #elif np.abs(intensity_slope) > intensity_dslope2:
    #    plt.title('Intensity, ' + '{:.4}'.format(10*intensity_slope) + ' *')
    #else:
    #    plt.title('Intensity, ' + '{:.4}'.format(10*intensity_slope))
    plt.title('Intensity, ' + '{:.4}'.format(meanDiff['intensity_max_max']) + ', {:.4}'.format(pDiff['intensity_max_max']))
    ax = plt.subplot(3,Nst,i+1+2*Nst)
    #plt.plot(years, duration_line, 'r-')
    plt.plot(years, mhwBlock['duration'], '-', color=col_obs)
    plt.plot(years, mhwBlock['duration'], 'ko', markersize=4)
    plt.plot(years[ttPre], meanDiffMeans['duration'][0]*np.ones(len(years[ttPre])), 'r-')
    plt.plot(years[ttPost], meanDiffMeans['duration'][1]*np.ones(len(years[ttPost])), 'r-')
    plt.ylim(0, 50)
    plt.xlim(1900, 2020)
    plt.grid()
    if i==0:
        plt.ylabel('[days]')
    else:
        ax.set_yticklabels([])
    #if np.abs(duration_slope) > duration_dslope1:
    #    plt.title('Duration, ' + '{:.4}'.format(10*duration_slope) + ' **')
    #elif np.abs(duration_slope) > duration_dslope2:
    #    plt.title('Duration, ' + '{:.4}'.format(10*duration_slope) + ' *')
    #else:
    #    plt.title('Duration, ' + '{:.4}'.format(10*duration_slope))
    plt.title('Duration, ' + '{:.4}'.format(meanDiff['duration']) + ', {:.4}'.format(pDiff['duration']))
    #
    # Save MHW data?
    #
    if sw_saveData:
        outfile = 'data/longTimeseries/mhwData_' + station
        np.savez(outfile, years=years, mhwBlock=mhwBlock, pred_duration=pred_duration, pred_frequency=pred_frequency)

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/long_timeseries_orig.pdf', bbox_inches='tight', pad_inches=0.5)
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/long_timeseries_pred_orig.pdf', bbox_inches='tight', pad_inches=0.5)

if sw_write:
    outfile_train.close()
    outfile_valid.close()
