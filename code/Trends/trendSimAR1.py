'''

  Software which uses the MHW definition
  of Hobday et al. (2015) applied to
  AR1 statistical simulated SST

'''

# Load required modules

import numpy as np
import scipy as sp
from scipy import signal
from scipy import linalg
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa as tsa


import deseason as ds
import ecoliver as ecj

import marineHeatWaves as mhw

# https://github.com/ndronen/misc/blob/master/python/ar1.py
def ar1fit(ts):
    '''
    Fits an AR(1) model to the time series data ts.  AR(1) is a
    linear model of the form
       x_t = beta * x_{t-1} + c + e_{t-1}
    where beta is the coefficient of term x_{t-1}, c is a constant
    and x_{t-1} is an i.i.d. noise term.  Here we assume that e_{t-1}
    is normally distributed. 
    Returns the tuple (beta, c, sigma).
    '''
    # Fitting AR(1) entails finding beta, c, and the noise term.
    # Beta is well approximated by the coefficient of OLS regression
    # on the lag of the data with itself.  Since the noise term is
    # assumed to be i.i.d. and normal, we must only estimate sigma,
    # the standard deviation.
 
    # Estimate beta
    x = ts[0:-1]
    y = ts[1:]
    p = sp.polyfit(x,y,1)
    beta = p[0]
 
    # Estimate c
    c = sp.mean(ts)*(1-beta)
 
    # Estimate the variance from the residuals of the OLS regression.
    yhat = sp.polyval(p,x)
    variance = sp.var(y-yhat)
    sigma = sp.sqrt(variance)
 
    return beta, c, sigma

def simulate(t, sst_obs, seas_obs, sst_trend_obs, N_ens, params=None):
    '''
    Fit AR1 model to sst time series  and simulate MHW property trends
    t is time vector, daily
    sst_trend_obs is trend in units of decade^-1
    N_ens is Number of ensembles, per trend value
    params=(a, sig_eps) specified AR1 model parameters, None by defaule
                        which forces the AR1 model to be fit to sst data
    '''

    # Variables for AR1 process (simulated SST)

    if params == None:
        a, tmp, sig_eps = ar1fit(signal.detrend(sst_obs - seas_obs))
    else:
        a = params[0]
        sig_eps = params[1]

    tau = -1/np.log(a)
    var_eps = sig_eps**2
    
    var_sst = var_eps*a/(1-a**2) # AR1 process variance
    sig_sst = np.sqrt(var_sst)
    
    # Variables for large-ensemble experiment with multiple trend values
    
    keys = ['count', 'intensity_mean', 'duration', 'intensity_max_max']
    N_keys = len(keys)
    trends = {}
    means = {}
    for key in keys:
        trends[key] = np.zeros((N_ens))
        means[key] = np.zeros((N_ens))
    
    # Loop over trend values and ensemble members, save MHW property trends
    
    T = len(t)
    for i_ens in range(N_ens): 
        # Initialize sst and noise variables
        #sst = np.zeros(T)
        #eps = sig_eps*np.random.randn(T)
        # Initial condition of sst is Normal random variable with mean 0, variance given by theoretical AR1 variance
        #sst[0] = sig_sst*np.random.randn(1)
        # Generate AR1 process
        #for tt in range(1,T):
        #    sst[tt] = a*sst[tt-1] + eps[tt]
        sst = tsa.arima_process.arma_generate_sample([1,-a], [1], T, sigma=sig_eps, burnin=100)
        # Add trend
        sst = sst + sst_trend_obs*(t-t[0])/10./365.25
        # Apply Marine Heat Wave definition
        mhws, clim = mhw.detect(t, sst)
        mhwBlock = mhw.blockAverage(t, mhws)
        mean, trend, dtrend = meanTrend_TS(mhwBlock) #mhw.meanTrend(mhwBlock)
        # Save trends
        for key in keys:
            trends[key][i_ens] = trend[key]
            means[key][i_ens] = mean[key]

    # Output results

    return tau, sig_eps, trends, means

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

