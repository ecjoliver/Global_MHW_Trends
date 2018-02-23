'''
    Plot ENSO states on time series plot
'''

# Load required modules

import numpy as np
from scipy import io
from scipy import signal
from datetime import date
from netCDF4 import Dataset

from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm

import marineHeatWaves as mhw
import trendSimAR1

import ecoliver as ecj

# Load MEI

MEI = np.genfromtxt('../../data/modes/mei.dat', skip_header=10)[:,1:].flatten()
t_mth, tmp, tmp, year_mth, month_mth, day_mth, tmp = ecj.timevector([1950, 1, 1], [2016, 12, 31])
t_mth = t_mth[day_mth==15]
dates_mth = [date.fromordinal(t_mth[tt]) for tt in range(len(t_mth))]
MEI = ecj.runavg(signal.detrend(MEI), 3)

# Find EN and LN periods
sc = 1. #0.75
EN = ecj.runavg(MEI>= sc*np.std(MEI), 3) > 0
LN = ecj.runavg(MEI<=-sc*np.std(MEI), 3) > 0

# Plot

plt.clf()
plt.fill_between(dates_mth, np.ones(MEI.shape), where=LN, color='b')
plt.fill_between(dates_mth, np.ones(MEI.shape), where=EN, color='r')
plt.xlim(date(1982,1,1).toordinal(), date(2017,1,1).toordinal())
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/timeSeries_ENSOState.pdf', bbox_inches='tight', pad_inches=0.5)


