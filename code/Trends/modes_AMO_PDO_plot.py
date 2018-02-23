'''
    Generate plots of the patterns and time series associated with the PDO and AMO
'''

import numpy as np
from scipy import signal
from netCDF4 import Dataset
from datetime import date
import ecoliver as ecj
from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm

# Load AMO and PDO spatial patterns

modefile = '/home/ecoliver/Desktop/MHW/data/modes/AMO.nc'
fileobj = Dataset(modefile, mode='r')
AMO_R1 = np.flipud(fileobj.variables['skt'][0,:,:])
lon_R1 = fileobj.variables['lon'][:]
lat_R1 = np.flipud(fileobj.variables['lat'][:])
fileobj.close()
modefile = '/home/ecoliver/Desktop/MHW/data/modes/PDO.nc'
fileobj = Dataset(modefile, mode='r')
PDO_R1 = np.flipud(fileobj.variables['skt'][0,:,:])
fileobj.close()
# Interpolate to SST grid and apply land mask
#AMO = datamask*interp.RectBivariateSpline(lon_R1, lat_R1, AMO_R1.T)(lon_map, lat_map).T
#PDO = datamask*interp.RectBivariateSpline(lon_R1, lat_R1, PDO_R1.T)(lon_map, lat_map).T

# Load time series

PDO_ts = np.genfromtxt('../../data/modes/pdo.ncdcnoaa.csv', delimiter=',', skip_header=2)[:,1]
t_PDO, tmp, tmp, year_PDO, month_PDO, day_PDO, tmp = ecj.timevector([1854, 1, 1], [2016, 12, 31])
t_PDO = t_PDO[day_PDO==15]
dates_PDO = [date.fromordinal(t_PDO[tt]) for tt in range(len(t_PDO))]
Ny = 5 # Number of years over which to smooth index
PDO_ts = ecj.runavg(PDO_ts, 12*Ny+1)
PDO_ts = PDO_ts - PDO_ts.mean()
PDO_ts = signal.detrend(PDO_ts)

AMO_ts = np.genfromtxt('../../data/modes/amon.us.long.data', skip_header=1, skip_footer=4)[:-1,1:].flatten()
t_AMO, tmp, tmp, year_AMO, month_AMO, day_AMO, tmp = ecj.timevector([1856, 1, 1], [2016, 12, 31])
t_AMO = t_AMO[day_AMO==15]
dates_AMO = [date.fromordinal(t_AMO[tt]) for tt in range(len(t_AMO))]
Ny = 5 # Number of years over which to smooth index
AMO_ts = ecj.runavg(AMO_ts, 12*Ny+1)
AMO_ts = AMO_ts - AMO_ts.mean()
AMO_ts = signal.detrend(AMO_ts)

# Plots

domain = [-65, 20, 70, 380]
domain = [-65, 0, 70, 360]
domain_draw = [-60, 60, 60, 380]
dlat = 30
dlon = 60
llon, llat = np.meshgrid(lon_R1, lat_R1)
bg_col = '0.6'
cont_col = '1.0'
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
lonproj, latproj = proj(llon, llat)

plt.clf()
plt.subplot(2,2,1, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[6,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[6,900])
plt.contourf(lonproj, latproj, PDO_R1, levels=[-1.5,-0.75,-0.5,-0.25,-0.1,0.1,0.25,0.5,0.75,1.5], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[$^\circ$C]')
plt.clim(-1, 1)
plt.title('(A) PDO Pattern')

plt.subplot(2,2,2, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[6,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[6,900])
plt.contourf(lonproj, latproj, AMO_R1, levels=[-12,-3,-2,-1,-0.5,0.5,1,2,3,12], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[$^\circ$C]')
plt.clim(-3, 3)
plt.title('(B) AMO Pattern')

plt.subplot(2,2,3)
plt.fill_between(dates_PDO, PDO_ts, where=PDO_ts>0, color=(1,0.2,0.2))
plt.fill_between(dates_PDO, PDO_ts, where=PDO_ts<0, color=(0.2,0.2,1))
plt.plot(dates_PDO, PDO_ts, 'k-')
plt.xlim(date.toordinal(date(1870,1,1)), date.toordinal(date(2020,1,1)))
plt.ylabel('PDO')
plt.title('(C) PDO Time Series')
plt.ylim(-1.5, 1.5)
#plt.grid()

plt.subplot(2,2,4)
plt.fill_between(dates_AMO, AMO_ts, where=AMO_ts>0, color=(1,0.2,0.2))
plt.fill_between(dates_AMO, AMO_ts, where=AMO_ts<0, color=(0.2,0.2,1))
plt.plot(dates_AMO, AMO_ts, 'k-')
plt.xlim(date.toordinal(date(1870,1,1)), date.toordinal(date(2020,1,1)))
plt.ylabel('AMO')
plt.title('(D) AMO Time Series')
plt.ylim(-0.35, 0.35)
#plt.grid()

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/PDO_AMO_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)



