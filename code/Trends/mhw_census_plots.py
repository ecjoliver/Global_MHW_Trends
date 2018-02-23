'''

  Software which uses the MHW definition
  of Hobday et al. (2015) applied to 
  select SST time series around the globe

'''

import numpy as np
import scipy.signal as sig
from scipy import linalg
from scipy import stats
from scipy import io
from scipy import interpolate as interp
from netCDF4 import Dataset

import ecoliver as ecj

from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm


#
# Load data and make plots
#

#outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_census'
#outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_census.2015'
outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_census.2016.p95'
outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_census.2016.p98'
outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_census.2016'

data = np.load(outfile+'.npz')
lon_map = data['lon_map']
lat_map = data['lat_map']
SST_mean = data['SST_mean']
MHW_total = data['MHW_total']
MHW_cnt = data['MHW_cnt']
MHW_dur = data['MHW_dur']
MHW_max = data['MHW_max']
MHW_mean = data['MHW_mean']
MHW_cum = data['MHW_cum']
MHW_td = data['MHW_td']
MHW_tc = data['MHW_tc']
SST_tr = data['SST_tr']
MHW_cnt_tr = data['MHW_cnt_tr']
MHW_dur_tr = data['MHW_dur_tr']
MHW_max_tr = data['MHW_max_tr']
MHW_mean_tr = data['MHW_mean_tr']
MHW_cum_tr = data['MHW_cum_tr']
MHW_td_tr = data['MHW_td_tr']
MHW_tc_tr = data['MHW_tc_tr']
SST_dtr = data['SST_dtr']
MHW_cnt_dtr = data['MHW_cnt_dtr']
MHW_dur_dtr = data['MHW_dur_dtr']
MHW_max_dtr = data['MHW_max_dtr']
MHW_mean_dtr = data['MHW_mean_dtr']
MHW_cum_dtr = data['MHW_cum_dtr']
MHW_td_dtr = data['MHW_td_dtr']
MHW_tc_dtr = data['MHW_tc_dtr']
N_ts = data['N_ts']
years = data['years']
SST_ts = data['SST_ts']
MHW_cnt_ts = data['MHW_cnt_ts']
MHW_dur_ts = data['MHW_dur_ts']
MHW_max_ts = data['MHW_max_ts']
MHW_mean_ts = data['MHW_mean_ts']
MHW_cum_ts = data['MHW_cum_ts']
MHW_td_ts = data['MHW_td_ts']
MHW_tc_ts = data['MHW_tc_ts']

outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_census.2016.noENSO.leadLag.1yr.monthly.ssta'
data = np.load(outfile+'.npz')
MHW_total_noENSO = data['MHW_total']
MHW_cnt_noENSO = data['MHW_cnt']
MHW_dur_noENSO = data['MHW_dur']
MHW_max_noENSO = data['MHW_max']
MHW_mean_noENSO = data['MHW_mean']
MHW_cum_noENSO = data['MHW_cum']
MHW_td_noENSO = data['MHW_td']
SST_ts_noENSO = data['SST_ts']
MHW_cnt_ts_noENSO = data['MHW_cnt_ts']
MHW_dur_ts_noENSO = data['MHW_dur_ts']
MHW_max_ts_noENSO = data['MHW_max_ts']
MHW_mean_ts_noENSO = data['MHW_mean_ts']
MHW_cum_ts_noENSO = data['MHW_cum_ts']
MHW_td_ts_noENSO = data['MHW_td_ts']
MHW_tc_ts_noENSO = data['MHW_tc_ts']

outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_census.2016.excessTrends.lores.combine'
data = np.load(outfile+'.npz')
MHW_cnt_tr_lr = data['MHW_cnt_tr']
MHW_mean_tr_lr = data['MHW_mean_tr']
MHW_max_tr_lr = data['MHW_max_tr']
MHW_dur_tr_lr = data['MHW_dur_tr']
ar1_tau = data['ar1_tau']
ar1_sig_eps = data['ar1_sig_eps']
ar1_putrend_cnt = data['ar1_putrend_cnt']
ar1_putrend_mean = data['ar1_putrend_mean']
ar1_putrend_max = data['ar1_putrend_max']
ar1_putrend_dur = data['ar1_putrend_dur']
ar1_pltrend_cnt = data['ar1_pltrend_cnt']
ar1_pltrend_mean = data['ar1_pltrend_mean']
ar1_pltrend_max = data['ar1_pltrend_max']
ar1_pltrend_dur = data['ar1_pltrend_dur']
ar1_mean_cnt = data['ar1_mean_cnt']
ar1_mean_mean = data['ar1_mean_mean']
ar1_mean_max = data['ar1_mean_max']
ar1_mean_dur = data['ar1_mean_dur']
lon_map_lr = data['lon_map']
lat_map_lr = data['lat_map']
#lon_map_lr = lon_map[::4].copy()

SST_mean[SST_mean==0] = np.nan
MHW_cnt[MHW_cnt==0] = np.nan
MHW_dur[MHW_cnt==0] = np.nan
MHW_max[MHW_cnt==0] = np.nan
MHW_mean[MHW_cnt==0] = np.nan
MHW_cum[MHW_cnt==0] = np.nan
MHW_td[MHW_cnt==0] = np.nan
MHW_tc[MHW_cnt==0] = np.nan
MHW_cnt_noENSO[MHW_cnt_noENSO==0] = np.nan
MHW_dur_noENSO[MHW_cnt_noENSO==0] = np.nan
MHW_max_noENSO[MHW_cnt_noENSO==0] = np.nan
MHW_mean_noENSO[MHW_cnt_noENSO==0] = np.nan
MHW_cum_noENSO[MHW_cnt_noENSO==0] = np.nan
MHW_td_noENSO[MHW_cnt_noENSO==0] = np.nan
SST_tr[SST_tr==0] = np.nan
MHW_cnt_tr[MHW_cnt_tr==0] = np.nan
MHW_dur_tr[MHW_cnt_tr==0] = np.nan
MHW_max_tr[MHW_cnt_tr==0] = np.nan
MHW_mean_tr[MHW_cnt_tr==0] = np.nan
MHW_cum_tr[MHW_cnt_tr==0] = np.nan
MHW_td_tr[MHW_cnt_tr==0] = np.nan
MHW_tc_tr[MHW_cnt_tr==0] = np.nan
#MHW_var_tr[MHW_cnt_tr==0] = np.nan
#MHW_ro_tr[MHW_cnt_tr==0] = np.nan
#MHW_rd_tr[MHW_cnt_tr==0] = np.nan
SST_ts[SST_ts==0] = np.nan
MHW_dur_ts[MHW_cnt_ts==0] = np.nan
MHW_max_ts[MHW_cnt_ts==0] = np.nan
MHW_mean_ts[MHW_cnt_ts==0] = np.nan
MHW_cum_ts[MHW_cnt_ts==0] = np.nan
MHW_td_ts[MHW_cnt_ts==0] = np.nan
MHW_tc_ts[MHW_cnt_ts==0] = np.nan
SST_ts_noENSO[SST_ts_noENSO==0] = np.nan
MHW_dur_ts_noENSO[MHW_cnt_ts_noENSO==0] = np.nan
MHW_max_ts_noENSO[MHW_cnt_ts_noENSO==0] = np.nan
MHW_mean_ts_noENSO[MHW_cnt_ts_noENSO==0] = np.nan
MHW_cum_ts_noENSO[MHW_cnt_ts_noENSO==0] = np.nan
MHW_td_ts_noENSO[MHW_cnt_ts_noENSO==0] = np.nan

MHW_cnt_tr_lr[MHW_cnt_tr_lr==0] = np.nan
MHW_mean_tr_lr[MHW_mean_tr_lr==0] = np.nan
MHW_max_tr_lr[MHW_max_tr_lr==0] = np.nan
MHW_dur_tr_lr[MHW_dur_tr_lr==0] = np.nan

#np.savetxt('MHW_totalDays_mean.dat', MHW_td, delimiter=',')
#np.savetxt('MHW_totalDays_trend.dat', MHW_td_tr, delimiter=',')
#np.savetxt('MHW_totalCum_mean.dat', MHW_tc, delimiter=',')
#np.savetxt('MHW_totalCum_trend.dat', MHW_tc_tr, delimiter=',')
#np.savetxt('lon.dat', lon_map, delimiter=',')
#np.savetxt('lat.dat', lat_map, delimiter=',')

# Load iceMean map for making mask
matobj = io.loadmat('/home/ecoliver/Desktop/data/MHWs/Trends/NOAAOISST_iceMean.mat')
ice_mean = matobj['ice_mean']
ice_longestRun = matobj['ice_longestRun']
# Make data mask based on land and ice
datamask = np.ones(SST_mean.shape)
datamask[np.isnan(SST_mean)] = np.nan
#datamask[~np.isnan(ice)] = np.nan # mask where there has been any ice
datamask[ice_longestRun>=6.] = np.nan # mask where ice had runs of 6 days or longer
datamask[lat_map<=-65.,:] = np.nan
datamask_ts = np.swapaxes(np.swapaxes(np.tile(datamask, (len(years),1,1)), 0, 1), 1, 2)
dl = len(lon_map)/len(lon_map_lr)
datamask_lr = datamask[::dl,::dl]

# Load AMO and PDO patterns for correlation analysis
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
AMO = datamask*interp.RectBivariateSpline(lon_R1, lat_R1, AMO_R1.T)(lon_map, lat_map).T
PDO = datamask*interp.RectBivariateSpline(lon_R1, lat_R1, PDO_R1.T)(lon_map, lat_map).T

# Load proxy means for correlation analysis
proxyfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_proxies_GLM.1900.2016.HadISST'
data = np.load(proxyfile+'.npz')
lon_had = data['lon_data']
lat_had = np.flipud(data['lat_data'])
MHW_cnt_proxy = np.flipud(data['MHW_m'].item()['count']['threshCount'])
MHW_dur_proxy = np.flipud(data['MHW_m'].item()['duration']['maxAnom'])
MHW_td_proxy = np.flipud(data['MHW_m'].item()['duration']['maxAnom'])
# Shift maps over 180 deg.
lon_had = np.append(lon_had[180:], lon_had[:180]+360)
MHW_cnt_proxy = np.append(MHW_cnt_proxy[:,180:], MHW_cnt_proxy[:,:180], axis=1)
MHW_dur_proxy = np.append(MHW_dur_proxy[:,180:], MHW_dur_proxy[:,:180], axis=1)
MHW_td_proxy = np.append(MHW_td_proxy[:,180:], MHW_td_proxy[:,:180], axis=1)
# Interpolate to SST grid and apply land mask
MHW_cnt_proxy[np.isnan(MHW_cnt_proxy)] = 0.
MHW_dur_proxy[np.isnan(MHW_dur_proxy)] = 0.
MHW_td_proxy[np.isnan(MHW_td_proxy)] = 0.
MHW_cnt_proxy = interp.RectBivariateSpline(lon_had, lat_had, MHW_cnt_proxy.T)(lon_map, lat_map).T
MHW_dur_proxy = interp.RectBivariateSpline(lon_had, lat_had, MHW_dur_proxy.T)(lon_map, lat_map).T
MHW_td_proxy = interp.RectBivariateSpline(lon_had, lat_had, MHW_td_proxy.T)(lon_map, lat_map).T

# Re-map to run 20E to 380E
i_20E = np.where(lon_map>20)[0][0]
lon_map = np.append(lon_map[i_20E:], lon_map[:i_20E]+360)
SST_mean = np.append(SST_mean[:,i_20E:], SST_mean[:,:i_20E], axis=1)
ice_mean = np.append(ice_mean[:,i_20E:], ice_mean[:,:i_20E], axis=1)
ice_longestRun = np.append(ice_longestRun[:,i_20E:], ice_longestRun[:,:i_20E], axis=1)
datamask = np.append(datamask[:,i_20E:], datamask[:,:i_20E], axis=1)
AMO = np.append(AMO[:,i_20E:], AMO[:,:i_20E], axis=1)
PDO = np.append(PDO[:,i_20E:], PDO[:,:i_20E], axis=1)
MHW_cnt_proxy = np.append(MHW_cnt_proxy[:,i_20E:], MHW_cnt_proxy[:,:i_20E], axis=1)
MHW_dur_proxy = np.append(MHW_dur_proxy[:,i_20E:], MHW_dur_proxy[:,:i_20E], axis=1)
MHW_td_proxy = np.append(MHW_td_proxy[:,i_20E:], MHW_td_proxy[:,:i_20E], axis=1)
MHW_total = np.append(MHW_total[:,i_20E:], MHW_total[:,:i_20E], axis=1)
MHW_cnt = np.append(MHW_cnt[:,i_20E:], MHW_cnt[:,:i_20E], axis=1)
MHW_dur = np.append(MHW_dur[:,i_20E:], MHW_dur[:,:i_20E], axis=1)
MHW_max = np.append(MHW_max[:,i_20E:], MHW_max[:,:i_20E], axis=1)
MHW_mean = np.append(MHW_mean[:,i_20E:], MHW_mean[:,:i_20E], axis=1)
MHW_cum = np.append(MHW_cum[:,i_20E:], MHW_cum[:,:i_20E], axis=1)
MHW_td = np.append(MHW_td[:,i_20E:], MHW_td[:,:i_20E], axis=1)
MHW_tc = np.append(MHW_tc[:,i_20E:], MHW_tc[:,:i_20E], axis=1)
MHW_total_noENSO = np.append(MHW_total_noENSO[:,i_20E:], MHW_total_noENSO[:,:i_20E], axis=1)
MHW_cnt_noENSO = np.append(MHW_cnt_noENSO[:,i_20E:], MHW_cnt_noENSO[:,:i_20E], axis=1)
MHW_dur_noENSO = np.append(MHW_dur_noENSO[:,i_20E:], MHW_dur_noENSO[:,:i_20E], axis=1)
MHW_max_noENSO = np.append(MHW_max_noENSO[:,i_20E:], MHW_max_noENSO[:,:i_20E], axis=1)
MHW_mean_noENSO = np.append(MHW_mean_noENSO[:,i_20E:], MHW_mean_noENSO[:,:i_20E], axis=1)
MHW_cum_noENSO = np.append(MHW_cum_noENSO[:,i_20E:], MHW_cum_noENSO[:,:i_20E], axis=1)
MHW_td_noENSO = np.append(MHW_td_noENSO[:,i_20E:], MHW_td_noENSO[:,:i_20E], axis=1)
SST_tr = np.append(SST_tr[:,i_20E:], SST_tr[:,:i_20E], axis=1)
MHW_cnt_tr = np.append(MHW_cnt_tr[:,i_20E:], MHW_cnt_tr[:,:i_20E], axis=1)
MHW_dur_tr = np.append(MHW_dur_tr[:,i_20E:], MHW_dur_tr[:,:i_20E], axis=1)
MHW_max_tr = np.append(MHW_max_tr[:,i_20E:], MHW_max_tr[:,:i_20E], axis=1)
MHW_mean_tr = np.append(MHW_mean_tr[:,i_20E:], MHW_mean_tr[:,:i_20E], axis=1)
MHW_cum_tr = np.append(MHW_cum_tr[:,i_20E:], MHW_cum_tr[:,:i_20E], axis=1)
MHW_td_tr = np.append(MHW_td_tr[:,i_20E:], MHW_td_tr[:,:i_20E], axis=1)
MHW_tc_tr = np.append(MHW_tc_tr[:,i_20E:], MHW_tc_tr[:,:i_20E], axis=1)
SST_dtr = np.append(SST_dtr[:,i_20E:], SST_dtr[:,:i_20E], axis=1)
MHW_cnt_dtr = np.append(MHW_cnt_dtr[:,i_20E:], MHW_cnt_dtr[:,:i_20E], axis=1)
MHW_dur_dtr = np.append(MHW_dur_dtr[:,i_20E:], MHW_dur_dtr[:,:i_20E], axis=1)
MHW_max_dtr = np.append(MHW_max_dtr[:,i_20E:], MHW_max_dtr[:,:i_20E], axis=1)
MHW_mean_dtr = np.append(MHW_mean_dtr[:,i_20E:], MHW_mean_dtr[:,:i_20E], axis=1)
MHW_cum_dtr = np.append(MHW_cum_dtr[:,i_20E:], MHW_cum_dtr[:,:i_20E], axis=1)
MHW_td_dtr = np.append(MHW_td_dtr[:,i_20E:], MHW_td_dtr[:,:i_20E], axis=1)
MHW_tc_dtr = np.append(MHW_tc_dtr[:,i_20E:], MHW_tc_dtr[:,:i_20E], axis=1)
datamask_ts = np.append(datamask_ts[:,i_20E:,:], datamask_ts[:,:i_20E,:], axis=1)
N_ts = np.append(N_ts[:,i_20E:,:], N_ts[:,:i_20E,:], axis=1)
SST_ts = np.append(SST_ts[:,i_20E:,:], SST_ts[:,:i_20E,:], axis=1)
MHW_cnt_ts = np.append(MHW_cnt_ts[:,i_20E:,:], MHW_cnt_ts[:,:i_20E,:], axis=1)
MHW_dur_ts = np.append(MHW_dur_ts[:,i_20E:,:], MHW_dur_ts[:,:i_20E,:], axis=1)
MHW_max_ts = np.append(MHW_max_ts[:,i_20E:,:], MHW_max_ts[:,:i_20E,:], axis=1)
MHW_mean_ts = np.append(MHW_mean_ts[:,i_20E:,:], MHW_mean_ts[:,:i_20E,:], axis=1)
MHW_cum_ts = np.append(MHW_cum_ts[:,i_20E:,:], MHW_cum_ts[:,:i_20E,:], axis=1)
MHW_td_ts = np.append(MHW_td_ts[:,i_20E:,:], MHW_td_ts[:,:i_20E,:], axis=1)
MHW_tc_ts = np.append(MHW_tc_ts[:,i_20E:,:], MHW_tc_ts[:,:i_20E,:], axis=1)
SST_ts_noENSO = np.append(SST_ts_noENSO[:,i_20E:,:], SST_ts_noENSO[:,:i_20E,:], axis=1)
MHW_cnt_ts_noENSO = np.append(MHW_cnt_ts_noENSO[:,i_20E:,:], MHW_cnt_ts_noENSO[:,:i_20E,:], axis=1)
MHW_dur_ts_noENSO = np.append(MHW_dur_ts_noENSO[:,i_20E:,:], MHW_dur_ts_noENSO[:,:i_20E,:], axis=1)
MHW_max_ts_noENSO = np.append(MHW_max_ts_noENSO[:,i_20E:,:], MHW_max_ts_noENSO[:,:i_20E,:], axis=1)
MHW_mean_ts_noENSO = np.append(MHW_mean_ts_noENSO[:,i_20E:,:], MHW_mean_ts_noENSO[:,:i_20E,:], axis=1)
MHW_cum_ts_noENSO = np.append(MHW_cum_ts_noENSO[:,i_20E:,:], MHW_cum_ts_noENSO[:,:i_20E,:], axis=1)
MHW_td_ts_noENSO = np.append(MHW_td_ts_noENSO[:,i_20E:,:], MHW_td_ts_noENSO[:,:i_20E,:], axis=1)


i_20E = np.where(lon_map_lr>20)[0][0]
lon_map_lr = np.append(lon_map_lr[i_20E:], lon_map_lr[:i_20E]+360)
datamask_lr = np.append(datamask_lr[:,i_20E:], datamask_lr[:,:i_20E], axis=1)
ar1_tau = np.append(ar1_tau[:,i_20E:], ar1_tau[:,:i_20E], axis=1)
ar1_sig_eps = np.append(ar1_sig_eps[:,i_20E:], ar1_sig_eps[:,:i_20E], axis=1)
ar1_putrend_cnt = np.append(ar1_putrend_cnt[:,i_20E:], ar1_putrend_cnt[:,:i_20E], axis=1)
ar1_putrend_mean = np.append(ar1_putrend_mean[:,i_20E:], ar1_putrend_mean[:,:i_20E], axis=1)
ar1_putrend_max = np.append(ar1_putrend_max[:,i_20E:], ar1_putrend_max[:,:i_20E], axis=1)
ar1_putrend_dur = np.append(ar1_putrend_dur[:,i_20E:], ar1_putrend_dur[:,:i_20E], axis=1)
ar1_pltrend_cnt = np.append(ar1_pltrend_cnt[:,i_20E:], ar1_pltrend_cnt[:,:i_20E], axis=1)
ar1_pltrend_mean = np.append(ar1_pltrend_mean[:,i_20E:], ar1_pltrend_mean[:,:i_20E], axis=1)
ar1_pltrend_max = np.append(ar1_pltrend_max[:,i_20E:], ar1_pltrend_max[:,:i_20E], axis=1)
ar1_pltrend_dur = np.append(ar1_pltrend_dur[:,i_20E:], ar1_pltrend_dur[:,:i_20E], axis=1)
ar1_mean_cnt = np.append(ar1_mean_cnt[:,i_20E:], ar1_mean_cnt[:,:i_20E], axis=1)
ar1_mean_mean = np.append(ar1_mean_mean[:,i_20E:], ar1_mean_mean[:,:i_20E], axis=1)
ar1_mean_max = np.append(ar1_mean_max[:,i_20E:], ar1_mean_max[:,:i_20E], axis=1)
ar1_mean_dur = np.append(ar1_mean_dur[:,i_20E:], ar1_mean_dur[:,:i_20E], axis=1)
MHW_cnt_tr_lr = np.append(MHW_cnt_tr_lr[:,i_20E:], MHW_cnt_tr_lr[:,:i_20E], axis=1)
MHW_mean_tr_lr = np.append(MHW_mean_tr_lr[:,i_20E:], MHW_mean_tr_lr[:,:i_20E], axis=1)
MHW_max_tr_lr = np.append(MHW_max_tr_lr[:,i_20E:], MHW_max_tr_lr[:,:i_20E], axis=1)
MHW_dur_tr_lr = np.append(MHW_dur_tr_lr[:,i_20E:], MHW_dur_tr_lr[:,:i_20E], axis=1)

# Apply datamask to all fields
SST_mean = datamask*SST_mean
MHW_total = datamask*MHW_total
MHW_cnt = datamask*MHW_cnt
MHW_dur = datamask*MHW_dur
MHW_max = datamask*MHW_max
MHW_mean = datamask*MHW_mean
MHW_cum = datamask*MHW_cum
MHW_td = datamask*MHW_td
MHW_tc = datamask*MHW_tc
MHW_total_noENSO = datamask*MHW_total_noENSO
MHW_cnt_noENSO = datamask*MHW_cnt_noENSO
MHW_dur_noENSO = datamask*MHW_dur_noENSO
MHW_max_noENSO = datamask*MHW_max_noENSO
MHW_mean_noENSO = datamask*MHW_mean_noENSO
MHW_cum_noENSO = datamask*MHW_cum_noENSO
MHW_td_noENSO = datamask*MHW_td_noENSO
SST_tr = datamask*SST_tr
MHW_cnt_tr = datamask*MHW_cnt_tr
MHW_dur_tr = datamask*MHW_dur_tr
MHW_max_tr = datamask*MHW_max_tr
MHW_mean_tr = datamask*MHW_mean_tr
MHW_cum_tr = datamask*MHW_cum_tr
MHW_td_tr = datamask*MHW_td_tr
MHW_tc_tr = datamask*MHW_tc_tr
SST_dtr = datamask*SST_dtr
MHW_cnt_dtr = datamask*MHW_cnt_dtr
MHW_dur_dtr = datamask*MHW_dur_dtr
MHW_max_dtr = datamask*MHW_max_dtr
MHW_mean_dtr = datamask*MHW_mean_dtr
MHW_cum_dtr = datamask*MHW_cum_dtr
MHW_td_dtr = datamask*MHW_td_dtr
MHW_tc_dtr = datamask*MHW_tc_dtr
N_ts = datamask_ts*N_ts
N_ts[np.isnan(N_ts)] = 0.
SST_ts = datamask_ts*SST_ts
N_ts[np.isnan(N_ts)] = 0.
MHW_cnt_ts = datamask_ts*MHW_cnt_ts
MHW_cnt_ts[np.isnan(MHW_cnt_ts)] = 0.
MHW_dur_ts = datamask_ts*MHW_dur_ts
MHW_max_ts = datamask_ts*MHW_max_ts
MHW_mean_ts = datamask_ts*MHW_mean_ts
MHW_cum_ts = datamask_ts*MHW_cum_ts
MHW_td_ts = datamask_ts*MHW_td_ts
MHW_tc_ts = datamask_ts*MHW_tc_ts
MHW_cnt_ts_noENSO = datamask_ts*MHW_cnt_ts_noENSO
MHW_cnt_ts_noENSO[np.isnan(MHW_cnt_ts_noENSO)] = 0.
MHW_dur_ts_noENSO = datamask_ts*MHW_dur_ts_noENSO
MHW_max_ts_noENSO = datamask_ts*MHW_max_ts_noENSO
MHW_mean_ts_noENSO = datamask_ts*MHW_mean_ts_noENSO
MHW_cum_ts_noENSO = datamask_ts*MHW_cum_ts_noENSO
MHW_td_ts_noENSO = datamask_ts*MHW_td_ts_noENSO

MHW_cnt_proxy = datamask*MHW_cnt_proxy
MHW_dur_proxy = datamask*MHW_dur_proxy
MHW_td_proxy = datamask*MHW_td_proxy

ar1_tau = datamask_lr*ar1_tau
ar1_sig_eps = datamask_lr*ar1_sig_eps
ar1_putrend_cnt = datamask_lr*ar1_putrend_cnt
ar1_putrend_mean = datamask_lr*datamask_lr*ar1_putrend_mean
ar1_putrend_max = datamask_lr*datamask_lr*ar1_putrend_max
ar1_putrend_dur = datamask_lr*datamask_lr*ar1_putrend_dur
ar1_pltrend_cnt = datamask_lr*ar1_pltrend_cnt
ar1_pltrend_mean = datamask_lr*datamask_lr*ar1_pltrend_mean
ar1_pltrend_max = datamask_lr*datamask_lr*ar1_pltrend_max
ar1_pltrend_dur = datamask_lr*datamask_lr*ar1_pltrend_dur
ar1_mean_cnt = datamask_lr*ar1_mean_cnt
ar1_mean_mean = datamask_lr*datamask_lr*ar1_mean_mean
ar1_mean_max = datamask_lr*datamask_lr*ar1_mean_max
ar1_mean_dur = datamask_lr*datamask_lr*ar1_mean_dur
MHW_cnt_tr_lr = datamask_lr*MHW_cnt_tr_lr
MHW_mean_tr_lr = datamask_lr*datamask_lr*MHW_mean_tr_lr
MHW_max_tr_lr = datamask_lr*datamask_lr*MHW_max_tr_lr
MHW_dur_tr_lr = datamask_lr*datamask_lr*MHW_dur_tr_lr

# Maps

domain = [-65, 20, 70, 380]
domain_draw = [-60, 20, 60, 380]
domain_draw = [-60, 60, 60, 380]
dlat = 30
dlon = 60
llon, llat = np.meshgrid(lon_map, lat_map)
llon_lr, llat_lr = np.meshgrid(lon_map_lr, lat_map_lr)
bg_col = '0.6'
cont_col = '1.0'

plt.clf()
plt.subplot(2,1,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
#plt.contourf(lonproj, latproj, MHW_cnt, levels=np.arange(0.5,3.5+0.5,0.5), cmap=plt.cm.afmhot_r)
plt.contourf(lonproj, latproj, MHW_cnt, levels=np.arange(0.5,3.5+0.5,0.5), cmap=plt.cm.YlOrRd)
plt.colorbar()
plt.clim(0.75,3.75)
plt.title('Count')
plt.subplot(2,1,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cnt_tr*10, levels=[-3.5,-2,-1,-0.5,0.5,1,2,3.5], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('Count trend')

# plt.savefig('figures/Census/MHW_cnt.png', bbox_inches='tight', pad_inches=0.5)

plt.clf()
plt.subplot(2,1,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_dur, levels=[5,10,15,20,30,60], cmap=plt.cm.gist_heat_r)
plt.colorbar()
plt.clim(8,80)
plt.title('Duration')
plt.subplot(2,1,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_dur_tr*10, levels=[-50,-20,-10,-2.5,2.5,10,20,50], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.clim(-18,18)
plt.title('Duration trend')

# plt.savefig('figures/Census/MHW_dur.png', bbox_inches='tight', pad_inches=0.5)

plt.clf()
plt.subplot(2,1,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_max, levels=np.arange(0,5+0.5,0.5), cmap=plt.cm.gist_heat_r)
plt.colorbar()
plt.clim(0.5,5.75)
plt.title('Intensity max')
plt.subplot(2,1,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_max_tr*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.clim(-0.5,0.5)
plt.title('Intensity max trend')

# plt.savefig('figures/Census/MHW_max.png', bbox_inches='tight', pad_inches=0.5)

plt.clf()
plt.subplot(2,1,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_mean, levels=np.arange(0,5+0.5,0.5), cmap=plt.cm.gist_heat_r)
plt.colorbar()
plt.clim(0.5,5.75)
plt.title('Intensity mean')
plt.subplot(2,1,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_mean_tr*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.clim(-0.5,0.5)
plt.title('Intensity mean trend')

# plt.savefig('figures/Census/MHW_mean.png', bbox_inches='tight', pad_inches=0.5)

plt.clf()
plt.subplot(2,1,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cum, levels=[0,10,20,30,40,80,160], cmap=plt.cm.gist_heat_r)
plt.colorbar()
plt.clim(10,80)
plt.title('Intensity cumulative')
plt.subplot(2,1,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cum_tr*10, levels=[-100,-20,-10,-2.5,2.5,10,20,100], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.clim(-22,22)
plt.title('Intensity cumulative trend')

# plt.savefig('figures/Census/MHW_cum.png', bbox_inches='tight', pad_inches=0.5)

plt.clf()
plt.subplot(2,1,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_td, levels=[10,15,20,25,30,35,40,45], cmap=plt.cm.gist_heat_r)
plt.colorbar()
plt.clim(15,70)
plt.title('Annual total MHW days [days]')
plt.subplot(2,1,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_td_tr*10, levels=[-200,-30,-20,-10,-5,5,10,20,30,200], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.clim(-30,30)
plt.title('Trend [per decade]')

# plt.savefig('figures/Census/MHW_totDays.png', bbox_inches='tight', pad_inches=0.5)

plt.clf()
plt.subplot(2,1,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_tc, levels=[0,10,20,30,40,60,80,100,200], cmap=plt.cm.gist_heat_r)
plt.colorbar()
plt.clim(10,225)
plt.title('Annual total cumulative intensity [deg.C-day]')
plt.subplot(2,1,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_tc_tr*10, levels=[-200,-40,-20,-10,-5,5,10,20,40,200], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.clim(-45,45)
plt.title('Trend [per decade]')

# plt.savefig('figures/Census/MHW_totCum.png', bbox_inches='tight', pad_inches=0.5)

# Proportion of ocean surface w/ positive trends
1.*np.sum(MHW_cnt_tr >= 0) / np.sum(~np.isnan(MHW_cnt_tr))
1.*np.sum(MHW_dur_tr >= 0) / np.sum(~np.isnan(MHW_dur_tr))
1.*np.sum(MHW_mean_tr >= 0) / np.sum(~np.isnan(MHW_mean_tr))
1.*np.sum(MHW_max_tr >= 0) / np.sum(~np.isnan(MHW_max_tr))
1.*np.sum(MHW_td_tr >= 0) / np.sum(~np.isnan(MHW_td_tr))

#
# Time series
#

# take annual means, detrend
SST_ts_glob = np.zeros(SST_ts.shape[2])
MHW_cnt_ts_glob = np.zeros(MHW_cnt_ts.shape[2])
MHW_dur_ts_glob = np.zeros(MHW_dur_ts.shape[2])
MHW_max_ts_glob = np.zeros(MHW_max_ts.shape[2])
MHW_mean_ts_glob = np.zeros(MHW_mean_ts.shape[2])
MHW_cum_ts_glob = np.zeros(MHW_cum_ts.shape[2])
MHW_td_ts_glob = np.zeros(MHW_cum_ts.shape[2])
MHW_tc_ts_glob = np.zeros(MHW_cum_ts.shape[2])
SST_ts_notrop = np.zeros(SST_ts.shape[2])
MHW_cnt_ts_notrop = np.zeros(MHW_cnt_ts.shape[2])
MHW_dur_ts_notrop = np.zeros(MHW_dur_ts.shape[2])
MHW_max_ts_notrop = np.zeros(MHW_max_ts.shape[2])
MHW_mean_ts_notrop = np.zeros(MHW_mean_ts.shape[2])
MHW_cum_ts_notrop = np.zeros(MHW_cum_ts.shape[2])
MHW_td_ts_notrop = np.zeros(MHW_cum_ts.shape[2])
MHW_tc_ts_notrop = np.zeros(MHW_cum_ts.shape[2])
SST_ts_woENSO = np.zeros(SST_ts.shape[2])
MHW_cnt_ts_woENSO = np.zeros(MHW_cnt_ts.shape[2])
MHW_dur_ts_woENSO = np.zeros(MHW_dur_ts.shape[2])
MHW_max_ts_woENSO = np.zeros(MHW_max_ts.shape[2])
MHW_mean_ts_woENSO = np.zeros(MHW_mean_ts.shape[2])
MHW_cum_ts_woENSO = np.zeros(MHW_cum_ts.shape[2])
MHW_td_ts_woENSO = np.zeros(MHW_cum_ts.shape[2])
MHW_tc_ts_woENSO = np.zeros(MHW_cum_ts.shape[2])

# Cosine scaling
scaling = np.cos(llat*np.pi/180)
# Sum / average over globe
for tt in range(MHW_cnt_ts.shape[2]):
    # Create mask
    mask = np.ones(llat.shape)
    mask[np.isnan(SST_ts[:,:,tt])] = np.nan
    mask_notrop = 1.*mask
    mask_notrop[np.abs(llat)<=20] = np.nan
    # SST
    SST_ts_glob[tt] = np.average(SST_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    SST_ts_woENSO[tt] = np.average(SST_ts_noENSO[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    SST_ts_notrop[tt] = np.average(SST_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Create mask
    mask = np.ones(llat.shape)
    mask[np.isnan(MHW_dur_ts[:,:,tt])] = np.nan
    mask_notrop = 1.*mask
    mask_notrop[np.abs(llat)<=20] = np.nan
    mask_noENSO = np.ones(llat.shape)
    mask_noENSO[np.isnan(MHW_dur_ts_noENSO[:,:,tt])] = np.nan
    # Count
    MHW_cnt_ts_glob[tt] = np.average(MHW_cnt_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    MHW_cnt_ts_woENSO[tt] = np.average(MHW_cnt_ts_noENSO[:,:,tt][~np.isnan(mask_noENSO)], weights=scaling[~np.isnan(mask_noENSO)])
    MHW_cnt_ts_notrop[tt] = np.average(MHW_cnt_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Duration
    MHW_dur_ts_glob[tt] = np.average(MHW_dur_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    MHW_dur_ts_woENSO[tt] = np.average(MHW_dur_ts_noENSO[:,:,tt][~np.isnan(mask_noENSO)], weights=scaling[~np.isnan(mask_noENSO)])
    MHW_dur_ts_notrop[tt] = np.average(MHW_dur_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Maximum intensity
    MHW_max_ts_glob[tt] = np.average(MHW_max_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    MHW_max_ts_woENSO[tt] = np.average(MHW_max_ts_noENSO[:,:,tt][~np.isnan(mask_noENSO)], weights=scaling[~np.isnan(mask_noENSO)])
    MHW_max_ts_notrop[tt] = np.average(MHW_max_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Mean intensity
    MHW_mean_ts_glob[tt] = np.average(MHW_mean_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    MHW_mean_ts_woENSO[tt] = np.average(MHW_mean_ts_noENSO[:,:,tt][~np.isnan(mask_noENSO)], weights=scaling[~np.isnan(mask_noENSO)])
    MHW_mean_ts_notrop[tt] = np.average(MHW_mean_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Cumulative intensity
    MHW_cum_ts_glob[tt] = np.average(MHW_cum_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    MHW_cum_ts_woENSO[tt] = np.average(MHW_cum_ts_noENSO[:,:,tt][~np.isnan(mask_noENSO)], weights=scaling[~np.isnan(mask_noENSO)])
    MHW_cum_ts_notrop[tt] = np.average(MHW_cum_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Total MHW days
    MHW_td_ts_glob[tt] = np.average(MHW_td_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    MHW_td_ts_woENSO[tt] = np.average(MHW_td_ts_noENSO[:,:,tt][~np.isnan(mask_noENSO)], weights=scaling[~np.isnan(mask_noENSO)])
    MHW_td_ts_notrop[tt] = np.average(MHW_td_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])
    # Total MHW cumulative intensity
    MHW_tc_ts_glob[tt] = np.average(MHW_tc_ts[:,:,tt][~np.isnan(mask)], weights=scaling[~np.isnan(mask)])
    #MHW_tc_ts_woENSO[tt] = np.average(MHW_tc_ts_noENSO[:,:,tt][~np.isnan(mask_noENSO)], weights=scaling[~np.isnan(mask_noENSO)])
    MHW_tc_ts_notrop[tt] = np.average(MHW_tc_ts[:,:,tt][~np.isnan(mask_notrop)], weights=scaling[~np.isnan(mask_notrop)])

# save
# outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_census.2016_ts'
# outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_census.2016_ts.p95'
# outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_census.2016_ts.p98'
# np.savez(outfile, years=years, MHW_cnt_ts_glob=MHW_cnt_ts_glob, MHW_cnt_ts_woENSO=MHW_cnt_ts_woENSO, MHW_dur_ts_glob=MHW_dur_ts_glob, MHW_dur_ts_woENSO=MHW_dur_ts_woENSO, MHW_max_ts_glob=MHW_max_ts_glob, MHW_max_ts_woENSO=MHW_max_ts_woENSO, MHW_mean_ts_glob=MHW_mean_ts_glob, MHW_mean_ts_woENSO=MHW_mean_ts_woENSO, MHW_cum_ts_glob=MHW_cum_ts_glob, MHW_cum_ts_woENSO=MHW_cum_ts_woENSO, MHW_td_ts_glob=MHW_td_ts_glob, MHW_td_ts_woENSO=MHW_td_ts_woENSO, MHW_tc_ts_glob=MHW_tc_ts_glob, MHW_tc_ts_woENSO=MHW_tc_ts_woENSO)

plt.clf()
plt.subplot(2,2,2)
plt.plot(years, MHW_cnt_ts_glob, 'k-')
plt.plot(years, MHW_cnt_ts_glob, 'k.')
plt.plot(years, MHW_cnt_ts_woENSO, 'k--')
plt.plot(years, MHW_cnt_ts_woENSO, 'k.')
plt.xlim(years.min()-1, years.max()+1)
plt.grid()
plt.ylabel('[count]')
plt.title('Frequency')
plt.subplot(2,2,1)
plt.plot(years, MHW_dur_ts_glob, 'k-')
plt.plot(years, MHW_dur_ts_glob, 'k.')
plt.plot(years, MHW_dur_ts_woENSO, 'k--')
plt.plot(years, MHW_dur_ts_woENSO, 'k.')
plt.xlim(years.min()-1, years.max()+1)
plt.grid()
plt.ylabel('[days]')
plt.title('Duration')
plt.subplot(2,2,4)
plt.plot(years, MHW_max_ts_glob, 'r-')
plt.plot(years, MHW_max_ts_glob, 'r.')
plt.plot(years, MHW_max_ts_woENSO, 'r--')
plt.plot(years, MHW_max_ts_woENSO, 'r.')
plt.plot(years, MHW_mean_ts_glob, 'k-')
plt.plot(years, MHW_mean_ts_glob, 'k.')
plt.plot(years, MHW_mean_ts_woENSO, 'k--')
plt.plot(years, MHW_mean_ts_woENSO, 'k.')
plt.xlim(years.min()-1, years.max()+1)
plt.grid()
plt.ylabel(r'[$^\circ$C]')
plt.title('Maximum (red) and mean (black) intensity')
plt.subplot(2,2,3)
plt.plot(years, MHW_cum_ts_glob, 'k-')
plt.plot(years, MHW_cum_ts_glob, 'k.')
plt.plot(years, MHW_cum_ts_woENSO, 'k--')
plt.plot(years, MHW_cum_ts_woENSO, 'k.')
plt.xlim(years.min()-1, years.max()+1)
plt.grid()
plt.ylabel(r'[$^\circ$C$\times$days]')
plt.title('Cumulative intensity')

# plt.savefig('figures/MHW_ts.png', bbox_inches='tight', pad_inches=0.5)

plt.figure()
plt.subplot(2,1,1)
plt.plot(years, MHW_td_ts_glob, 'k-')
plt.plot(years, MHW_td_ts_woENSO, 'k--')
plt.plot(years, MHW_td_ts_glob, 'k.')
plt.plot(years, MHW_td_ts_woENSO, 'k.')
plt.xlim(years.min()-1, years.max()+1)
plt.grid()
plt.title('Annual total MHW days')
plt.ylabel('[days]')
plt.legend(['Global average', 'Excluding tropics (20S-20N)'], 4)
plt.subplot(2,1,2)
plt.plot(years, MHW_tc_ts_glob, 'k-')
plt.plot(years, MHW_tc_ts_woENSO, 'k--')
plt.plot(years, MHW_tc_ts_glob, 'k.')
plt.plot(years, MHW_tc_ts_woENSO, 'k.')
plt.xlim(years.min()-1, years.max()+1)
plt.grid()
plt.title('Annual total cumulative intensity')
plt.ylabel('[deg C-day]')
plt.legend(['Global average', 'Excluding tropics (20S-20N)'], 4)

# plt.savefig('figures/MHW_ts_tot.png', bbox_inches='tight', pad_inches=0.5)

plt.figure(figsize=(6,3))
plt.clf()
plt.plot(years, MHW_td_ts_glob, 'k-')
plt.plot(years, MHW_td_ts_woENSO, 'r-')
plt.xlim(years.min()-1, years.max()+1)
#plt.grid()
plt.ylabel('Annual MHW days')
plt.legend(['Global average', 'Excluding ENSO'], loc='upper left', fontsize=12)
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_ts_totDays_orig.pdf', bbox_inches='tight', pad_inches=0.5)

# Calculate trends
#def trend(x, y, alpha):
#    '''
#    Calculates the trend of y given the linear
#    independent variable x. Outputs the mean,
#    trend, and alpha-level (e.g., 0.05 for 95%)
#    confidence limit on the trend.
#    '''
#    X = np.array([np.ones(x.shape), x-x.mean()])
#    beta = linalg.lstsq(X.T, y)[0]
#    yhat = np.sum(beta*X.T, axis=1)
#    t_stat = stats.t.isf(alpha/2, len(years)-2)
#    s = np.sqrt(np.sum((y - yhat)**2) / (len(years)-2))
#    Sxx = np.sum(X[:,1]**2) - (np.sum(X[:,1])**2)/len(years) # np.var(X, axis=1)[1]
#    return beta[0], beta[1], t_stat * s / np.sqrt(Sxx)

alpha = 0.05 # 0.01, 0.10
mean, slope, dslope = ecj.trend_TheilSen(years, MHW_cnt_ts_glob, alpha)
print dslope[0], slope, dslope[1] # slope*len(years) # linear increase over record
mean, slope, dslope = ecj.trend_TheilSen(years, MHW_cnt_ts_woENSO, alpha)
print dslope[0], slope, dslope[1]
mean, slope, dslope = ecj.trend_TheilSen(years, MHW_mean_ts_glob, alpha)
print dslope[0], slope, dslope[1]
mean, slope, dslope = ecj.trend_TheilSen(years, MHW_mean_ts_woENSO, alpha)
print dslope[0], slope, dslope[1]
mean, slope, dslope = ecj.trend_TheilSen(years, MHW_dur_ts_glob, alpha)
print dslope[0], slope, dslope[1]
mean, slope, dslope = ecj.trend_TheilSen(years, MHW_dur_ts_woENSO, alpha)
print dslope[0], slope, dslope[1]
# Average number of MHW days
mean, slope, dslope = ecj.trend_TheilSen(years, MHW_td_ts_glob, alpha)
print dslope[0], slope, dslope[1]
mean, slope, dslope = ecj.trend_TheilSen(years, MHW_td_ts_woENSO, alpha)
print dslope[0], slope, dslope[1]
# Total annual cumulative intensity
mean, slope, dslope = ecj.trend_TheilSen(years, MHW_tc_ts_glob, alpha)
print dslope[0], slope, dslope[1]
mean, slope, dslope = ecj.trend_TheilSen(years, MHW_tc_ts_woENSO, alpha)
print dslope[0], slope, dslope[1]
# Max intensity
mean, slope, dslope = ecj.trend_TheilSen(years, MHW_max_ts_glob, alpha)
print dslope[0], slope, dslope[1]
mean, slope, dslope = ecj.trend_TheilSen(years, MHW_max_ts_woENSO, alpha)
print dslope[0], slope, dslope[1]
# SST
mean, slope, dslope = ecj.trend_TheilSen(years, SST_ts_glob, alpha)
print dslope[0], slope, dslope[1]
mean, slope, dslope = ecj.trend_TheilSen(years, SST_ts_woENSO, alpha)
print dslope[0], slope, dslope[1]


#
# Spatial correlations
#

ecj.pattern_correlation(SST_tr, MHW_cnt_tr)
ecj.pattern_correlation(SST_tr, MHW_mean_tr)
ecj.pattern_correlation(SST_tr, MHW_max_tr)
ecj.pattern_correlation(SST_tr, MHW_dur_tr)

ecj.pattern_correlation(MHW_cnt, MHW_dur)
ecj.pattern_correlation(np.std(SST_ts, axis=2), MHW_mean)
ecj.pattern_correlation(np.std(SST_ts, axis=2), MHW_max)

ecj.pattern_correlation(SST_tr, AMO)
ecj.pattern_correlation(MHW_cnt_tr, AMO)
ecj.pattern_correlation(MHW_mean_tr, AMO)
ecj.pattern_correlation(MHW_max_tr, AMO)
ecj.pattern_correlation(MHW_dur_tr, AMO)
ecj.pattern_correlation(SST_tr, PDO)
ecj.pattern_correlation(MHW_cnt_tr, PDO)
ecj.pattern_correlation(MHW_mean_tr, PDO)
ecj.pattern_correlation(MHW_max_tr, PDO)
ecj.pattern_correlation(MHW_dur_tr, PDO)

ecj.pattern_correlation(MHW_cnt, MHW_cnt_proxy)
ecj.pattern_correlation(MHW_dur, MHW_dur_proxy)
ecj.pattern_correlation(MHW_td, MHW_td_proxy)

#
# Figure 1 for MHW MS2 (OLD: TREND VERSION!)
#

plt.figure(figsize=(18,10))
plt.clf()
N = 1
#hatch = 'xx'
hatch = '//'
hatch_excess = '\\\\'
smooth_cut = 10.
sign_SST = mask*(((np.abs(SST_tr)-SST_dtr)>0).astype(float))
sign_cnt = mask*(((np.abs(MHW_cnt_tr)-MHW_cnt_dtr)>0).astype(float))
sign_mean = mask*(((np.abs(MHW_mean_tr)-MHW_mean_dtr)>0).astype(float))
sign_max = mask*(((np.abs(MHW_max_tr)-MHW_max_dtr)>0).astype(float))
sign_dur = mask*(((np.abs(MHW_dur_tr)-MHW_dur_dtr)>0).astype(float))
#sign_SST = np.round(sign_SST - ecj.spatial_filter(sign_SST, 0.25, smooth_cut, smooth_cut))
#sign_cnt = np.round(sign_cnt - ecj.spatial_filter(sign_cnt, 0.25, smooth_cut, smooth_cut))
#sign_mean = np.round(sign_mean - ecj.spatial_filter(sign_mean, 0.25, smooth_cut, smooth_cut))
#sign_dur = np.round(sign_dur - ecj.spatial_filter(sign_dur, 0.25, smooth_cut, smooth_cut))
excess_cnt = (MHW_cnt_tr_lr > ar1_putrend_cnt) + (MHW_cnt_tr_lr < ar1_pltrend_cnt)
excess_mean = (MHW_mean_tr_lr > ar1_putrend_mean) + (MHW_mean_tr_lr < ar1_pltrend_mean)
excess_max = (MHW_max_tr_lr > ar1_putrend_max) + (MHW_max_tr_lr < ar1_pltrend_max)
#excess_mean = np.round(excess_mean - ecj.spatial_filter(excess_mean, 0.25, smooth_cut, smooth_cut))
excess_dur = (MHW_dur_tr_lr > ar1_putrend_dur) + (MHW_dur_tr_lr < ar1_pltrend_dur)
# SST
plt.subplot2grid((4,8), (3,0), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, SST_mean, levels=np.arange(-2,30+1,4)) #, cmap=plt.cm.YlOrRd)
H = plt.colorbar()
#plt.clim(0.75,3.75)
H.set_label(r'[$^\circ$C]')
plt.subplot2grid((4,8), (3,3), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, SST_tr*10, levels=[-0.9,-0.6,-0.3,-0.1,0.1,0.3,0.6,0.9], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
plt.contourf(lonproj, latproj, sign_SST, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
H.set_label(r'[$^\circ$C / decade]')
AX = plt.subplot2grid((4,8), (3,6), colspan=2)
plt.plot(years, SST_ts_glob, 'k-')
plt.plot(years, SST_ts_glob, 'k.')
#plt.plot(years, SST_ts_woENSO, 'k--')
#plt.plot(years, SST_ts_woENSO, 'k.')
plt.grid()
plt.xlim(years.min()-1, years.max()+1)
plt.ylabel(r'[$^\circ$C]')
AX.yaxis.tick_right()
AX.yaxis.set_label_position('right')
# MHW Frequency
plt.subplot2grid((4,8), (0,0), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cnt, levels=np.arange(0.5,3.5+0.5,0.5), cmap=plt.cm.YlOrRd)
H = plt.colorbar()
H.set_label(r'Annual number [count]')
plt.clim(0.75,3.75)
plt.subplot2grid((4,8), (0,3), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cnt_tr*10, levels=[-3.5,-2,-1,-0.5,0.5,1,2,3.5], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
plt.contourf(lonproj, latproj, sign_cnt, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
lonproj, latproj = proj(llon_lr, llat_lr)
plt.contourf(lonproj, latproj, excess_cnt, hatches=['', hatch_excess], levels=[0., 0.5, 1.0], colors='none')
H.set_label(r'[count / decade]')
AX = plt.subplot2grid((4,8), (0,6), colspan=2)
plt.plot(years, MHW_cnt_ts_glob, 'k-')
plt.plot(years, MHW_cnt_ts_glob, 'k.')
plt.plot(years, MHW_cnt_ts_woENSO, 'k--')
plt.plot(years, MHW_cnt_ts_woENSO, 'k.')
plt.grid()
plt.xlim(years.min()-1, years.max()+1)
plt.ylabel('[count]')
AX.yaxis.tick_right()
AX.yaxis.set_label_position('right')
AX.set_xticklabels([])
# MHW Intensity (max) XXX:(mean)
plt.subplot2grid((4,8), (1,0), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
#plt.contourf(lonproj, latproj, MHW_mean, levels=np.arange(0,5+0.5,0.5), cmap=plt.cm.gist_heat_r)
plt.contourf(lonproj, latproj, MHW_max, levels=np.arange(0,5+0.5,0.5), cmap=plt.cm.gist_heat_r)
H = plt.colorbar()
H.set_label(r'[$^\circ$C]')
plt.clim(0.5,5.75)
plt.subplot2grid((4,8), (1,3), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
#plt.contourf(lonproj, latproj, MHW_mean_tr*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.contourf(lonproj, latproj, MHW_max_tr*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[$^\circ$C / decade]')
plt.clim(-0.5,0.5)
#plt.contourf(lonproj, latproj, sign_mean, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.contourf(lonproj, latproj, sign_max, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
lonproj, latproj = proj(llon_lr, llat_lr)
#plt.contourf(lonproj, latproj, excess_mean, hatches=['', hatch_excess], levels=[0., 0.5, 1.0], colors='none')
plt.contourf(lonproj, latproj, excess_max, hatches=['', hatch_excess], levels=[0., 0.5, 1.0], colors='none')
AX = plt.subplot2grid((4,8), (1,6), colspan=2)
#plt.plot(years, MHW_mean_ts_glob, 'k-')
#plt.plot(years, MHW_mean_ts_glob, 'k.')
#plt.plot(years, MHW_mean_ts_woENSO, 'k--')
#plt.plot(years, MHW_mean_ts_woENSO, 'k.')
plt.plot(years, MHW_max_ts_glob, 'k-')
plt.plot(years, MHW_max_ts_glob, 'k.')
plt.plot(years, MHW_max_ts_woENSO, 'k--')
plt.plot(years, MHW_max_ts_woENSO, 'k.')
plt.grid()
plt.xlim(years.min()-1, years.max()+1)
plt.ylabel(r'[$^\circ$C]')
AX.yaxis.tick_right()
AX.yaxis.set_label_position('right')
AX.set_xticklabels([])
# MHW Duration
plt.subplot2grid((4,8), (2,0), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_dur, levels=[5,10,15,20,30,60], cmap=plt.cm.gist_heat_r)
H = plt.colorbar()
H.set_label(r'[days]')
plt.clim(8,80)
plt.subplot2grid((4,8), (2,3), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_dur_tr*10, levels=[-50,-20,-10,-2.5,2.5,10,20,50], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[days / decade]')
plt.clim(-18,18)
plt.contourf(lonproj, latproj, sign_dur, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
lonproj, latproj = proj(llon_lr, llat_lr)
plt.contourf(lonproj, latproj, excess_dur, hatches=['', hatch_excess], levels=[0., 0.5, 1.0], colors='none')
AX = plt.subplot2grid((4,8), (2,6), colspan=2)
plt.plot(years, MHW_dur_ts_glob, 'k-')
plt.plot(years, MHW_dur_ts_glob, 'k.')
plt.plot(years, MHW_dur_ts_woENSO, 'k--')
plt.plot(years, MHW_dur_ts_woENSO, 'k.')
plt.grid()
plt.xlim(years.min()-1, years.max()+1)
plt.ylabel('[days]')
AX.yaxis.tick_right()
AX.yaxis.set_label_position('right')
AX.set_xticklabels([])

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_mean_trend_ts_orig.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_mean_trend_ts_orig.pdf', bbox_inches='tight', pad_inches=0.5)

#
# AR1 Model Results
#

plt.figure(figsize=(9,8))
plt.clf()
plt.subplot(2,1,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[6,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[6,900])
lonproj, latproj = proj(llon_lr, llat_lr)
plt.contourf(lonproj, latproj, ar1_tau, levels=[0,5,10,15,20,25,40,60], cmap=plt.cm.gist_heat_r)
H = plt.colorbar()
plt.clim(0,70)
H.set_label(r'[days]')
plt.title(r'(A) Autoregressive time scale ($\tau$)')
plt.subplot(2,1,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[6,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[6,900])
lonproj, latproj = proj(llon_lr, llat_lr)
plt.contourf(lonproj, latproj, ar1_sig_eps, levels=np.arange(0,0.65+0.05,0.05), cmap=plt.cm.gist_heat_r)
H = plt.colorbar()
plt.clim(0.05,0.65)
H.set_label(r'[$^\circ$C]')
plt.title(r'(B) Error standard deviation ($\sigma_\epsilon$)')
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/ar1fit_orig.pdf', bbox_inches='tight', pad_inches=0.5)
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/ar1fit_orig.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.figure(figsize=(7,10))
plt.clf()
#hatch = 'xx'
hatch = '////'
sign_cnt = (MHW_cnt_tr_lr > ar1_putrend_cnt) + (MHW_cnt_tr_lr < ar1_pltrend_cnt) #(ar1_ptrend_cnt<0.05).astype(float)
sign_mean = (MHW_mean_tr_lr > ar1_putrend_mean) + (MHW_mean_tr_lr < ar1_pltrend_mean) #(ar1_ptrend_mean<0.05).astype(float)
sign_max = (MHW_max_tr_lr > ar1_putrend_max) + (MHW_max_tr_lr < ar1_pltrend_max) #(ar1_ptrend_max<0.05).astype(float)
sign_dur = (MHW_dur_tr_lr > ar1_putrend_dur) + (MHW_dur_tr_lr < ar1_pltrend_dur) #(ar1_ptrend_dur<0.05).astype(float)
# 
plt.subplot(3,1,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[6,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[6,900])
lonproj, latproj = proj(llon_lr, llat_lr)
plt.contourf(lonproj, latproj, MHW_cnt_tr_lr*10, levels=[-3.5,-2,-1,-0.5,0.5,1,2,3.5], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[count / decade]')
plt.title('(A) MHW Frequency Linear Trend')
plt.contourf(lonproj, latproj, sign_cnt, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.subplot(3,1,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[6,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[6,900])
lonproj, latproj = proj(llon_lr, llat_lr)
tmp = MHW_max_tr_lr*10
tmp[tmp>0.7] = 0.7
tmp[tmp<-0.7] = -0.7
plt.contourf(lonproj, latproj, tmp, levels=[-0.75,-0.5,-0.25,-0.10,0.10,0.25,0.5,0.75], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[$^\circ$C / decade]')
plt.title('(B) MHW Intensity Linear Trend')
plt.clim(-0.6,0.6)
plt.contourf(lonproj, latproj, sign_max, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.subplot(3,1,3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[6,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[6,900])
lonproj, latproj = proj(llon_lr, llat_lr)
plt.contourf(lonproj, latproj, MHW_dur_tr_lr*10, levels=[-50,-20,-10,-2.5,2.5,10,20,50], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[days / decade]')
plt.title('(C) MHW Duration Linear Trend')
plt.clim(-18,18)
plt.contourf(lonproj, latproj, sign_dur, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/ar1fit_trends_orig.pdf', bbox_inches='tight', pad_inches=0.5)
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/ar1fit_trends_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)

wet = ~np.isnan(ar1_putrend_cnt)
100*(1-1.*sign_cnt[wet].sum()/len(sign_cnt[wet]))
100*(1-1.*sign_mean[wet].sum()/len(sign_mean[wet]))
100*(1-1.*sign_max[wet].sum()/len(sign_max[wet]))
100*(1-1.*sign_dur[wet].sum()/len(sign_dur[wet]))

plt.figure(figsize=(7,10))
plt.clf()
hatch = 'xx'
#sign_cnt = (ar1_ptrend_cnt<0.05)*((np.abs(MHW_cnt_tr)-MHW_cnt_dtr)>0).astype(float)
#sign_mean = (ar1_ptrend_mean<0.05)*((np.abs(MHW_mean_tr)-MHW_mean_dtr)>0).astype(float)
#sign_dur = (ar1_ptrend_dur<0.05)*((np.abs(MHW_dur_tr)-MHW_dur_dtr)>0).astype(float)
sign_cnt = (MHW_cnt_tr_lr > ar1_putrend_cnt) + (MHW_cnt_tr_lr < ar1_pltrend_cnt) #(ar1_ptrend_cnt<0.05).astype(float)
sign_mean = (MHW_mean_tr_lr > ar1_putrend_mean) + (MHW_mean_tr_lr < ar1_pltrend_mean) #(ar1_ptrend_mean<0.05).astype(float)
sign_dur = (MHW_dur_tr_lr > ar1_putrend_dur) + (MHW_dur_tr_lr < ar1_pltrend_dur) #(ar1_ptrend_dur<0.05).astype(float)
# 
plt.subplot(3,1,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon_lr, llat_lr)
plt.contourf(lonproj, latproj, 0.5*(ar1_pltrend_cnt + ar1_putrend_cnt)*10, levels=[-3.5,-2,-1,-0.5,0.5,1,2,3.5], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[count / decade]')
plt.title('(A) MHW Frequency Linear Trend')
plt.subplot(3,1,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon_lr, llat_lr)
plt.contourf(lonproj, latproj, 0.5*(ar1_pltrend_mean + ar1_putrend_mean)*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[$^\circ$C / decade]')
plt.title('(B) MHW Intensity Linear Trend')
plt.clim(-0.5,0.5)
plt.subplot(3,1,3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon_lr, llat_lr)
plt.contourf(lonproj, latproj, 0.5*(ar1_pltrend_dur + ar1_putrend_dur)*10, levels=[-50,-20,-10,-2.5,2.5,10,20,50], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[days / decade]')
plt.title('(C) MHW Duration Linear Trend')
plt.clim(-18,18)

#
# Time slice differences
#

# t-test
Ny = len(years)/2
p_SST = np.nan*np.zeros((SST_ts.shape[0], SST_ts.shape[1]))
p_cnt = np.nan*np.zeros((SST_ts.shape[0], SST_ts.shape[1]))
p_mean = np.nan*np.zeros((SST_ts.shape[0], SST_ts.shape[1]))
p_max = np.nan*np.zeros((SST_ts.shape[0], SST_ts.shape[1]))
p_dur = np.nan*np.zeros((SST_ts.shape[0], SST_ts.shape[1]))
for j in range(p_SST.shape[0]):
    print j
    for i in range(p_SST.shape[1]):
        if (MHW_cnt_ts[j,i,0:Ny].sum() == 0) + (MHW_cnt_ts[j,i,Ny+1:].sum() == 0):
            continue
        t, p_SST[j,i] = ecj.ttest_serialcorr(SST_ts[j,i,Ny+1:], SST_ts[j,i,0:Ny])
        t, p_cnt[j,i] = ecj.ttest_serialcorr(MHW_cnt_ts[j,i,Ny+1:], MHW_cnt_ts[j,i,0:Ny])
        t, p_mean[j,i] = ecj.ttest_serialcorr(ecj.nonans(MHW_mean_ts[j,i,Ny+1:]), ecj.nonans(MHW_mean_ts[j,i,0:Ny]))
        t, p_max[j,i] = ecj.ttest_serialcorr(ecj.nonans(MHW_max_ts[j,i,Ny+1:]), ecj.nonans(MHW_max_ts[j,i,0:Ny]))
        t, p_dur[j,i] = ecj.ttest_serialcorr(ecj.nonans(MHW_dur_ts[j,i,Ny+1:]), ecj.nonans(MHW_dur_ts[j,i,0:Ny]))

sign_p_SST = mask*(p_SST<=0.05).astype(float)
sign_p_cnt = mask*(p_cnt<=0.05).astype(float)
sign_p_mean = mask*(p_mean<=0.05).astype(float)
sign_p_max = mask*(p_max<=0.05).astype(float)
sign_p_dur = mask*(p_dur<=0.05).astype(float)

# KS-test
Ny = len(years)/2
p_KS_SST = np.nan*np.zeros((SST_ts.shape[0], SST_ts.shape[1]))
p_KS_cnt = np.nan*np.zeros((SST_ts.shape[0], SST_ts.shape[1]))
p_KS_mean = np.nan*np.zeros((SST_ts.shape[0], SST_ts.shape[1]))
p_KS_max = np.nan*np.zeros((SST_ts.shape[0], SST_ts.shape[1]))
p_KS_dur = np.nan*np.zeros((SST_ts.shape[0], SST_ts.shape[1]))
for j in range(p_KS_SST.shape[0]):
    print j
    for i in range(p_KS_SST.shape[1]):
        if (MHW_cnt_ts[j,i,0:Ny].sum() == 0) + (MHW_cnt_ts[j,i,Ny+1:].sum() == 0):
            continue
        t, p_KS_SST[j,i] = stats.ks_2samp(SST_ts[j,i,Ny+1:], SST_ts[j,i,0:Ny])
        t, p_KS_cnt[j,i] = stats.ks_2samp(MHW_cnt_ts[j,i,Ny+1:], MHW_cnt_ts[j,i,0:Ny])
        t, p_KS_mean[j,i] = stats.ks_2samp(ecj.nonans(MHW_mean_ts[j,i,Ny+1:]), ecj.nonans(MHW_mean_ts[j,i,0:Ny]))
        t, p_KS_max[j,i] = stats.ks_2samp(ecj.nonans(MHW_max_ts[j,i,Ny+1:]), ecj.nonans(MHW_max_ts[j,i,0:Ny]))
        t, p_KS_dur[j,i] = stats.ks_2samp(ecj.nonans(MHW_dur_ts[j,i,Ny+1:]), ecj.nonans(MHW_dur_ts[j,i,0:Ny]))

sign_p_SST = mask*(p_KS_SST<=0.05).astype(float)
sign_p_cnt = mask*(p_KS_cnt<=0.05).astype(float)
sign_p_mean = mask*(p_KS_mean<=0.05).astype(float)
sign_p_max = mask*(p_KS_max<=0.05).astype(float)
sign_p_dur = mask*(p_KS_dur<=0.05).astype(float)

outfile = '/home/ecoliver/Desktop/data/MHWs/Trends/mhw_census.2016.p'
#np.savez(outfile, p_SST=p_SST, p_cnt=p_cnt, p_mean=p_mean, p_max=p_max, p_dur=p_dur, sign_p_SST=sign_p_SST, sign_p_cnt=sign_p_cnt, sign_p_mean=sign_p_mean, sign_p_max=sign_p_max, sign_p_dur=sign_p_dur, p_KS_SST=p_KS_SST, p_KS_cnt=p_KS_cnt, p_KS_mean=p_KS_mean, p_KS_max=p_KS_max, p_KS_dur=p_KS_dur, Ny=Ny)
data = np.load(outfile + '.npz')
p_SST = data['p_SST']
p_cnt = data['p_cnt']
p_mean = data['p_mean']
p_max = data['p_max']
p_dur = data['p_dur']
sign_p_SST = data['sign_p_SST']
sign_p_cnt = data['sign_p_cnt']
sign_p_mean = data['sign_p_mean']
sign_p_max = data['sign_p_max']
sign_p_dur = data['sign_p_dur']
p_KS_SST = data['p_KS_SST']
p_KS_cnt = data['p_KS_cnt']
p_KS_mean = data['p_KS_mean']
p_KS_max = data['p_KS_max']
p_KS_dur = data['p_KS_dur']
Ny = data['Ny']

#smooth_cut = 10.
#sign_SST = np.round(sign_p_SST - ecj.spatial_filter(sign_p_SST, 0.25, smooth_cut, smooth_cut))
#sign_cnt = np.round(sign_p_cnt - ecj.spatial_filter(sign_p_cnt, 0.25, smooth_cut, smooth_cut))
#sign_mean = np.round(sign_p_mean - ecj.spatial_filter(sign_p_mean, 0.25, smooth_cut, smooth_cut))
#sign_dur = np.round(sign_p_dur - ecj.spatial_filter(sign_p_dur, 0.25, smooth_cut, smooth_cut))

#land = np.ones(sign_p_SST.shape)
#land[np.isnan(SST_ts[:,:,20])] = np.nan

#
# New figure 1 for MHW MS2 (time slice diffs)
#

plt.figure(figsize=(18,10))
plt.clf()
hatch = '//////'
hatch_excess = '\\\\\\\\'
smooth_cut = 2.
# SST
plt.subplot2grid((4,8), (3,0), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, SST_mean, levels=np.arange(-2,30+1,4), cmap=plt.cm.jet)
H = plt.colorbar()
H.set_label(r'[$^\circ$C]')
plt.subplot2grid((4,8), (3,3), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, np.mean(SST_ts[:,:,Ny+1:], axis=2) - np.mean(SST_ts[:,:,0:Ny], axis=2), levels=[-2,-1,-0.5,-0.25,-0.1,0.1,0.25,0.5,1,2], cmap=plt.cm.RdBu_r)
plt.clim(-1,1)
H = plt.colorbar()
plt.contourf(lonproj, latproj, sign_p_SST, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
H.set_label(r'[$^\circ$C]')
AX = plt.subplot2grid((4,8), (3,6), colspan=2)
plt.plot(years, SST_ts_glob, 'k-')
plt.plot(years, SST_ts_woENSO, 'r-')
plt.xlim(years.min()-1, years.max()+1)
plt.ylim(20.6, 21.5)
plt.ylabel(r'[$^\circ$C]')
AX.yaxis.tick_right()
AX.yaxis.set_label_position('right')
AX.set_xticks(np.arange(1985, 2015+1, 5))
# MHW Frequency
plt.subplot2grid((4,8), (0,0), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cnt, levels=np.arange(0.5,3.5+0.5,0.5), cmap=plt.cm.YlOrRd)
H = plt.colorbar()
H.set_label(r'Annual number [count]')
plt.clim(0.75,3.75)
plt.subplot2grid((4,8), (0,3), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, datamask*(np.mean(MHW_cnt_ts[:,:,Ny+1:], axis=2) - np.mean(MHW_cnt_ts[:,:,0:Ny], axis=2)), levels=[-6,-4,-3,-2,-1,1,2,3,4,6], cmap=plt.cm.RdBu_r)
plt.clim(-5,5)
H = plt.colorbar()
#plt.contourf(lonproj, latproj, sign_p_cnt, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.contourf(lonproj, latproj, np.round(sign_p_cnt - ecj.spatial_filter(sign_p_cnt, 0.25, smooth_cut, smooth_cut)), hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
lonproj, latproj = proj(llon_lr, llat_lr)
H.set_label(r'[count]')
AX = plt.subplot2grid((4,8), (0,6), colspan=2)
plt.plot(years, MHW_cnt_ts_glob, 'k-')
plt.plot(years, MHW_cnt_ts_woENSO, 'r-')
plt.xlim(years.min()-1, years.max()+1)
plt.ylim(1.5, 4.6)
plt.ylabel('[count]')
AX.yaxis.tick_right()
AX.yaxis.set_label_position('right')
AX.set_xticks(np.arange(1985, 2015+1, 5))
AX.set_xticklabels([])
# MHW Intensity (max) XXX:(mean)
plt.subplot2grid((4,8), (1,0), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_max, levels=np.arange(0,5+0.5,0.5), cmap=plt.cm.gist_heat_r)
H = plt.colorbar()
H.set_label(r'[$^\circ$C]')
plt.clim(0.5,5.75)
plt.subplot2grid((4,8), (1,3), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, np.nanmean(MHW_max_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_max_ts[:,:,0:Ny], axis=2), levels=[-2.5,-1.5,-1,-0.5,-0.2,0.2,0.5,1,1.5,2.5], cmap=plt.cm.RdBu_r)
plt.clim(-1.6, 1.6)
H = plt.colorbar()
H.set_label(r'[$^\circ$C]')
#plt.contourf(lonproj, latproj, sign_p_mean, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.contourf(lonproj, latproj, np.round(sign_p_max - ecj.spatial_filter(sign_p_max, 0.25, smooth_cut, smooth_cut)), hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
lonproj, latproj = proj(llon_lr, llat_lr)
AX = plt.subplot2grid((4,8), (1,6), colspan=2)
plt.plot(years, MHW_max_ts_glob, 'k-')
plt.plot(years, MHW_max_ts_woENSO, 'r-')
plt.xlim(years.min()-1, years.max()+1)
plt.ylim(1.7, 2.4)
plt.ylabel(r'[$^\circ$C]')
AX.yaxis.tick_right()
AX.yaxis.set_label_position('right')
AX.set_xticks(np.arange(1985, 2015+1, 5))
AX.set_xticklabels([])
# MHW Duration
plt.subplot2grid((4,8), (2,0), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_dur, levels=[5,10,15,20,30,60], cmap=plt.cm.gist_heat_r)
H = plt.colorbar()
H.set_label(r'[days]')
plt.clim(8,80)
plt.subplot2grid((4,8), (2,3), colspan=3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, np.nanmean(MHW_dur_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_dur_ts[:,:,0:Ny], axis=2), levels=[-75,-30,-20,-10,-5,5,10,20,30,75], cmap=plt.cm.RdBu_r)
plt.clim(-30,30)
H = plt.colorbar()
H.set_label(r'[days]')
#plt.contourf(lonproj, latproj, sign_p_dur, hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
plt.contourf(lonproj, latproj, np.round(sign_p_dur - ecj.spatial_filter(sign_p_dur, 0.25, smooth_cut, smooth_cut)), hatches=['', hatch], levels=[0., 0.5, 1.0], colors='none')
lonproj, latproj = proj(llon_lr, llat_lr)
AX = plt.subplot2grid((4,8), (2,6), colspan=2)
plt.plot(years, MHW_dur_ts_glob, 'k-')
plt.plot(years, MHW_dur_ts_woENSO, 'r-')
plt.xlim(years.min()-1, years.max()+1)
plt.ylim(8, 28)
plt.ylabel('[days]')
AX.yaxis.tick_right()
AX.yaxis.set_label_position('right')
AX.set_xticks(np.arange(1985, 2015+1, 5))
AX.set_xticklabels([])

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_mean_timeslice_ts_tsOnly_orig.pdf', bbox_inches='tight', pad_inches=0.05)
# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_mean_timeslice_ts_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)

1.*np.sum((np.mean(SST_ts[:,:,Ny+1:], axis=2) - np.mean(SST_ts[:,:,0:Ny], axis=2)) >= 0) / np.sum(~np.isnan(SST_tr))
1.*np.sum((datamask*(np.mean(MHW_cnt_ts[:,:,Ny+1:], axis=2) - np.mean(MHW_cnt_ts[:,:,0:Ny], axis=2))) >= 0) / np.sum(~np.isnan(MHW_cnt_tr))
1.*np.sum((datamask*(np.nanmean(MHW_mean_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_mean_ts[:,:,0:Ny], axis=2))) >= 0) / np.sum(~np.isnan(MHW_mean_tr))
1.*np.sum((datamask*(np.nanmean(MHW_max_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_max_ts[:,:,0:Ny], axis=2))) >= 0) / np.sum(~np.isnan(MHW_max_tr))
1.*np.sum((datamask*(np.nanmean(MHW_dur_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_dur_ts[:,:,0:Ny], axis=2))) >= 0) / np.sum(~np.isnan(MHW_dur_tr))

1.*np.sum(((np.mean(SST_ts[:,:,Ny+1:], axis=2) - np.mean(SST_ts[:,:,0:Ny], axis=2)) >= 0)*(p_SST<=0.05)) / np.sum(~np.isnan(SST_tr))
1.*np.sum(((datamask*(np.mean(MHW_cnt_ts[:,:,Ny+1:], axis=2) - np.mean(MHW_cnt_ts[:,:,0:Ny], axis=2))) >= 0)*(p_cnt<=0.05)) / np.sum(~np.isnan(MHW_cnt_tr))
1.*np.sum(((datamask*(np.nanmean(MHW_mean_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_mean_ts[:,:,0:Ny], axis=2))) >= 0)*(p_mean<=0.05)) / np.sum(~np.isnan(MHW_mean_tr))
1.*np.sum(((datamask*(np.nanmean(MHW_max_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_max_ts[:,:,0:Ny], axis=2))) >= 0)*(p_max<=0.05)) / np.sum(~np.isnan(MHW_max_tr))
1.*np.sum(((datamask*(np.nanmean(MHW_dur_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_dur_ts[:,:,0:Ny], axis=2))) >= 0)*(p_dur<=0.05)) / np.sum(~np.isnan(MHW_dur_tr))

1.*np.sum(((np.mean(SST_ts[:,:,Ny+1:], axis=2) - np.mean(SST_ts[:,:,0:Ny], axis=2)) >= 0)*(p_KS_SST<=0.05)) / np.sum(~np.isnan(SST_tr))
1.*np.sum(((datamask*(np.mean(MHW_cnt_ts[:,:,Ny+1:], axis=2) - np.mean(MHW_cnt_ts[:,:,0:Ny], axis=2))) >= 0)*(p_KS_cnt<=0.05)) / np.sum(~np.isnan(MHW_cnt_tr))
1.*np.sum(((datamask*(np.nanmean(MHW_mean_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_mean_ts[:,:,0:Ny], axis=2))) >= 0)*(p_KS_mean<=0.05)) / np.sum(~np.isnan(MHW_mean_tr))
1.*np.sum(((datamask*(np.nanmean(MHW_max_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_max_ts[:,:,0:Ny], axis=2))) >= 0)*(p_KS_max<=0.05)) / np.sum(~np.isnan(MHW_max_tr))
1.*np.sum(((datamask*(np.nanmean(MHW_dur_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_dur_ts[:,:,0:Ny], axis=2))) >= 0)*(p_KS_dur<=0.05)) / np.sum(~np.isnan(MHW_dur_tr))

ecj.pattern_correlation(np.mean(SST_ts[:,:,Ny+1:], axis=2) - np.mean(SST_ts[:,:,0:Ny], axis=2), datamask*(np.mean(MHW_cnt_ts[:,:,Ny+1:], axis=2) - np.mean(MHW_cnt_ts[:,:,0:Ny], axis=2)))
ecj.pattern_correlation(np.mean(SST_ts[:,:,Ny+1:], axis=2) - np.mean(SST_ts[:,:,0:Ny], axis=2), np.nanmean(MHW_mean_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_mean_ts[:,:,0:Ny], axis=2))
ecj.pattern_correlation(np.mean(SST_ts[:,:,Ny+1:], axis=2) - np.mean(SST_ts[:,:,0:Ny], axis=2), np.nanmean(MHW_max_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_max_ts[:,:,0:Ny], axis=2))
ecj.pattern_correlation(np.mean(SST_ts[:,:,Ny+1:], axis=2) - np.mean(SST_ts[:,:,0:Ny], axis=2), np.nanmean(MHW_dur_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_dur_ts[:,:,0:Ny], axis=2))

ecj.pattern_correlation(MHW_cnt, MHW_dur)
ecj.pattern_correlation(np.std(SST_ts, axis=2), MHW_mean)
ecj.pattern_correlation(np.std(SST_ts, axis=2), MHW_max)

ecj.pattern_correlation(np.mean(SST_ts[:,:,Ny+1:], axis=2) - np.mean(SST_ts[:,:,0:Ny], axis=2), AMO)
ecj.pattern_correlation(datamask*(np.mean(MHW_cnt_ts[:,:,Ny+1:], axis=2) - np.mean(MHW_cnt_ts[:,:,0:Ny], axis=2)), AMO)
ecj.pattern_correlation(np.nanmean(MHW_mean_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_mean_ts[:,:,0:Ny], axis=2), AMO)
ecj.pattern_correlation(np.nanmean(MHW_max_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_max_ts[:,:,0:Ny], axis=2), AMO)
ecj.pattern_correlation(np.nanmean(MHW_dur_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_dur_ts[:,:,0:Ny], axis=2), AMO)
ecj.pattern_correlation(np.mean(SST_ts[:,:,Ny+1:], axis=2) - np.mean(SST_ts[:,:,0:Ny], axis=2), PDO)
ecj.pattern_correlation(datamask*(np.mean(MHW_cnt_ts[:,:,Ny+1:], axis=2) - np.mean(MHW_cnt_ts[:,:,0:Ny], axis=2)), PDO)
ecj.pattern_correlation(np.nanmean(MHW_mean_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_mean_ts[:,:,0:Ny], axis=2), PDO)
ecj.pattern_correlation(np.nanmean(MHW_max_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_max_ts[:,:,0:Ny], axis=2), PDO)
ecj.pattern_correlation(np.nanmean(MHW_dur_ts[:,:,Ny+1:], axis=2) - np.nanmean(MHW_dur_ts[:,:,0:Ny], axis=2), PDO)

def pattern_correlation(x1, x2, centred=True):
    '''
    Calculates the pattern correlation of x1 and x2.
    Assumes and x1 and x2 are 2D numpy arrays. Can
    handle missing values, even if missing values are
    distributed differently in x1 and x2. By default
    calculated the centred pattern correlation (centred
    =True) in which the spatial means of x1 and x2 are
    removed prior to calculation. Can calculated uncentred
    pattern correlation (centred=False) in which these
    means are not removed.
    .
    Written by Eric Oliver, IMAS/UTAS, Nov 2015
    '''
    # Flatten 2D arrays and find shared valid (non-nan) indices
    X1 = x1.flatten()
    X2 = x2.flatten()
    valid = ~(np.isnan(X1) + np.isnan(X2))
    # Create Nx2 array of valid data
    X = np.zeros((valid.sum(), 2))
    X[:,0] = X1[valid]
    X[:,1] = X2[valid]
    # Centre data if desired
    if centred:
        X[:,0] = X[:,0] - np.mean(X[:,0])
        X[:,1] = X[:,1] - np.mean(X[:,1])
    #
    # Effective sample size (serial correlation)
        # Autocorrelation Function (pad NaN values for an approximation)
    rho1 = ecj.acf(X1[valid] - np.nanmean(X1[valid]))
    rho2 = ecj.acf(X2[valid] - np.nanmean(X2[valid]))
    # Equivalent sample lengths
    n = valid.sum()
    n1 = n / (1 + ((1-np.arange(1, int(n))/n)*rho1[:-1]).sum())
    n2 = n / (1 + ((1-np.arange(1, int(n))/n)*rho2[:-1]).sum())
    n = np.floor(np.min([n1, n2]))
    #
    # Calculate pattern correlation
    pcorr = np.corrcoef(X.T)[0,1]
    # 95% CI
    z = 0.5*np.log((1+pcorr)/(1-pcorr))
    zL = z - 1.96/np.sqrt(n-3)
    zU = z + 1.96/np.sqrt(n-3)
    CI = [np.tanh(zL), np.tanh(zU)]
    return pcorr, CI

#
# Comparison of wENSO and woENSO
#

plt.figure(figsize=(18,10))
plt.clf()
# MHW Frequency
plt.subplot(3,3,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cnt, levels=np.arange(0.5,3.5+0.5,0.5), cmap=plt.cm.YlOrRd)
H = plt.colorbar()
H.set_label(r'Annual number [count]')
plt.clim(0.75,3.75)
plt.subplot(3,3,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cnt_noENSO, levels=np.arange(0.5,3.5+0.5,0.5), cmap=plt.cm.YlOrRd)
H = plt.colorbar()
H.set_label(r'Annual number [count]')
plt.clim(0.75,3.75)
plt.subplot(3,3,3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cnt - MHW_cnt_noENSO, levels=[-2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2,2.5], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'Annual number [count]')
plt.clim(-2.2,2.2)
# MHW Intensity
plt.subplot(3,3,4, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_max, levels=np.arange(0,5+0.5,0.5), cmap=plt.cm.gist_heat_r)
H = plt.colorbar()
H.set_label(r'[$^\circ$C]')
plt.clim(0.5,5.75)
plt.subplot(3,3,5, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_max_noENSO, levels=np.arange(0,5+0.5,0.5), cmap=plt.cm.gist_heat_r)
H = plt.colorbar()
H.set_label(r'[$^\circ$C]')
plt.clim(0.5,5.75)
plt.subplot(3,3,6, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_max - MHW_max_noENSO, levels=[-1.5,-1.25,-1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,1.25,1.5], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[$^\circ$C]')
plt.clim(-1.2, 1.2)
# MHW Duration
plt.subplot(3,3,7, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_dur, levels=[5,10,15,20,30,60], cmap=plt.cm.gist_heat_r)
H = plt.colorbar()
H.set_label(r'[days]')
plt.clim(8,80)
plt.subplot(3,3,8, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_dur_noENSO, levels=[5,10,15,20,30,60], cmap=plt.cm.gist_heat_r)
H = plt.colorbar()
H.set_label(r'[days]')
plt.clim(8,80)
plt.subplot(3,3,9, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_dur - MHW_dur_noENSO, levels=[-55,-35,-15,-5,5,15,35,55], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[days]')
plt.clim(-45,45)

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_ENSO_orig.png', bbox_inches='tight', pad_inches=0.5, dpi=300)


plt.figure(figsize=(18,10))
plt.clf()
# MHW Frequency (mean)
plt.subplot(3,2,1, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cnt - MHW_cnt_noENSO, levels=[-2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2,2.5], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'Annual number [count]')
plt.clim(-2.2,2.2)
# MHW Frequency (std)
plt.subplot(3,2,2, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, datamask*np.nanstd(MHW_cnt_ts - MHW_cnt_ts_noENSO, axis=2), levels=np.arange(0.,4.0+0.5,0.5), cmap=plt.cm.rainbow)
H = plt.colorbar()
H.set_label(r'Annual number [count]')
plt.clim(0, 3.5)
# MHW Intensity (mean)
plt.subplot(3,2,3, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_max - MHW_max_noENSO, levels=[-1.5,-1.25,-1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,1.25,1.5], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[$^\circ$C]')
plt.clim(-1.2, 1.2)
# MHW Intensity (std)
plt.subplot(3,2,4, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, np.nanstd(MHW_max_ts - MHW_max_ts_noENSO, axis=2), levels=[0,0.25,0.5,0.75,1,1.5,2,3.5], cmap=plt.cm.rainbow)
H = plt.colorbar()
H.set_label(r'[$^\circ$C]')
plt.clim(0, 2)
# MHW Duration (mean)
plt.subplot(3,2,5, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_dur - MHW_dur_noENSO, levels=[-55,-35,-15,-7.5,-2.5,2.5,7.5,15,35,55], cmap=plt.cm.RdBu_r)
H = plt.colorbar()
H.set_label(r'[days]')
plt.clim(-45,45)
# MHW Duration (std)
plt.subplot(3,2,6, axisbg=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
lonproj, latproj = proj(llon, llat)
#plt.contourf(lonproj, latproj, np.log10(datamask*np.nanstd(MHW_dur_ts - MHW_dur_ts_noENSO, axis=2)))#, levels=np.arange(0.5,3.5+0.5,0.5), cmap=plt.cm.rainbow)
plt.contourf(lonproj, latproj, datamask*np.nanstd(MHW_dur_ts - MHW_dur_ts_noENSO, axis=2), levels=[0,5,10,15,20,40,100,400], cmap=plt.cm.rainbow)
H = plt.colorbar()
H.set_label(r'[days]')
plt.clim(0,100)

# plt.savefig('../../documents/05_MHW_Spatial_Distribution_And_Trends/figures/MHW_ENSO_meanStd_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)


plt.plot(years, MHW_dur_ts_glob - MHW_dur_ts_woENSO, 'k-')




