
  clear all
  addpath(genpath('/home/ecoliver/Desktop/include/'));
  rehash;

  % Data location
  header = '/mnt/erebor/data/sst/noaa_oi_v2/avhrr/timeseries/';

  % Set up ice matrix
  i0 = 1;
  file0 = [header 'avhrr-only-v2.ts.' num2str(i0, '%.4d') '.mat'];
  load(file0, 'lon');
  load(file0, 'lat');
  ice_mean = zeros(length(lat), length(lon));
  ice_longestRun = zeros(length(lat), length(lon));

  % Loop over lons and get mean ice field from timeseries files
  for i = 1:length(lon)
    [i length(lon)]
    file = [header 'avhrr-only-v2.ts.' num2str(i, '%.4d') '.mat'];
    load(file, 'ice_ts');
    ice_mean(:,i) = nanmean(ice_ts, 2);
    ice_presence = isnan(ice_ts);
    for j = 1:length(lat)
      if sum(ice_presence(j,:)) == size(ice_presence,2), continue, end;
      iceRuns = contiguous(ice_presence(j,:), 0);
      ice_longestRun(j,i) = max(diff(iceRuns{1,2}')+1);
    end
  end

  % Save data
  save('/home/ecoliver/Desktop/data/extreme_SSTs/NOAAOISST_iceMean.mat', 'lon', 'lat', 'ice_mean', 'ice_longestRun')
