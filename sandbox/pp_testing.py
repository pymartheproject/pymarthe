import sys 
import os 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import geopandas 

sys.path.append('/Users/apryet/Programmes/python/pyemu/')
import pyemu

sys.path.append('/Users/apryet/Programmes/python/adeqwat')
from pymarthe import * 

# load Marthe model
mm = MartheModel('../MONA_V3/mona.rma')
nlay, nrow, ncol = mm.nlay, mm.nrow, mm.ncol
xx, yy = np.meshgrid(mm.x_vals,mm.y_vals)

# add new parameter 

# ---- developement for pilot points

# NOTE param, lay  will be iterated

par = 'kepon'
mm.add_param(par,1e-6)
izone = -1*np.ones( (nlay, nrow, ncol) )
izone[1,:,:] = 1
mm.param[par].set_izone(izone=izone)

lay = 1
zone = 1

# load pilot points for given layer and zone
prefix = 'pp_{0}_l{1:02d}'.format(par,lay)
pp_df = pp_utils.ppoints_from_shp('./data/points_pilotes_eponte2.shp', prefix, zone)

# set up variogram 
a = 16 # km
v = pyemu.utils.ExpVario(contribution=1.0,a=a)
gs = pyemu.utils.GeoStruct(variograms=v,transform="log")
# set up krigging
ok = pyemu.utils.OrdinaryKrige(geostruct=gs,point_data=pp_df)
sr = SpatialReference(mm) # only for compatibility
ok.spatial_reference = sr 

# pandas dataframe of point where interpolation shall be conducted
# set up index for current zone and lay
idx = mm.param[par].izone[lay,:,:] > 0
x_select = xx[idx].ravel()
y_select = yy[idx].ravel()
zone_select = mm.param[par].izone[lay,:,:].ravel()

# compute kriging factors
kfac_df = ok.calc_factors(x_select, y_select, minpts_interp=1, maxpts_interp=20,
                     search_radius=500, verbose=False,
                     pt_zone=zone, forgive=False)

# write kriging factors to file
kfac_file = './data/kfac_{0}_l{1:02d}.csv'.format(par,lay+1)

ok.to_grid_factors_file(kfac_file)

# write template file (all layers and zones)

# write initial parameter value file (all zones)
pp_file = './data/pp_{0}_l{1:02d}.csv'.format(par,lay+1)

pp_df.to_csv(pp_file)

# -----------------------------------

# STEP 2 : KRIGE AND RUN 

# kriging from ppoint and factor files
kriged_values_df = pp_utils.fac2real(pp_file = pp_df ,factors_file = kfac_file)
#kriged_values_df.vals.fillna(kriged_values_df.vals.mean(),inplace=True)

idx = mm.param[par].izone[lay,:,:] > 0
mm.param[par].array[lay][idx] = kriged_values_df.vals

plt.imshow(mm.param[par].array[lay])

