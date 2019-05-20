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
imask = mm.imask

# ----------------------------------------------------------------------------
# ------------------------------ kepon ---------------------------------------
# ----------------------------------------------------------------------------

# --- set up variables

# base data 
parname = 'kepon' # parameter name 
pp_shpfiles_prefix = 'points_pilotes_eponte' # shapefile prefix 
value = 1e-7 # initial, default value m/s

# geostatistical data 
a = 16 # variogram range, km
minpts_interp=1
maxpts_interp=20
search_radius=25

# --- parameter setup 

mm.grids['permh'] = 1e-6

mm.write_grids() #

# izone (one id per layer)
izone = np.stack([np.ones((mm.nrow, mm.ncol))*lay  for lay in range(mm.nlay) ])*mm.imask

# setup kepon parameter 
# new class MartheParam
mm.add_par(parname, value)

# load shapefiles 
pp_shpfiles = [ pp_shpfiles_prefix + str(lay) +'.shp' for lay in range(nlay)]

# build up pp_df from shapefiles 
for lay in range(nlay) : 
    kepon_par.set_pp_df( pp_shpfiles[i], lay = lay )

# set PyEMU geostats 
v = pyemu.utils.ExpVario(contribution=1.0,a=a)
gs = pyemu.utils.GeoStruct(variograms=v,transform="log")
ok = pyemu.utils.OrdinaryKrige(geostruct=gs,point_data=pp_df)
sr = SpatialReference(mm) # for compatibility only
ok.spatial_reference = sr 

# ------ compute kriging factors 

for lay in range(nlay) :
    zone_ids = np.unique(np.unique(par.izone[lay,:,:]))
    pp_zone_ids = 
    for pp_zone_id in pp_zone_ids :

        # get coordinates where to perform interpolation
        x_select, y_select = kepon_par.interp_coords(zone = pp_zone_id)

        # compute kriging factors
        kfac_df = ok.calc_factors(x_select, y_select, minpts_interp=1, maxpts_interp=20,
                             search_radius=25, verbose=False,
                             pt_zone=zone, forgive=False)

        # set kriging factor file
        kfac_filename = 'kfac' + kepon_par.name + str(lay) + '_' + str(pp_zone_id) +'.csv'
        kfac_file = os.path.join(path,kfac_filename)

        # write kriging factors to file
        ok.to_grid_factors_file(kfac_file)

# ------- krige 

# kriging from ppoint and factor files
kriged_values_df = pest_utils.fac2real(pp_file = './data/ppoints.csv', factors_file = './data/grid_factor.csv')

data_df = pd.merge(kfac_df,kriged_values_df,how = 'left',left_index=True, right_index=True)

data_df.vals = data_df.vals.fillna( data_df.vals.mean() )

# initialize new array
kriged_array_2d = np.array(mm.imask[lay,:,:],dtype = float)
# build up 2d array, seems to match without sorting 
kriged_array_2d[idx] = data_df['vals']


# ----------------------------------------------------------------------------
# ----------------------- PEST control file  ---------------------------------
# ----------------------------------------------------------------------------



