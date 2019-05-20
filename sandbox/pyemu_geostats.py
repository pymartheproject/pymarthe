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


# new parameter

mm.add_parameter('kepon',1e-3)

# load kepon
mm.load_grid('kepon')




# load pilot points 
pp_shp_file = '/Users/apryet/recherche/adeqwat/dev/adeqwat/sandbox/data/points_pilotes_eponte2.shp'

pp_df = geopandas.read_file(pp_shp_file)

pp_df['x'] = pp_df.geometry.x
pp_df['y'] = pp_df.geometry.y
pp_df['name'] =  [ 'pp' + str(id) for id in pp_df['id'] ]
pp_df['zone'] = 1
pp_df['parval1'] = pp_df['y']


# set up variogram 
a = 16 # km
v = pyemu.utils.ExpVario(contribution=1.0,a=a)
gs = pyemu.utils.GeoStruct(variograms=v,transform="log")
# set up krigging
ok = pyemu.utils.OrdinaryKrige(geostruct=gs,point_data=pp_df)
sr = SpatialReference(mm) # only for compatibility
ok.spatial_reference = sr 




# define current parameter 
par = 'kepon'

# define current zone 
zones = np.unique(mm.izone[par])

for zone in zones :
    if zone == 0 : # inactive cells
        continue
    elif zone < 0 :  # zones of piecewise constancy
        continue
    elif zone > 0 : # parameterization with pilot points
        continue

# ---- developement for pilot points

# NOTE in the future will param, zone and  will be iterated
zone = 1
lay = 0

mm.set_izone(par,1)

# set up index for curret zone and lay
idx = mm.izone[par][lay,:,:] == zone

# pandas dataframe of point where interpolation shall be conducted
xx, yy = np.meshgrid(mm.x_vals,mm.y_vals)
x_select = xx[idx].ravel()
y_select = yy[idx].ravel()

# compute kriging factors
kfac_df = ok.calc_factors(x_select, y_select, minpts_interp=1, maxpts_interp=20,
                     search_radius=25, verbose=False,
                     pt_zone=zone, forgive=False)

# write kriging factors to file
ok.to_grid_factors_file('./data/grid_factor.csv')

# -----------------------------------

# kriging from ppoint and factor files
kriged_values_df = pest_utils.fac2real(pp_file = './data/ppoints.csv',factors_file = './data/grid_factor.csv')

data_df = pd.merge(kfac_df,kriged_values_df,how = 'left',left_index=True, right_index=True)

data_df.vals = data_df.vals.fillna( data_df.vals.mean() )

# initialize new array
kriged_array_2d = np.array(mm.imask[lay,:,:],dtype = float)
# build up 2d array, seems to match without sorting 
kriged_array_2d[idx] = data_df['vals']

plt.imshow(kriged_array_2d)




mm.grids[par] = kriged_array_3d

kepon_interp[idx] = kriged_values

marthe_utils.write_grid_file('/data/mona.kepon',[kepon_interp*,x,y)

pst = pyemu.pst_utils.pst_from_io_files(tpl_files,input_files,ins_files,output_files)


