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


# ---- developement for pilot points

# NOTE param, lay  will be iterated

par = 'kepon'

mm.add_param(par,1e-4)
izone = -1*np.ones( (nlay, nrow, ncol) )
izone[1,:,:] = 1 # 2nd layer parameterized with pilot points
mm.param[par].set_izone(izone=izone)

# -------- example for one layer and one zone
lay = 1
zone = 1

# load pilot points for given layer and zone
path_to_shp = os.path.join(mm.mldir,'sig','points_pilotes_eponte2.shp')
mm.param[par].pp_df_from_shp(path_to_shp, lay, zone)
mm.param[par].pp_from_rgrid(lay, n_cell=8)

plt.plot(mm.param[par].pp_dic[lay].x,mm.param[par].pp_dic[lay].y,'+')

# write template file (all layers and zones)
mm.param[par].write_pp_tpl()

# set upvariogram 
v = pyemu.utils.ExpVario(contribution=1.0,a=32) # sill and range (km)
gs = pyemu.utils.GeoStruct(variograms=v,transform="log")

# set up kriging
pp_df = mm.param[par].pp_dic[lay] # pointer to pilot point dataframe
ok = pyemu.utils.OrdinaryKrige(geostruct=gs,point_data=pp_df)
sr = SpatialReference(mm) # only for compatibility
ok.spatial_reference = sr 

# pandas dataframe of point where interpolation shall be conducted
# set up index for current zone and lay
x_coords, y_coords = mm.param[par].zone_interp_coords(lay,zone)

# compute kriging factors
kfac_df = ok.calc_factors(x_coords, y_coords, minpts_interp=1, maxpts_interp=50,
                     search_radius=800,pt_zone=1, num_threads = 4 )

# write kriging factors to file
kfac_file = os.path.join(mm.mldir,'param','kfac_{0}_l{1:02d}.dat'.format(par,lay+1))
ok.to_grid_factors_file(kfac_file)


# write initial parameter value file (all zones)
mm.param[par].write_pp_df()

# -----------------------------------

# STEP 2 : INTERPOLATE AND RUN

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

mm = MartheModel('../MONA_V3/mona.rma')
nlay, nrow, ncol = mm.nlay, mm.nrow, mm.ncol

# ---- NOTE param, lay and zone can be iterated

par = 'kepon'
mm.add_param(par,1e-6)
izone = -1*np.ones( (nlay, nrow, ncol) )
izone[1,:,:] = 1 # 2nd layer parameterized with pilot points
mm.param[par].set_izone(izone=izone)

lay = 1
zone = 1

# read pp_df files to get values at pilot points
mm.param[par].read_pp_df()

# update grid values for current par, lay, zone
mm.param[par].interp_from_factors()

map_2d =np.ma.masked_where(mm.grids['kepon'][1,:,:]==0, mm.grids['kepon'][1,:,:])
plt.imshow(np.log10(map_2d))
plt.ion()
plt.show()
plt.colorbar()


