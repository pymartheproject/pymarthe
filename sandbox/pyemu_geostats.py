import sys 
import os 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import geopandas 

sys.path.append('/Users/apryet/Programmes/python/pyemu/')
import pyemu

sys.path.append('/Users/apryet/Programmes/python/')
from adeqwat.pymarthe import * 

from adeqwat.utils import pest_utils
from adeqwat.utils import marthe_utils

# load Marthe model
mm = MartheModel('../MONA_V3/mona.rma')

# load kepon
mm.load_grid('kepon')


# load pilot points 

pp_shp_file = '/Users/apryet/recherche/adeqwat/dev/adeqwat/sandbox/data/points_pilotes_eponte2.shp'

pp_df = geopandas.read_file(pp_shp_file)

pp_df['x'] = pp_df.geometry.x
pp_df['y'] = pp_df.geometry.y
pp_df['name'] =  [ 'pp' + str(id) for id in pp_df['id'] ]
pp_df['zone'] = 1
pp_df['parval1'] = np.random.rand( pp_df.shape[0] ) * 10**(-3)

pp_df.to_csv('./data/ppoints.csv',sep=' ',index = False, header=False, columns=['name','x','y','zone','parval1'])

# set up variogram 
a = 16 # km
v = pyemu.utils.ExpVario(contribution=1.0,a=a)
gs = pyemu.utils.GeoStruct(variograms=v,transform="log")
# set up krigging
ok = pyemu.utils.OrdinaryKrige(geostruct=gs,point_data=pp_df)
sr = SpatialReference(mm) # only for compatibility
ok.spatial_reference = sr 

# import interpolation support 

zone = 1

xx, yy = np.meshgrid(mm.x_vals,mm.y_vals)

df = pd.DataFrame(data={'x':xx.ravel(),'y':yy.ravel(),'parval':mm.izone[lay,:,:].ravel()})

lay = 0

idx = mm.imask[lay,:,:] != 0

df = 


# compute kriging factors
kfac_df = ok.calc_factors(x_select,y_select,minpts_interp=1,maxpts_interp=20,
                     search_radius=25,verbose=False,
                     pt_zone=None,forgive=False)

# write kriging factor file
ok.to_grid_factors_file('./data/grid_factor.csv')

# kriging from ppoint and factor files
kriged_values = pest_utils.fac2real(pp_file = './data/ppoints.csv',factors_file = './data/grid_factor.csv')

kepon_interp = np.array(mm.imask[lay,:,:],dtype=np.float)
kepon_interp[idx] = kriged_values

marthe_utils.write_grid_file('/data/mona.kepon',[kepon_interp*,x,y)

pst = pyemu.pst_utils.pst_from_io_files(tpl_files,input_files,ins_files,output_files)


