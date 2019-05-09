import sys 
import pandas as pd

sys.path.append('/Users/apryet/Programmes/python/pyemu/')
import pyemu

sys.path.append('/Users/apryet/Programmes/python/')
from adeqwat.pymarthe import * 

mm = MartheModel('/Users/apryet/recherche/adeqwat/dev/adeqwat/sandbox/mona/v3/mona.rma')


# import grid data 
x_list, y_list, grid_list = read_grid_file('./mona/v3/mona.permh')


# mask to get points to interpolate 


# load pilot points 
pp_df = pd.read_csv('./data/ppoints.csv')

# range 
a = 16000 #m

v = pyemu.utils.ExpVario(contribution=1.0,a=a)
gs = pyemu.utils.GeoStruct(variograms=v,transform="log")

ok = pyemu.utils.OrdinaryKrige(geostruct=gs,point_data=pp_df)



sr = SpatialReference(1,420)

ok.spatial_reference = sr 


grid_df = pd.read_csv('./data/regular_grid_points.csv')

x = grid_df['x'].values
y = grid_df['y'].values

kfac_df = ok.calc_factors(x,y,minpts_interp=1,maxpts_interp=20,
                     search_radius=1.0e+10,verbose=False,
                     pt_zone=None,forgive=False)


ok.to_grid_factors_file('./data/grid_factor.csv')

pst = pyemu.pst_utils.pst_from_io_files(tpl_files,input_files,ins_files,output_files)

pyemu.utils.geostats.fac2real('./data/ppoints.csv',out_file='hk_layer_1.ref')



'/data/grid_factor.csv'

with open(filename, 'w') as f:
    f.write(points_file + '\n')
    f.write(zone_file + '\n')
    f.write("{0} {1}\n".format(self.spatial_reference.ncol, self.spatial_reference.nrow))
    f.write("{0}\n".format(self.point_data.shape[0]))
    [f.write("{0}\n".format(name)) for name in self.point_data.name]
    t = 0
    if self.geostruct.transform == "log":
        t = 1
    pt_names = list(self.point_data.name)
    for idx,names,facts in zip(self.interp_data.index,self.interp_data.inames,self.interp_data.ifacts):
        if len(facts) == 0:
            continue
        n_idxs = [pt_names.index(name) for name in names]
        f.write("{0} {1} {2} {3:8.5e} ".format(idx+1, t, len(names), 0.0))
        [f.write("{0} {1:12.8g} ".format(i+1, w)) for i, w in zip(n_idxs, facts)]
        f.write("\n")

