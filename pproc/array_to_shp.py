import numpy as np 
import fiona
from collections import OrderedDict

# -----------------------
#  grid_data_to_shp
# -----------------------

# moved 

# -----------------------
# testing 
# -----------------------

# dummy data for testing 
x_values = np.arange(386034,386044)
y_values = np.arange(6434859,6434869)

nrow = len(x_values)
ncol = len(y_values)

data1 = np.ones((nrow,ncol))*2.2
data2 = np.ones((nrow,ncol))*3.2


field_name_list = ['f1','f2']
data_list = [data1,data2]

file_path = './shapefile/grid.shp'

driver = 'ESRI Shapefile'
crs = fiona.crs.from_epsg(2154)
#crs = {'lon_0': 3, 'ellps': 'GRS80', 'y_0': 6600000, 'no_defs': True, 'proj': 'lcc', 'x_0': 700000, 'units': 'm', 'lat_2': 44, 'lat_1': 49, 'lat_0': 46.5}

schema = {'geometry': 'Polygon', 'properties': OrderedDict([('id', 'int:10'),(field_name,'float')])}

# write grid 
grid_data_to_shp(data, x_values, y_values, file_path, field_name='head')

