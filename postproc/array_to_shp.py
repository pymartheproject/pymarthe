import numpy as np 
import fiona
from collections import OrderedDict

# -----------------------
#  grid_data_to_shp
# -----------------------

def grid_data_to_shp(data_list, x_values, y_values, file_path, field_name_list=['data']):
    """
    -----------
    Description:
    -----------
    Writes shapefile from structured rectangular grid data
    Based on Fiona library
    
    Parameters: 
    -----------
    data : List of numpy array with data.shape =  len(x_values), len(y_values)
    x_values : 1D numpy array with x coordinates of grid cell centers
    y_values : 1D numpy array with y coordinates of grid cell centers

    Returns:
    -----------
    True if successful, False otherwise 

    Example
    -----------
    data_list = [array1, array2]
    field_name_list = ['f1','f2']
    grid_data_to_shp(data_list, x_values, y_values, file_path, field_name_list)
    """
    # open collection 
    try : 
        collection = fiona.open(
                file_path,
                'w',
                driver=driver,
                crs=crs,
                schema=schema
                )
    except : 
        print('I/O error, check file path')
        return(False)
    
    # counter (record id)
    n=0
    # iterate over rows 
    for i in range(len(x_values)-1):
        # iterate over columns
        for j in range(len(y_values)-1):
            # data value
            val = data[i,j]
            # centroid coordinates
            x_c = x_values[i]
            y_c = y_values[j]
            # cell size
            dx = float(x_values[i+1] - x_c)
            dy = float(y_values[j+1] - y_c)
            # rectangle coordinates
            # from top left corner, clockwise
            rec_coor = [ (x_c - dx/2, y_c + dy/2), 
                         (x_c + dx/2, y_c + dy/2), 
                         (x_c + dx/2, y_c - dy/2), 
                         (x_c - dx/2, y_c - dy/2), 
                         (x_c - dx/2, y_c + dy/2) 
                       ]
            # set up record geometry and properties
            geometry = {'type': 'Polygon', 'coordinates':[rec_coor]}
            #properties = {'id':n}

            # fill data fields from list 
            #for data,fieldname in zip(data_list,field_name_list):
            #    properties[fieldname] = data

            # set up record 
            record = {'id':n, 'geometry':geometry, 'properties':properties}
            # write record
            collection.write(record)
            # counter increment
            n = n+1
            
    res = collection.close()
    
    return(res)


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

