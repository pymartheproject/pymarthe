# -*- coding: utf-8 -*-
import numpy as np
from itertools import islice
import pandas as pd
from pathlib import Path
from collections import OrderedDict

############################################################
#        Functions for reading and writing grid files
############################################################

# fixed a problem for grid files written from WinMarthe
# but not the best option. 
encoding = 'latin-1'

def read_grid_file(path_file):

    '''
    Description
    -----------

    This function reads Marthe grid files 
   
    Parameters
    ----------
    path_file : Directory path with parameter files
 
    Return
    ------
    A tuple containing following elements. 
    x_vals  : np.array grid x coordinates  
    y_vals  : np.array grid y coordinates 
    grid : 3d numpy array with shape (nlay,nrow,ncol) 

    Example
    -----------
    x, y, grid = read_grid_file(file_path)
    
    '''
    x_list = []
    y_list = []   
    grid_list = []

    # -- set lookup strings
    # begin of each grid
    lookup_begin  = '[Data]' 
    lookup_begin_constant = '[Constant_Data]' 
    # end of each grid
    lookup_end = '[End_Grid]'

    # -- open the file  
    data = open(path_file,"r",encoding = encoding)

    # --iterate over lines
    for num, line in enumerate(data, 1): #NOTE quel intérêt de commencer à 1 ? enumerate(data) mieux non ? 
        #search for line number with begin mark
        if lookup_begin in line:
            begin = num + 1
            constant = False 
        if lookup_begin_constant in line:
            begin = num 
            constant  = True # for uniform data
        #search for line number with end mark
        if lookup_end in line :
            if num  > begin: 
                end = num -1
                # case with different values 
                if constant == False :
                    full_grid  = []
                    # extract full grid from file
                    with open(path_file,"r",encoding = encoding) as text_file:
                        for line in islice(text_file, begin,  end ):
                             full_grid.append([float(v) for v in line.split()])
                    # select yrows, xcols, delr, delc, param in full_grid
                    x_vals = full_grid[0]
                    del x_vals[0:2] # remove first two zeros
                    x_vals = np.array(x_vals, dtype = np.float)
                    full_grid = full_grid[1:-1] #remove the first and the last line (x and delc) 
                    full_grid = np.array(full_grid)
                    y_vals = full_grid[:,1] 
                    grid_data = full_grid[:,2:-1]                  
                # case with constant (homogeneous) values
                if constant == True :
                    table_split = []
                    full_grid  = []
                    # select table
                    with open(path_file,"r",encoding = encoding) as text_file:
                        for line in islice(text_file, begin,  end ):
                            table_split.append(line.split())
                    constant_value = (float(table_split[0][0].split("=")[1]))
                    # select yrows, xcols, delr, delc, param in full_grid
                    x_vals = table_split[3]
                    x_vals = np.array(x_vals, dtype = np.float)
                    y_vals = table_split[7]
                    y_vals = np.array(y_vals, dtype = np.float)
                    grid_data = np.full((len(y_vals),len(x_vals)), constant_value)
                grid_list.append(grid_data)
                x_list.append(x_vals)
                y_list.append(y_vals)

    grid = np.stack(grid_list)

    return (x_vals,y_vals,grid)
    

def write_grid_file(path_file, x, y, grid):
    
    '''
    Description
    -----------
    Writer for regular grids, square cells.     
    This function writes text file with the same structure than parameter file in grid form
    
    Parameters
    ----------
    path_file : directory path to write the file. 
    The extension file must match the name of the parameter. Example : 'model.kepon'
    x : np.array grid x coordinates  
    y : np.array grid y coordinates  
    grid  : 3d numpy array with shape (nlay,nrow,ncol) 
   

    Example
    -----------
    write_grid_file(path_file,grid,x,y)
        
    '''
    # check regular mesh with square cell
    assert abs(x[1] - x[0]) == abs(y[1] - y[0])

    # infer square cell size 
    m_size = x[1] - x[0]

    grid_pp = open(path_file , "w")

    nrow, ncol = grid[0].shape

    nprow =  np.arange(1,nrow+1,1)
    npcol =  np.arange(0,ncol+1,1)

    #create a list of widths of the columns
    delc = [0,0] + [int(m_size)]*ncol
    #create a list of heights of the lines
    delr = [int(m_size)]*nrow

    xmin = x[0]  - 1 # NOTE why 1 ?
    ymin = y[-1] - 1 # NOTE why 1 ? 


    i = 0

    #Extract the name of the parameter from the file path
    parse_path = Path(path_file).parts
    file_name = parse_path[-1]
    param = file_name.split('.')[-1]

    for layer in grid:
        i = i + 1
        parameter = zip(*layer)
        grid_pp.write('Marthe_Grid Version=9.0 \n')
        grid_pp.write('Title=Travail                                                        '+param+'            '+str(i)+'\n')
        grid_pp.write('[Infos]\n')
        grid_pp.write('Field=\n')
        grid_pp.write('Type=\n')
        grid_pp.write('Elem_Number=0\n')
        grid_pp.write('Name=\n')
        grid_pp.write('Time_Step=-9999\n')
        grid_pp.write('Time=0\n')
        grid_pp.write('Layer=0\n')
        grid_pp.write('Max_Layer=0\n')
        grid_pp.write('Nest_grid=0\n')
        grid_pp.write('Max_NestG=0\n')
        grid_pp.write('[Structure]\n')
        grid_pp.write('X_Left_Corner='+str(xmin)+'\n')
        grid_pp.write('Y_Lower_Corner='+str(ymin)+'\n')
        grid_pp.write('Ncolumn='+str(ncol)+'\n')
        grid_pp.write('Nrows='+str(nrow)+'\n')
        grid_pp.write('[Data]\n')
        grid_pp.write('0 \t')
        [grid_pp.write(str(i)+'\t') for i in npcol]
        grid_pp.write('\n')
        grid_pp.write(('0 \t 0 \t'))
        [grid_pp.write(str(i)+'\t') for i in x]
        grid_pp.write('\n')
        for row, cols, param_line, col_size in zip(nprow, y,layer, delr) :
            grid_pp.write(str(row)+'\t'+str(cols)+'\t')
            [grid_pp.write(str(i)+'\t') for i in param_line]
            grid_pp.write(str(col_size) +'\t \n')
        [grid_pp.write(str(j)+'\t') for j in delc]
        grid_pp.write('\n')
        grid_pp.write('[End_Grid]\n')

    return

def read_obs(path_file):
    
    df_obs = pd.read_csv(path_file, sep='\t', skiprows=1, decimal =",",low_memory=False) 
    loc_ids = list(df_obs.columns[1:])
    df_obs.columns = ['date'] + loc_ids 
    df_obs.date = pd.to_datetime(df_obs.date,  format="%d/%m/%Y")
    df_obs = df_obs.set_index(df_obs.date)
    df_obs = df_obs.iloc[0:-2,:]

    return loc_ids,df_obs


def read_prn(prn_file):
    '''
    Description
    -----------

    This function reads file of simulated data (historiq.prn)
    and returns files for every points. Each file contains two columns : date and its simulation value
   
    Parameters
    ----------
    path_file : Directory path with simulated data 

    Return
    ------
    df_sim : Dataframe containing the same columns than in the reading file

    
    Example
    -----------
    read_prn(path_file)
        
    '''
    df_sim = pd.read_csv(prn_file, sep='\t', skiprows=3)  # Dataframe
    loc_ids = [x.rstrip('.1') for x in df_sim.columns][1:] 
    loc_ids = [x.rstrip(' ') for x in loc_ids]
    df_sim.columns = ['date'] + loc_ids  # Indicates the name of the columns in the DataFrame
    df_sim.date = pd.to_datetime(df_sim.date,  format="%d/%m/%Y")
    df_sim = df_sim.set_index(df_sim.date)
    df_sim = df_sim.iloc[:,2:-1]
    df_fluct = df_sim - df_sim.mean()
    return  df_sim,df_fluct


def extract_prn(prn_file, out_dir ="./", obs_dir = None):
    '''
    Description
    -----------
    Reads model.prn read_prn() and writes individual files for each locations. 
    Each file contains two columns : date and its simulation value
   
    Parameters
    ----------
    prn_file : Directory path with simulated data
    out_dir  : Directory path to write data
    obs_dir : Directory of observed values used for sim subset
    
    Return
    ------
        
    Example
    -----------
    extract_prn(path_file,'./output_data/')
        
    '''
    # read prn file
    df_sim,df_fluct = read_prn(prn_file)

    # if obs_dir is not provided, write all simulated dates  
    if obs_dir == None :
        for loc in df_sim.columns :
            # write individual files of simulated records
            df_sim.to_csv(out_dir+loc+'_abs.dat', columns = [loc], sep='\t', index=True, header=False)

    # if obs_dir is provided, get observed dates for each loc 
    else :         
        # iterate over simulated locations and get observed data 
        for obs_loc in df_sim.columns : 
            obs_file = os.path.join(obs_dir, obs_loc + '.dat')
            df_obs = pd.read_csv(obs_file, delim_whitespace=True,header=None,skiprows=1)
            df_obs.rename(columns ={0 : 'date', 1 :'value'}, inplace =True)
            df_obs.date = pd.to_datetime(df.date, format="%Y-%m-%d")
            df_obs.set_index('date', inplace = True)
            dates_out_dic[obs_loc] = df_obs.index
            # write individual files of simulated records
            df_sim.loc[df_obs.index].to_csv(out_dir+loc+'_abs.dat', columns = [loc], sep='\t', index=True, header=False)
    return



def extract_prn_fluct(prn_file, out_dir ="./", obs_dir = None):
    '''
    Description
    -----------
    Reads model.prn read_prn() and writes individual files for each locations. 
    Each file contains two columns : date and its simulation value
   
    Parameters
    ----------
    prn_file : Directory path with simulated data
    out_dir  : Directory path to write data
    obs_dir : Directory of observed values used for sim subset
    
    Return
    ------
        
    Example
    -----------
    extract_prn(path_file,'./output_data/')
        
    '''
    # read prn file
    df_sim,df_fluct = read_prn(prn_file)

    # if obs_dir is not provided, write all simulated dates  
    if obs_dir == None :
        for loc in df_sim.columns :
            # write individual files of simulated records
            df_fluct.to_csv(out_dir+loc+'_fluct.dat', columns = [loc], sep='\t', index=True, header=False)

    # if obs_dir is provided, get observed dates for each loc 
    else :         
        # iterate over simulated locations and get observed data 
        for obs_loc in df_fluct.columns : 
            obs_file = os.path.join(obs_dir, obs_loc + '.dat')
            df_obs = pd.read_csv(obs_file, delim_whitespace=True,header=None,skiprows=1)
            df_obs.rename(columns ={0 : 'date', 1 :'value'}, inplace =True)
            df_obs.date = pd.to_datetime(df.date, format="%Y-%m-%d")
            df_obs.set_index('date', inplace = True)
            dates_out_dic[obs_loc] = df_obs.index
            # write individual files of simulated records
            df_fluct.loc[df_obs.index].to_csv(out_dir+loc+'_fluct.dat', columns = [loc], sep='\t', index=True, header=False)
    return

def read_histo_file(path_file):

    '''
    Description
    -----------

    This function reads model.histo
   
    Parameters
    ----------
    path_file : Directory path with observation file
    

    Return
    ------
    df_histo : Dataframe containing the same columns than in the reading file

    ''' 
    # read fixed-width format file 
    df_histo = pd.read_fwf(path_file, skiprows = 1, widths= [30,7,2,7,2,7,1,12,30], header=None)
    # drop last row
    df_histo.drop(df_histo.tail(1).index,inplace=True)
    # drop dummy columns
    df_histo.drop([0,2,4,6], axis=1, inplace=True)
    # rename columns
    df_histo.columns = ['x','y','layer','id','label']
    # set types 
    df_histo = df_histo.astype({'x': float, 'y': float, 'layer': int})
    # set id as index 
    df_histo.set_index('id',inplace=True)
    return df_histo


def grid_data_to_shp(data_list, x_values, y_values, file_path, field_name_list,crs):
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
    crs = fiona.crs.from_epsg(2154)
    grid_data_to_shp(data_list, x_values, y_values, file_path, field_name_list,crs)
    """
    # open collection 
    driver = 'ESRI Shapefile'
    data_field_properties = [(field_name, 'float') for field_name in field_name_list]
    id_field_properties = [('id','int:10')]
    field_properties = OrderedDict(id_field_properties+ data_field_properties)
    schema = {'geometry': 'Polygon', 'properties':field_properties}
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
            properties = {'id':n}

            # fill data fields from list 
            for data,fieldname in zip(data_list,field_name_list):
                properties[fieldname] = data[i,j]

            # set up record 
            record = {'id':n, 'geometry':geometry, 'properties':properties}
            # write record
            collection.write(record)
            # counter increment
            n = n+1
            
    res = collection.close()
    
    return(res)


