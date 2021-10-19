# -*- coding: utf-8 -*-
import os
import numpy as np
from itertools import islice
import pandas as pd
import re
from pathlib import Path
from collections import OrderedDict
import re, ast

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

def read_histobil_file(path_file,pastsp):
    '''
    Description
    ----------
    This function reads Marthe grid files
    Parameters
    ----------
    path_file : Directory path with parameter files
    pastsp : Time steps number
    Return
    -----
    dfzone_list : list of zone datframes
    Example
    -----------
    dfzone_list = read_histobil_file(file_path,pastsp)
    '''
    dfzone_list = []
    # -- set lookup strings
    # begin of each grid
    lookup_begin  = "Bilan de débit d'aquifère : Zone"
    # end of each grid
    pastsp = pastsp
    # -- open the file
    data = open(path_file,"r",encoding = encoding)
    # --iterate over lines
    zone_ids = []
    for num, line in enumerate(data, 1): 
    #search for line number with begin mark
        if line.startswith(lookup_begin):
            zone_id = line.split('\t')[0]
            zone_ids.append(zone_id)
            begin = num +1
            end = begin+1+pastsp
            # extract full grid from file
            full_grid = []
            with open(path_file,"r",encoding = encoding) as text_file:
                for line in islice(text_file, begin-1,  end ):
                    full_grid.append([(v) for v in line.split()])
            # select yrows, xcols, delr, delc, param in full_grid
            columns = full_grid[0]
            df_zone = pd.DataFrame(full_grid,columns = columns)
            df_zone.drop(df_zone.index[0:2],axis = 0,inplace = True)
            df_zone.set_index(pd.DatetimeIndex(df_zone.Date.iloc[:,0]),inplace = True)
            df_zone.drop(df_zone.Date,axis=1,inplace = True)
            dfzone_list.append(df_zone)
    #zone_dic = {k:v for k,v in zip(zone_ids, dfzone_list)}
    return (dfzone_list,zone_ids)

def extract_variable(path_file,pastsp,variable,dti_present,dti_future,period,out_dir = None):
    '''
    Description
    ----------
    This function extact the variable of interest from histobil file 
    and writes individual files for each zone 
    Each file contains two columns : date and its simulation value
    Parameters
    ----------
    path_file : Directory path with parameter files
    pastsp : Time steps number
    variable : the variable of interest : one of the columns of histobil dataframe
    dti_present : date from which to start the present period Y-M-D
    dti_future : date from which to start the future period : formar Y-M-D
    period : duration period 
    out_dir : directory where to save files 
    Return
    -----
    Saved files 
    Example
    -----------
    dfzone_list = extract_variable(file_path,pastsp = 40,'1990-12-08','2000-12-08',period = 2)
    '''
    dfzone_list,zone_ids = read_histobil_file(path_file,pastsp)
    for i in range(len(dfzone_list)):
        df_variable = pd.to_numeric(dfzone_list[i][variable]).cumsum() # take the column of interest variable
        present_period = pd.date_range(dti_present, periods=period, freq='A') # Extarct present period 
        future_period = pd.date_range(dti_future, periods=period, freq='A') # Extract future period 
        scum_present_mean = df_variable[present_period].mean() # Mean of cumulative sum for the present period
        scum_future_mean  = df_variable[future_period].mean() # Mean of cumulative sum for the future period
        delta_s_relative = (scum_future_mean - scum_present_mean) / abs(scum_present_mean) # Relative delta  
        df = pd.DataFrame([scum_present_mean,scum_future_mean,delta_s_relative],index = [dti_present,dti_future,dti_future])
        df.to_csv( out_dir+'stock_'+(zone_ids[i].split()[-1])+'.dat', header=False,sep ='\t') 



    

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

def read_obs(path_file, starting_date, ending_date):
    '''
    Description
    ----------
    This function reads initial obs data
    Parameters
    ----------
    path_file     : Directory path with obs data 
    starting_data : Year of starting period (str)
    ending_data   : Year of ending period (str)
    Return
    -----
    list of observations and the dataframe 
    -----------
    '''
    df_obs = pd.read_csv(path_file, sep='\t',index_col = 'Date',parse_dates = True)
    df_obs = df_obs[starting_date:ending_date]

    return df_obs


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
    return  df_sim


def extract_prn(prn_file,fluct, out_dir ="./", obs_dir = None):
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
    if fluct == True:
        df_sim = read_prn(prn_file)
        df_fluct = df_sim - df_sim.mean()
    else : 
        df_sim = read_prn(prn_file)

    # if obs_dir is not provided, write all simulated dates  
    if obs_dir == None :
        for loc in df_sim.columns :
            # write individual files of simulated records
            if fluct == True:
                df_sim.to_csv(out_dir+loc+'_abs.dat', columns = [loc], sep='\t', index=True, header=False)
                df_fluct.to_csv(out_dir+loc+'_fluct.dat', columns = [loc], sep='\t', index=True, header=False)
            else :
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
            if fluct == True:
                df_sim.loc[df_obs.index].to_csv(out_dir+loc+'_abs.dat', columns = [loc], sep='\t', index=True, header=False)
                df_fluct.loc[df_obs.index].to_csv(out_dir+loc+'_fluct.dat', columns = [loc], sep='\t', index=True, header=False)
            else:
                df_sim.loc[df_obs.index].to_csv(out_dir+loc+'_abs.dat', columns = [loc], sep='\t', index=True, header=False) 

    return



def read_histo_file (path_file):

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

    histo_file = open(path_file,"r",encoding = 'latin-1')
    x_list, y_list, lay_list, id_list, label_list = [],[],[],[],[]

    for line in histo_file :
        # skip lines without slash or without number slash (gigogne)
        try :
            if (line[2] != '/' ):
                continue
        except : 
            continue
        
        # check histo definition
        if (re.search(r'(?<=(/   =   /))\w+', line).group(0) == 'XCOO') :
            # get positions within line string
            xpos = line.find('X=')
            ypos = line.find('Y=')
        else        :
            xpos = line.find('C=')
            ypos = line.find('L=')

        ppos = line.find('P=')
        scpos = line.find(';')
        # extract x, y, lay from line string
        x_list.append(float(line[xpos+2:ypos]))
        y_list.append(float(line[ypos+2:ppos]))
        lay_list.append(int(line[ppos+2:scpos]))
        # split id string and get label if any
        id_string = line.split('Name=')[1].strip().split()
        id_list.append(id_string[0])
        if len(id_string)>1:
            label_list.append(' '.join(id_string[1:]))
        else :
            label_list.append('')
    df_histo = pd.DataFrame({'id':id_list,'x': x_list, 'y': y_list, 'layer': lay_list, 'label':label_list})
    df_histo.set_index('id',inplace=True)
    histo_file.close()
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




def read_listm_qfile(qfile, mode = 'aquifer'):
    """
    -----------
    Description:
    -----------
    Read single flow rate text file for a list of cells
    
    Parameters: 
    -----------
    qfile (str) : simple file containing pumping data
                  Format: 4 columns without headers
                          V (value, float), C (column, int),
                          L (line, int), P (layer, int) 
    mode (str) : type of cell localisation pumping
                Can be 'aquifer' (columns, line, layer)
                or 'river' ('affluent', 'troncon')
                Default is 'aquifer'

    Returns:
    -----------
    arr (numpy array) : withdrawal data in structured array

    Example
    -----------
    arr = read_qfile('qfile.txt')
    """
    if mode == 'aquifer':
        # ---- Set data types
        dt = [('v','f8'), ('c', 'i4'), ('l', 'i4'), ('p', 'i4')]
        # ---- Read qfile as array (separator = any whitespace)
        base_arr = np.loadtxt(qfile, dtype = dt)
        # ---- Add boundnames
        data = [tuple([v,c,l,p ] + ['_'.join(map(str,[c,l,p]))]) for v,c,l,p in base_arr]
        arr = np.array(data, dtype = dt + [('boundname', '<U25')])
    if mode == 'river':
        print('LISTM OPERATION NOT AVAILABLE YET FOR RIVER PUMPING')
        # # ---- Set data types
        # dt = [('v','f8'), ('a', 'i4'), ('t', 'i4')]
        # # ---- Read qfile as array (separator = any whitespace)
        # base_arr = np.loadtxt(qfile, dtype = dt)
        # # ---- Add boundnames
    # ---- Return data
    return arr




def read_record_qfile(qfile, qcol):
    """
    -----------
    Description:
    -----------
    Read single cell pumping record
    
    Parameters: 
    -----------
    qfile (str) : simple file containing pumping data
                  Format: 4 columns with headers
                          V (value, float), C (column, int),
                          L (line, int), P (layer, int)
    qcol (int) : column number to read in qfile (0-based)

    Returns:
    -----------
    header (str) : single column name (column n° qcol)
    arr (numpy array) : withdrawal data in structured array

    Example
    -----------
    arr = read_qfile('qfile.txt')
    """
    # ---- Read header (='boundname')
    header = np.loadtxt(qfile, usecols=qcol, dtype = np.str, max_rows=1).tolist()
    # ---- Set data types
    dt = [('v','f8')]
    # ---- Read qfile as array (separator = any whitespace)
    arr = np.loadtxt(qfile, usecols=qcol, skiprows=1, dtype = dt)
    # ---- Return data
    return header, arr




def read_pastp(pastp_file):
    """
    -----------
    Description:
    -----------
    Read .pastp file
    (For now, just pumping data can be extracted from .pastp file) 
    
    Parameters: 
    -----------
    pastp_file (str) : path to .pastp marthe file

    Returns:
    -----------
    content (dict) : content of .pastp file by timestep blocks
                     Format : {istep0 :[line0, ..., lineN], .., }
    nstep (int) : number of timestep

    Example
    -----------
    lines, nstep = read_pastp('mm.pastp')
    """
    # ---- Initiate lookups
    fstep_begin = '*** Début de la simulation'
    istep_begin = '*** Le pas'
    istep_end = '/*****/***** Fin de ce pas'

    # ---- Initiate content variables
    istep, content, data = None, {}, []

    # ----- Collect pastp file lines
    with open(pastp_file,"r", encoding = encoding) as f:
        # ---- Fetch line content by timestep block
        for line in f.readlines():
            # -- Start recording data
            if fstep_begin in line:
                istep = 0
            # -- Start new timestep
            if istep_begin in line:
                istep += 1
            # -- End of the current timestep
            if istep_end in line:
                content[istep] = data
                data = []
                # -- Collect line
            else:
                if not istep is None:
                    data.append(line)

    # ---- Get number of timestep
    nstep = len(content)

    # ---- Return pumping data as dictionary
    return nstep, content






def extract_pastp_pumping(content, mode):
    """
    -----------
    Description:
    -----------
    Extract pumping data from .pastp file content (by timestep blocks)
    NOTE : 2 types of pumping condition can be read:
                - LIST_MAIL (regional model)
                - MAILLE (local model) single value or record
    
    Parameters: 
    -----------
    content (dict) : content of .pastp file by timestep blocks
                     Come from read_pastp() function
    mode (str) : type of cell localisation pumping
                 Can be 'aquifer' (columns, line, layer)
                 or 'river' ('affluent', 'troncon')
                 Default is 'aquifer'
    Returns:
    -----------
    pumping_data (dict) : all pumping data by timestep
    qfilenames (dict) : all qfiles (full) names by timestep

    Example
    -----------
    lines, nstep = read_pastp('mm.pastp')
    pumping_data, qfilenames = extract_pastp_aqpumping(lines, nstep)
    """
    # ---- Set pastp tag according to pumping mode
    mode_tag = '/DEBIT/' if mode == 'aquifer' else '/Q_EXTER_RIVI/'

    # ---- Initialize dictionaries
    pumping_data, qfilenames = [{i:None for i in range(len(content))} for _ in range(2)]

    # -- Set regular expression of numeric string (int and float)
    re_num = r"[-+]?\d*\.?\d+|\d+" 

    # ---- Initialize array data type
    dt = [('v','f8'), ('c', 'i4'), ('l', 'i4'), ('p', 'i4'), ('boundname', '<U25')]

    for istep, lines in content.items():

        # ---- Iterate over all pastp file lines
        for line in lines:

            # -- Check if a pumping condition is applied
             if mode_tag in line:
                
                # -- Check which type of pumping condition is provided 

                # 1) Manage LIST_MAIL (list of pumping cells in external file)
                if '/LIST' in line:
                    # -- Get path to the qfile normalized
                    path = line.partition('N:')[-1]
                    qfilename = os.path.normpath(path.strip())
                    # -- Extract data from LISTM qfile (as a structure array)
                    arr = read_listm_qfile(qfilename, mode = mode)
                    # -- Set qfilename with data localisation
                    qfilename_arr = np.array([f'{qfilename}&ListmLin={icell}' for icell in range(len(arr))], dtype = np.str)
                    # -- Set pumping data
                    if isinstance(pumping_data[istep], np.ndarray):
                        pumping_data[istep] = np.append(pumping_data[istep], arr)
                        qfilenames[istep] = np.append(qfilenames[istep], qfilename_arr)
                    else:
                        pumping_data[istep] = arr
                        qfilenames[istep] = qfilename_arr

                # 2) Manage MAILLE (unique pumping cell)
                if '/MAILLE' in line:

                    # -- 2.1) Manage single cell pumping record  
                    if "File=" in line:
                        # -- If column is provided (otherwise it's the first column)
                        if "Col=" in line:
                            loc, file, col = map(str.strip, line[line.index('C='):].split(';'))
                        else:
                            loc, file = map(str.strip, line[line.index('C='):].split(';'))
                            col = 'Col=1'

                        # -- Get col, line, plan, value, qfilename and qcol
                        c,l,p,v = map(ast.literal_eval, re.findall(re_num, loc))
                        qfilename =  os.path.normpath(file.split('=')[1].strip())
                        qcol = int(col.split('=')[1])

                        # -- Extract data from pumping record file /!\columns number is 0-based in python/!\
                        bdname, record = read_record_qfile(qfilename, qcol-1)

                        # -- Set pumping data
                        pump_steady = [np.array([(v,c,l,p, bdname)], dtype = dt)]
                        pump_transient = [np.array([(record['v'][iistep],c,l,p,bdname)], dtype = dt) for iistep in range(len(content)-1)]
                        pump_arr = np.array(pump_steady + pump_transient, dtype = dt)

                        for iistep in content.keys():
                            if isinstance(pumping_data[iistep], np.ndarray):
                                pumping_data[iistep] = np.append(pumping_data[iistep], pump_arr[iistep])
                            else:
                                pumping_data[iistep] = pump_arr[iistep]

                        # -- Set qfilenames
                        qfile_steady = [np.array([None])]
                        qfile_transient = [np.array([f'{qfilename}&RecordCol={qcol-1}RecordLin={iistep+1}'], dtype = np.str) for iistep in range(len(content)-1)]
                        qfile_arr = np.array(qfile_steady + qfile_transient)

                        for iistep in content.keys():
                            if isinstance(qfilenames[iistep], np.ndarray):
                                qfilenames[iistep] = np.append(qfilenames[iistep], qfile_arr[iistep])
                            else:
                                qfilenames[iistep] = qfile_arr[iistep]

                    # -- 2.2) Manage single cell single pumping
                    else:
                        # -- Parse pumping informations (localisation, value)
                        parse = line[line.index('C='):].split(';')[0]
                        c,l,p,v = map(ast.literal_eval, re.findall(re_num, parse))
                        bdname = '_'.join(map(str,[c,l,p]))
                        arr = np.array((v,c,l,p), dtype = dt)

                        # -- Set pumping data & qfilenames
                        if isinstance(pumping_data[istep], np.ndarray):
                            pumping_data[istep] = np.append(pumping_data[istep], arr)
                            qfilenames[istep] = np.append(qfilenames[istep], np.array([None]))
                        else:
                            pumping_data[istep] = arr
                            qfilenames[istep] = np.array([None])


    # -- Return filename and data (as array)
    return pumping_data, qfilenames



def replace_text_in_file(file, match, subs, flags=0):
    """
    Function to replace string by another in text file

    Parameters:
    ----------
    file (str) : file name or full path
    match (str) : str/regex to match
    subs (str) : replaced string to write
    flags (re object) : flag to add in match expression
                        This is a regex object like:
                         - re.IGNORECASE
                         - re.VERBOSE
                         - ...
                        Default is 0

    Returns:
    --------
    Rewrite replaced text in file (inplace)

    Examples:
    --------
    file = 'replace_text_in_file.txt'
    match = 'Imass'
    subs = 'Initial_mass'
    replace_text_in_file(file, match, subs)
    """
    # ---- Open file in r+ mode
    with open(file, "r+") as f:
        # ---- Extract file content as string
        text = f.read()
        # ---- Search matches
        pattern = re.compile(re.escape(match), flags)
        # ---- Replace matches
        new_text = pattern.sub(subs, text)
        # ---- Overwrite modification
        f.seek(0)
        f.truncate()
        f.write(new_text)







def convert_at2clp(aff, trc, mlname, mldir = ''):
    """
    Function convert 'affluent' / 'tronçon' to column, line, plan (layer)

    Parameters:
    ----------
    aff (int) : number of the reach
    trc (int) : number of the section in tributary stream
    mlname (str) : model name
    mldir (str) : model directory
                  Dafault is ''

    Returns:
    --------
    c,l,p (int) : column, line, plan (layer)

    Examples:
    --------
    aff, trc = 243, 154
    c,l,p = convert_at2clp(aff, trc, 'mymodel')
    """
    # ---- Fetch reach and section file from model name
    aff_file, trc_file = [os.path.join(mldir, f'{mlname}.{x}_r') for x in ['aff','trc']]
    # ---- Set regular expression of numeric string (int and float)
    re_num = r"[-+]?\d*\.?\d+|\d+" 
    # ---- Fetch layer (p)
    with open(aff_file) as f:
        for line in  f:
            if line.startswith('Layer='):
                p = ast.literal_eval(re.findall(re_num, line)[0])
    # ---- Read grids
    aff_grid, trc_grid = [read_grid_file(f)[-1] for f in [aff_file, trc_file]]
    # ---- Get row and column corresponding to aff/trc arguments
    row, col = np.where((aff_grid == aff) & (trc_grid == trc))[1:]
    # ---- Convert to c, l
    c, l = [arr[0] for arr in [col, row]]
    # ---- Return basic c,l,p
    return c,l,p+1




def convert_at2clp_pastp(pastp_file, content, mlname=None, mldir = ''):
    """
    Function convert 'affluent' / 'tronçon' to column, line, plan (layer)
    and rewrite it in pastp file

    Parameters:
    ----------
    pastp_file (str) : path to pastp file
    content (list) : content of pastp file (from read_pastp() function)
    mlname (str) : model name
                   If None, mlname = pastp_file without extension
                   Default is None
    mldir (str) : model directory
                  Default is ''

    Returns:
    --------
    Replace lines inplace in pastp file

    Examples:
    --------
    pastp_file = 'mymodel.pastp'
    _, content = read_pastp(pastp_file)
    convert_at2clp_pastp(pastp_file, content)
    """
    # ---- Set regular expression of numeric string (int and float)
    re_num = r"[-+]?\d*\.?\d+|\d+"

    # ---- Get content at every timestep
    for istep, lines in content.items():
        # ---- Iterate over each line
        for line in lines:
            # ---- Check if the line contain aff/trc
            if all(s in line for s in ['Q_EXTER_RIVI','A=','T=']):
                # ---- Replace TRONCON for MAILLE
                mail_line = line.replace('TRONCON', 'MAILLE')
                # ---- Get substring to replace
                s2replace = line[line.index('A='):line.index('V=')]
                # ---- Fetch aff/trc as number
                a, t = map(ast.literal_eval, re.findall(re_num, s2replace))
                # ---- Convert aff/trc to column, line, plan (layer)
                if mlname is None:
                    c,l,p = convert_at2clp(aff=a, trc=t, mlname = pastp_file.replace('.pastp', ''))
                else:
                    c,l,p = convert_at2clp(aff=a, trc=t, mlname = os.path.join(mldir,mlname))
                # ---- Build substring to replace
                sub = 'C={:>7}L={:>7}P={:>7}'.format(c,l,p)
                # ---- Build entire line to be replace for
                l2replace = mail_line.replace(s2replace,sub)
                # ---- Replace text in pastp file
                replace_text_in_file(pastp_file, line, l2replace)




