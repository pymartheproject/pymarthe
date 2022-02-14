# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import re, ast
import warnings

from pymarthe import *
from .grid_utils import MartheGrid

# Set encoding
encoding = 'latin-1'


# Set commun no data values
NO_DATA_VALUES = [-9999., -8888.]


def get_mlfiles(rma_file):
    """
    -----------
    Description:
    -----------
    Extract all Marthe file paths frop .rma file

    Parameters: 
    -----------
    rma_file (str): Marthe .rma file path

    Returns:
    -----------
    mfiles_dic (dict) : all .rma file paths
                        Format: {'permh': 'model.permh', ...}

    Example
    -----------
    rma_file = 'mymarthemodel.rma'
    mlfiles = get_mlfiles(rma_file)
    """
    # ---- Get .rma content as text
    mldir, mlname = os.path.split(rma_file)
    with open(rma_file, 'r', encoding=encoding) as f:
        content = f.read()
    # ---- Fetch all marthe file paths
    re_mlfile = r'\n(\w+.\w+)\s*'
    mlfiles = re.findall(re_mlfile, content)
    # ---- Build dictionary with all marthe file path
    mlfiles_dic = {mlfile.split('.')[1]: 
                        os.path.normpath(
                            os.path.join(mldir, mlfile)) for mlfile in mlfiles}
    return mlfiles_dic



def progress_bar(percent, barlen=50):
    """
    Function to print progress bar by percent according
    to a given bar length.

    Parameters:
    ----------
    percent (float) : progress percent
                      Must be between 0 and 1.
    barlen (int,optional) : bar length showed on console screen.
                            Default is 50

    Returns:
    --------
    Write in sys.stdout

    Examples:
    --------
    l = ['azerty'] * 100000
    for i, v in enumerate(l):
        percent = i/len(l)
        var = l[1:] + ''.join(v) + l[:-1]
        progress_bar(percent, berlen=70)

    """
    
    # ---- Write graphical percent according to barlen
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'='*int(barlen * percent):{barlen}s}] {int(100*percent)}%")
    sys.stdout.flush()



def get_layers_infos(layfile, base = 1):
    """
    -----------
    Description:
    -----------
    Extract layer informations from Marthe .layer file 

    Parameters: 
    -----------
    layfile (str): Marthe .rma file path
    base (int) : base for layer counting.
                 Python is 0-based.
                 Marthe is compiled in 1-based (Fortran)
                 Default is 1.

    Returns:
    -----------
    nnest (int) : number of nested mesh ("gigogne")
    layers_infos (DataFrame) : layer informations like
                              layer numbers, thickness, ...
    Format:

        layer  thickness  epon_sup   ke  anisotropy     name
    0       1       50.0         0  0.0         0.0  layer_0
    1       2      150.0         1  0.0         0.0  layer_1


    Example
    -----------
    layfile = 'mymarthemodel.layer'
    nnest, layers_infos = get_layers_infos(layfile, base = 1)
    """
    # ---- Get .layer content as text
    with open(layfile, 'r', encoding=encoding) as f:
        content = f.read()
    # ---- set regular expressions
    regex = [r"Cou=\s*(\d+)", r"[Epais|Épais]=\s*([-+]?\d*\.?\d+|\d+)",
             r"[Epon|Épon] Sup =\s*(\d+)", r"Ke=\s*(\d+)", r"Anisot=\s*(\d+)"]
    # ---- Build Dataframe of all layer informations
    df = pd.DataFrame([re.findall(r, content) for r in regex],
                      index =['layer', 'thickness', 'epon_sup', 'ke', 'anisotropy'],
                      dtype=float).T
    # ---- Manage dtypes of informations
    dtypes = [int, float, int, float, float]
    layers_infos = df.astype({col:dt for col, dt in zip(df.columns, dtypes)})
    # ---- Manage layer counting from base
    layers_infos['layer'] = layers_infos['layer'].add(base-1)

    # ---- Add layer name if exist
    re_lnmes =  r";\s*Name=\s*(.+)\n"
    if re.search(re_lnmes, content) is None:
        layers_infos['name'] = 'layer_' + layers_infos['layer'].astype(str)
    else:
        layers_infos['name'] = re.findall(re_lnmes, content)

    # ---- Determine number of "gigogne" 
    nnest = int(re.findall(r"(\d+)=Nombre", content)[0])
    # ---- Return infos
    return nnest, layers_infos


def read_zonsoil_prop(martfile):
    """
    -----------
    Description:
    -----------
    Read soil properties in .mart file. 

    Parameters: 
    -----------
    martfile (str): Marthe .mart file path

    Returns:
    -----------
    df (DataFrame) : soil data with apply zone ids.

    Format:

             property       zone    value
    0   cap_sol_progr         54      10.2
    1          ru_max        126      41.4  

    Example
    -----------
    martfile = 'mymarthemodel.mart'
    soil_df = read_zonsoil_prop(martfile)

    """
    # ---- Read .mart file content
    with open(martfile, 'r',encoding=encoding) as f:
        content = f.read()

    # ---- Assert that some soil property exist
    err_msg = f'No soil properties found in {martfile}.'
    assert '/ZONE_SOL' in content, err_msg

    # ---- Set usefull regex
    re_init = r"\*{3}\s*Initialisation avant calcul\s*\*{3}\n(.+)\*{5}"
    re_num = r"[-+]?\d*\.?\d+|\d+"
    re_prop = r"\/(.+)\/ZONE_SOL\s*Z=\s*({0})V=\s*({0});".format(re_num)

    # ---- Get initialisation block as string
    block = re.findall(re_init,content, re.DOTALL)[0]

    # ---- Extract zonal soil properties
    dt_dic = {c:dt for c,dt in zip(['soilprop', 'zone', 'value'], [str, int, float])}
    df = pd.DataFrame(re.findall(re_prop, block),columns=dt_dic.keys()).astype(dt_dic)
    df['soilprop'] = df['soilprop'].str.lower()

    # ---- Return zonal soil properties data
    return df.sort_values('soilprop').reset_index(drop=True)




def write_zonsoil_prop(soil_df, martfile):
    """

    Description:
    -----------
    Write soil properties in .mart file. 

    Parameters: 
    -----------
    soil_df (DataFrame) : soil data with apply zone ids.
                            Format:
                                        property       zone    value
                                0   cap_sol_progr         54      10.2
                                1          ru_max        126      41.4

    martfile (str): Marthe .mart file path

    Returns:
    -----------
    Write property values in .mart file
  

    Example
    -----------
    martfile = 'mymarthemodel.mart'
    soil_df = read_zonsoil_prop(martfile)
    soil_df.value = 125
    write_zonsoil_prop(soil_df, martfile)
    """
    # ---- Fetch actual .mart file content as text
    with open(martfile, 'r', encoding=encoding) as f:
        content = f.read()
    # ---- Set useful regex
    re_num = r"[-+]?\d*\.?\d+|\d+"
    # ---- Iterate over each soil DataFrame line
    for d in soil_df.itertuples():
        # ---- Regex to match
        re_match = r"\/{0}\/ZONE_SOL\s*Z=\s*{1}V=\s*({2});".format(
                                                        d.soilprop.upper(),
                                                        d.zone, 
                                                        re_num)
        # ---- Search pattern
        match = re.search(re_match, content)
        # ---- Replace by new value
        sub = re.sub(match.group(1), str(d.value), match.group(0))
        # ---- Rewrite .mart file
        replace_text_in_file(martfile, match.group(0), sub)





def remove_autocal(martfile):
    """
    Function to make marthe auto calibration silent

    Parameters:
    ----------
    martfile (str) : path to .mart file.

    Returns:
    --------
    Write in .mart file inplace

    Examples:
    --------
    remove_autocal('mymodel.mart')
    """
    # ---- Fetch .mart file content
    with open(martfile, 'r', encoding=encoding) as f:
        lines = f.readlines()

    # ---- Define pattern to search
    re_cal = r"^\s*1=Optimisation"

    for line in lines:
        # ---- Search patterns
        cal_match = re.search(re_cal, line)
        # ---- Make calibration/optimisation silent 
        if cal_match is not None:
            wrong = cal_match.group()
            right = re.sub('1','0', wrong)
            new_line  = re.sub(wrong, right, line)
            replace_text_in_file(file, line, new_line)



def make_silent(martfile):
    """
    Function to make marthe run silent

    Parameters:
    ----------
    self : MartheModel instance
    martfile (str) : .mart file path
                      Default is None

    Returns:
    --------
    Write in .mart inplace

    Examples:
    --------
    make_silent('mymodel.mart')
    """
    # ---- Fetch .mart file content
    with open(martfile, 'r', encoding=encoding) as f:
        lines = f.readlines()

    # ---- Define pattern to search
    re_exe = r"^\s*(\s|\w)=Type d'exécution"

    for line in lines:
        # ---- Search patterns
        exe_match = re.search(re_exe, line)
        # ---- Make run silent 
        if exe_match is not None:
            wrong = exe_match.group()
            right = re.sub(r'(\s|\w)=','M=', wrong)
            new_line  = re.sub(wrong, right, line)
            replace_text_in_file(martfile, line, new_line)




def read_grid_file(grid_file):

    """
    Function to read Marthe grid data in file.
    Only structured grids are supported.

    Parameters:
    ----------
    grid_file (str) : Marthe Grid file full path

    Returns:
    --------
    grid_list (list) : contain one or more
                        MartheGrid instance
    
    Examples:
    --------
    grids = read_grid_file('mymodel.permh')

    """
    # ---- Extract data as large string
    with open(grid_file, 'r', encoding = encoding) as f:
        content = f.read()

    # ---- Define data regex
    sgrid, scgrid, egrid, cxdx0, cydy0, cxdx1, cydy1 =  [r'\[Data]',
                                                         r'\[Constant_Data]',
                                                         r'\[End_Grid]',
                                                         r'\[Columns_x_and_dx]',
                                                         r'\[Rows_y_and_dy]',
                                                         r'\[Num_Columns_/_x_/_dx]',
                                                         r'\[Num_Rows_/_y_/_dy]']

    # ---- Define infos regex
    re_headers = [r"Field=(\w*)",
                  r"\nLayer=(\d+)",
                  r"Nest_grid=(\d+)",
                  r"X_Left_Corner=([-+]?\d*\.?\d+|\d+)",
                  r"Y_Lower_Corner=([-+]?\d*\.?\d+|\d+)",
                  r"Ncolumn=(\d+)",
                  r"Nrows=(\d+)"]

    # ---- Collect headers as a list of strings
    headers = [re.findall(r, content) for r in re_headers]

    # ---- Collect data as a list of grids 
    r = r"({}|{})\n(.+?){}".format(sgrid, scgrid, egrid)
    str_grids = re.findall(r, content, re.DOTALL)

    # ---- Iterate over each grid
    grid_list = []

    for field, layer, inest, xl, yl , ncol, nrow, str_grid_tup in zip(*headers, str_grids):
        # ---- Get 
        data_type, str_grid = str_grid_tup
        # ---- Manage uniform value
        if data_type in scgrid:
            # -- Search and convert uniform value to float
            value = float(re.search(r"Uniform_Value=([-+]?\d*\.?\d+|\d+)",
                          str_grid).group(1))
            # ---- Support different version of uniform headers (cxdx0 or cxdx1)
            if all(s in str_grid for s in [cxdx0, cydy0]):
                cxdx, cydy = cxdx0, cydy0
            else:
                cxdx, cydy = cxdx1, cydy1

            # -- Search x/y cell centers and x/y cell resolution
            search = re.search(r"{0}\n{2}{1}\n{2}".format(cxdx, cydy, r'(.*?)\n'*3),
                               str_grid, re.DOTALL)
            
            # -- Fetch all rows, columns, cellcenters and dx, dy
            cols, xcc, dx, rows, ycc, dy = map(np.array,
                                           map(str.split,
                                            [search.group(i+1) for i in range(6)]))
            # -- Build uniform array
            array = np.full((len(rows),len(cols)), value)
        else:
            # -- Extract 2D-array from string
            arr_list = [np.fromstring(line.strip(), sep='\t', dtype = float)
                       for line
                       in str_grid.splitlines()]
            array = np.stack([a[2:-1] for a in arr_list[2:-1]], axis=0)
            # -- Convert to mask array if nested
            # if float(inest) > 0:
            #     array = np.ma.masked_array(array,
            #             mask= array == -9999.,
            #             fill_value = -9999.)
            # -- Search x/y cell centers and x/y cell resolution
            xcc, dx = arr_list[1][2:], arr_list[-1][2:]
            ycc, dy = np.dstack([l[[1,-1]] for l in arr_list[2:-1]])[0]
        # -- Store grid arguments
        layer = int(layer) - 1 # switch to 0-based
        args = (layer, inest, nrow, ncol, xl, yl, dx, dy, xcc, ycc, array, field)
        # -- Append MartheGrid instance to the grid list
        grid_list.append(MartheGrid(*args))

    # ---- Return all MartheGrid instances and xcc, ycc if required
    return tuple(grid_list)




def write_grid_file(grid_file, grid_list, maxlayer=None, maxnest=None):

    """
    Function to read Marthe grid data in file.
    Only structured grids are supported.

    Parameters:
    ----------
    grid_file (str) : Marthe Grid file full path

    Returns:
    --------
    grid_list (list) : contain one or more
                        MartheGrid instance
    
    Examples:
    --------
    grids = read_grid_file('mymodel.permh')

    """
    with open(grid_file, 'w', encoding = encoding) as f:
        for mg in grid_list:
            f.write(mg.to_string(maxlayer, maxnest))




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
    with open(file, "r+", encoding=encoding) as f:
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




def get_units_dic(mart_file):
    """
    -----------
    Description:
    -----------
    Extract units from "Unités des données"
    block in .mart file.
    
    Parameters: 
    -----------
    mart_file (str): .mart file name

    Returns:
    -----------
    units_dic (dict) : dictionary of unit converters values
                      Format: {'permh': 1, 'flow': 1.15e-5 ,...}

    Example
    -----------
    units = get_units_dic('mymodel.mart')
    """
    # ---- Set unit names
    unit_names = ['permh', 'flow', 'head', 'emmca', 'emmli', 'climh',
                  'irrigh', 'climtime', 'modeltime', 'modeldist', 
                  'vani', 'hani', 'emmcatype', 'salinity', 'concentration',
                  'porosity', 'stock', 'mass', 'permtype', 'volume', 'massflowtype']

    # ---- Set time unit dictionary 
    tu_dic = {'SEC':'S', 'MIN':'T', 'HEU':'H', 'JOU':'D',
              'SEM': 'W', 'MOI': 'M', 'ANN':'Y',
              'seconde':'S', 'minute': 'T', 'heure':'H',
              'jour':'J', 'semaine':'W', 'mois': 'M', 'année':'Y'}

    # ---- Fetch .mart file content
    with open(mart_file, 'r', encoding=encoding) as f:
        content = f.read()

    # ---- Set block regex
    re_block = r'Unités des données\s*\*{3}\n(.*?)\*{3}'

    # ---- Extract string block in .mart file
    block = re.findall(re_block, content, re.DOTALL)[0]

    # ---- Build unit dictionary
    units_dic = {}
    for unit_name, line in zip(unit_names, block.splitlines()):
        # -- Get value as string
        val_str = line.split('=')[0].strip()
        # -- Set None if not provided
        if len(val_str) == 0:
            v = None
        # -- Manage time units
        elif val_str in tu_dic.keys():
                v = tu_dic[val_str]
        # -- Manage val_str as value
        else:
            # -- Correct bad scientific notation
            if re.search(r'\d[-+]\d', val_str) is not None:
                sign = re.search(r'[-+]', val_str).group()
                val_str = val_str.replace(sign, 'e' + sign)
            # -- Convert into numeric
            v = ast.literal_eval(val_str)
        units_dic[unit_name] = v

    # ---- Return units
    return units_dic




def get_dates(pastp_file, mart_file):
    """
    -----------
    Description:
    -----------
    Extract dates from .pastp file
    
    Parameters: 
    -----------
    pastp_file (str) : path to .pastp marthe file
    mart_file (str): .mart file name

    Returns:
    -----------
    dates (DateTimeIndex): list of model dates

    Example
    -----------
    mm = MartheModel('mymodel.rma')
    dates = mm.get_dates()
    """
    # ---- Set date regular expression
    re_anydate = r'date\s*:\s*(\d{2}/\d{2}/\d{4}|[-+]?\d*\.?\d+|\d+)\s*;'
    # ---- Fetch pastp file content as string
    with open(pastp_file, 'r', encoding=encoding) as f:
        dates_str = re.findall(re_anydate, f.read())
    # ---- Distinguish classic dates / timedelta dates
    if not '/' in ''.join(dates_str):
        # -- Get time unit
        tu = get_units_dic(mart_file)['modeltime']
        # -- Build dates
        dates = pd.TimedeltaIndex([s + tu for s in dates_str])
    else:
        dates = pd.DatetimeIndex(dates_str, dayfirst = True)

    # ---- Return dates
    return dates



def read_prn(prnfile):
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
    df_sim = pd.read_csv(prnfile,  sep='\t',skiprows = 3,encoding=encoding, 
                         index_col = 0, parse_dates = True)
    df_sim.index.names = ['Date']
    df_sim.columns = df_sim.columns.str.replace(' ','')
    df_sim = df_sim.iloc[:,1:-1]
    return  df_sim



def read_mi_prn(prnfile = 'historiq.prn'):
    """
    Function to read simulated prn file

    Parameters:
    ----------
    prnfile (str) : prn file full path
                     Default is 'historiq.prn'

    Returns:
    --------
    df (DataFrame) : Multi-index DataFrame
                     index = 'date' (DateTimeIndex)
                     columns = MultiIndex(level_0 = 'type',         # Data type ('Charge', 'Débit', ...)
                                          level_1 = 'gigogne')      # (optional) Refined grid number 
                                                                      (0 <= gigogne <= N_gigogne)
                                          level_2 = 'boundname',    # Custom name 

    Examples:
    --------
    prn_df = read_mi_prn(prnfile = 'historiq.prn')
    """
    # ---- Check if prnfile exist
    path, file = os.path.split(prnfile)
    msg = f'{prnfile} file not found.'
    assert file in os.listdir(os.path.normpath(path)), msg
    # ---- Build Multiple index columns
    with open(prnfile, 'r', encoding=encoding) as f:
        # ----Fetch 5 first lines of prn file 
        flines_arr = np.array([f.readline().split('\t')[:-1] for i in range(5)], dtype=list)
        # ---- Create a boolean mask to read only usefull header lines
        mask = [False, True, False, True , False]
        # ---- Select only usefull first lines by mask
        if any('Main_Grid' in elem for elem in flines_arr[-2]):
            nest = True 
            # -- Transform to fancy integer 'inest' number
            flines_arr[-2] = ['0' if not 'Gigogne' in g else g.split(':')[1].strip()
                                  for g in flines_arr[-2]]
            # -- Add -gigone- boolean to mask
            mask[-1] = nest
        else:
            nest = False
        # ---- Fetch headers
        headers = list(flines_arr[mask])
    # ---- Get all headers as tuple
    tuples = [tuple(map(str.strip,list(t)) ) for t in list(zip(*headers))][2:]
    # ---- Set multi-index names
    if nest:
        idx_names = ['type', 'inest', 'boundname']
    else:
        idx_names = ['type', 'boundname']
    # ---- Build multi-index
    midx = pd.MultiIndex.from_tuples(tuples, names=idx_names)
    # ---- Read prn file without headers (with date format)
    skiprows = mask.count(True) + 1
    df = pd.read_csv(prnfile, sep='\t', encoding=encoding, 
                     skiprows=mask.count(True) + 1, index_col = 0,
                     parse_dates = True, dayfirst=True)
    df.drop(df.columns[0], axis=1,inplace=True)
    df.dropna(axis=1, how = 'all', inplace = True)  # drop empty columns if exists
    # ---- Format DateTimeIndex
    df.index.name = 'date'
    # ---- Set columns as multi-index as columns
    df.columns = midx
    # ---- Trandform inest id to integer
    if nest:
        levels = df.columns.get_level_values('inest').astype(int).unique()
        df.columns.set_levels(levels = levels, level='inest', inplace=True)
    # ---- Return prn DataFrame
    return df




def read_histo_file(histo_file):
    """
    Function to read .histo file.
    Support localisation by coordinates (XCOO) or by column, row (CL)
    Support additional label.
    Support older versions of Marthe .histo file.

    Parameters:
    ----------
    histo_file (str) : .histo file full path

    Returns:
    --------
    df (DataFrame) : information about data
                     to save by Marthe

    Examples:
    --------
    histo_df = read_histo_file('mymodel.histo')
    """
    # ---- Set usefull regex
    re_num = r"[-+]?\d*\.?\d+|\d+"
    re_nest = r'^\s*\d*/\w+'
    re_names = r";\s*(.*)"
    # ---- Fetch histo file content by line
    with open(histo_file, encoding = encoding) as f:
        lines = [line.rstrip() for line in f]
    # ---- Iterate over all lines
    data = []
    for line in lines:
        if '/HISTO/' in line:
            # ---- Fetch 'inest' and data type
            inest_str, typ = map(str.strip, re.search(re_nest, line).group(0).split('/'))
            inest = int(inest_str) if inest_str else 0
            # ---- Fetch cell localisation
            if '/XCOO' in line:
                loc_str = line[line.index('X='):line.index(';')] 
                loc_type = 'xyz'
            if '/MAIL' in line:
                loc_str = line[line.index('C='):line.index(';')]
                loc_type = 'clp'
            loc = list(map(ast.literal_eval, re.findall(re_num, loc_str)))
            # ---- Fetch id and label (void older version with 'Name='' tag)
            names = re.split(r"\s{5,}", re.split(';', re.sub('Name=','',line))[-1].strip())
            # ---- Manage undefined label
            names = names*2 if len(names) == 1 else names
            # ---- Store cell data
            data.append([typ, inest, loc_type] + loc + names)
    # ---- Build histo DataFrame
    cols = ['type','inest','loc_type','x','y','layer','id', 'label']
    df = pd.DataFrame(data, columns = cols)
    df.set_index('id', inplace = True)
    # ---- Return histo DataFrame
    return df



def isiterable(object):
    """
    Detect if a object is a iterable.
    String are not considered as iterable.

    Parameters:
    ----------
    object (?): instance to check.

    Returns:
    --------
    (bool) : True : object is iterable.
             False : object is not iterable.

    Examples:
    --------
    l = [4,5,6]
    if _isiterable(l):
        print(sum(l))
    
    """
    if isinstance(object, str):
        return False
    else:
        try:
            it = iter(object)
        except TypeError: 
            return False
        return True



def make_iterable(var):
    """
    Make any variable iterable

    Parameters:
    ----------
    var (?): instance transform

    Returns:
    --------
    it (iterable) : any iterable object

    Examples:
    --------
    i = 9
    it = make_iterable(var)
    
    """
    it = var if isiterable(var) else [var]
    return it




def read_listm_qfile(qfile, istep, fmt):
    """
    """
    if (len(fmt) == 0) | (fmt == 'Somm_Mail|C_L_P|Keep_9999'): 
        # ---- Set data types
        dt = {'value':'f8','j':'i4','i':'i4','layer':'i4'}
        # ---- Read qfile as DataFrame (separator = any whitespace)
        df = pd.read_csv(qfile, header=None,  delim_whitespace=True,
                            names=list(dt.keys()), dtype=dt)
        # ---- Add istep
        df['istep'] = istep
        df['boundname'] = 'boundname'
        # ---- Manage metadata
        metacols = ['qfilename', 'qtype', 'qrow', 'qcol']
        df[metacols] = np.array([qfile, 'listm', df.index, 0], dtype=object)
        # ---- Return data
        cols = ['istep', 'layer', 'i', 'j', 'value', 'boundname']
        _cols = cols + metacols 
        return df[cols], df[_cols]

    if fmt == 'X_Y_C|Somm_Mail|Keep_9999':
        # ---- Set data types
        dt = {'x':'f8','y':'f8','layer':'i4', 'value':'f8'}
        # ---- Read qfile as DataFrame (separator = any whitespace)
        df = pd.read_csv(qfile, header=None,  delim_whitespace=True,
                            names=list(dt.keys()), dtype=dt)
        # ---- Add istep
        df['istep'] = istep
        df['boundname'] = 'boundname'
        # ---- Manage metadata
        metacols = ['qfilename', 'qtype', 'qrow', 'qcol']
        df[metacols] = np.array([qfile, 'listm', df.index, 3], dtype=object)
        # ---- Return data
        cols = ['istep', 'layer', 'x', 'y', 'value', 'boundname']
        _cols = cols + metacols 
        return df[cols], df[_cols]



def read_record_qfile(i,j,k,v,qfile,qcol):
    """
    """
    # ---- Read just qcol column in qfile
    data = pd.read_csv(qfile, usecols=[qcol], dtype='f8',  delim_whitespace=True)
    # ---- Extract boundname and value
    bdnme = 'boundname'
    value = [v] + data[bdnme].tolist()
    istep = qrow = list(range(len(value)))
    df = pd.DataFrame({'istep':istep,'layer':k,'i': i,'j':j,
                       'value': value, 'boundname': bdnme,
                       'qfilename': qfile, 'qtype': 'record',
                       'qrow': qrow, 'qcol': qcol})
    # ---- Return data
    cols = ['istep', 'layer', 'i', 'j', 'value', 'boundname']
    metacols =  ['qfilename', 'qtype', 'qrow', 'qcol']
    _cols = cols + metacols
    return df[cols], df[_cols]




def extract_pastp_pumping(pastpfile, mode = 'aquifer'):
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

    # ---- Prepare some usefull regex
    re_block = r";\s*\*{3}\n(.*?)/\*{5}"
    re_num = r"[-+]?\d*\.?\d+|\d+"
    re_jikv = r"C=\s*({})L=\s*({})P=\s*({})V=\s*({});".format(*[re_num]*4)
    re_file = r"\s*File=\s*(.*);"
    re_col = r"\s*Col=\s*({})".format(re_num)
    re_listm_fmt = r'(?<=<)[^<:]+(?=:?>)'

    # ---- Get working directory full path
    mm_ws = os.path.split(pastpfile)[0]

    # ---- Fetch pastp data by block (each block = data for step i)
    with open(pastpfile, 'r', encoding=encoding) as f:
        content = f.read()
        # ---- Set special tag according to pumping mode
        mode_tag = '/DEBIT/' if mode == 'aquifer' else '/Q_EXTER_RIVI/'
        err_msg = f"ERROR: No '{mode}' pumping found in .pastp file."
        assert mode_tag in content, err_msg
        # ---- Extract pastp by block 
        blocks = re.findall(re_block, content, re.DOTALL)

    # ---- Initialize DataFrame and metaDataFrame
    dfs, _dfs = [], []
    # ---- iterate over istep blocks
    for istep, block in enumerate(blocks):
        # ---- Iterate over block content (lines)
        for line in block.splitlines():
            # -- Check if a pumping condition is applied
             if mode_tag in line:
                # -- Check which type of pumping condition is provided 
                # 1) Manage LIST_MAIL (list of pumping cells in external file)
                if any(s in line for s in ['/LISTM', '/LIST_M']):
                    fmt = '|'.join(re.findall(re_listm_fmt, line))
                    path = line.split('N: ')[-1].split('<')[0]
                    qfilename = os.path.normpath(os.path.join(mm_ws, path.strip()))
                    # -- Extract data from LISTM qfile (as DataFrame)
                    df, _df = read_listm_qfile(qfilename, istep, fmt)
                # 2) Manage MAILLE (unique pumping cell)
                if '/MAIL' in line:
                    j,i,k,v = map(ast.literal_eval, re.findall(re_jikv, line)[0])
                    _file, _col = [re.search(r, line) for r in [re_file, re_col]]
                    qcol = 0 if _col is None else int(_col.group(1)) -1 # convert to 0-based
                    if _file is None:
                        bdnme = 'boundname'
                        df = pd.DataFrame([[istep,k,i,j,v,bdnme]],
                             columns=['istep','layer','i','j','value','boundname'])
                        _df = df.copy(deep=True)
                        _df[['qfilename', 'qtype', 'qrow','qcol']] = [None, 'mail', None, None]
                    else:

                        qfilename = os.path.normpath(os.path.join(mm_ws, _file.group(1)))
                        df, _df = read_record_qfile(i,j,k,v, qfilename, qcol)

                # ---- Append (meta)DataFrame list
                dfs.append(df)
                _dfs.append(_df)

    # ---- Return concatenate (meta)DataFrame
    data, metadata = [df.reset_index(drop=True) for df in list(map(pd.concat, [dfs, _dfs]))]
    return data, metadata





def convert_at2clp_pastp(pastpfile, mm):
    """
    Function convert 'affluent' / 'tronçon' to column, line, plan (layer)
    and rewrite it in pastp file

    Parameters:
    ----------
    pastpfile (str) : path to pastp file
    mm (object) : MartheModel instance
    
    Returns:
    --------
    Replace lines inplace in pastp file

    Examples:
    --------
    convert_at2clp_pastp(pastpfile, mm)
    
    """
    # ---- Set regular expression of numeric string (int and float)
    re_num = r"[-+]?\d*\.?\d+|\d+"
    re_block = r";\s*\*{3}\n(.*?)/\*{5}"

    # ---- Get convertisor (i,j) -> (a,t) as df
    at = []
    for s in ['aff_r', 'trc_r']:
        arr = read_grid_file(mm.mlfiles[s])[0].array
        ij = list(np.where(arr != 9999.))
        at.append(arr[arr != 9999.])
    conv_df = pd.DataFrame({k:v for k,v in zip(list('ijat'), [*ij, *at])}, dtype=int)

    # ---- Extract .pastp file content
    with open(pastpfile, 'r', encoding=encoding) as f:
        # ---- Extract pastp by block 
        blocks = re.findall(re_block, f.read(), re.DOTALL)

    for block in blocks:
        # ---- Iterate over block content (lines)
        for line in block.splitlines():
            # ---- Check if the line contain aff/trc
            if all(s in line for s in ['Q_EXTER_RIVI','A=','T=']):
                # ---- Replace TRONCON for MAILLE
                mail_line = line.replace('TRONCON', 'MAIL')
                # ---- Get substring to replace
                s2replace = line[line.index('A='):line.index('V=')]
                # ---- Fetch aff/trc as number
                a,t = map(ast.literal_eval, re.findall(re_num, s2replace))
                # ---- Convert aff/trc to column, line, plan (layer)
                i,j = conv_df.query(f"a=={a} & t=={t}")[list('ij')].to_numpy()[0]
                layer = mm.get_outcrop()[i,j]
                # ---- Build substring to replace
                sub = '{:>8}C={:>7}L={:>7}P={:>7}'.format(' ',j+1, i+1, int(layer))
                # ---- Build entire line to be replace for
                l2replace = mail_line.replace(s2replace,sub)
                # ---- Replace text in pastp file
                replace_text_in_file(pastpfile, line, l2replace)




def remove_no_data_values(df, column = 'value', nodata = NO_DATA_VALUES):
    """
    -----------
    Description
    -----------
    Remove no data values in DataFrame on specific column

    -----------
    Parameters
    -----------
    - df (DataFrame) : table with potential no data values inside
    - column (str) : column name to search the no data values
    - nodata (list) : list of no data values to remove
                      Default is [-9999., -8888., 9999.]

    -----------
    Returns
    -----------
    df (DataFrame) : clean table without no data value

    -----------
    Examples
    -----------
    clean_df = remove_no_data_values(df, nodata = [1e+30, -9999.])
    """
    # ---- Return same DataFrame
    if nodata is None:
        return df
    else:
        # ---- Transform specific no data value to NaN
        for nd in nodata:
            df.loc[df[column] == nd, column] = pd.NA
        # ---- Return clean DataFrame
        return df.dropna()



def read_obsfile(obsfile, nodata = None):
    """
    Simple function to read observation file.
    Format : header0        header1
             09/05/1996     0.12
             10/05/1996     0.88
    Note: separator is anywhite space (tabulation is prefered)

    Parameters:
    ----------
    obsfile (str): observation filename to read value.
                   Note: if locnme is not provided,
                   the locnme is set as obsfile without file extension.
    nodata (list/None) : no data values to remove.
                         Default is None.

    Returns:
    --------
    df (DataFrame) : observation table
                     Format : date          value
                              1996-05-09    0.12
                              1996-05-10    0.88

    Examples:
    --------
    obs_df = read_obsfile(obsfile = 'myobs.dat')
    """
    # ---- Read obsfile
    data = pd.read_csv(obsfile, delim_whitespace=True, header=None, skiprows=1, index_col = 0, parse_dates=True)
    # ---- Set standard column names
    df = data.rename(columns = {1 :'value'})
    df.index.name = 'date'
    # ----- Return clean DataFrame
    return remove_no_data_values(df, nodata = nodata)



def write_obsfile(date, value, obsfile):
    """
    Write a standard obsfile from observation dates and value.

    Parameters:
    ----------
    date (DateTimeIndex) : observation date index 
    value (iterable) : observation values 
    obsfile (str): observation filename to read value.

    Returns:
    --------
    Write observation values.
    Format : date          value
             1996-05-09    0.12
             1996-05-10    0.88

    Examples:
    --------
    locnme = '07065X0002'
    obs_df = write_obsfile(date, values, obsfile = locnme + '.dat')
    """
    # ---- Build standard DataFrame
    df = pd.DataFrame(dict(value = list(value)), index = date)
    # ---- Write DataFrame
    df.to_csv(obsfile,  sep = '\t', header = True, index = True)

