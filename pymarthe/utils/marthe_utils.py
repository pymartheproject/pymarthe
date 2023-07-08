# -*- coding: utf-8 -*-
import os, sys
from copy import deepcopy
import numpy as np
import pandas as pd
from pathlib import Path
import re, ast
import warnings
import functools

import pymarthe
from pymarthe import *
from .grid_utils import MartheGrid

# Set encoding
encoding = 'latin-1'


# Set commun no data values
NO_DATA_VALUES = [-9999., -8888.]




def deprecated(func):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used. Addtionnal message 'use funcname
    instead.' is written if provided.
    """
    # -- Build decorator
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # -- Turn off filter
        warnings.simplefilter('always', DeprecationWarning)
        # -- Build warning msg
        msg = "Function `.{}`() is now deprecated. ".format(func.__name__)
        # -- Write deprecation warning
        warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
          # -- Reset filter
        warnings.simplefilter('default', DeprecationWarning)
        return func(*args, **kwargs)
    return new_func





def unanimous(obj):
    """
    Check if all elements in object (obj) has same length.
    """
    # -- Check if object is iterable 
    err_msg = 'ERROR : `obj` is not iterable. ' \
              f'Given : {obj}.'
    assert isiterable(obj), err_msg
    # -- Transform dictionary to valid sequence (list, array, tuple, ..)
    seq = list(obj.values()) if isinstance(obj, dict) else obj
    # -- Check if items are iterable
    err_msg = 'ERROR : some item in `obj` are not iterable. ' \
              f'Given : {obj}.'
    assert all(isiterable(item) for item in seq), err_msg
    # -- Map the length of each items and verify the unique length
    res = True if len(set(map(len, seq))) == 1 else False
    # -- Return boolean
    return res



def get_mlfiles(rma_file):
    """
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
    re_mlfile = r"\n(.+?)="
    mlfiles = [s.strip() for s in re.findall(re_mlfile, content)
                         if (not s.isspace()) & ('=' not in s)]

    # ---- Build dictionary with all marthe file path
    mlfiles_dic = {mlfile.split('.')[-1]: 
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



def has_soilprop(text):
    """
    Description:
    -----------
    Boolean response of zone soil data in string. 

    Parameters: 
    -----------
    text (str) : string to test.

    Returns:
    -----------
    res (bool) : Whatever the text contains soil property data
  

    Example
    -----------
    s = "my string"
    if has_soilprop(s):
        print('This string contains soil data.')
    else:
        print('This string does not contain soil data')
    """
    res = True if '/ZONE_SOL' in text else False
    return res



def extract_soildf(text, istep=None):
    """
    Description:
    -----------
    Extract soil data in text to DataFrame.

    Parameters: 
    -----------
    text (str) : string to test.
    istep (int, optional): insert a additional 'istep' column
                           at first position.
                           Default is None.

    Returns:
    -----------
    soil_df (DataFrame) : soil data with apply zone ids and value.
                          Format:
                          (if istep is None)

                                     property       zone    value
                            0   cap_sol_progr         54      10.2
                            1          ru_max        126      41.4

                          (if istep is not None)

                                istep       property       zone    value
                            0       0  cap_sol_progr         54      10.2
                            1       0         ru_max        126      41.4

    Example
    -----------
    soil_df = extract_soildf(mymartfielcontent, istep=0)
    """
    # -- Set useful regex to identify soil data contain by line
    re_num = r"[-+]?\d*\.?\d+|\d+"
    re_prop = r"\s*\/(.+)\/ZONE_SOL\s*Z=\s*({0})V=\s*(.+);".format(re_num)
    # -- Define data type
    dt_dic = {c:dt for c,dt
               in zip(['soilprop', 'zone', 'value'],
                      [str, int, float])}
    # -- Extract soil data with regex (line-by-line)
    data = re.findall(re_prop, text)
    # -- Build soil DataFrame with correct data types
    soil_df = pd.DataFrame(data, columns=dt_dic.keys()).astype(dt_dic)
    # -- Sort by lower case soil properties names
    soil_df['soilprop'] =  soil_df['soilprop'].str.lower()
    soil_df = soil_df.sort_values('soilprop').reset_index(drop=True)
    # -- Add istep column if required
    if not istep is None:
        soil_df.insert(0, 'istep', istep)
    # -- Return soil properties DataFrame
    return soil_df





def read_zonsoil_prop(martfile, pastpfile):
    """
    Description:
    -----------
    Detetct existence and location of zone soil property data
    and convert it to single DataFrame.

    Parameters: 
    -----------
    martfile (str) : path to the .mart file.
    pastpfile (str) : path to the .pastp file.

    Returns:
    -----------
    mode (str) : flag of the zone soil data implementation.
                 Can be :
                    - 'mart-c'  (constant soil data in .mart file)
                    - 'pastp-c' (constant soil data in .pastp file)
                    - 'pastp-t' (transient soil data in .pastp file)
    soil_df (DataFrame) : soil data with apply zone ids and property values.
                          Format:
                          (if istep is None)

                                     property       zone    value
                            0   cap_sol_progr         54      10.2
                            1          ru_max        126      41.4

                          (if istep is not None)

                                istep       property       zone    value
                            0       0  cap_sol_progr         54      10.2
                            1       0         ru_max        126      41.4

    Example
    -----------
    mode, soil_df = read_zonsoil_prop(martfile, pastpfile)

    """
    # -- Check if martfile contains soil property(ies)
    with open(martfile, 'r',encoding=encoding) as f:
        mart_content = f.read()
    _mart = True if has_soilprop(mart_content) else False

    # -- Check if pastp contains soil property(ies)
    with open(pastpfile, 'r', encoding=encoding) as f:
        pastp_content = f.read()
    _pastp = True if has_soilprop(pastp_content) else False

    # -- Raise error for none or both soil property implementations
    err_both = "ERROR : soil properties provided " \
               f"in both {martfile} and {pastpfile} files."
    err_none = "ERROR : No soil properties found " \
               f"in both {martfile} and {pastpfile} files."
    assert any(x is False for x in [_mart,_pastp]), err_both
    assert any(x is True  for x in [_mart,_pastp]), err_none

    # -- Manage constante soil properties store in .mart file
    if _mart:
        # -- Set soil properties mode
        mode = 'mart-c'
        # -- Get constant soil DataFrame
        soil_df = extract_soildf(mart_content, istep=0)

    # -- Manage constante/transient soil properties store in .pastp file
    else:
        # -- Extract pastp text blocks (by timestep)
        re_block = r";\s*\*{3}\n(.*?)/\*{5}"
        blocks = re.findall(re_block, pastp_content, re.DOTALL)
        nstep = len(blocks)
        # -- Soil properties counter
        counter = list(map(has_soilprop, blocks)).count(True)
        # -- Set soil properties mode
        mode = 'pastp-t' if counter > 1 else 'pastp-c'
        # -- Extract soil DataFrame for each time step if provided
        soil_dfs = [extract_soildf(block, istep) 
                    for istep, block in enumerate(blocks)
                    if has_soilprop(block)]
        # -- Concatenate all time step DataFrame
        soil_df = pd.concat(soil_dfs, ignore_index=True)

    # -- Return implementation mode and soil properties DataFrame
    return mode, soil_df




def remove_autocal(rmafile, martfile):
    """
    Function to make marthe auto calibration / optimisation silent

    Parameters:
    ----------
    martfile (str) : path to .mart file.
    martfile (str) : path to .mart file.

    Returns:
    --------
    Write in .mart/.rma file inplace

    Examples:
    --------
    remove_autocal('mymodel.rma', 'mymodel.mart')
    """

    # ---- Fetch .rma file content
    with open(rmafile, 'r', encoding=encoding) as f:
        lines = f.readlines()

    # ---- Define pattern to search
    re_fopt = r"(\w+)\.(\w+)"
    re_opt = r"{}\s*=(.*)Optimisation\n".format(re_fopt)

    for line in lines:
        # -- Search pattern
        opt_match = re.search(re_opt, line)
        if opt_match is not None:
            # -- Get optimisation file name
            fopt = '.'.join(re.findall(re_fopt, line)[0])
            # -- Replace bye empty strings
            new_line = re.sub(fopt,' '*len(fopt), line)
            # -- Set changes inplace
            replace_text_in_file(rmafile, line, new_line)

    # ---- Fetch .mart file content
    with open(martfile, 'r', encoding=encoding) as f:
        lines = f.readlines()

    # ---- Define pattern to search
    re_cal = r"^\s*1=Optimisation"

    for line in lines:
        # -- Search patterns
        cal_match = re.search(re_cal, line)
        # -- Make calibration/optimisation silent 
        if cal_match is not None:
            wrong = cal_match.group()
            right = re.sub('1','0', wrong)
            new_line  = re.sub(wrong, right, line)
            replace_text_in_file(martfile, line, new_line)





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



def get_chasim_indexer(chasim):
    """
    Return a DataFrame with start/end columns using as 
    character indexes to read grid in Marthe simulated grid file.

    Parameters
    -----------
    chasim (str): simulated field(s) file
                  If None, the 'chasim.out' in model path
                  will be considered.
                  Default is None.

    Returns
    --------
    indexer (DataFrame) : table indexer.
                          Format:

                                    field       istep       start       end     
                            0      CHARGE           0           0     16789
                            1      CHARGE           0       16790     33578
                            2      CHARGE           0       33579     50367
                           ..          ..          ..          ..        ..

    Examples
    -----------
    mm = MartheField('mona.rma')
    mfs = MartheFieldSeries(mm, chasim=None)

    """
    # -- Extract chasim content as string
    with open(chasim, 'r', encoding = encoding) as f:
        content = f.read()

    # -- Set regular expressions
    re_igrid = r"Marthe_Grid(.*?)\[End_Grid]"
    re_headers = [r"Field=(.+)\n", r"Time_Step=([-+]?\d*\.?\d+|\d+)"]

    # -- Extract only usefull grid informations into a indexer DataFrame
    indexer = pd.DataFrame(
                np.column_stack(
                    [ 
                    np.column_stack(
                        [re.findall(r, content) for r in re_headers]
                        ),
                    np.column_stack(
                        [[m.start(0), m.end(0)] for m in re.finditer(re_igrid, content, re.DOTALL)]
                        ).T
                    ]
                ),
            columns = ['field', 'istep', 'start', 'end']
            ).astype(
                {col: int for col in ['istep', 'start', 'end']}
                )

    # -- Return indexer
    return indexer



def read_grid_file(grid_file, keep_adj=False, start=None, end=None):

    """
    Function to read Marthe grid data in file.
    Only structured grids are supported.

    Parameters:
    ----------
    grid_file (str) : Marthe Grid file full path

    keep_adj (bool) : whatever conserving adjacent cells
                      for nested grids.
                      Default is False.
    
    start/end (int, optional) : first/last character index to 
                                consider in whole grid file.
                                If None, `start`=0 and `end`= len(N)
                                (where N is the number of character
                                 in the provided grid file)
                                Default is None.

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
        allcontent = f.read()
        # -- Manage start/end input 
        starts = [0] if start is None else make_iterable(start)
        ends = [len(allcontent)] if end is None else make_iterable(end)
        # -- Get subseted content
        content = ''.join([allcontent[s:e] for s,e in zip(starts, ends)])

    # ---- Define data regex
    sgrid, scgrid, egrid, cxdx0, cydy0, cxdx1, cydy1 =  [r'\[Data]',
                                                         r'\[Constant_Data]',
                                                         r'\[End_Grid]',
                                                         r'\[Columns_x_and_dx]',
                                                         r'\[Rows_y_and_dy]',
                                                         r'\[Num_Columns_/_x_/_dx]',
                                                         r'\[Num_Rows_/_y_/_dy]']

    # ---- Define infos regex
    re_headers = [r"Field=(.+)\n",
                  r"Time_Step=([-+]?\d*\.?\d+|\d+)",
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

    for field, istep, layer, inest, xl, yl , ncol, nrow, str_grid_tup in zip(*headers, str_grids):
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
        # -- Switch to 0-based
        layer = int(layer) - 1
        # -- Store arguments in tuple
        if keep_adj:
            args = (istep, layer, inest, nrow, ncol, xl, yl, dx, dy, xcc, ycc, array, field)
        else:
            if int(inest) > 0:
                # -- Remove first and last element on i and j data 
                #    corresponding to neighbor cell of the main grids
                args = ( istep, layer, inest, int(nrow)-2, int(ncol)-2, xl, yl,
                         dx[1:-1], dy[1:-1], xcc[1:-1],ycc[1:-1], array[1:-1,1:-1], field )
            else:
                args = (istep, layer, inest, nrow, ncol, xl, yl, dx, dy, xcc, ycc, array, field)
        # args = (istep, layer, inest, nrow, ncol, xl, yl, dx, dy, xcc, ycc, array, field)
        # -- Append MartheGrid instance to the grid list
        grid_list.append(pymarthe.utils.grid_utils.MartheGrid(*args))

    # ---- Raise error if marthe grid file headers are not provided
    err_msg = f"ERROR : uncorrect headers values in `{grid_file}` grid file.\n" \
               "Check and correct the following lines:\n" \
               "\t-'Layer='\n" \
               "\t-'Max_Layer='\n" \
               "\t-'Nest_grid='\n" \
               "\t-'Max_NestG='\n"
    assert all(mg.layer >= 0 for mg in grid_list), err_msg

    # ---- Return all MartheGrid instances and xcc, ycc if required
    return tuple(grid_list)





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




def get_units_dic(martfile):
    """
    -----------
    Description:
    -----------
    Extract units from "Unités des données"
    block in .mart file.
    
    Parameters: 
    -----------
    martfile (str): .mart file name

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
    with open(martfile, 'r', encoding=encoding) as f:
        content = f.read()

    # ---- Set block regex
    re_block = r'Unités des données\s*\*{3}\n(.*?)\*{3}'

    # ---- Extract string block in .mart file
    block = re.findall(re_block, content, re.DOTALL)[0]

    # ---- Build unit dictionary (add try loop to avoid errors)

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
        # -- Manage val_str as value or str unit
        else:
            # -- Correct bad scientific notation
            if re.search(r'\d[-+]\d', val_str) is not None:
                sign = re.search(r'[-+]', val_str).group()
                val_str = val_str.replace(sign, 'e' + sign)
            try:
                v = ast.literal_eval(val_str)
            except:
                v = val_str
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



def read_prn(prnfile = 'historiq.prn'):
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
                                          level_2 = 'name',    # Custom name 

    Examples:
    --------
    prn_df = read_mi_prn(prnfile = 'historiq.prn')
    """
    # ---- Check if prnfile exist
    path, file = os.path.split(prnfile)
    msg = f'{prnfile} file not found.'
    assert file in os.listdir(os.path.normpath(path)), msg
    # ---- Build Multiple index columns
    add_skip = 1
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
            if any('Niveau_Lac' in elem for elem in flines_arr[-4]):
                flines_arr[-2] = flines_arr[-1]
                add_skip = 2
            nest = False
        # ---- Fetch headers
        headers = list(flines_arr[mask])
    # ---- Set multi-index names
    if nest:
        idx_names = ['type', 'inest', 'name']
    else:
        idx_names = ['type', 'name']
    # identify whether a date columns is provided in addition to simulation time
    # (this implies 2 subsequent '#_Date' columns in the prn)
    date_col = headers[0][0].strip(' ')==headers[0][1].strip(' ')
    # remove time column and parse dates when date column is present
    if date_col:
        # ---- Get all headers as tuple
        tuples = [tuple(map(str.strip,list(t)) ) for t in list(zip(*headers))][2:]
        # ---- Read prn file without headers (with date format)
        df = pd.read_csv(prnfile, sep='\t', encoding=encoding, 
                         skiprows=mask.count(True) + add_skip, index_col = 0,
                         parse_dates = True, dayfirst=True)
        df.drop(df.columns[0], axis=1,inplace=True)
    else :
        # ---- Get all headers as tuple
        tuples = [tuple(map(str.strip,list(t)) ) for t in list(zip(*headers))][1:]
        # ---- Read prn file without headers (time is not a date)
        df = pd.read_csv(prnfile, sep='\t', encoding=encoding, 
                         skiprows=mask.count(True) + add_skip, index_col = 0,
                         )
    df.dropna(axis=1, how = 'all', inplace = True)  # drop empty columns if exists
    # ---- Format DateTimeIndex or float
    df.index.name = 'date'
    # ---- Set columns as multi-index as columns
    midx = pd.MultiIndex.from_tuples(tuples, names=idx_names)
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
    if (len(fmt) == 0) | ('Somm_Mail|C_L_P|Keep_9999' in fmt): 
        # ---- Set data types
        dt = {'value':'f8','j':'i4','i':'i4','layer':'i4'}
        # ---- Read qfile as DataFrame (separator = any whitespace)
        df = pd.read_csv(qfile, encoding=encoding, delim_whitespace=True,
                         header=None, names=list(dt.keys()), dtype=dt)
        # ---- Pass t0 0-based
        df[['j','i','layer']] = df[['j','i','layer']].sub(1)
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

    if 'X_Y_C|Somm_Mail|Keep_9999' in fmt:
        # ---- Set data types
        dt = {'x':'f8','y':'f8','layer':'i4', 'value':'f8'}
        # ---- Read qfile as DataFrame (separator = any whitespace)
        df = pd.read_csv(qfile, encoding=encoding, delim_whitespace=True,
                         header=None, names=list(dt.keys()), dtype=dt)
        # ---- Pass t0 0-based
        df['layer'] = df['layer'].sub(1)
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

    if  'X_Y_V|Somm_Mail|Keep_9999' in fmt:
        # ---- Set data types
        dt = {'x':'f8','y':'f8', 'value':'f8'}
        # ---- Read qfile as DataFrame (separator = any whitespace)
        df = pd.read_csv(qfile, encoding=encoding, delim_whitespace=True,
                         header=None, names=list(dt.keys()), dtype=dt)
        # ---- Add layer info (=0)
        df['layer'] = 0
        # ---- Add istep
        df['istep'] = istep
        df['boundname'] = 'boundname'
        # ---- Manage metadata
        metacols = ['qfilename', 'qtype', 'qrow', 'qcol']
        df[metacols] = np.array([qfile, 'listm', df.index, 2], dtype=object)
        # ---- Return data
        cols = ['istep', 'layer', 'x', 'y', 'value', 'boundname']
        _cols = cols + metacols 
        return df[cols], df[_cols]





def read_record_qfile(i,j,k,v,qfile,qcol):
    """
    """
    # ---- Read just qcol column in qfile
    data = pd.read_csv(qfile, encoding=encoding, delim_whitespace=True)
    # ---- Extract boundname and value
    bdnme = 'boundname'
    value = [v] + data.iloc[:,qcol].to_list()
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
    re_file = r"\s*File=\s*(.*)\.(\w{3})"
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
                    # -- Convert to 0-based
                    qcol = 0 if _col is None else int(_col.group(1)) -1
                    k,i,j = k-1,i-1,j-1
                    if _file is None:
                        bdnme = 'boundname'
                        df = pd.DataFrame([[istep,k,i,j,v,bdnme]],
                             columns=['istep','layer','i','j','value','boundname'])
                        _df = df.copy(deep=True)
                        _df[['qfilename', 'qtype', 'qrow','qcol']] = [None, 'mail', None, None]
                    else:

                        qfilename = os.path.normpath(os.path.join(mm_ws, f"{_file.group(1)}.{_file.group(2)}"))
                        df, _df = read_record_qfile(i,j,k,v, qfilename, qcol)

                # ---- Append (meta)DataFrame list
                dfs.append(df)
                _dfs.append(_df)

    # ---- Return concatenate (meta)DataFrame
    data, metadata = [df.reset_index(drop=True) for df in list(map(pd.concat, [dfs, _dfs]))]
    return data, metadata





def convert_at2clp(pastpfile, mm):
    """
    Function convert 'affluent' / 'tronçon' to column, line, plan (layer)
    and rewrite it inplace in pastp file.
    Warn user about the conversion.

    Parameters:
    ----------
    pastpfile (str) : path to pastp file
    mm (object) : MartheModel instance
    
    Returns:
    --------
    Replace lines inplace in pastp file

    Examples:
    --------
    convert_at2clp(pastpfile, mm)
    
    """
    # ---- Set regular expression of numeric string (int and float)
    re_num = r"[-+]?\d*\.?\d+|\d+"
    re_block = r";\s*\*{3}\n(.*?)/\*{5}"

    # ---- Build modelgrid if not exists
    if mm.modelgrid is None:
        mm.build_modelgrid()

    # ---- Add aff and trc column in modelgrid
    for ext in ['aff_r', 'trc_r']:
        fname = ext.replace('_r', '')
        field = pymarthe.mfield.MartheField(fname, mm.mlfiles[ext], mm)
        mm.modelgrid[fname] = field.data['value']

    # ---- Extract .pastp file content
    with open(pastpfile, 'r', encoding=encoding) as f:
        # ---- Extract pastp by block 
        blocks = re.findall(re_block, f.read(), re.DOTALL)
    
    # ---- Set boolean marker to know if at list one convertion had been done
    any_at = []

    # ---- Iterate over timestep block data
    for block in blocks:
        # ---- Iterate over block content (lines)
        for line in block.splitlines():
            # ---- Check if the line contain aff/trc
            if all(s in line for s in ['Q_EXTER_RIVI','A=','T=']):
                # -- Detect conversion
                any_at.append(True)
                # -- Replace TRONCON to MAILLE
                mail_line = line.replace('TRONCON', 'MAILLE')
                # -- Get substring to replace
                s2replace = line[line.index('A='):line.index('V=')]
                # -- Fetch aff/trc as number
                a,t = map(ast.literal_eval, re.findall(re_num, s2replace))
                # ---- Convert aff/trc to column, line, plan (layer) by querying the grid
                i, j, layer = mm.query_grid(aff=a, trc=t, target=['i','j','layer']).to_numpy()[0]
                # -- Build substring to replace
                sub = '{:>8}C={:>7}L={:>7}P={:>7}'.format(' ',j+1, i+1, layer+1)
                # -- Build entire line to be replaced for
                l2replace = mail_line.replace(s2replace,sub)
                # -- Replace text in pastp file
                replace_text_in_file(pastpfile, line, l2replace)

    # ---- User warning about aff/trc conversion
    if len(any_at) > 0:
        msg = f"WARNINGS : some river pumpings in {pastpfile} file are " \
               "located by there 'tributary' and 'reach' ids which is not supported. " \
               "They will be converted to 'row', 'column', 'layer' format instead (inplace)."
        warnings.warn(msg)




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
    data = pd.read_csv(obsfile, delim_whitespace=True,
                                header=None, skiprows=1,
                                index_col=0, parse_dates=True)
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




def get_run_times(logfile = 'bilandeb.txt'):
    """
    Extract run times for model processes from log file.

    Parameters:
    ----------
    logfile (str) : log filename.
                    Default 'bilandeb.txt'

    Returns:
    --------
    df (DataFrame) : Summary run times
                     Format :   
                                  Process        CPU time
                                process_1          time_1
                                process_2          time_2
                                      ...             ...

    Examples:
    --------
    rtdf = get_run_times(logfile='model/bilandeb.txt')
    """
    # -- Extract logs
    with(open(logfile, 'r', encoding='latin-1')) as f:
        content = f.read()

    # -- Regex pattern to find in log
    re_rt = r"Temps CPU\s*(pour|\s*)(.+)=\s*([-+]?\d*\.?\d+|\d+)"

    # -- Search pattern in logs and collect times
    process, times = [],[]
    for _, p, t in re.findall(re_rt, content):
        pro = p.strip().capitalize()
        t = int(ast.literal_eval(t))
        h = t // 3600
        m = t % 3600 // 60
        s = t % 3600 % 60
        time = ''
        if h > 0:
            time += f'{h:02d}h '
        elif m > 0 or h > 0:
            time += f'{m:02d}m '
        time += f'{s:02d}s'
        process.append(pro)
        times.append(time)

    # -- Return output DataFrame
    return pd.DataFrame({'Process':process,'CPU time':times}).set_index('Process')





def bordered_array(a, v):
    """
    Border an array with constant value.

    Parameters:
    ----------
    a (ndarray) : ndarray to border.
                  Format: 
                  array([[1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1]])

    Returns:
    --------
    array (ndarray) : bordered array.
                      Format:
                      array([[v, v, v, v, v, v, v],
                             [v, 1, 1, 1, 1, 1, v],
                             [v, 1, 1, 1, 1, 1, v],
                             [v, 1, 1, 1, 1, 1, v],
                             [v, 1, 1, 1, 1, 1, v],
                             [v, 1, 1, 1, 1, 1, v],
                             [v, v, v, v, v, v, v]])

    Examples:
    --------
    ba = bordered_array(np.ones((5,5)), 0)
    """
    car = np.c_[np.ones(a.shape[0])*v, a, np.ones(a.shape[0])*v]
    bar = np.c_[np.ones(car.shape[1])*v, car.T, np.ones(car.shape[1])*v]
    return bar.T




def read_budget(filename= 'histobil_nap_pastp.prn'):
    """
    Global budget reader (.prn).
    The `io.StringIO` module is required.

    Parameters:
    ----------
    filename (str, optional) : budget text file to read.
                               Can be :
                                    - aquifer budget by timesteps -->  '~/histobil_nap_pastp.prn' 
                                    - cumulative aquifer budget   -->  '~/histobil_nap_cumu.prn'
                                    - climatic budget             -->  '~/histoclim.prn'
                                    - flow budjet                 -->  '~/histobil_debit.prn'
                               Default is 'histobil_nap_pastp.prn'.

    Returns:
    --------
    1) filename IS a flow budget file ('histobil_debit.prn'):
            bud_dfs (list) : budget DataFrames.
                             Format: [global_flow_df, river_flow_df, cumulative_river_flow_df]

    2) filename IS NOT a flow budget file:
            dub_df (DataFrame) : global budget DataFrame. 

    Examples:
    --------
    nap_cumul_df = read_budget(filename= 'histobil_nap_cumu.prn')
    nap_pastp_df = read_budget(filename= 'histobil_nap_pastp.prn')
    clim_df = read_budget(filename= 'histoclim.prn')
    flow_df, riv_df, criv_df = read_budget(filename= 'histobil_debit.prn')

    """
    # ---- Try to import io package (needed)
    try:
        from io import StringIO
    except:
        err_msg = 'Could not load python `io` module. '
        err_msg += 'Try `pip install io`.'
        raise(ImportError(err_msg))

    # ---- Read raw data as string
    with open(filename, 'r', encoding=encoding) as f:
        content = f.read()

    # ---- Match specific tables
    re_budget = r"Date(.+?)\n(Zone|Légende|Volumes|Bilan|$)"
    tables = re.findall(re_budget, content, re.DOTALL)

    # ---- Extract budget(s)
    bud_dfs = []
    for table in tables:
        if '<Date>' not in table[0]: 
            # -- Convert string budget data to DataFrame (drop numeric date and NaN columns)
            bud_df = pd.read_table( StringIO(table[0]),
                                    index_col=0, parse_dates=True, dayfirst = True
                                        ).dropna(
                                            axis=1).iloc[:,1:]
            # -- Manage column names
            bud_df.columns = bud_df.columns.str.strip()
            bud_df.index.name = 'date'
            # -- Add budget to main list
            bud_dfs.append(bud_df)

    # ---- Return budget DataFrame(s)
    if len(bud_dfs) == 1:
        return bud_dfs[0]
    else:
        return bud_dfs







def read_zonebudget(filename= 'histobil_debit.prn'):
    """
    Zone budget reader.
    The `io.StringIO` module is required.

    Parameters:
    ----------
    filename (str, optional) : flow budget text file to read.
                               Default is 'histobil_debit.prn'.

    Returns:
    --------
    zb_df (DataFrame) : zone budget MultiIndex DataFrame.
                        Format :

                                            Entr_Limit_ext  Sort_Limit_ext      ..
                        zone        date    
                         100  01/01/2006           4658895         4667213      ..
                         100  02/01/2006           4658796         4649391      ..
                          ..          ..                ..              ..      ..
                         920  01/01/2009           4697856         4263942      ..

    Examples:
    --------
    zb_df = read_zonebudget()

    """
    # ---- Try to import io package (needed)
    try:
        from io import StringIO
    except:
        err_msg = 'Could not load python `io` module. '
        err_msg += 'Try `pip install io`.'
        raise(ImportError(err_msg))

    # ---- Read raw data as string
    with open(filename, 'r', encoding=encoding) as f:
        content = f.read()

    # ---- Manage regex expression/generation
    re_zone = r"\n(\s*|.+)Zone(\s+|\s+=\s+)(\d+)\s*\n"

    get_re_zb = lambda z: ''.join(
                            [ re_zone.replace(r'(\d+)', str(z)),
                              r"(.+?)\n(Bilan|Zone|Débits|Légende)" ]
                        )

    # ---- Extract zone budget for each zones
    zb_dfs = []

    for match_zone in re.findall(re_zone, content):
        # -- Convert match to interger zone number
        zone = int(match_zone[-1]) 
        # -- Extract string table with raw data
        zb_table = re.search(
                        get_re_zb(zone), content, re.DOTALL
                        ).group(3)
        # -- Convert to DataFrame skiping the useless additional headers and NaN columns
        zb_df = pd.read_table(StringIO(zb_table), skiprows=[1]).dropna(axis=1)
        # -- Drop numeric date column
        zb_df.drop(columns=zb_df.columns[1], inplace=True)
        # -- Manage column names
        zb_df.columns = ['date'] + list(zb_df.columns[1:].str.strip())
        # -- Convert date column to datetime format
        zb_df['date'] = pd.to_datetime(zb_df['date'], dayfirst = True)
        # -- Add a column with the zone id
        zb_df['zone'] = zone
        # -- Add zone budget to main list
        zb_dfs.append(zb_df)

    # ---- Manage absent zone budgets
    err_msg = f"ERROR : there are no existing zone budgets in '{filename}'."
    assert len(zb_dfs) > 0, err_msg

    # ---- Concatenate all zone budget DataFrames with MultiIndex (zone, date)
    zb_df = pd.concat(zb_dfs).set_index(['zone', 'date'])

    # ---- Return MultiIndex DataFrame
    return zb_df





def hydrodyn_periodicity(pastpfile, istep, external=False, new_pastpfile=None):
    """
    Helper function to manage hydrodynamic computation periodicity in .pastp file.

    Parameters:
    ----------

    pastpfile (str) : path to the required model .pastp file.

    istep (str/int/iterable) : required istep to activate hydrodynamic computation.
                               Can be :
                                    - 'all' : activate for all timesteps
                                    - 'none': desactivate for all timesteps
                                    - 'start:end:step' : string sequence
                                    - [0,1,2,..] : any integer iterables

    external (bool, optional) : whatever create a external file with required
                                hydrodynamic computation periodicity.
                                Note: external optional will not be considered
                                      for global `istep` such as 'all', 'none'
                                Default is False.

    new_pastpfile (str, optional) : path to the new pastp file to write.
                                    If None, will overwrite the input pastp file.
                                    Default is None.

    Returns:
    --------
    (Over-)write pastp file.
    If `external` == True, will also write 'cacul_hydro.txt'.
                      .

    Examples:
    --------
    f = 'mymodel.pastp'
    # -- All timesteps
    hydrodyn_periodicity(f, istep= 'all', external=False)
    # -- Weekly
    hydrodyn_periodicity(f, istep= '::7', external=True)
    # -- Annual
    hydrodyn_periodicity(f, istep= '::365', external=False)
    # -- Specific
    hydrodyn_periodicity(f, istep= [0,5,6,7,9,11], external=True)

    """
    # ---- Read .pastp file by lines
    with open(pastpfile, 'r', encoding = encoding) as f:
        init = f.readlines()

    # ---- Clear existing hydrodynamic action
    lines = [l for l in init if not 'CALCUL_HDYNAM' in l]
    nlines = deepcopy(lines)

    # -- Manage string istep
    if isinstance(istep, str):
        # ---- Activate all timesteps
        if istep == 'all':
            # -- Find index `i` where insert required line
            for i, l in enumerate(lines):
                if 'Fin de ce pas' in l:
                    break
            # -- Insert line to activate all timesteps (default) 
            _l ='  /CALCUL_HDYNAM/ACTION    I= 0;\n'
            nlines.insert(i, _l)

        # ---- Desactivate all timesteps
        if istep == 'none':
            # -- Find index `i` where insert required line
            for i, l in enumerate(lines):
                if 'Fin de ce pas' in l:
                    break
            # -- Insert line to activate all timesteps (default)
            _l ='  /CALCUL_HDYNAM/ACTION    I= -1;\n'
            nlines.insert(i, _l)

        # ---- Manage string sequence to get istep 
        if ':' in istep:
            # -- Extract total number of timesteps
            nstep = len(re.findall(r'Fin de ce pas', ''.join(lines)))
            # -- Extract time bounds (start, end, step) as integers
            re_seq = r'(\d+|\s*):(\d+|\s*):(\d+)'
            s, e, ii = re.findall(re_seq,istep)[0]
            start = eval(s) if s != '' else 0
            end = eval(e) if e != '' else nstep
            step = eval(ii) if ii != '' else 1
            # -- Set numerical istep as array
            if start < end:
                istep = np.arange(start, end, step)
            else:
                istep = np.arange(end, start, step)

    # -- Manage iterable istep
    if not isinstance(istep, str):
        # -- Make istep iterable if is not already
        isteps = make_iterable(istep)
        # -- Manage external mode
        if external:
            # -- Set external filename (generic) in first timestep
            ext = 'calcul_hydro.txt'
            _l = f'  /CALCUL_HDYNAM/ACTION    I= 0; File= {ext}\n'
            ext_path = os.path.join(os.path.split(pastpfile)[0], ext)
            for i, l in enumerate(lines):
                if 'Fin de ce pas' in l:
                    break
            nlines.insert(i, _l)
            # -- Fetch timesteps calendar dates
            dates = re.findall(r'\d{2}\/\d{2}\/\d{4}', ''.join(lines))
            # -- Set external file header
            s =  "HYDRODYNAMIC COMPUTATION PLANNING\n"
            # -- Iterate over all time steps (2 = active, 0 = inactive)
            for n, date in enumerate(dates):
                # -- Active if in required timesteps desactive otherwise
                if n in isteps:
                    s += f'2\t{date}\n'
                else:
                    s += f'0\t{date}\n'

            # -- Write external file
            with open(ext_path, 'w', encoding =encoding) as f:
                f.write(s)

        # -- Manage internal mode
        else:
            # -- Set active/desactive line to write
            _l = '  /CALCUL_HDYNAM/ACTION    I= 2;\n'
            __l = '  /CALCUL_HDYNAM/ACTION    I= 0;\n'
            n = -1
            # -- Iterate over pastp lines
            for i, l in enumerate(lines):
                # -- Detect end of time step line index
                if 'Fin de ce pas' in l:
                    n += 1
                    # -- Insert active/desative line
                    if n in isteps:
                        nlines.insert(i+n,_l)
                    else:
                        nlines.insert(i+n,__l)

    # -- (Over-)write pastp file 
    out = pastpfile if new_pastpfile is None else new_pastpfile
    with open(out, 'w', encoding=encoding) as f:
        f.write(''.join(nlines))

    # -- Print final message
    print("==> Hydrodynamic computation periodicity " \
          f"had been set successfully in '{out}'") 




def set_tw(start=None, end=None, mm=None, martfile=None, pastpfile=None):
    """
    Function to set/change model time window in .mart file.
    Note: the .pastp file will not be modify.

    Parameters:
    ----------
    start (str/int, optional) : string date or istep number of required
                                first timestep to consider.
                                If None, the first istep (in .pastp file)
                                will be considered.
                                Default is None.

    end (str/int, optional) : string date or istep number of required 
                              last timestep to consider.
                              If None, the last istep (in .pastp file)
                              will be considered.
                              Default is None.

    mm (MartheModel, optional) : MartheModel with correct .mldates and .mlfiles.
                                 If None, `martfile` and `pastpfile` arguments
                                 will be considered.
                                 Default is None.

    martfile (str, optional) : related model .mart file
                               Default is None.

    pastpfile (str, optional) : related model .mart file
                               Default is None.


    Returns:
    --------
    Change .mart file inplace with required time window.

    Examples:
    --------
    # -- From isteps
    set_tw(start=10, end=35, mm=mm)

    # -- From dates
    set_tw(start='1999/01/28', end=65, mm=mm)

    # -- From external files
    set_tw(start=10, end=35,
                martfile='mymodel.mart',
                pastpfile='mymodel.pastp')
    """
    # ---- Manage provided required Marthe files
    if mm is not None:
        pastpfile = mm.mlfiles['pastp']
        martfile = mm.mlfiles['mart']
        dates = mm.mldates
    else:
        err_msg = "ERROR : could not reach .mart or .pastp file. " \
                  "Make sure to provide either a `MartheModel` " \
                  "instance or both .mart and .pastp file names."
        assert all(f is not None for f in [martfile, pastpfile]), err_msg
        dates = get_dates(pastpfile, martfile)

    # ---- Manage default start/end input
    start = 0 if start is None else start
    end = 0 if end is None else end

    # ---- Manage time bounds
    tbounds = []
    for tb in [start, end]:

        # -- Manage date input (str)
        if isinstance(tb, str):
            # -- Convert string to pandas Timestamp
            ts = pd.Timestamp(tb)
            # -- Check if fall in model time window
            err_msg = "ERROR : provided date (str) is outside the model time window. " \
                      f"Given: {tb}."
            assert ts in dates, err_msg
            # -- Convert to istep (int)
            itb = dates.get_loc(ts)

        # -- Manage istep input (int)
        elif isinstance(tb, int):
            # -- Check if fall in model time window
            err_msg = "ERROR : provided istep (int) is outside the model time window. " \
                      f"Given: {tb}."
            assert tb in list(range(len(dates))), err_msg
            itb = tb

        # # -- Manage not provided end
        # elif tb is None:
        #     itb = 0 #défault

        # -- Add time bound
        tbounds.append(itb)

    istart, iend = tbounds

    # ---- Assert that start < end
    err_msg = "ERROR : `end` timestep must be greater than `start`. " \
              "Given : {}, {}.".format(*tbounds)
    assert np.logical_or(istart < iend, iend == 0) , err_msg

    # ---- Extract .mart file content
    with open(martfile, encoding=encoding) as f:
        lines = f.readlines()

    # ---- Set usefull regex
    re_block = r"\*{3}\s*(Pas|Pas de Temps) et"
    re_repl = r"([-+]?\d*\.?\d+|\d+)|(\*|\s*)=N"
    re_bef = r"(^\s*)[\*|\d]"

    # ---- Extract required lines
    for i, line in enumerate(lines):
        if re.search(re_block, line) is not None:
            start_line = lines[i+4]
            end_line = lines[i+1]
            new_start_line = '{}{}={}'.format(re.search(re_bef, start_line).group(1),
                                              istart,
                                              '='.join(start_line.split('=')[1:]))
            new_end_line = '{}{}={}'.format(re.search(re_bef, end_line).group(1),
                                              iend,
                                              '='.join(end_line.split('=')[1:]))
            break

    # ---- Replace start line
    replace_text_in_file(martfile, start_line, new_start_line)

    # ---- Replace end line
    replace_text_in_file(martfile, end_line, new_end_line)

    # -- Print final message
    print(f"==> Model time window had been set from istep " \
          f"{istart} to {iend} successfully. ")





def get_tw(mm=None, martfile=None, pastpfile=None, tw_type='date'):
    """
    Function to extract model time window in .mart file.

    Parameters:
    ----------
    mm (MartheModel, optional) : MartheModel with correct .mldates and .mlfiles.
                                 If None, `martfile` and `pastpfile` arguments
                                 will be considered.
                                 Default is None.

    martfile (str, optional) : related model .mart file
                               Default is None.

    pastpfile (str, optional) : related model .mart file
                               Default is None.

    tw_type (str, optional) : time window output type.
                              Can be :
                                - 'date' : return pd.timestamp objects
                                - 'istep': return integers
                              Default is 'date'.

    Returns:
    --------
    tw_min, tw_max (tuple): time window bounds (start/end)

    Examples:
    --------
    # -- From isteps
    istart, iend = get_tw(mm=mm, tw_type='istep')
    # -- Get time window dates
    start, end = get_tw(mm=mm, tw_type='date')
    # -- From external files
    start, end = get_tw(martfile='mymodel.mart', pastpfile='mymodel.pastp')

    """
    # ---- Manage provided required Marthe files
    if mm is not None:
        pastpfile = mm.mlfiles['pastp']
        martfile = mm.mlfiles['mart']
    else:
        err_msg = "ERROR : could not reach .mart or .pastp file. " \
                  "Make sure to provide either a `MartheModel` " \
                  "instance or both .mart and .pastp file names."
        assert all(f is not None for f in [martfile, pastpfile]), err_msg

    # ---- Extract .mart file content
    with open(martfile, encoding=encoding) as f:
        lines = f.readlines()

    # ---- Set usefull regex
    re_block = r"\*{3}\s*(Pas|Pas de Temps) et"
    re_istep = r"([-+]?\d*\.?\d+|\d+)|(\*|\s*)=N"

    # ---- 0 if there has no match
    tw_infer = lambda m: 0 if m.group(1) is None else ast.literal_eval(m.group(1))

    # ---- Extract required lines
    for i, line in enumerate(lines):
        if re.search(re_block, line) is not None: 
            tw_min = tw_infer(re.search(re_istep, lines[i+4]))
            tw_max = tw_infer(re.search(re_istep, lines[i+1]))
            break

    # ---- Fetch model dates from .pastp file
    dates = get_dates(pastpfile, martfile) if mm is None else mm.mldates

    # ---- Rectify Marthe default max date
    if tw_max == 0:
        tw_max = len(dates) - 1

    # ---- Return time window as tuple
    if tw_type == 'date':
        return dates[tw_min], dates[tw_max]

    elif tw_type == 'istep':
        return tw_min, tw_max

