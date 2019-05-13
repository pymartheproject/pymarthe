# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import dates as mdates
import datetime
import subprocess
from matplotlib.dates import bytespdate2num
import matplotlib.dates as mdates
import pandas as pd

############################################################
#        Utils for pest preprocessing for Marthe
############################################################

def write_obs_data(obs_points, input_dir, output_dir, dates_out, date_string_format = '%Y', lin_interp=False):
    '''
    Description
    -----------
    Write PEST instruction files and observation section of the PEST control file (.pst)
    This function explores input_dir for csv files containing obs_points.
    File names with corresponding obs_points will be used to write the corresponding instruction file.
    Observed data within each file will be considered as an observation group in the pst file.

    Parameters
    ----------
    obs_points : list of observation names (IDs)

    input_dir : directory with observation record files,
                a two-column (date and value) csv file with header.
                The observation files must not contain NA values.

    lin_interp : If True, missing observations are generated at
                simulated values (dates_out) by linear interpolation
                See qgridder_utils_tseries.interp_from_file()

    date_string_format : format of date strings in observation files (e.g.  '%Y-%m-%d')

    dates_out : Sequence of fixed frequency DatetimeIndex 

    Return
    ------
    Number of observations (obs_num)
    Number of observation groups (nobs_grp)
    Dictionary with  observation dates :  obs_dates = {'ID1':[date1,...daten], ... }
    '''

    # initialize dictionary of observation dates

    obs_dates = {}
    # open pst files
    try :
        pst_observation_groups_file = open(output_dir + 'pst_observation_groups.txt', 'w')
        pst_io_file   = open(output_dir +   'pst_io_obs.txt', 'w')
        obs_data_file = open(output_dir + 'pst_obs_list.txt', 'w')


    except :
        print(('Write error in ' + output_dir ))
        return

    # init observations and observation groups counters
    obs_num  = 0
    nobs_grp = 0

    # width of values (number of characters)
    VAL_CHAR_LEN = 21
    VAL_START = 12

    # Start function for each observation file
    for obs_point in obs_points:

        # read observed data for obs_point
        obs_point_file_path = input_dir + obs_point + '.txt'

        try :
            df_obs_point = pd.read_csv(obs_point_file_path,sep='\t')
            df_obs_point.Year = pd.to_datetime(df_obs_point.Year,  format="%Y-%m-%d")
            df_obs_point = df_obs_point.set_index(df_obs_point.Year)
            print(('Successfully read ' + obs_point + ' observation file.'))
        except :
            print(('Cannot find ' + obs_point + ' observation file.'))
            continue

        obs_point_datenums = df_obs_point.Year
        obs_point_values   = df_obs_point.Mean
        obs_point_weight   = df_obs_point.Weight

        # Interpolate / subset observed values
        # Interpolate missing values
        if lin_interp == True :
            select_obs_values = interp_from_file(obs_point_file, dates_out, date_string_format = date_string_format)
        # Look for available observed values at simulation dates (dates_out)
        else :
            
            select_obs_values = obs_point_values[dates_out]
            select_obs_weight = obs_point_weight[dates_out]

           

        # fill obs_dates dictionary
        obs_dates[obs_point]= obs_point_datenums

        # open instruction and observation files for obs_point
        ins_file = open(output_dir + obs_point + '.ins', 'w')
        print(('Writing in ' + output_dir + obs_point + '.ins'))
        # write instruction file header
        ins_file.write('pif #'+ '\n')

        # Write instruction and observatin line for each value
        for obs_val, obs_weight in zip(select_obs_values, select_obs_weight):
            obs_num += 1
            ins_file.write('l1 ' + '(o' + str(obs_num) + ')' + str(VAL_START) +':' + str(VAL_CHAR_LEN) + '\n')
            obs_data_file.write('o' + str(obs_num) + ' ' + str(obs_val) + ' ' + str(obs_weight) + ' ' + str(obs_point) + '\n')

        # add point entry to pst files
        pst_observation_groups_file.write( obs_point + '\n')
        pst_io_file.write( 'pest_files/' + obs_point + '.ins' + ' ' + 'pest_files/' + obs_point + '.txt' + '\n' )

        # close point files
        ins_file.close()

        # increment number of observation groups
        nobs_grp += 1

    # close pst files
    pst_observation_groups_file.close()
    pst_io_file.close()
    obs_data_file.close()

    return( obs_num, nobs_grp, obs_dates)



def write_tpl_files(dic_params, output_dir, dic_params_init = None):
    '''
    #Description
    #-----------
    #Computation of template files for pest

    #Parameters
    #----------
    #dic_params : dict
     #       Dict with groups of calibration in keys with parameters names inside
      #  ex: dic_params = {'Hydrodynamic':['T','S'], 'Recharge':['CRT','DCRT','FN','QImax','QRmax','CQI','CQR']}
    #output_dir : str
     #       path to the output directory where template files will be written
    #dic_params_init : dict
    #        initial parameter values
     #       ex: dic_params_init = {'Hydrodynamic':[0.01,0.1], 'Recharge':[80, 5, 10, 130, 80, 0.05, 0.3]}
    #Returns
   #-------
    '''
    # iterate over parameter groups
    pst_par_grp_file_path = output_dir + 'pst_par_grp.txt'
    pst_par_data_file_path = output_dir + 'pst_par_data.txt'
    pst_io_par_file_path = output_dir + 'pst_io_par.txt'

    npar = 0
    npar_grp = 0

    try :
        file_pst_par_grp = open(pst_par_grp_file_path, 'w')
        file_pst_par_data = open(pst_par_data_file_path, 'w')
        file_pst_io_par = open(pst_io_par_file_path, 'w')

    except :
        print(('Write error in ' + output_dir + '.'))

    for param_grp in list(dic_params.keys()):

        # write parameter group and parameter I/O files data
        file_pst_par_grp.write(param_grp + '    relative 0.01  0.0  switch  2.0 parabolic\n')
        file_pst_io_par.write('pest_files/'+ 'param_' + param_grp + '.tpl' + ' ' + 'param_' + param_grp + '.dat' + '\n'  )

        # set up and fill template file
        tpl_file_path = output_dir + 'param_' + param_grp + '.tpl'

        #try :
        file_tpl = open(tpl_file_path,'w')
        npar_grp += 1
        file_tpl.write('ptf #' + '\n')
        # iterate over parameters within this group
        for i in range(len(dic_params[param_grp])) :
            npar += 1
            param = dic_params[param_grp][i]
            file_tpl.write('# ' + param_grp + '_' + param + ' #' + '\n')
            if not dic_params_init is None :
                par_init_val = float ( dic_params_init[param_grp][i] )
                par_dat_line =  param_grp + '_' + param +  '  log  factor     ' + str(par_init_val) + '      1.000000E-10   1.000000E+10 ' + \
                        param_grp + '        1.0000        0.0000      1\n'
            else :
                par_dat_line =  param_grp + '_' + param +  '  log  factor     -99999      1.000000E-10   1.000000E+10 ' + \
                        param_grp + '        1.0000        0.0000      1\n'
            file_pst_par_data.write(par_dat_line)
        file_tpl.close()
        #except :
        #    print('Write error in ' + output_dir + '.')

    file_pst_par_grp.close()
    file_pst_par_data.close()
    file_pst_io_par.close()

    return(npar,npar_grp)


def write_pst_io(path_output_files) :
    try :
        subprocess.call('cat ./pest_files/pst_io_par.txt ./pest_files/pst_io_obs.txt > pest_files/pst_io.txt',shell=True)

    except :
        print('Cannot find pst_io_par.txt or pst_io_obs.txt')

# ----------------------------------------------------------------------------------------------------------
#Extraction from PyEMU 
#https://github.com/jtwhite79/pyemu/blob/develop/pyemu/
# ----------------------------------------------------------------------------------------------------------

PP_NAMES = ["name","x","y","zone","parval1"]

def fac2real(pp_file=None,factors_file="factors.dat",
             upper_lim=1.0e+30,lower_lim=-1.0e+30,fill_value=1.0e+30):
    """A python replication of the PEST fac2real utility for creating a
    structure grid array from previously calculated kriging factors (weights)
    Parameters
    ----------
    pp_file : (str)
        PEST-type pilot points file
    factors_file : (str)
        PEST-style factors file
    upper_lim : (float)
        maximum interpolated value in the array.  Values greater than
        upper_lim are set to fill_value
    lower_lim : (float)
        minimum interpolated value in the array.  Values less than lower_lim
        are set to fill_value
    fill_value : (float)
        the value to assign array nodes that are not interpolated
    Returns
    -------
    arr : numpy.ndarray
        if out_file is None
    out_file : str
        if out_file it not None
    Example
    -------
    ``>>>import pyemu``
    ``>>>pyemu.utils.geostats.fac2real("hkpp.dat",out_file="hk_layer_1.ref")``
    """

    if pp_file is not None and isinstance(pp_file,str):
        assert os.path.exists(pp_file)
        pp_data = pp_file_to_dataframe(pp_file)
        pp_data.loc[:,"name"] = pp_data.name.apply(lambda x: x.lower())
    elif pp_file is not None and isinstance(pp_file,pd.DataFrame):
        assert "name" in pp_file.columns
        assert "parval1" in pp_file.columns
        pp_data = pp_file
    else:
        raise Exception("unrecognized pp_file arg: must be str or pandas.DataFrame, not {0}"\
                        .format(type(pp_file)))
    assert os.path.exists(factors_file)
    f_fac = open(factors_file,'r')
    fpp_file = f_fac.readline()
    if pp_file is None and pp_data is None:
        pp_data = pp_file_to_dataframe(fpp_file)
        pp_data.loc[:, "name"] = pp_data.name.apply(lambda x: x.lower())

    fzone_file = f_fac.readline()
    ncol,nrow = [int(i) for i in f_fac.readline().strip().split()]
    npp = int(f_fac.readline().strip())
    pp_names = [f_fac.readline().strip().lower() for _ in range(npp)]

    # check that pp_names is sync'd with pp_data
    diff = set(list(pp_data.name)).symmetric_difference(set(pp_names))
    if len(diff) > 0:
        raise Exception("the following pilot point names are not common " +\
                        "between the factors file and the pilot points file " +\
                        ','.join(list(diff)))

    pp_dict = {int(name):val for name,val in zip(pp_data.index,pp_data.parval1)}
    try:
        pp_dict_log = {name:np.log10(val) for name,val in zip(pp_data.index,pp_data.parval1)}
    except:
        pp_dict_log = {}

    out_index = []
    out_values = []
    while True:
        line = f_fac.readline()
        if len(line) == 0:
            break
        try:
            inode,itrans,fac_data = parse_factor_line(line)
        except Exception as e:
            raise Exception("error parsing factor line {0}:{1}".format(line,str(e)))
        if itrans == 0:
            fac_sum = sum([pp_dict[pp] * fac_data[pp] for pp in fac_data])
        else:
            fac_sum = sum([pp_dict_log[pp] * fac_data[pp] for pp in fac_data])
        if itrans != 0:
            fac_sum = 10**fac_sum
        out_values.append(fac_sum)
        out_index.append(inode)

    df = pd.DataFrame(data={'vals':out_vals},index=out_index)

    return(df)


def parse_factor_line(line):
    """ function to parse a factor file line.  Used by fac2real()
    Parameters
    ----------
    line : (str)
        a factor line from a factor file
    Returns
    -------
    inode : int
        the inode of the grid node
    itrans : int
        flag for transformation of the grid node
    fac_data : dict
        a dictionary of point number, factor
    """

    raw = line.strip().split()
    inode,itrans,nfac = [int(i) for i in raw[:3]]
    fac_data = {int(raw[ifac])-1:float(raw[ifac+1]) for ifac in range(4,4+nfac*2,2)}
    return inode,itrans,fac_data


def pp_file_to_dataframe(pp_filename):

    """ read a pilot point file to a pandas Dataframe
    Parameters
    ----------
    pp_filename : str
        pilot point file
    Returns
    -------
    df : pandas.DataFrame
        a dataframe with pp_utils.PP_NAMES for columns
    """

    df = pd.read_csv(pp_filename, delim_whitespace=True,
                     header=None, names=PP_NAMES,usecols=[0,1,2,3,4])
    df.loc[:,"name"] = df.name.apply(str).apply(str.lower)
    return df

