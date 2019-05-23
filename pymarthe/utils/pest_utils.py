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

# ----------------------------------------------------------------------------------------------------------
#Modified from PyEMU 
#https://github.com/jtwhite79/pyemu/
# ----------------------------------------------------------------------------------------------------------

# ----- from PyEMU ----------------
def SFMT(item):
    try:
        s = "{0:<20s} ".format(item.decode())
    except:
        s = "{0:<20s} ".format(str(item))
    return s

SFMT_LONG = lambda x: "{0:<50s} ".format(str(x))
IFMT = lambda x: "{0:<10d} ".format(int(x))
FFMT = lambda x: "{0:<20.10E} ".format(float(x))


PP_FMT = {"parname": SFMT, "x": FFMT, "y": FFMT, "zone": IFMT, "tpl": SFMT,
"value": FFMT}


def write_tpl_from_df(tpl_file,df) :  
    f_tpl = open(tpl_file,'w')
    f_tpl.write("ptf ~\n")
    f_tpl.write(df.to_string(col_space=0,
        columns=["parname","tpl"],
        formatters=PP_FMT,
        justify="left",
        header=False,
        index=False) + '\n')

# -----------------------------------
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
