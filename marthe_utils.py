# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import dates as mdates
import datetime
import subprocess

############################################################
#        Utils for pest preprocessing for Marthe
############################################################

def write_grid_file():
    '''
    Description
    -----------
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

    Return
    ------
    Number of observations (obs_num)
    Number of observation groups (nobs_grp)
    Dictionary with  observation dates :  obs_dates = {'ID1':[date1,...daten], ... }
    '''
    # initialize dictionary of observation dates