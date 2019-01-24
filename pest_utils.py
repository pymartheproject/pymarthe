# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import dates as mdates
import datetime
import subprocess

############################################################
#        Utils for pest preprocessing #test
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
	pst_observation_groups_file= open(output_dir + 'pst_observation_groups.txt', 'w')
	pst_io_file = open(output_dir +   'pst_io_obs.txt', 'w')
	obs_data_file = open(output_dir + 'pst_obs_list.txt', 'w')

    except : 
	print('Write error in ' + output_dir )
	return

    # init observations and observation groups counters
    obs_num = 0
    nobs_grp = 0

    # width of values (number of characters)
    VAL_CHAR_LEN = 12

    # Start function for each observation file
    for obs_point in obs_points:	

	# read observed data for obs_point
        obs_point_file_path = input_dir + obs_point + '.dat'
	
	try : 
	    obs_point_array = np.genfromtxt(obs_point_file_path, 
		delimiter='	',skip_header=True,unpack=True,
		converters={ 0: mdates.strpdate2num(date_string_format)}
		)
	    print('Successfully read ' + obs_point + ' observation file.')
	except : 
	    print('Cannot find ' + obs_point + ' observation file.')
	    continue

	obs_point_datenums = obs_point_array[0,:]
	obs_point_values = obs_point_array[1,:]
	obs_point_weight = obs_point_array[2,:]

        # Interpolate / subset observed values
	# Interpolate missing values
	if lin_interp == True :
	    select_obs_values = interp_from_file(obs_point_file, dates_out, date_string_format = date_string_format)
	# Look for available observed values at simulation dates (dates_out)
	else : 
	    idx = np.in1d(obs_point_datenums, obs_point_datenums)# mdates.date2num(dates_out))
	    select_obs_values = obs_point_values[idx]
	    select_obs_weight = obs_point_weight[idx]

        # fill obs_dates dictionary
        obs_dates[obs_point]= obs_point_datenums

	# open instruction and observation files for obs_point
	ins_file = open(output_dir + obs_point + '.ins', 'w')
	print('Writing in ' + output_dir + obs_point + '.ins')
	# write instruction file header 
	ins_file.write('pif #'+ '\n')

	# Write instruction and observatin line for each value
	for obs_val, obs_weight in zip(select_obs_values, select_obs_weight):
	    obs_num += 1	
	    ins_file.write('l1 ' + '(o' + str(obs_num) + ')' + '1:' + str(VAL_CHAR_LEN) + '\n')
	    obs_data_file.write('o' + str(obs_num) + ' ' + str(obs_val) + ' ' + str(obs_weight) + ' ' + str(obs_point) + '\n')
	
	# add point entry to pst files 
	pst_observation_groups_file.write( obs_point + '\n')
	pst_io_file.write( 'pest_files/' + obs_point + '.ins' + ' ' + 'pest_files/' + obs_point + '.dat' + '\n' )

	# close point files
	ins_file.close()

	# increment number of observation groups
	nobs_grp += 1

    # close pst files
    pst_observation_groups_file.close() 
    pst_io_file.close() 
    obs_data_file.close()

    return( obs_num, nobs_grp, obs_dates )


def write_tpl_files(dic_params, output_dir, dic_params_init = None):
    '''
    Description
    -----------
    Computation of template files for pest
    
    Parameters
    ----------
    dic_params : dict
            Dict with groups of calibration in keys with parameters names inside
        ex: dic_params = {'Hydrodynamic':['T','S'], 'Recharge':['CRT','DCRT','FN','QImax','QRmax','CQI','CQR']}
    output_dir : str
            path to the output directory where template files will be written
    dic_params_init : dict
	    initial parameter values  
	    ex: dic_params_init = {'Hydrodynamic':[0.01,0.1], 'Recharge':[80, 5, 10, 130, 80, 0.05, 0.3]}
    Returns
    -------

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
	print('Write error in ' + output_dir + '.')
	
    for param_grp in dic_params.keys():

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


# ---------------------------------------------------------------
def gen_time_seq(date_start_string, tstep, n_tstep, date_string_format = '%Y-%m-%d %H:%M' ) :
    """
    Description.

    Parameters
    ----------
    date_start_string : date from which to start the sequence in string format
                        '%Y-%m-%d %H:%M', e.g. '2015-03-02 12:00'.
    tstep : length of time step in seconds.
    n_tstep : number of time steps
    date_string_format : date string format used in input file, 
                         see https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior
    
    Returns
    -------

    dates_out : list of datetime objects

    Examples
    --------
    # generate a sequence of 20 datetimes instances starting from 2015-03-02 12:00 at 1-minute-interval. 
    >>> dates_out = gen_time_seq(date_start_string = '2015-03-02 12:00' ,tstep = 60 ,n_tstep = 20)

    """
    # convert date_start_string to datetime format
    date_start = dt.datetime.strptime(date_start_string, date_string_format)
    # estimate date_end in datetime format
    date_end = date_start + dt.timedelta(seconds = tstep*n_tstep)
    # generate sequence of dates
    datenums_out = mdates.drange(date_start,date_end - dt.timedelta(seconds=0.001), dt.timedelta(seconds=tstep))
    # convert to list of datetime 
    dates_out = mdates.num2date(datenums_out)

    return(dates_out)


