import sys 
import os 
import pandas as pd
import numpy as np

# https://github.com/jtwhite79/pyemu
sys.path.append(os.path.expanduser('~/Programmes/python/pyemu/'))
import pyemu

# https://github.com/apryet/adeqwat
sys.path.append(os.path.expanduser('~/Programmes/python/adeqwat/'))
from pymarthe import * 

# ---------------------------------------------------
# ------- STEP 2 : use (PEST parameter estimation)
# ---------------------------------------------------

# GO TO MONA WORKING DIRECTORY
# load existing Marthe model
mm = MartheModel('./mona.rma')
nlay, nrow, ncol = mm.nlay, mm.nrow, mm.ncol

# load parameterization and read parameter values
mm.load_param()

# write parameter values to Marthe files 
mm.write_param()

"""
# parameter names 
params = ['emmca','permh','kepon','emmli']

# log-transformation 
log_transform_dic = {'kepon': True,'permh': True,'emmca' : True,'emmli': False}

for param in params :
    # read izone data from disk
    x, y, izone  = marthe_utils.read_grid_file('{0}.i{1}'.format(mm.mlname,param))
    mm.add_param(param, izone = izone, default_value=1e-5, log_transform=log_transform_dic[param])
    # parameters with pilot points (izone with positive values)
    if np.max(np.unique(izone)) > 0  :
        # read pp_df files to get values at pilot points
        mm.param[param].read_pp_df()
        # update grid values for current par, lay, zone
        mm.param[param].interp_from_factors()
    # parameters with zpcs (izones with negative values)
    if np.min(np.unique(izone)) < 0  :
        # parameters ith zpcs  
        mm.param[param].read_zpc_df()
        mm.param[param].set_array_from_zpc_df()
    # write grid to disk
    mm.write_grid(param)
"""

# -- model run
mm.run_model()
# extract prn and write simulation files (read by PEST)
mm.extract_prn(fluct = False)
