import sys 
import os 
import pandas as pd
import numpy as np

# pyemu and adeqwat modules should be placed in ~/Programmes/python/

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

params = ['kepon','permh','emmca','emmli']
log_transform_dic = {'kepon': True,'permh': True,'emmca' : True,'emmli': False}

# -- parameter
for param in params : 
    # add parameter
    mm.add_param(param,log_transform = log_transform_dic[param])
    # load zpc parameter values from file (written by PEST)
    mm.param[param].read_zpc_df()
    # set corresponding mm grid array from zpc_values 
    mm.param[param].set_array_from_zpc_df()
    # write mm grid array to file 
    mm.write_grid(param)
    
# -- model run
mm.run_model()
# extract prn and write simulation files (read by PEST)
mm.extract_prn()

