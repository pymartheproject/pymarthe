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

# -- parameter
mm.add_param('kepon')

# load zpc parameter values from file
mm.param['kepon'].read_zpc_df()

# set corresponding mm grid array from zpc_values 
mm.param['kepon'].set_array_from_zpc_df()

# write mm grid array to file 
mm.write_grid('kepon')

# -- model run
mm.run_model()

# -- load simulation

# extract prn
mm.extract_prn()


