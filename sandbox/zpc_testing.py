import sys 
import os 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import geopandas 

sys.path.append('/Users/apryet/Programmes/python/pyemu/')
import pyemu

sys.path.append('/Users/apryet/Programmes/python/adeqwat')
from pymarthe import * 

# ------- STEP 1 : setup
# load existing Marthe model
mm = MartheModel('../MONA_V3/mona.rma')

# new parameter
mm.add_parameter('kepon',1e-3)

# pointer to MartheParam instance
kepon = mm.parameters['kepon']

# write template file
kepon.write_zpc_tpl()


# --- from PyEMU
# https://github.com/jtwhite79/pyemu/blob/c7f25f945033916fbe631766a2cd86fd74fba29b/examples/notest_modflow_to_pest_like_a_boss.ipynb
tpl_files = [os.path.join(ml.model_ws,f) for f in os.listdir(ml.model_ws) if f.endswith(".tpl")]
input_files = [f.replace(".tpl",'.dat') for f in tpl_files]
ins_files = [os.path.join(ml.model_ws,f) for f in os.listdir(ml.model_ws) if f.endswith(".ins")]
output_files = [f.replace(".ins",'.dat') for f in ins_files]


# ------- STEP 2 : use

mm = MartheModel('../MONA_V3/mona.rma')

mm.add_parameter('kepon',1e-3)

# load zpc parameter values from file
kepon.read_zpc_df()

# set parameter array
kepon.set_array_from_zpc_df()


