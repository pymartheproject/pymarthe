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
mm = MartheModel('/Users/apryet/recherche/adeqwat/dev/adeqwat/MONA_V3/mona.rma')


# -- observations

obs_dir = os.path.join(mm.mldir,'obs','')
obs_files = [os.path.join(obs_dir, f) for f in os.listdir( obs_dir ) if f.endswith(".txt")]

mm.add_obs(obs_file = obs_files[0])

mm.obs['07065X0002'].write_ins()





# -- parameters

# new parameter
mm.add_param('kepon',1e-3)

# pointer to MartheParam instance
kepon = mm.param['kepon']

# write template file
kepon.write_zpc_tpl()


# --- from PyEMU
# https://github.com/jtwhite79/pyemu/blob/c7f25f945033916fbe631766a2cd86fd74fba29b/examples/notest_modflow_to_pest_like_a_boss.ipynb
tpl_files = [os.path.join(mm.mldir,f) for f in os.listdir(mm.mldir) if f.endswith(".tpl")]
input_files = [f.replace(".tpl",'.dat') for f in tpl_files]
ins_files = [os.path.join(ml.model_ws,f) for f in os.listdir(ml.model_ws) if f.endswith(".ins")]
output_files = [f.replace(".ins",'.dat') for f in ins_files]


marthe_utils.read_write_file_sim( os.path.join(mm.mldir, 'historiq.prn'), os.path.join(mm.mldir,'obs',''))

# ------- STEP 2 : use

mm = MartheModel('/Users/apryet/recherche/adeqwat/dev/adeqwat/MONA_V3/mona.rma')

# -- parameter
mm.add_parameter('kepon',1e-3)

# load zpc parameter values from file
kepon.read_zpc_df()

# set parameter array
kepon.set_array_from_zpc_df()

# -- model run


# -- load simulation

# extract prn
mm.extract_prn()

