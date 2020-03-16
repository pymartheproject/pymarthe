import sys 
import os 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import geopandas 

import pyemu
from pymarthe import * 

# load Marthe model
mm = MartheModel('./mona.rma')
nlay, nrow, ncol = mm.nlay, mm.nrow, mm.ncol

# ----------------------------------------------------------------
# Load geometry and identify confined/unconfined domains
# ----------------------------------------------------------------
# load model geometry
x, y, sepon  = marthe_utils.read_grid_file( mm.mlname + '.sepon')
x, y, topog  = marthe_utils.read_grid_file(mm.mlname + '.topog')
x, y, hsubs  = marthe_utils.read_grid_file(mm.mlname + '.hsubs')

# load heads from historical calibration for the definition of 
# confined/unconfined domains
x, y, chasim = marthe_utils.read_grid_file('./chasim_cal_histo.out')

top = sepon 
nper = 40 #number of time steps

NO_EPON_VAL = 9999 #In layer domain but not in eponte
NO_EPON_OUT_VAL = 8888 # Not in layer domain

#Defining Geometry
for lay in range(nlay-2,1,-1):
    #Define indices where values are equal to 9999 or 8888
    idx = np.logical_or(top[lay:nlay,:,:] == NO_EPON_VAL, top[lay:nlay,:,:] == NO_EPON_OUT_VAL  ) 
    #Replace 9999 and 8888 values with layer values just before
    top[lay:nlay][idx] = np.stack([sepon[lay,:,:]]*(nlay-lay))[idx]  

idx = np.logical_or(chasim == NO_EPON_VAL, chasim == NO_EPON_OUT_VAL    )
#Replace 9999 and 8888 values by nan
chasim[idx] = np.nan 

#Create 4d numpy array by joining arrays of the same time step
heads = np.stack ([chasim[i:i+nlay] for i in range(0,nlay*nper,nlay)])  

# maximum head value
hmax   = np.nanmax(heads,0) # Defining hmax
# minimum head value 
hmin   = np.nanmin(heads,0) # Defining hmin 

# setting confined / unconfined areas
# these areas intersect
# the first is NOT the complement of the other
confined   = hmax > top 
unconfined = hmin <= top    

# save confined/unconfined zones to disk 
marthe_utils.write_grid_file(mm.mlname+'.conf',x,y,confined.astype(np.int))
marthe_utils.write_grid_file(mm.mlname+'.unconf',x,y,unconfined.astype(np.int))

# ---------------------------------------------------------------
# -------- STEP 2 : import prior data from excel file -----------
# ---------------------------------------------------------------

# set initial values
xls = pd.ExcelFile('./param.xlsx')
column_names = ['lay', 'parlbnd', 'parubnd','parval1']

# read prior data of each param dfrom param.xlsx as a dataframe
# assign new columns names and add a new column with the param name (assign)
df_kepon = pd.read_excel(xls, 'kepon', names= column_names).assign(parname='kepon')
df_permh = pd.read_excel(xls, 'permh', names= column_names).assign(parname = 'permh')
df_Ss = pd.read_excel(xls, 'Ss', names= column_names).assign(parname = 'emmca')
df_w  = pd.read_excel(xls, 'w', names= column_names).assign(parname = 'emmli')

#Concatenate all the param dataframes 
prior_data_df = pd.concat([df_Ss,df_w,df_permh,df_kepon])

# set parameter name and layer as indexes
prior_data_df.set_index(['parname','lay'],inplace=True)

# --------------------------------------------------------------
# -------- clear folder  -----------
# --------------------------------------------------------------

# path to output folders 
tpl_dir = os.path.join('.','tpl')
par_dir = os.path.join('.','param')
ins_dir = os.path.join('.','ins')
sim_dir = os.path.join('.','sim')

# clear output folders
l = [ os.remove(os.path.join(tpl_dir,f)) for f in os.listdir(tpl_dir) if f.endswith(".tpl") ]
l = [ os.remove(os.path.join(par_dir,f)) for f in os.listdir(par_dir) if f.endswith(".dat") ]
l = [ os.remove(os.path.join(ins_dir,f)) for f in os.listdir(ins_dir) if f.endswith(".ins") ]


# --------------------------------------------------------------
# ------------------- parameter settings  ----------------------
# --------------------------------------------------------------

# log-transformation 
log_transform_dic = {'kepon': True,'permh': True,'emmca' : True,'emmli': False}

# ---------  set up izones dictionary  --------
# reload if necessary
#x, y, confined  = marthe_utils.read_grid_file(mm.mlname+'.conf')
#x, y, unconfined  = marthe_utils.read_grid_file(mm.mlname+'.unconf')

# only in confined zone for emmca and first layer with ZPC 
izone_emmca = np.zeros( (nlay, nrow, ncol) )
izone_emmca[ confined ] = 1
izone_emmca[0,:,:] = -1

# everywhere for permh
izone_permh = np.ones( (nlay, nrow, ncol) ) 

# -1 or zpc :
izone_zpc = -1*np.ones( (nlay, nrow, ncol) )

# izone dic set to none for default values (layer-based ZPC)
izone_dic = {'emmca':izone_emmca , 'permh':izone_permh,'kepon': izone_zpc,'emmli': izone_zpc}

# set up pilot point spacings with pp_ncells
pp_ncells_dic = {}
for par in izone_dic.keys():
    pp_ncells_dic[par] = {}
    for lay in range(0,11) : 
        pp_ncells_dic[par][lay] = 12
    for lay in range(11,nlay) : 
        pp_ncells_dic[par][lay] = 25

# setup template files 
mm.setup_pst_tpl(izone_dic, log_transform = log_transform_dic, pp_ncells=pp_ncells_dic)

# --------------------------------------------------------------
# -------------------- observations set up   ---------------------
# --------------------------------------------------------------

# --- Observations ---
# observation files (generated by obs_pproc.py)
obs_dir = os.path.join(mm.mldir,'obs','')

# NOTE : added sorted() to match with sorted list of sim files
obs_files = [os.path.join(obs_dir, f) for f in sorted(os.listdir( obs_dir )) if f.endswith('.dat')]

# add observations
for obs_file in obs_files :
    mm.add_obs(obs_file = obs_file)

# write instruction files
for obs_loc in mm.obs.keys() :
    mm.obs[obs_loc].write_ins()

# -------------------------------------------------------------
# -------------- STEP 3: setup PEST control file  -------------
# -------------------------------------------------------------

# -------------- list files ------------------------------------

# template files
tpl_files = [os.path.join(tpl_dir, f) for f in sorted(os.listdir( tpl_dir )) if f.endswith('.tpl')]

# input parameter files 
par_files = [os.path.join(par_dir, f) for f in sorted(os.listdir( par_dir )) if f.endswith('.dat')]

# instruction files 
ins_files = [os.path.join(ins_dir, f) for f in sorted(os.listdir( ins_dir )) if f.endswith('.ins')]

# output simulation files 
sim_files = [os.path.join(sim_dir, f + '.dat') for f in sorted(list(mm.obs.keys()))  ]

# --------------------- set up pst file -----------------------------
pst = pyemu.helpers.pst_from_io_files(tpl_files, par_files, ins_files, sim_files)


# set observation values and weights in the pst
for obs_loc in list(mm.obs.keys()):
    pst.observation_data.loc[mm.obs[obs_loc].df.index,'obsval'] = mm.obs[obs_loc].df.value
    pst.observation_data.loc[mm.obs[obs_loc].df.index,'weight'] = mm.obs[obs_loc].df.weight
    pst.observation_data.loc[mm.obs[obs_loc].df.index,'obgnme'] = obs_loc

# initialize values (initial and boundary values)
for stat in ['parlbnd','parval1','parubnd']:
    for par in izone_dic.keys() :
        # ZPCs : iterate over ZPCs in zpc_df
        for parname in mm.param[par].zpc_df.index :
            # set parameter group
            pst.parameter_data.loc[ parname , 'pargp'] = par
            # get value from prior data 
            lay = mm.param[par].zpc_df.loc[parname,'lay']
            val = prior_data_df.loc[(par,lay+1),stat]
            # log-transform values when necessary
            # and write to PEST
            if log_transform_dic[par] == True : 
                pst.parameter_data.loc[ parname , stat] = np.log10(val)
            else : 
                pst.parameter_data.loc[ parname , stat] = val
        # pilot points :iterate over layers with pilot points
        for lay in mm.param[par].pp_dic.keys() :
            # iterate over pp_df for given layer 
            for parname in mm.param[par].pp_dic[lay].index :
                pst.parameter_data.loc[ parname , 'pargp'] = par
                # get value from prior data 
                val = prior_data_df.loc[(par,lay+1),stat]
                # log-transform values when necessary
                # and write to PEST
                if log_transform_dic[par] == True : 
                    pst.parameter_data.loc[ parname , stat] = np.log10(val)
                else : 
                    pst.parameter_data.loc[ parname , stat] = val


# set parameter transformation to none (log-transformation handled by PyMarthe)
pst.parameter_data.loc[:,'partrans'] = 'none'

# set parameter change limit to relative (factor not accepted va
pst.parameter_data.loc[:,'parchglim'] = 'relative'

# further settings (with log-transformed values) 
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmca' , 'parubnd'] =  2
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmli' , 'parubnd'] =  1
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'permh' , 'parubnd'] =  2
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'kepon' , 'parubnd'] =  2

pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'kepon' , 'parlbnd'] =  -20
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'permh' , 'parlbnd'] =  -20
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmca' , 'parlbnd'] =  -20
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmli' , 'parlbnd'] =  -20

# fix insensitive parameters 
pst.parameter_data.loc['kepon_l01_zpc01','partrans'] = 'fixed'
pst.parameter_data.loc['emmca_l01_zpc01','partrans'] = 'fixed'

# update parameter_groups from parameter_data 
pst.rectify_pgroups()

# Zero-order Tikhonov reg
pyemu.helpers.zero_order_tikhonov(pst)
pyemu.utils.helpersfirst_order_pearson_tikhonov(pst) 

# First-order Tikhonov reg for pilot points
for par in izone_dic.keys() :
    for lay in mm.param[par].ppcov_dic.keys() :
        cov_mat = mm.param[par].ppcov_dic[lay]
        pyemu.helpers.first_order_pearson_tikhonov(pst,cov_mat,reset=False,abs_drop_tol=0.2)


# derivative calculation type 
pst.parameter_groups.loc[ pst.parameter_groups.index,'forcen'] = 'always_3'
pst.parameter_groups.loc[ pst.parameter_groups.index,'inctyp'] = 'absolute'
pst.parameter_groups.loc[ pst.parameter_groups.index,'derinc'] = 0.15
pst.parameter_groups.loc[ pst.parameter_groups.index,'derincmul'] = 1.0

# pestpp-glm options 
pst.pestpp_options['svd_pack'] = 'propack'
pst.pestpp_options['lambdas'] = str(list(np.power(10,np.linspace(-2,3,6)))).strip('[]')
pst.pestpp_options['lambda_scale_fac'] = str(list(np.power(10,np.linspace(-2,0,12)))).strip('[]')

pst.control_data.jcosaveitn

# write pst 
pst.write('calreg_test2.pst')

