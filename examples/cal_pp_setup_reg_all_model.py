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
# ---------------------------------------------------------------
# -------- STEP 1  General  settings   -----------
# ---------------------------------------------------------------

# -------- PEST  settings   -----------

# name of parameter estimation case
# name of parameter estimation case
case_name = 'cal_pp_permh_20_with_reg'
# prefered value regularization (zero order Tikhonov)
reg_pref_value = True
# prefered value regularization for pilot points (1st order Tikhonov)
reg_pref_homog = True

# ---------- parameter settings --------

# log-transformation
log_transform_dic = {'permh': True,'emmca': True,'kepon': True,'emmli': False}

# ---------  set up izones dictionary  --------
# reload if necessary
x, y, confined  = marthe_utils.read_grid_file(mm.mlname+'.conf')
x, y, unconfined  = marthe_utils.read_grid_file(mm.mlname+'.unconf')

# only in confined zone for emmca and first layer with ZPC 
izone_emmca = np.zeros( (nlay, nrow, ncol))
izone_emmca[ unconfined ==1 ] = -1
izone_emmca[ confined == 1 ] = 1
izone_emmca[0,:,:] = -1

izone_emmli = np.zeros( (nlay, nrow, ncol))
izone_emmli[ confined ==1 ] = -1 #zpc where confined condition 
izone_emmli[ unconfined ==1 ] =1 #pp in unconf condition / the overlapping domain where the aquifer may change of condition is handlled with pp

# everywhere for permh
izone_permh = np.ones( (nlay, nrow, ncol) ) 

# everywhere for kepon
izone_kepon = np.ones( (nlay, nrow, ncol) ) 
izone_kepon[0,:,:] = -1
# -1 or zpc :
izone_zpc = -1*np.ones( (nlay, nrow, ncol) )

# izone dic set to none for default values (layer-based ZPC)
izone_dic = {'emmca':izone_emmca , 'permh':izone_permh,'kepon': izone_zpc,'emmli': izone_zpc}

# set fixed parameters 
fixed_parameters = ['kepon_l01_zpc01','emmca_l01_zpc01']

# set up pilot point spacings with pp_ncells
pp_ncells_dic = {'permh':20,'emmca':20, 'emmli':30, 'kepon':30}

# ---------------------------------------------------------------
# ---- STEP 2 : generate PEST template and instruction files  -----------
# ---------------------------------------------------------------

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

# pymarthe helper function
mm.setup_tpl(izone_dic = izone_dic, log_transform = log_transform_dic, pp_ncells = pp_ncells_dic, 
             save_settings = '{}.settings'.format(case_name))

# --------------------------------------------------------------
# -------------------- observations set up   -------------------
# --------------------------------------------------------------

# pymarthe helper function
mm.setup_ins()



# ---------------------------------------------------------------
# -------- STEP 3 : import prior data from excel file -----------
# ---------------------------------------------------------------

# set initial values
xls = pd.ExcelFile('./param.xlsx')
column_names = ['lay', 'parlbnd', 'parubnd','parval1','Commentaire','Commentaire','Couche']

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

# -------------------------------------------------------------
# -------------- STEP 4: setup PEST control file  -------------
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
# further settings (with log-transformed values) 
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'permh' , 'parubnd'] =  1
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmca' , 'parubnd'] =  1
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmli' , 'parubnd'] =  1
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'kepon' , 'parubnd'] =  1

pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'permh' , 'parlbnd'] =  -20
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmca' , 'parlbnd'] =  -20
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'kepon' , 'parlbnd'] =  -20
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmli' , 'parlbnd'] =  10 ** -5

# fix insensitive parameters 
pst.parameter_data.loc['kepon_l01_zpc01','partrans'] = 'fixed'
pst.parameter_data.loc['emmca_l01_zpc01','partrans'] = 'fixed'


# update parameter_groups from parameter_data 
pst.rectify_pgroups()

# Zero-order Tikhonov reg
if reg_pref_value == True:
    pyemu.helpers.zero_order_tikhonov(pst)

# First-order Tikhonov reg for pilot points
if reg_pref_homog == True:
    for par in izone_dic.keys() :
        for lay in mm.param[par].ppcov_dic.keys() :
            cov_mat = mm.param[par].ppcov_dic[lay]
            pyemu.helpers.first_order_pearson_tikhonov(pst,cov_mat,reset=False,abs_drop_tol=0.2)

# regularization settings
pst.reg_data.phimlim = 1.0
pst.reg_data.phimaccept = 1.05
pst.reg_data.fracphim = 0.05
pst.reg_data.wfmin = 1.0e-5
pst.reg_data.wfinit = 1.0
pst.reg_data.iregadj = 1

# derivative calculation type 
pst.parameter_groups.loc[ pst.parameter_groups.index,'forcen'] = 'always_3'
pst.parameter_groups.loc[ pst.parameter_groups.index,'inctyp'] = 'absolute'
pst.parameter_groups.loc[ pst.parameter_groups.index,'derinc'] = 0.15
pst.parameter_groups.loc[ pst.parameter_groups.index,'derincmul'] = 1.0

# pestpp-glm options 
pst.pestpp_options['svd_pack'] = 'redsvd'
pst.pestpp_options['uncertainty'] = 'false'





# 8 lambdas, 12 scalings =>  96 upgrade vectors tested
pst.pestpp_options['lambdas'] = str([0]+list(np.power(10,np.linspace(-2,3,7)))).strip('[]')
pst.pestpp_options['lambda_scale_fac'] = str(list(np.power(10,np.linspace(-2,0,12)))).strip('[]')


# set overdue rescheduling factor to twice the average model run
pst.pestpp_options['overdue_resched_fac'] = 2
pst.control_data.jcosave = 'jcosave'
pst.control_data.jcosaveitn = 'jcosaveitn'
pst.control_data.relparmax = 1
#pst.control_data.relparmax = 2


#pst.observation_data['obgnme'] = pst.observation_data['obgnme'].str.lower()
# write pst 
pst.write('{}.pst'.format(case_name), version=1)

