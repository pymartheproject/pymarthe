import sys 
import os 
import pandas as pd
import numpy as np

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
case_name = 'cal_pp_reg_coarse_article'
# prefered value regularization (zero order Tikhonov)
reg_pref_value = True
# prefered value regularization for pilot points (1st order Tikhonov)
reg_pref_homog = True
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

# ---------- parameter settings --------

# log-transformation
log_transform_dic = {'permh':True,'emmca': True,'kepon': True,'emmli': False}

# ---------  set up izones dictionary  --------
# reload if necessary
x, y, confined  =   marthe_utils.read_grid_file(mm.mlname+'.conf')
x, y, unconfined  = marthe_utils.read_grid_file(mm.mlname+'.unconf')

# only in confined zone for emmca and first layer with ZPC 
izone_emmca = np.zeros( (nlay, nrow, ncol))
izone_emmca[ unconfined ==1 ] = -1
izone_emmca[ confined == 1 ] = 1
izone_emmca[0,:,:] = -1

izone_emmli = np.zeros( (nlay, nrow, ncol))
izone_emmli[ confined ==1 ] = -1 #zpc where confined condition 
#izone_emmli[ unconfined ==1 ] =1 #pp in unconf condition / the overlapping domain where the aquifer may change of condition is handlled with pp
izone_emmli[ unconfined ==1 ] =-1 #zpc for all layers
izone_emmli[0,:,:] = 1 # pp for the first layer

# everywhere for permh
izone_permh = np.ones( (nlay, nrow, ncol) ) 

# everywhere for kepon
izone_kepon = np.ones( (nlay, nrow, ncol) ) 
izone_kepon[0,:,:] = -1
# -1 or zpc :
izone_zpc = -1*np.ones( (nlay, nrow, ncol) )

# izone dic set to none for default values (layer-based ZPC)
izone_dic = {'emmca':izone_emmca , 'permh':izone_permh,'kepon': izone_kepon,'emmli': izone_emmli}

# set fixed parameters 
fixed_parameters = ['kepon_l01_zpc01','emmca_l01_zpc01']

# set up pilot point spacings with pp_ncells
#pp_ncells_dic = {'permh':20,'emmca':20, 'emmli':20, 'kepon':20}


pp_ncells_dic = {}
for par in izone_dic.keys():
    pp_ncells_dic[par] = {}
    if (par == 'kepon') or (par =='emmca'):
        for lay in range(nlay):
            pp_ncells_dic[par][lay] = 40
    elif par == 'emmli':
        for lay in range(nlay):
            pp_ncells_dic[par][lay] = 20
    else : 
        for lay in range(7) : 
            pp_ncells_dic[par][lay] = 20
        for lay in range(7,nlay) : 
            pp_ncells_dic[par][lay] = 20

'''
###Define refine value
#Get crit dataframe of all params 
df_crit_file = os.path.join('crit','df_crit.dat')
df_crit = pd.read_csv(df_crit_file, delim_whitespace=True, index_col='param')
'''
# initialize refinement dic
refine_crit_dic = {par: None for par in izone_dic.keys()}
refine_crit_type_dic = {par: None for par in izone_dic.keys()}
refine_value_dic = {par: None for par in izone_dic.keys()}
params_to_refine = ['permh']

# for each parameter, refinement of the third of 
# pp with higher css values, over all layers 
for par in params_to_refine :
    refine_crit_dic[par] = 'css'
    refine_crit_type_dic[par] = 'quantile'
    #df_crit_par = df_crit.loc[df_crit.index.str.startswith(par),:]
    #df_crit_par =df_crit.loc['permh_l01_z01_000':'permh_l07_z01_018']
    #df_par = df_crit_par[refine_crit_dic[par]]
    #df_sort = df_par.sort_values(ascending = True)
    # points with css within the upper third will be refined 
    #refine_value_dic[par] = df_sort.iloc[int(2*df_sort.shape[0]/3)]
    refine_value_dic[par] = 0.4
# ---------------------------------------------------------------
# ---- STEP 2 : generate PEST template and instruction files  -----------
# ---------------------------------------------------------------
mm.setup_tpl(
        izone_dic = izone_dic, 
        log_transform = log_transform_dic,
        pp_ncells = pp_ncells_dic, 
        #refine_crit = refine_crit_dic, 
        #refine_crit_type = refine_crit_type_dic, 
        #refine_value = refine_value_dic,
        #refine_level = 1,
        #refine_layers = [0,1,2,3,4,5,6],
        save_settings = '{}.settings'.format(case_name)
        #reload_settings= 'cal_pp_base_article_jacobienne.settings'
        )
'''
mm.setup_tpl(
        izone_dic = izone_dic, 
        log_transform = log_transform_dic,
        pp_ncells = pp_ncells_dic, 
        refine_crit = refine_crit_dic, 
        refine_crit_type = refine_crit_type_dic, 
        refine_value = refine_value_dic,
        refine_level = 1,
        refine_layers = [0,1,2,3,4,5,6,7],
        #save_settings = '{}.settings'.format(case_name),
        reload_settings= '{}.settings'.format(case_name)
        )
'''


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

# Concatenate all the param dataframes 
prior_data_df = pd.concat([df_Ss,df_w,df_permh,df_kepon])

# set parameter name and layer as indexes
prior_data_df.set_index(['parname','lay'],inplace=True)

# -------------------------------------------------------------
# -------------- STEP 4: setup PEST control file  -------------
# -------------------------------------------------------------

# -------------- list files ------------------------------------

# path to output folders 
tpl_dir = os.path.join('.','tpl')
par_dir = os.path.join('.','param')
ins_dir = os.path.join('.','ins')
sim_dir = os.path.join('.','sim')
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
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmli' , 'parlbnd'] =  10 ** -5
'''
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmca' , 'parubnd'] =  -2
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmli' , 'parubnd'] =  1
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'permh' , 'parubnd'] =  -1
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'kepon' , 'parubnd'] =  -3

pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'kepon' , 'parlbnd'] =  -10
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'permh' , 'parlbnd'] =  -6
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmca' , 'parlbnd'] =  -10
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmli' , 'parlbnd'] =  10 ** -5
'''
# -- get full prior parameter covariance matrix 
# set up dic of geostructs and pilot point template files 
sd = {}
for par in izone_dic.keys() :
    for lay,gs in mm.param[par].gs_dic.items() :
        sd[gs] = os.path.join('tpl','{0}_pp_l{1:02d}.tpl'.format(par,lay+1))
# get prior cov matrix from pst and geostructs
# non pilot points will be considered as not correlated
cov = pyemu.helpers.geostatistical_prior_builder(pst,struct_dict=sd, sigma_range=4, verbose=True)

# suggestions of outputs 
plt.imshow(cov.x) # see what's in
cov.to_binary("prior.jcb")
cov.to_ascii("prior_parcov.mat")

# set fixed parameters
for parname in fixed_parameters :
    if parname in pst.parameter_data.index.values :
        pst.parameter_data.loc[parname,'partrans'] = 'fixed'

# update parameter_groups from parameter_data 
pst.rectify_pgroups()

# Zero-order Tikhonov reg
if reg_pref_value == True:
    pyemu.helpers.zero_order_tikhonov(pst)

# First-order Tikhonov reg for pilot points!
if reg_pref_homog == True:
    for par in izone_dic.keys() :
        for lay in mm.param[par].ppcov_dic.keys() :
            cov_mat = mm.param[par].ppcov_dic[lay]
            pyemu.helpers.first_order_pearson_tikhonov(pst,cov_mat,reset=False,abs_drop_tol=0.2)

# regularization settings
pst.reg_data.phimlim = 423
pst.reg_data.phimaccept = 430
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
#pst.control_data.relparmax = 1

# 12 lambdas, 8 scalings =>  96 upgrade vectors tested
pst.pestpp_options['lambdas'] = str([0]+list(np.power(10,np.linspace(-3,3,11)))).strip('[]')
pst.pestpp_options['lambda_scale_fac'] = str(list(np.power(10,np.linspace(-2,0,8)))).strip('[]')

# set overdue rescheduling factor to twice the average model run
pst.pestpp_options['overdue_resched_fac'] = 2
pst.pestpp_options['panther_agent_no_ping_timeout_secs'] = 36000

# write pst 
pst.write('{}.pst'.format(case_name), version=1)


