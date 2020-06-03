import sys 
import os 
import pandas as pd
import numpy as np

import pyemu
from pymarthe import * 

# ---------------------------------------------------------------
# -------- load Marthe model
# ---------------------------------------------------------------
mm = MartheModel('./mona.rma')
nlay, nrow, ncol = mm.nlay, mm.nrow, mm.ncol

# ---------------------------------------------------------------
# -------- General  settings   -----------
# ---------------------------------------------------------------

# -------- PEST  settings   -----------

# output pest control file

control_file = 'cal_pp_eoc_reg6.pst'
# prefered value regularization (zero order Tikhonov)
reg_pref_value = False
# prefered value regularization for pilot points (1st order Tikhonov)
reg_pref_homog = False

# --------- layer settings -------------

# layers from which observations will be considered
obs_layers = [5]

# layers for  with  parameters will be adjusted
par_layers = [5]

# ---------- parameter settings --------

# log-transformation 
log_transform_dic = {'kepon': True,'permh': True,'emmca' : True,'emmli': False}

fixed_parameters = []
#fixed_parameters = ['kepon_l01_zpc01','emmca_l01_zpc01']

#  izone  settings  

# delination of permanently  confined/unconfined areas
# emmca not adjusted where a layer is permanently unconfined
# emmli not adjusted where a layer is permanently confined
x, y, confined  = marthe_utils.read_grid_file(mm.mlname+'.conf')
x, y, unconfined  = marthe_utils.read_grid_file(mm.mlname+'.unconf')

# only in confined zone for emmca and first layer with ZPC 
izone_emmca = np.zeros( (nlay, nrow, ncol) )
izone_emmca[ confined.astype(bool) ] = 1
izone_emmca[0,:,:] = -1

# everywhere for permh
izone_permh = np.ones( (nlay, nrow, ncol) ) 

# ZPC for emmli and kepon :
izone_zpc = -1*np.ones( (nlay, nrow, ncol) )

# izone dic set to none for default values (layer-based ZPC)
izone_dic = {'emmca':izone_emmca , 'permh':izone_permh,'kepon': izone_zpc,'emmli': izone_zpc}

# -------- pilot point  settings   -----------

# set up pilot point spacings with pp_ncells
pp_ncells_dic = {}
every_ncell = 6
for par in izone_dic.keys():
    pp_ncells_dic[par] = {lay:every_ncell for lay in range(nlay)}

# vario_range [L]  = model_mesh*ncell*vario_range_mult   
vario_range_mult = 3 

# refinement criteria (optional)
# base regular grid refined where refine_crit > refine_threshold
#refine_crit = 'nobs' # set not None to disable re!finement
refine_crit = None # set not None to disable refinement
refine_threshold = 2

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

#-- disable parameter estimation for par_layers (izone set to 0)
excluded_layers = [i for i in range(nlay) if i not in par_layers]
for par in izone_dic.keys():
    for lay in excluded_layers:
        izone_dic[par][lay,:,:] = 0


# get observation locations for pilot point adaptive refinement 
df_histo = marthe_utils.read_histo_file('mona.histo')

# setup template files 
# iterate over parameters with izone data  
for par in izone_dic.keys():
    print('Processing parameter {0}...'.format(par))
    # add parameter 
    mm.add_param(par, izone = izone_dic[par], default_value = 1e-5, log_transform=log_transform_dic[par])
    # write izone to disk for future use by model_run
    marthe_utils.write_grid_file('{0}.i{1}'.format(mm.mlname,par),mm.x_vals,mm.y_vals,izone_dic[par])
    # parameters with pilot points (izone with positive values)
    if np.max(np.unique(izone_dic[par])) > 0  :
        print('Setting up pilot points for parameter {0}'.format(par))
        # iterate over layers 
        for lay in range(mm.nlay):
            # layers with pilot points (izone with positive values)
            if np.max(np.unique(izone_dic[par][lay,:,:])) > 0 :
                # layer-dependent variogram settings
                mm.param[par].pp_from_rgrid(lay, n_cell=pp_ncells_dic[par][lay], n_cell_buffer = True)
                # pointer to pilot point dataframe generated by pp_from_rgrid
                pp_df = mm.param[par].pp_dic[lay]
                print('{0} pilot points seeded for parameter {1}, layer {2}'.format(pp_df.shape[0],par,lay+1))
                # refine pilot point grid where criteria is met
                if refine_crit is not None:
                    loc_df = df_histo.loc[df_histo['layer'] == lay+1]
                    df = mm.param[par].get_pp_nobs(lay, loc_df)
                    df['refine'] = df[refine_crit] > refine_threshold 
                    mm.param[par].pp_refine(lay,df)
                # Update pp_df after refinement 
                pp_df = mm.param[par].pp_dic[lay]
                # desired variogram range (3 times pilot base point spacing)
                vario_range = vario_range_mult*mm.cell_size*pp_ncells_dic[par][lay]
                # variogram setup :
                #   - parameter a is a proxy for range
                #   - the contribution has no effect
                v = pyemu.utils.ExpVario(contribution=1, a=vario_range)
                # build up GeoStruct
                gs = pyemu.utils.GeoStruct(variograms=v,transform="log")
                # get covariance matrix 
                mm.param[par].ppcov_dic[lay] = gs.covariance_matrix(pp_df.x,pp_df.y,pp_df.name)
                # set up kriging
                ok = pyemu.utils.OrdinaryKrige(geostruct=gs,point_data=pp_df)
                # spatial reference (for pyemu compatibility only)
                ok.spatial_reference = SpatialReference(mm) 
                # pandas dataframe of point where interpolation shall be conducted
                # set up index for current zone and lay
                x_coords, y_coords = mm.param[par].zone_interp_coords(lay,zone=1)
                # compute kriging factors
                kfac_df = ok.calc_factors(x_coords, y_coords,pt_zone=1, num_threads=6)
                # write kriging factors to file
                kfac_file = os.path.join(mm.mldir,'kfac_{0}_l{1:02d}.dat'.format(par,lay+1))
                ok.to_grid_factors_file(kfac_file)
        # write initial parameter value file (all zones)
        mm.param[par].write_pp_df()
        mm.param[par].write_pp_tpl()
    # set up ZPCs if present
    # will have not effect for parameters and layers with pilot points
    mm.param[par].write_zpc_data()
    mm.param[par].write_zpc_tpl()

# --------------------------------------------------------------
# -------------------- observations set up   ---------------------
# --------------------------------------------------------------

# observation files (generated by obs_pproc.py)
obs_dir = os.path.join(mm.mldir,'obs','')

#  added sorted() to match with sorted list of sim files
all_obs_files = [os.path.join(obs_dir, f) for f in sorted(os.listdir( obs_dir )) if f.endswith('.dat')]

#  observation locations (BSS ids) identified in model output file (mona.histo)
sim_obs_loc = []
#  observation files selected for history matching (with simulated counterparts)
obs_files = []

# iterate over selected layers and get simulated observation locations
for lay in obs_layers:
    sim_obs_loc.extend(df_histo.loc[(df_histo.layer - 1)==lay].index)

# iterate over observation files
for obs_file in all_obs_files:
    # infer observation loc (BSS id) from filename
    obs_filename = obs_file.split('/')[-1]
    obs_loc = obs_filename[0:10] # BSS id is ten char long
    # select that observation file if obs_loc found in simulated outputs
    if obs_loc in sim_obs_loc:
        obs_files.append(obs_file)

# add selected observations
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
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmli' , 'parlbnd'] =  10 ** -5

# set fixed parameters
for parname in fixed_parameters :
    if parname in pst.parameter_data.index.values :
        pst.parameter_data.loc[parname,'partrans'] = 'fixed'

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

# svd-assist options
#pst.pestpp_options['n_iter_base'] = 1
#pst.pestpp_options['n_iter_super'] = 3
#pst.pestpp_options['super_eigthresh'] = 1e-6

# 8 lambdas, 12 scalings =>  96 upgrade vectors tested
pst.pestpp_options['lambdas'] = str([0]+list(np.power(10,np.linspace(-2,3,7)))).strip('[]')
pst.pestpp_options['lambda_scale_fac'] = str(list(np.power(10,np.linspace(-2,0,12)))).strip('[]')

# set overdue rescheduling factor to twice the average model run
pst.pestpp_options['overdue_resched_fac'] = 2
pst.pestpp_options['panther_agent_no_ping_timeout_secs'] = 36000


#pst.observation_data['obgnme'] = pst.observation_data['obgnme'].str.lower()
# write pst 
pst.write(control_file, version=2)

