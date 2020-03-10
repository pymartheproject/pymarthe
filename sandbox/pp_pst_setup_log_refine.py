import sys 
import os 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import geopandas 

# https://github.com/jtwhite79/pyemu
sys.path.append(os.path.expanduser('~/Programmes/python/pyemu/'))
import pyemu

# https://github.com/apryet/adeqwat
sys.path.append(os.path.expanduser('~/Programmes/python/adeqwat/'))
from pymarthe import * 

# load Marthe model
mm = MartheModel('./mona.rma')
nlay, nrow, ncol = mm.nlay, mm.nrow, mm.ncol

# ---------------------------------------------------------------
# -------- STEP 2 : import prior data from excel file -----------
# ---------------------------------------------------------------

# set initial values
# layer-based parameter assignation
xls = pd.ExcelFile('./param.xlsx')
df_kepon = pd.read_excel(xls, 'kepon')
df_permh = pd.read_excel(xls, 'permh')
df_Ss = pd.read_excel(xls, 'Ss')
df_w  = pd.read_excel(xls, 'w')

prior_data_dic = {}
dic_kepon = {}
dic_permh = {}
dic_emmca = {}
dic_emmli = {}

dic_kepon['parval1'] = df_kepon.Dmoy.to_dict()
dic_permh['parval1'] = df_permh.moyk.to_dict()
dic_emmca['parval1'] = df_Ss.MoySs.to_dict()
dic_emmli['parval1'] = df_w.Wmoy.to_dict()

dic_kepon['parlbnd'] = df_kepon.Dmin.to_dict()
dic_permh['parlbnd'] = df_permh.kmin.to_dict()
dic_emmca['parlbnd'] = df_Ss.Ssmin.to_dict()
dic_emmli['parlbnd'] = df_w.Wmin.to_dict()

dic_kepon['parubnd'] = df_kepon.Dmax.to_dict()
dic_permh['parubnd'] = df_permh.kmax.to_dict()
dic_emmca['parubnd'] = df_Ss.Ssmax.to_dict()
dic_emmli['parubnd'] = df_w.Wmax.to_dict()

prior_data_dic['kepon'] = dic_kepon
prior_data_dic['emmli'] = dic_emmli
prior_data_dic['emmca'] = dic_emmca
prior_data_dic['permh'] = dic_permh


# --------------------------------------------------------------
# -------- clear folder  -----------
# --------------------------------------------------------------

# path to output folders 
tpl_dir = os.path.join('.','tpl')
par_dir = os.path.join('.','param')
ins_dir = os.path.join('.','ins')
sim_dir = os.path.join('.','sim')

# clear output folders (but NOT parameter data)
l = [ os.remove(os.path.join(tpl_dir,f)) for f in os.listdir(tpl_dir) if f.endswith(".tpl") ]
l = [ os.remove(os.path.join(ins_dir,f)) for f in os.listdir(ins_dir) if f.endswith(".ins") ]

# ---------------------------------------------------------------
# --------import criteria data from former pest run   -----------
# ---------------------------------------------------------------

pst = 'caleval_v1_it13.pst'
jco = 'caleval_v1_it13.jcb'

# load sensititivies (example)
mla = pyemu.la.LinearAnalysis( pst = pst, jco = jco)
crit_df = mla.get_par_css_dataframe()

# set boolean refine column (pilot point will be refined if true)
# select only permh parameters 
pp_permh_idx = [parname.startswith('permh') for parname in crit_df.index]
pp_emmca_idx = [parname.startswith('emmca') for parname in crit_df.index]
# select only points with sensitivity value greater than 3d quartile
crit_val = crit_df.loc[pp_permh_idx,'pest_css'].quantile(0.75)
crit_df['refine'] = pp_permh_idx & (crit_df['pest_css'] > crit_val)
print(str(crit_df['refine'].sum()) + ' points to refine...')

# --------------------------------------------------------------
# ------------------- parameter settings  ----------------------
# --------------------------------------------------------------

# parameter names
params = ['emmca','permh','kepon','emmli']
# log-transformation 
log_transform_dic = {'kepon': True,'permh': True,'emmca' : True,'emmli': False}

# --------------------------------------------------------------
# -------------------- parameter set up   ---------------------
# --------------------------------------------------------------

# set up parameterization 
for par in params:
    # read izone data from disk
    x, y, izone  = marthe_utils.read_grid_file('{0}.i{1}'.format(mm.mlname,par))
    # add parameter
    mm.add_param(par, izone = izone, default_value = 1e-5, log_transform=log_transform_dic[par])
    # parameters with pilot points (izone with positive values)
    if np.max(np.unique(izone)) > 0  :
        # read former pp dataframe for all layers 
        mm.param[par].read_pp_df()
        # iterate over layers 
        for lay in range(nlay):
            # layers with pilot points (izone with positive values)
            if np.max(np.unique(izone[lay,:,:])) > 0 :
                # pointer to pp_df for current layer 
                pp_df = mm.param[par].pp_dic[lay]
                print('former pp_df')
                print(pp_df)
                # join refine attributes
                pp_df_refine = pp_df.join(crit_df['refine'])
                print(str(pp_df_refine['refine'].sum()) + ' points will be refined.')
                # refine pilot points (updates pp_df)
                mm.param[par].pp_refine(lay, df = pp_df_refine, base_spacing = 32.)
                # build up GeoStruct
                v = pyemu.utils.ExpVario(contribution=2,a=32)
                gs = pyemu.utils.GeoStruct(variograms=v,transform="log")
                # set up kriging (pp_df points to refined set of pilot points
                # pointer to pp_df for current layer 
                pp_df = mm.param[par].pp_dic[lay]
                print('new pp_df')
                print(pp_df)
                ok = pyemu.utils.OrdinaryKrige(geostruct=gs,point_data=pp_df)
                # Spatial reference (for compatibility only)
                ok.spatial_reference = SpatialReference(mm) 
                # pandas dataframe of point where interpolation shall be conducted
                x_coords, y_coords = mm.param[par].zone_interp_coords(lay,zone=1)
                # compute kriging factors
                kfac_df = ok.calc_factors(x_coords, y_coords, minpts_interp=1, 
                        maxpts_interp=50,search_radius=800,pt_zone=1 , num_threads=6
                        )
                # write kriging factors to file
                kfac_file = os.path.join(mm.mldir,'kfac_{0}_l{1:02d}.dat'.format(par,lay+1))
                ok.to_grid_factors_file(kfac_file)
        # write new pp_df files (all layers) 
        mm.param[par].write_pp_df()
        mm.param[par].write_pp_tpl()
    # set up ZPCs if present (unchanged)
    if izone.min() < 0 : 
        mm.param[par].read_zpc_df()
        mm.param[par].write_zpc_tpl()

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
# ----------- STEP 3: PyEMU pre-processing for PEST ------------
# -------------------------------------------------------------

# NOTE : added sorted() to file lists to match with each others 

# template files
tpl_files = [os.path.join(tpl_dir, f) for f in sorted(os.listdir( tpl_dir )) if f.endswith('.tpl')]

# input parameter files 
par_files = [os.path.join(par_dir, f) for f in sorted(os.listdir( par_dir )) if f.endswith('.dat')]

# instruction files 
ins_files = [os.path.join(ins_dir, f) for f in sorted(os.listdir( ins_dir )) if f.endswith('.ins')]

# output simulation files 
sim_files = [os.path.join(sim_dir, f + '.dat') for f in sorted(list(mm.obs.keys()))  ]

# set up pst file
pst = pyemu.helpers.pst_from_io_files(tpl_files, par_files, ins_files, sim_files)

# set observation values and weights in the pst
for obs_loc in list(mm.obs.keys()):
    pst.observation_data.loc[mm.obs[obs_loc].df.index,'obsval'] = mm.obs[obs_loc].df.value
    pst.observation_data.loc[mm.obs[obs_loc].df.index,'weight'] = mm.obs[obs_loc].df.weight
    pst.observation_data.loc[mm.obs[obs_loc].df.index,'obgnme'] = obs_loc


# set lower and upper bounds (but not values, taken from the former parameter value files)
for stat in ['parlbnd','parval1','parubnd']:
    for par in params :
        # ZPCs : iterate over ZPCs in zpc_df
        for parname in mm.param[par].zpc_df.index :
            # set parameter group
            pst.parameter_data.loc[ parname , 'pargp'] = par
            lay = mm.param[par].zpc_df.loc[parname,'lay']
            if stat=='parval1':
                # get value from former estimation
                val = mm.param[par].zpc_df.loc[parname,'value']
            else : 
                # get value from prior data 
                val = prior_data_dic[par][stat][lay] 
            # log-transform values when necessary
            # and write to PEST
            if log_transform_dic[par] == True : 
                pst.parameter_data.loc[ parname , stat] = np.log10(val)
            else : 
                pst.parameter_data.loc[ parname , stat] = val
        # Pilot points :iterate over layers with pilot points
        for lay in mm.param[par].pp_dic.keys() :
            # iterate over pp_df for given layer 
            for parname in mm.param[par].pp_dic[lay].index :
                pst.parameter_data.loc[ parname , 'pargp'] = par
                if stat == 'parval1':
                    # get value from former estimation
                    val = mm.param[par].pp_dic[lay].loc[parname,'value']
                else : 
                    # get value from prior data
                    val = prior_data_dic[par][stat][lay]
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
pyemu.utils.helpers.zero_order_tikhonov(pst)

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
pst.write('cal_v1_refine.pst')

