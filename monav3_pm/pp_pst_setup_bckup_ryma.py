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
# Define geometry 
# ---------------------------------------------------------------


x, y, sepon  = marthe_utils.read_grid_file('./mona.sepon')
x, y, topog  = marthe_utils.read_grid_file('./mona.topog')
x, y, hsubs  = marthe_utils.read_grid_file('./mona.hsubs')
x, y, chasim = marthe_utils.read_grid_file('./chasim_cal_histo.out')


top = sepon 
nlay, nrow, ncol = top.shape
nper = 40 #Time step 

NO_EPON_VAL = 9999 #In layer domain but not in eponte
NO_EPON_OUT_VAL = 8888 # Not in layer domain

#Definig Geometry
for lay in range(nlay-2,1,-1):
    idx = np.logical_or(top[lay:nlay,:,:] == NO_EPON_VAL, top[lay:nlay,:,:] == NO_EPON_OUT_VAL  ) #Define inices where values are equal to 9999 or 8888
    top[lay:nlay][idx] = np.stack([sepon[lay,:,:]]*(nlay-lay))[idx] #Replace 9999 and 8888 values with layer values just before   
idx = np.logical_or(chasim == NO_EPON_VAL, chasim == NO_EPON_OUT_VAL    )
chasim[idx] = np.nan #Replace 9999 and 8888 values by nan 
heads = np.stack ([chasim[i:i+nlay] for i in range(0,nlay*nper,nlay)]) #Create 4d numpy array by joining arrays of the same time step 
hmax   = np.nanmax(heads,0) # Defining hmax
hmin   = np.nanmin(heads,0) # Defining hmin 

confined   = hmax > top  #Defining confind parts of each layer
unconfined = hmin <= top #Defining unconfind parts of each layer   

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


#df_emmli = pd.read_excel(xls, 'emmli')
#df_emmli = df_emmli.drop(4).reset_index(drop=True)

#epaisseur = pd.read_excel('./param_all_aq.xlsx')
#epaisseur = epaisseur.replace(9999,nan)
#epaisseur.loc[epaisseur['code'] =='BACX']['epaisseur']


prior_data_dic = {}
dic_kepon = {}
dic_permh = {}
dic_emmca = {}
dic_emmli = {}

#{'kepon':{'min':1e-12;'max':1e-2;'parval1':1e-6}}
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
# -------- STEP 2 : PyMarthe pre-processing for PEST -----------
# --------------------------------------------------------------

# clear output folders

tpl_dir = os.path.join('.','tpl')
par_dir = os.path.join('.','param')
ins_dir = os.path.join('.','ins')
sim_dir = os.path.join('.','sim')

l = [ os.remove(os.path.join(tpl_dir,f)) for f in os.listdir(tpl_dir) if f.endswith(".tpl") ]
l = [ os.remove(os.path.join(par_dir,f)) for f in os.listdir(par_dir) if f.endswith(".dat") ]
l = [ os.remove(os.path.join(ins_dir,f)) for f in os.listdir(ins_dir) if f.endswith(".ins") ]


'''
# --- Parameters handled with ZPC ---
params = {'kepon'}
# process parameters with PyMarthe 
for param in params :
    # add new parameter
    mm.add_param(param)
    # set initial values from prior information
    mm.param[param].set_zpc_values(prior_data_dic[param]['parval1'])
    # write template file
    mm.param[param].write_zpc_tpl()
    # write initial parameter value files
    mm.param[param].write_zpc_data()
'''
# ---- param handled with pilot points
params = ['emmca','permh','kepon','emmli']
for par in params : 
    # Setup izones
    if par == 'permh' :
        zone = 1
        izone = np.ones( (nlay, nrow, ncol) )
    # Emmca with pp where confined and default value elsewhere
    elif par == 'emmca':
        zone =1
        izone = np.zeros( (nlay, nrow, ncol) )
        izone[  confined ==1] = 1
        izone[0,:,:] = 0
    # Emmli and kepon with zpc
    else:
        izone = -1*np.ones( (nlay, nrow, ncol) )
    #Add param
    mm.add_param(par, default_value = 1e-5)
    # Set izones
    mm.param[par].set_izone(izone=izone)
    # Setup zpc
    if par == 'kepon' or par == "emmli":
        mm.param[par].set_zpc_values(prior_data_dic[par]['parval1'])
        mm.param[par].write_zpc_data()
        mm.param[par].write_zpc_tpl()
    # Set pp
    else: 
        for lay in range(nlay):
            if lay == 0 and par == 'emmca':
                continue
            if lay <= 11 :
                mm.param[par].pp_from_rgrid(lay, n_cell=16)
                v = pyemu.utils.ExpVario(contribution=2,a=32)
            else:
                mm.param[par].pp_from_rgrid(lay, n_cell=25)
                # set upvariogram 
                v = pyemu.utils.ExpVario(contribution=2,a=50) # sill and range (km)
            gs = pyemu.utils.GeoStruct(variograms=v,transform="log")
            # pointer to pilot point dataframe
            pp_df = mm.param[par].pp_dic[lay]
            # set initial values 
            pp_df.value  = prior_data_dic[par]['parval1'][lay]
            # set up kriging
            ok = pyemu.utils.OrdinaryKrige(geostruct=gs,point_data=pp_df)
            sr = SpatialReference(mm) # only for compatibility
            ok.spatial_reference = sr 
            # pandas dataframe of point where interpolation shall be conducted
            # set up index for current zone and lay
            x_coords, y_coords = mm.param[par].zone_interp_coords(lay,zone)
            # compute kriging factors
            kfac_df = ok.calc_factors(x_coords, y_coords, minpts_interp=1, maxpts_interp=50,search_radius=800,pt_zone=1 , num_threads=4)
            # write kriging factors to file
            kfac_file = os.path.join(mm.mldir,'kfac_{0}_l{1:02d}.dat'.format(par,lay+1))
            ok.to_grid_factors_file(kfac_file)
        # write initial parameter value file (all zones)
        mm.param[par].write_pp_df()
        mm.param[par].write_pp_tpl()
    

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

# --------------------------------------------------------------
# ----------- STEP 3: PyEMU pre-processing for PEST ------------
# -------------------------------------------------------------

# template files
tpl_files = [os.path.join(tpl_dir, f) for f in sorted(os.listdir( tpl_dir )) if f.endswith('.tpl')]

# input parameter files 
par_files = [os.path.join(par_dir, f) for f in sorted(os.listdir( par_dir )) if f.endswith('.dat')]
#par_files = [os.path.join(par_dir, f) for f in sorted(os.listdir( par_dir )) if (f.endswith('.dat') and not f.startswith('pp'))]

# instruction files 
ins_files = [os.path.join(ins_dir, f) for f in sorted(os.listdir( ins_dir )) if f.endswith('.ins')]

# output simulation files 
# NOTE : added sorted() to match with sorted list of obs files 
sim_files = [os.path.join(sim_dir, f + '.dat') for f in sorted(list(mm.obs.keys()))  ]

# set up pst file
pst = pyemu.helpers.pst_from_io_files(tpl_files, par_files, ins_files, sim_files)


# set observation values and weights in the pst
for obs_loc in list(mm.obs.keys()):
    pst.observation_data.loc[mm.obs[obs_loc].df.index,'obsval'] = mm.obs[obs_loc].df.value
    pst.observation_data.loc[mm.obs[obs_loc].df.index,'weight'] = mm.obs[obs_loc].df.weight
    pst.observation_data.loc[mm.obs[obs_loc].df.index,'obgnme'] = obs_loc


#for stat in ['parlbnd','parval1','parubnd']:
for stat in ['parval1']:
    for partype in {'kepon','emmli'}:
        pst.parameter_data.loc[ mm.param[partype].zpc_df.index , 'pargp'] = partype
        for lay in range(mm.nlay):
            parname = '{0}_l{1:02d}_zpc{2:02d}'.format(partype,lay+1,1)
            val = prior_data_dic[partype][stat][lay]
            pst.parameter_data.loc[ parname , stat] = val

#for stat in ['parlbnd','parval1','parubnd']:
for stat in ['parval1']:
    for partype in {'permh'} :
        pst.parameter_data.loc[ mm.param[partype].pp_dic[lay].index , 'pargp'] = partype
        for lay in range(mm.nlay):
            for parname in mm.param[partype].pp_dic[lay].index:
                val = prior_data_dic[partype][stat][lay]
                pst.parameter_data.loc[ parname , stat] = val

#for stat in ['parlbnd','parval1','parubnd']:
for stat in ['parval1']:
    for partype in {'emmca'} :
        pst.parameter_data.loc[ mm.param[partype].pp_dic[lay].index , 'pargp'] = partype
        for lay in range(1,14,1):
            for parname in mm.param[partype].pp_dic[lay].index:
                val = prior_data_dic[partype][stat][lay]
                pst.parameter_data.loc[ parname , stat] = val


pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmca' , 'parubnd'] =  10**(20)
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmli' , 'parubnd'] =  10**(20)
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'permh' , 'parubnd'] =  10**(20)
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'kepon' , 'parubnd'] =  10**(20)

pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'kepon' , 'parlbnd'] =  10**(-20)
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'permh' , 'parlbnd'] =  10**(-20)
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmca' , 'parlbnd'] =  10**(-20)
pst.parameter_data.loc[ pst.parameter_data['pargp'] == 'emmli' , 'parlbnd'] =  10**(-20)

pst.parameter_data.loc['kepon_l01_zpc01','partrans'] = 'fixed'
pst.parameter_data.loc['kepon_l01_zpc01','parlbnd']  = 0.

# first layer , no need to adjust kepon value
# add fixed trans to zpc for emmli and emmca
#pst.parameter_data.loc[pst.parameter_data.parnme.str.startswith('kepon_l00'),'partrans']= 'fixed'
#pst.parameter_data.loc[pst.parameter_data.parnme.str.startswith('kepon_l00'),'parlbnd'] = 0

#pst.parameter_data.loc[pst.parameter_data'kepon_l01_zpc01','partrans'] = 'fixed'
#pst.parameter_data.loc['kepon_l01_zpc01',"parlbnd"]  = 0.

#pst.parameter_data.loc['emmli_l{}_zpc01'.format(lay+1),'partrans'] = 'fixed'
#pst.parameter_data.loc['emmli_l{}_zpc01'.format(lay+1),'partrans'] = 'fixed'

# update parameter_groups from parameter_data 
pst.rectify_pgroups()


# Zero-order Tikhonov reg
pyemu.utils.helpers.zero_order_tikhonov(pst)
pst.prior_information.head()

# derivative calculation type 
pst.parameter_groups.loc[ pst.parameter_groups.index,'forcen'] = 'always_3'
pst.parameter_groups.loc[ pst.parameter_groups.index,'derinc'] = 0.10
#pst.parameter_groups.loc[ pst.parameter_groups.index,'dermthd'] = 'minvar'
pst.parameter_groups.loc[ pst.parameter_groups.index,'derincmul'] = 1.0

# pestpp-glm options 
pst.pestpp_options['svd_pack'] = 'propack'
pst.pestpp_options['lambdas'] = str(list(np.power(10,np.linspace(-2,3,10)))).strip('[]')
#pst.pestpp_options['lambda_scale_fac'] = str([0.9, 0.8, 0.7, 0.5]).strip('[]')


# write pst 
pst.write(mm.mlname + '_pp.pst')



