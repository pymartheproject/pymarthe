'''
**************************************************************************************
                    Copyright : Géoressources et Environnement ©
                    Auteur : Ryma AISSAT (ryma.aissat@ensegid.fr)
**************************************************************************************
'''
# This code compares simulated data with observed data
#======================================================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib import style
from math import sqrt
import seaborn as sns
import sys
sys.path.append('C:/Users/raissat/Programmes/python/adeqwat/pymarthe/utils')

# Directory path where the marthe utils library is located
from marthe_utils import*

#===============================================================================================================

#Read mona.histo 
df_histo = read_histo_file('./mona.histo')

# Read obs file 
id_points_obs,df_obs = read_obs('Piezo_2015_Ryma.txt')
obs_number= df_obs.groupby(pd.Grouper(freq='Y'))[id_points_obs].count()

#Read sim file 
df_sim = read_prn('historiq.prn')

#Extract common columns 
common_cols = list(set(df_histo['ID_FORAGE']).intersection(id_points_obs))
df_obs = df_obs[common_cols]
df_sim = df_sim[common_cols]
df_histo  = df_histo.loc[common_cols]
# ID_FORAGE in index
df_histo  = df_histo.set_index(df_histo.ID_FORAGE)
# Annual average
yearly_data_obs = df_obs.resample('Y').mean()
# Replace 9999 by nan
yearly_data_sim   =  df_sim.replace( 9999, np.nan)

# Create a list of id points
list_id =  yearly_data_obs.columns

res   = ((yearly_data_sim   -  yearly_data_obs)**2).mean()
biais = (yearly_data_sim    -  yearly_data_obs).mean()
biais_list = []
rmse_list  = []
for id in list_id :
    rmse = (res [id].mean())**.5
    rmse_list.append(round(rmse,2))
    b = biais [id].mean()
    biais_list.append(round(b,2))
nobs   = yearly_data_obs.count()
nobs   = pd.Series(nobs)
rmse   = pd.Series(rmse_list, index = list_id)
layer  = (df_histo[['Couche']].apply(pd.to_numeric)).Couche
bias   = pd.Series(biais_list, index =list_id)

# Criteria table by observation point
dfcrit = pd.DataFrame(data = dict(layer = layer , nobs = nobs, rmse = rmse, bias = bias ), index = layer.index,dtype = float)
dfcrit.to_csv('./'+'critères par forage'+'.txt',sep='\t',header=1, index = True)


# Mean of RMSE and Bias for each layer
list_mean_rmse = []
list_mean_bias = []
nb_layer = 15
for i in range(1,nb_layer+1):
    carac_layer = dfcrit.loc[dfcrit["layer"] == i ]
    mean_rmse   = carac_layer.rmse.mean()
    mean_bias   = carac_layer.bias.mean()
    list_mean_rmse.append(mean_rmse)
    list_mean_bias.append(mean_bias)

'''
#Plot boxplot : 
dfcrit.boxplot('rmse','layer', grid = False, showfliers = False)
plt.savefig('./boxplot_rmse.png', dpi = 1000)
dfcrit.boxplot('bias','layer', grid = False, showfliers = False)
plt.savefig('./boxplot_bias.png', dpi = 1000)
'''

# Plot comparaison
for id in list_id :
    rmse = (res [id].mean())**.5
    df_sim_column = yearly_data_sim[id]
    df_obs_column = yearly_data_obs[id]
    df_sim_column.plot(alpha =0.6, label = "SIM ")
    df_obs_column.plot(style = '.',  label = "OBS ")
    if type(layer[id]) ==  pd.core.series.Series:
        plt.title('RMSE = '+str(round(rmse,2))+'m '+ '   '+'BIAIS = '+str(round(biais[id].mean(),2))+'m '+'   '+id+'   '+'Couches'+'  '+str((layer[id].values[0]))+' et '+str((layer[id].values[1]))+' ', fontsize = 12)
    else:
        plt.title('RMSE = '+str(round(rmse,2))+'m  '+ '  '+'BIAIS = '+str(round(biais[id].mean(),2))+'m  '+'  '+id+'   '+'Couche'+' '+str(layer[id])+' ', fontsize = 12)
    plt.ylabel ('H (mNGF)')
    plt.xlabel ('Année')
    plt.legend()
    plt.savefig('./fig_comp/'+id+'.png', dpi = 1000)
    plt.close()