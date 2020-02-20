import matplotlib
matplotlib.use('Agg')
import sys 
import os 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import geopandas as gp
from shapely.geometry import Point, Polygon
import csv
from itertools import islice
import datetime as dt
import matplotlib.dates as mdates
from math import sqrt
import fileinput

# pyemu and adeqwat modules should be placed in ~/Programmes/python/
# https://github.com/jtwhite79/pyemu
sys.path.append(os.path.expanduser('~/Programmes/python/pyemu/'))
import pyemu

# https://github.com/apryet/adeqwat
sys.path.append(os.path.expanduser('~/Programmes/python/adeqwat/'))
from pymarthe import * 


#Read mona.histo 
df_histo = marthe_utils.read_histo_file('./mona.histo')
IDindex_layer   = df_histo[["Couche","ID_FORAGE"]]
ID_layer   = ID_layer.set_index(ID_layer.Couche)


# Read obs file 
id_points_obs,df_obs = marthe_utils.read_obs('./txt_files/Piezo_2015_Ryma.txt')

#Read sim file 
df_sim = marthe_utils.read_prn('./historiq.prn')

common_cols = list(set(df_histo['ID_FORAGE']).intersection(id_points_obs))
#comm = list(set(id_points_sim).intersection(id_points_obs))
df_obs = df_obs[common_cols]
df_sim = df_sim[common_cols]
df_sim.iloc[:,0:-1]
df_histo  = df_histo.loc[common_cols]
df_histo  = df_histo.set_index(df_histo.Couche)

yearly_data_obs = df_obs.resample('Y').mean()
yearly_data_obs = yearly_data_obs['1972-12-31':'2011-12-31']
yearly_data_sim = df_sim.replace(9999,np.nan)


# Calculate RMSE
list_id =  yearly_data_obs.columns
mse   = ((yearly_data_sim   -  yearly_data_obs)**2).mean()
biais = (yearly_data_sim    -  yearly_data_obs).mean()
biais = biais.loc[~biais.index.duplicated(keep ='first')]
mse = mse.loc[~mse.index.duplicated(keep ='first')]
rmse = mse**0.5


nobs   = yearly_data_obs.count() #Le nombre d'année ou j'ai de la donnée
nobs   = pd.Series(nobs)
layer  = (IDindex_layer[['Couche']].apply(pd.to_numeric)).Couche
dfcrit = pd.DataFrame(data = dict(layer = layer , nobs = nobs, rmse = rmse, bias = biais,mse = mse ), index = layer.index,dtype = float)
dfcrit.to_csv('./figs/'+'crit_postcalage_forage'+'.txt',sep='\t',header=1, index = True)
mse_mean = dfcrit.groupby('layer')['mse'].mean()
rmse_mean_layer = mse_mean**0.5
rmse_mean_tot = (mse_mean.mean())**0.5
biais_mean_lay = dfcrit.groupby('layer')['bias'].mean()
biais_tot = biais_mean_lay.mean()

#Plot obs sim comparaison
for id in list_id :
	sim_column = yearly_data_sim[id]
	obs_column = yearly_data_obs[id]
	sim_column.plot(alpha =0.6, label = "SIM ")
	obs_column.plot(style = '.',  label = "OBS ")
	if type(layer[id]) ==  pd.core.series.Series:
		plt.title('RMSE = '+str(round(rmse[id],2))+'m '+ '   '+'BIAIS = '+str(round(biais[id],2))+'m '+'   '+id+'   '+'Couches'+'  '+str((layer[id].values[0]))+' et '+str((layer[id].values[1]))+' ', fontsize = 12)
	else:
		plt.title('RMSE = '+str(round(rmse[id],2))+'m  '+ '  '+'BIAIS = '+str(round(biais[id],2))+'m  '+'  '+id+'   '+'Couche'+' '+str(layer[id])+' ', fontsize = 12)
	plt.ylabel ('H (mNGF)')
	plt.xlabel ('Année')
	plt.legend()
	plt.savefig('./figs/comp_sim_obs/'+id+'.png', dpi = 1000)
	plt.close()




#Plot histo diff sim (to appreciate the numerical noise for example)
diff = sim_base - sim_opt

diff_perm = diff.iloc[0,2:-1]
diff_2011 = diff.iloc[-1,2:-1]

diff_perm= diff_perm.astype(np.float64)
diff_2011= diff_2011.astype(np.float64)

diff_perm =  np.absolute(diff_perm)
log_diff_perm = diff_perm.apply(np.log10)
plt.hist(log_diff_perm[np.isfinite(log_diff_perm)].values)
plt.ion()

diff_2011 =  np.absolute(diff_2011)
log_diff_2011 = diff_2011.apply(np.log10)
plt.hist(log_diff_2011[np.isfinite(log_diff_2011)].values)
plt.ion()


plt.show()



#Plot piezo chronicles 

data_loc = df_histo.iloc[:,4:7]
data_loc = data_loc.set_index(data_loc.ID_FORAGE)

for id in list_id :
	data  = df_obs[id]
	#data  = data.reset_index().set_index('DATE',drop= False)
	start = dt.datetime(1972,12,30)
	end   = dt.datetime(2018,1,1)
	#data  = data.to_frame()
	data.plot.scatter(x = list(data.DATE), y = data[id])
	plt.xlim(start,end)
	plt.ylabel('H [mNGF]')
	plt.title(id, fontsize = 12)
	plt.savefig('./Results_script/chroniques_obs/'+str(id)+'.png', dpi = 1000)
	plt.close()



#Plot boxplot : 
dfcrit.boxplot('rmse','layer', grid = False, showfliers = False)
plt.savefig('./boxplot_rmse.png', dpi = 1000)
dfcrit.boxplot('bias','layer', grid = False, showfliers = False)
plt.savefig('./boxplot_bias.png', dpi = 1000)


# Write files (year obs sim) for each bss code
for id in list_id:

	obs_sim = pd.concat([yearly_data_obs[id], yearly_data_sim[id]], axis= 1)
	if len(obs_sim.columns) == 2: 
		obs_sim.columns = ['OBS','SIM']
	else:
		obs_sim.columns = ['OBS','SIM','SIM']
	obs_sim.index = obs_sim.index.strftime('%Y')
	obs_sim = obs_sim.round(2)
	y_obs_sim = obs_sim.rename_axis('Year')
	y_obs_sim.to_csv('./Results_script/fsim_obs/'+id+'.txt',sep=',',header=1, index = True)




# Density plots for each layer 
r = yearly_data_sim - yearly_data_obs
for i in range(1,nb_layer+1):
	wells_selection     = list(ID_layer.loc[str(i)]['ID_FORAGE'])
	res_layer = r[wells_selection]
	res_layer = res_layer.stack(0)
	sns.distplot(res_layer,hist = True,kde = True, bins = 10, color = 'darkblue', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
	#sns.distplot(res_layer,hist = True,kde = False, bins = 10, color = 'darkblue')
	#res_layer = res_layer.stack(0).hist()
	#plt.savefig('./Results_script/histo_error/'+str(i)+'.png', dpi = 1000)
	#plt.close()
	plt.show()

