import numpy as np
import pandas as pd
import csv
from itertools import islice
import datetime as dt
import matplotlib.dates as mdates
import os
from math import sqrt
import fileinput
import sys
sys.path.append('C:/Users/raissat/Programmes/python/adeqwat/pymarthe')
from utils import pest_utils 
from utils import marthe_utils

#Read mona.histo 
df_histo = marthe_utils.read_histo_file('./txt_file/mona.histo')

# Read obs file 
id_points_obs,df_obs = marthe_utils.read_obs('./txt_file/Piezo_2015_Ryma.txt')

#Read sim file 
df_sim = marthe_utils.read_prn('./txt_file/historiq.prn')

common_cols = list(set(df_histo['ID_FORAGE']).intersection(id_points_obs))
#comm = list(set(id_points_sim).intersection(id_points_obs))
df_obs = df_obs[common_cols]
df_sim = df_sim[common_cols]
df_sim.iloc[:,0:-1]
df_histo  = df_histo.loc[common_cols]
df_histo  = df_histo.set_index(df_histo.Couche)

#*******************************************************************************************
# Create a dataframe of weights
#*******************************************************************************************
yearly_data_obs = df_obs.resample('Y').mean()
yearly_data_obs = yearly_data_obs.loc[df_sim.index]
mean_yearly  = yearly_data_obs.fillna(-9999)
std_yearly   = df_obs.resample('Y').std()
std_yearly   = std_yearly.fillna(0)
std_yearly = std_yearly.loc[df_sim.index]
count_yearly = df_obs.resample('Y').count()
count_yearly = count_yearly.loc[df_sim.index]
dates =  count_yearly.index.strftime('%Y-%m-%d')

# Create dataframe containing the standard deviation of each point over years
std_all_years = df_obs.std()
var_all_years = std_all_years**2
nb_layer= 15
df_wlayer = pd.DataFrame()
for i in range(1,nb_layer+1):
	wells_selection = list(df_histo.loc[str(i)]['ID_FORAGE'])
	var_wells_selection = var_all_years[wells_selection]
	var_mean_layer      = (var_wells_selection.mean())
	std_mean_layer = sqrt(var_mean_layer)
	std_mean_layer      = pd.DataFrame({str(i) : std_mean_layer}, index=[0])
	df_std_mean_layer   = pd.concat([df_wlayer,std_mean_layer],axis = 1)
	df_wlayer = df_std_mean_layer

nobs_min = 6
std_mes  = 0.05
nobs  = yearly_data_obs.count()
nobs  = pd.Series(nobs)
dfw = pd.DataFrame()

for id in common_cols :
	weights_list = []
	means_list = []
	std_forage = std_yearly[id]
	mean_forage = mean_yearly[id]
	count_data=count_yearly[id]
	sim_data =  df_sim[id]
	# iterate over years 
	for j in range (len(count_data)):
		if type(sim_data) == pd.core.frame.DataFrame :
			if sim_data.iloc[j,0] == 9999:
				w = 0.
				m = mean_forage[j]
			else : 
			#case no obs for  the j-th year
				if (count_data[j] == 0)  :
					w = 0.
					m =mean_forage[j]
				# case enough obs for the compuation of the error on the mean
				elif count_data.iloc[j] > nobs_min  :
					std_m = std_forage[j] / sqrt(count_data[j])
					std = std_m + std_mes
					w = (1./std)/ nobs[id]
					m = mean_forage[j]
				# case not enough obs for the couputation of the error on the mean
				else :
					layer = df_histo.loc[df_histo.ID_FORAGE == id]
					layer = layer.Couche
					l = layer.iloc[0]
					std_m = df_wlayer [l][0]
					std   = std_m + std_mes
					w     = (1./(std))/ nobs[id]
					m = mean_forage[j]
		else :
			if sim_data[j] == 9999:
				w = 0.
				m =mean_forage[j]
			else: 
				#case no obs for  the j-th year
				if (count_data[j] == 0):
					w = 0.
					m = mean_forage[j]
				# case enough obs for the compuation of the error on the mean
				elif count_data.iloc[j] > nobs_min  :
					std_m = std_forage[j] / sqrt(count_data[j])
					std = std_m + std_mes
					w = (1./std)/ nobs[id]
					m = mean_forage[j]
				# case not enough obs for the couputation of the error on the mean	
				else : 
					layer = df_histo.loc[df_histo.ID_FORAGE == id]
					layer = layer.Couche
					l = layer.iloc[0]
					std_m = df_wlayer [l][0]
					std   = std_m + std_mes
					w     = (1./(std))/ nobs[id]
					m = mean_forage[j]
		# append new element to the lists
		weights_list.append(w)
		means_list.append(m)
	mean_weight = pd.DataFrame(np.column_stack([list(dates),means_list, weights_list]),columns=['Year','Mean','Weight'])
	mean_weight.to_csv('./txt_file/obs_data/'+id+'.dat',sep='\t', index = False)

'''
# --------------- Preamble ----------------------------------------------
path_obs_files    = './obs_data/'
path_output_files = './pest_files/'
# =======================================================================
# =============== reading obs files for pest ============================
# =======================================================================
#generate obs_points array
obs_points = []

for filename in os.listdir('./obs_data/'):
	if filename.endswith(".txt"):
		point_id = filename.replace(".txt", "")
		obs_points.append(point_id)

# =======================================================================
# =============== PEST Preprocessing tools ==============================
# =======================================================================
# -- Time discretization ---
date_start_string = '1972'
dates_out = pd.date_range(date_start_string, periods=40, freq= 'A')


# Generation of pest instruction files
nobs, nobs_grp, obs_dates = pest_utils.write_obs_data (obs_points, path_obs_files, path_output_files, dates_out)
'''
'''
# build parameter dictionary & set parameters initial values
params = {}
params_init = {}
ppoints = {}

# Generation of regular pilot point grid
pp_spacing = 50. #km 
x_dist = 579.5 - 284.5#TODO search for x_dist in .permh
y_dist = 411 - 156
x_min, x_max = 284.5, 579.5
y_min, y_max = 156, 411

nx, ny = round(x/pp_spacing,0) , round(y/pp_spacing,0)
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), 
		     np.linspace(y_min, y_max, ny))

x = np.reshape(xx,(nx*ny))
y = np.reshape(yy,(nx*ny))

# Generation of pilot point dict
ppoints_id = 0
for ppoint_x, ppoint_y in zip(x, y):
    ppoints_id += 1
    ppoints[ppoints_id] = (ppoint_x, ppoint_y)

ppoints_name_list = []
for point_id in sorted(ppoints.keys()) :
    ppoints_name_list.append( str(point_id) )
params['pp_S'] = ppoints_name_list
params['pp_T'] = ppoints_name_list 

params_init['pp_T'] = [T_init]*len(ppoints_name_list)
params_init['pp_S'] = [S_init]*len(ppoints_name_list)

# Generation of pest template files
npar, npar_grp = pest_utils.write_tpl_files(params, path_output_files, params_init)

# Generation of pst I/O section
pest_utils.write_pst_io(path_output_files)
'''

#write sim data
#marthe_utils.read_file_sim ('./txt_file/historiqprn.txt','./pest_files/')
