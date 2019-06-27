import numpy as np
import sys
sys.path.append('C:/Documents these/Dev/adeqwat/pymarthe/utils')
from marthe_utils import read_grid_file
from matplotlib import pyplot as plt


# Loading files 
x, y, sepon  = read_grid_file('./mona_grille.sepon')
x, y, topog  = read_grid_file('./mona_grille.topog')
x, y, hsubs  = read_grid_file('./mona_grille.hsubs')
x, y, chasim = read_grid_file('./chasim.out')


top = sepon 
nlay, nrow, ncol = top.shape
nper = 40 #Time step 

NO_EPON_VAL = 9999 #In layer domain but not in eponte
NO_EPON_OUT_VAL = 8888 # Not in layer domain

#Definig Geometry
for lay in range(nlay-2,1,-1):
	idx = np.logical_or(top[lay:nlay,:,:] == NO_EPON_VAL, top[lay:nlay,:,:] == NO_EPON_OUT_VAL	) #Define inices where values are equal to 9999 or 8888
	top[lay:nlay][idx] = np.stack([sepon[lay,:,:]]*(nlay-lay))[idx] #Replace 9999 and 8888 values with layer values just before   


idx = np.logical_or(chasim == NO_EPON_VAL, chasim == NO_EPON_OUT_VAL	)
chasim[idx] = np.nan #Replace 9999 and 8888 values by nan 
heads  = np.stack ([chasim[i:i+nlay] for i in range(0,nlay*nper,nlay)]) #Create 4d numpy array by joining arrays of the same time step 
hmax   = np.nanmax(heads,0) # Defining hmax
hmin   = np.nanmin(heads,0) # Defining hmin 



cap   = hmax > top  #Defining confind parts of each layer
libre = hmin <= top #Defining unconfind parts of each layer  
mixte = np.logical_and(cap,libre)

