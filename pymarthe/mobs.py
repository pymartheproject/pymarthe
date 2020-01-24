"""
Contains the MartheObs class
for handling observations by locations
One instance per obs location. 

"""
import os 
import numpy as np
from matplotlib import pyplot as plt 
from .utils import marthe_utils
from .utils import pest_utils
import pandas as pd 
import pyemu

# width of values (number of characters)
VAL_CHAR_LEN = 21
VAL_START = 12
NO_DATA_VALUES = [-9999,-8888]

class MartheObs() : 
    """
    Class for handling observations 

    Parameters
    ----------
    mm : MartheModel instance
        The MartheModel instance to which the observation pertains
    loc_name : str
        observation location name (ex. BSS id) 
        will correspond to the name of the observation group in PEST files
    value : int or np.array (time series)
        observed value(s)
    date : (optional) 1D numpy array

    Examples
    --------
    """
    def __init__(self, mm, prefix, obs_file, loc_name) :

        # pointer to parent model
        self.mm = mm
        self.prefix = prefix

        # set name of instance with location name
        self.loc_name = loc_name
                
        # open obs file for reading
        try :
            df = pd.read_csv(obs_file, delim_whitespace=True,header=None,skiprows=1)
        except : 
            print('Cannot open observation file ' + obs_file)
            return

        # get obs loc name from filename 
        self.obs_dir, self.obs_filename = os.path.split(obs_file)

        # case weights are not provided in the file
        if df.shape[1] == 2 :  # two columns date and value
            df.rename(columns ={0 : 'date', 1 :'value'}, inplace =True)
        else : 
            df.rename(columns ={0 : 'date', 1 :'value', 2 : 'weight'}, inplace =True)

        # set index to date
        try :
            df.date = pd.to_datetime(df.date, format="%Y-%m-%d")
        except : 
            print('Cannot convert date for ' + obs_file)

        df.set_index('date', inplace = True)

        # convert no data values to nan 
        for no_data_string in NO_DATA_VALUES :
            df.loc[df.value == no_data_string,'value'] = np.nan
            #df.dropna()
        # NOTE : issues with incomplete series simul/obs mismatch
	#for no_data_string in NO_DATA_VALUES :
        #    df.loc[df.value == no_data_string,'value'] = np.nan
	#		df.dropna()

        self.df = df

    def write_ins(self) :
        # open instruction file for obs_point
        ins_file = open( os.path.join(self.mm.mldir,'ins', self.loc_name + '.ins'), 'w')

        # write instruction file header
        ins_file.write('pif #\n')

        obs_ids = []

        # Write instruction and observation line for each value
        for i in range(self.df.shape[0]):
            obs_id = '{0}n{1:04d}'.format(self.prefix, i+1)
            ins_file.write('l1 ({0}){1}:{2}\n'.format(obs_id, VAL_START, VAL_CHAR_LEN))
            obs_ids.append(obs_id)

        self.df['obs_id'] = obs_ids
        self.df.set_index('obs_id', inplace=True)

        ins_file.close()

