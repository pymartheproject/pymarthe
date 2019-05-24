"""
Contains the MartheModel class
Designed for structured grid and
layered parameterization

"""
import os 
import numpy as np
from matplotlib import pyplot as plt 
from .utils import marthe_utils
import pandas as pd 

from .mparam import MartheParam
from .mobs import MartheObs


# ----- from PyEMU ----------------


class MartheModel():
    """
    Python wrapper for Marthe
    """
    def __init__(self,rma_path):
        """
        Parameters
        ----------
        path_to_rma : string
            The path to the Marthe rma file from which the model name 
            and working directory will be identified.

        Examples
        --------

        mm = MartheModel('/Users/john/models/model/mymodel.rma')

        """
        # initialize grids dictionary
        self.grid_keys = ['permh','kepon']
        self.grids = { key : None for key in self.grid_keys }

        # initialize parameter list 
        self.param = {}
        self.obs = {}

        # get model working directory and rma file 
        self.mldir, self.rma_file = os.path.split(rma_path)

        # get model name 
        self.mlname = self.rma_file.split('.')[0]

        # read permh data
        self.x_vals, self.y_vals, self.grids['permh'] = self.read_grid('permh')

        # get nlay nrow, ncol
        self.nlay, self.nrow, self.ncol = self.grids['permh'].shape

        # set up mask of active/inactive cells.
        # value : 1 for active cells. 0 for inactive cells
        # imask is a 3D array (nlay,nrow,ncol)
        self.imask = (self.grids['permh'] != 0).astype(int)

        # set up izone
        # similarly to self.grids, based on a dic with parameter name as key
        # values for izone arrays 3D arrays : 
        # - int, zone of piecewise constancy
        # + int, zone with pilot points
        # 0, for inactive cells
        # default value from imask, -1 zone of piecewise constancy
        self.izone = { key : -1*self.imask for key in self.grid_keys } 

        # set up pilot point data frame 
        self.pp_df = pd.DataFrame({'name': pd.Series(None, dtype=str),
            'x': pd.Series(None, dtype=np.float32),
            'y': pd.Series(None, dtype=np.float32),
            'lay': pd.Series(None, dtype=np.float32),
            'zone' : pd.Series(None, dtype=np.float32),
            'parval': pd.Series(None, dtype=np.float32)
            })

        # init number of observation locations
        self.nobs_loc = 0

    def add_param(self, name, default_value, izone = None, array = None) :

        # case an array is provided
        if isinstance(array, np.ndarray) :
            assert array.shape == (self.nlay, self.nrow, self.ncol)
            self.grids[name] = array
        else : 
            # fetch mask from mm
            self.grids[name] = np.array(self.imask, dtype=np.float)
            # fill array with nan within mask
            self.grids[name][ self.grids[name] != 0 ] = np.nan

        # create new instance of MartheParam
        self.param[name] = MartheParam(self, name, default_value, izone, array)       

    def add_obs(self, obs_file, loc_name = None) :
        
        self.nobs_loc += 1
        # prefix will be used to set individual obs name
        prefix = 'loc{0:02d}'.format(self.nobs_loc)

        # infer loc_name from file name if loc_name not provided
        if loc_name is None : 
            obs_dir, obs_filename = os.path.split(obs_file)
            loc_name = obs_filename.split('.')[0]

        self.obs[loc_name] = MartheObs(self, prefix, obs_file, loc_name)

    def load_grid(self,key) : 
        """
        Simple wrapper for read_grid.
        The grid is directly stored to grids dic

        Parameters
        ----------
        key : str
            grid data key

        Examples
        --------
        x_vals, y_vals, permh_grid = read_grid('permh')

        """
        x_vals, y_vals, self.grids[key] =  self.read_grid(key)
        
        return

    
    def read_grid(self,key):
        """
        Simple wrapper for read_grid file
        Builds up path to the file from the key

        Parameters
        ----------
        param : type
            text

        Examples
        --------
        x_vals, y_vals, permh_grid = read_grid('permh')

        """
        # get path to grid file 
        grid_path = os.path.join(self.mldir,self.mlname + '.' + key)

        # load grid file 
        x_vals, y_vals, grid = marthe_utils.read_grid_file(grid_path)

        return(x_vals,y_vals,grid)

        return

    def plot_grid(self,key,lay):
        """
        Parameters
        ----------
        param : type
            text

        Examples
        --------

        """
        plt.imshow(self.grids[key][lay,:,:])

    def data_to_shp(self, key, lay, filepath ) :
        data = self.grids[key][lay,:,:]
        marthe_utils.grid_data_to_shp(self.x_vals, self.y_vals, data, file_path, field_name_list=[key])

    def extract_prn(self, prn_file = None, out_dir = None):
        if prn_file == None : 
            prn_file = os.path.join(self.mldir,'historiq.prn')
        if out_dir == None : 
            out_dir = os.path.join(self.mldir,'obs','')

        marthe_utils.extract_prn(prn_file, out_dir)


class SpatialReference():
    """
    Inspired from FloPy, for compatibility with PyEMU
    """
    def __init__(self,mm):
        """
        Parameters
        ----------
        ml : instance of MartheModel
        """
        self.nrow = mm.nrow
        self.ncol = mm.ncol
