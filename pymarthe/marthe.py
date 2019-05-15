"""
Contains the MartheModel class

"""
import os 
import numpy as np
from matplotlib import pyplot as plt 
from .utils import marthe_utils
import pandas as pd 

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

    def set_izone(self,key,data):
        """
        Load izone array to MartheModel instance
        Former izone for given parameter, if present, will be reset. 

        Parameters
        ----------
        key : str
            parameter to which the array is related
        data : int or np array of int, shape (nlay,nrow,ncol)

        Examples
        --------

        """
        # reset izone for current parameter from imask
        self.izone[key] = self.imask
        # index of active cells
        idx_active_cells = self.imask == 1

        if isinstance(data,int) :
            self.izone[key][idx_active_cells] = data

        if isinstance(data,np.ndarray) : 
            assert data.shape == (nlay,nrow,ncol) 
            # only update active cells  
            self.izone[key][idx_active_cells] = data[idx_active_cells]

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

    def add_ppoints(self,new_pp_df) :
        """
        Parameters
        ----------
        param : type
            text

        Examples
        --------

        """




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
