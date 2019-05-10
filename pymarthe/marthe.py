"""
Contains the MartheModel class

"""
import os 
import numpy as np
from matplotlib import pyplot as plt 
from ..utils import marthe_utils

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
        # NOTE all the respective items of x_list and y_list are identical
        self.x_vals, self.y_vals, self.grids['permh'] = self.read_grid('permh')

        # get nlay nrow, ncol
        self.nlay, self.nrow, self.ncol = self.grids['permh'].shape

        # get list of mask of active/inactive cells. 1 active. 0 inactive. 
        self.imask = (self.grids['permh'] != 0).astype(int)

        # NOTE under development
        self.izone = self.imask

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

    def izone_from_shp(self,shp_file,lay):
        """
        Parameters
        ----------
        param : type
            text

        Examples
        --------

        """

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
