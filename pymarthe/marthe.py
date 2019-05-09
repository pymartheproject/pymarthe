"""
Contains the MartheModel class

"""
import os 
from ..utils.marthe_utils import read_grid_file

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
        # get model working directory and rma file 
        self.mldir, self.rma_file = os.path.split(rma_path)

        # get model name 
        self.mlname = rma_path.split('.')[0]

        # set path to other files 
        permh_path = os.path.join(self.mldir,self.mlname + '.permh')

        # read permh data
        # NOTE all the respective items of x_list and y_list are identical
        x_list, y_list, self.permh_grids = read_grid_file(permh_path)

        # get x and y coordinates (take the first, arbitrarily)
        self.x_vals = x_list[0]
        self.y_vals = y_list[0]

        # get nrow, ncol and nlay
        self.nrow = len(self.y_vals)
        self.ncol = len(self.x_vals)
        self.nlay = len(self.permh_grids)

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
