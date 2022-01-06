"""
Contains the MartheModel and Spatial Reference classes.
Designed for structured and nested grid.
"""

import os, sys
import subprocess as sp
from shutil import which
import queue 
import threading
import re
from copy import deepcopy
import numpy as np
import pandas as pd 


from .mfield import MartheField
from .mpump import MarthePump
from .utils import marthe_utils, shp_utils

encoding = 'latin-1'


class MartheModel():
    """
    Wrapper MARTHE --> Python
    """
    def __init__(self, rma_path):
        """
        Parameters
        ----------
        rma_path (str): path to the Marthe .rma file from which
                        the model name and working directory
                        will be identified.

        Examples
        --------
        mm = MartheModel('/Users/john/zone/model/mymodel.rma')

        """
        # ---- Get model working directory and rma file 
        self.mldir, self.rma_file = os.path.split(rma_path)

        # ---- Get model name 
        self.mlname = self.rma_file.split('.')[0]

        # ---- Get model files paths
        self.mlfiles = marthe_utils.get_mlfiles(os.path.join(rma_path))

        # ---- Get model units
        self.units = marthe_utils.get_units_dic(self.mlfiles['mart'])
        self.mldates = marthe_utils.get_dates(self.mlfiles['pastp'], self.mlfiles['mart'])

        # ---- Get infos about layers
        self.nnest, self.layers_infos = marthe_utils.get_layers_infos(self.mlfiles['layer'], base = 0)

        # ---- Initialize property grids with permeability data
        permh = MartheField('permh', self.mlfiles['permh'], self)
        self.prop = {'permh': permh}

        # ---- Store model grid infos from permh field
        self.imask = deepcopy(permh)
        self.imask.field = 'imask'
        self.imask.data['value'] = (self.imask.data['value'] != 0).astype(int)

        # ---- Set spatial reference (used for compatibility with pyemu geostat utils)
        self.spatial_reference = SpatialReference(self)

        # ---- Set number of simulated timestep
        self.nstep = len(self.mldates)



    def load_prop(self, prop):
        """
        Load MartheModel property by name.

        Parameters:
        ----------
        prop (str) : supported property name.
                     Can be :
                     - Field (MartheField)
                        - 'permh'
                        - 'emmca'
                        - 'emmli'
                        - 'kepon'
                     - Pumping (MarthePump)
                        - 'aqpump'
                        - 'rivpump'

        Returns:
        --------
        Stock property class in prop dictionary

        Examples:
        --------
        mm = MartheModel('mona.rma')
        mm.load_prop('emmca')
        """
        # ---- Manage fields
        if prop in self.mlfiles.keys():
            self.prop[prop] = MartheField(prop, self.mlfiles[prop], self)
        # ---- Manage pumping
        elif prop == 'aqpump':
            self.prop[prop] = MarthePump(self, mode = 'aquifer')
        elif prop == 'rivpump':
            self.prop[prop] = MarthePump(self, mode = 'river')
        # ---- Not supported property
        else:
            print(f"Property `{prop}` not supported.")




    def remove_autocal(self):
        """
        Function to make marthe auto calibration silent.
        wrapper to marthe_utils.remove_autocal().

        Parameters:
        ----------
        self : MartheModel instance

        Returns:
        --------
        Write in .mart file inplace

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.remove_autocal()
        """
        marthe_utils.remove_autocal(self.mlfiles['mart'])      



    def make_silent(self):
        """
        Function to make marthe run silent

        Parameters:
        ----------
        self : MartheModel instance

        Returns:
        --------
        Write in .mart inplace

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.make_silent()
        """
        marthe_utils.make_silent(self.mlfiles['mart']) 
        

    '''

    *** UNDER DEVELOPMENT ***

    def export_prop(self, filename, prop = 'permh', layer=None, inest=None, base = 0,  epsg=None, prj=None):
        """
        -----------
        Description:
        -----------
        Export grid values as shapefile.
        Only available for structured grid.
        
        Parameters: 
        -----------
        filename (str) : name of the output shapefile
        prop (str) : gridded property to export.
                      Default is 'permh'
        layer (int) : layer number to export.
                      If None, all layers considered.
                      Default is None.
        base (int) : base for 2D-arrays.
                     Python is 0-based.
                     Marthe is compiled in 1-based (Fortran)
                     Default is 1.
        Returns:
        -----------
        Write shapefile inplace
        Example
        -----------
        mm = MartheModel('mona.rma')
        mm.load_prop('emmli')
        
        """ 
        # ---- Assert that 'prop' is a gridded property of MartheModel
        err_msg = f"'{prop}' is not a gridded (exportable)."
        assert isinstance(self.prop[prop], MartheField), err_msg

        # ---- Fetch property geometries and records
        parts, rec = self.prop[prop].to_pyshp(layer=layer, inest=inest)

        # ---- Convert reccaray to shafile
        shp_utils.recarray2shp(rec, parts, shpname=filename, epsg=epsg, prj=prj)
        print("\n ---> Property '{}' wrote in {} succesfully.".format(prop , filename))

    '''





    def get_outcrop(self):
        """
        Function to get outcropping layer number
        (integer) as 2D-array.
        Not available for nested model.

        Parameters:
        ----------
        self : MartheModel instance

        Returns:
        --------
        outcrop (2D-array) : outcropping layer numbers.

        Examples:
        --------
        mm = MartheModel('mymodel.rma')
        outcrop_arr = mm.get_outcrop()
        """
        if self.nnest == 0:
            # ---- Set list of arrays with layer number on active cell
            layers = [ilay * imask for ilay, imask in enumerate(self.imask.as_array(), start=1)]
            # ---- Transform 0 to NaN
            nanlayers = []
            for layer in layers:
                arr = layer.astype('float')
                arr[arr == 0] = np.nan
                nanlayers.append(arr)
            # ---- Get minimum layer number excluding NaNs
            outcrop = np.fmin.reduce(nanlayers)
            # # ---- Back transform inactive zone to 0
            outcrop[np.isnan(outcrop)] = 0
            # ---- Return outcrop layer as array
            return outcrop.astype(int)
        else:
            print("'.get_outcrop() method not available for nested model yet.'")




    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheModel'



class SpatialReference():
    """
    Inspired from FloPy, for compatibility with PyEMU
    """
    def __init__(self, mm):
        """
        Parameters
        ----------
        ml : instance of MartheModel
        """
        mg = mm.imask.to_grids(layer=0, inest=0)[0]
        self.nrow, self.ncol = mg.nrow, mg.ncol


    def __str__(self):
        """
        Internal string method.
        """
        return 'SpatialReference'

