"""
Contains the MartheObs class
for handling observations by locations
One instance per obs location. 
"""


import os, sys
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # Ignore pandas SettingWithCopyWarning
from pymarthe import *
from .utils import marthe_utils, pest_utils


class MartheObs():
    """
    Class to manage Marthe or external observations for PEST coupling purpose
    """
    def __init__(self, datatype, obsfile, iloc, locnme = None, nodata = None, **kwargs):
        """
        Instance generator of MartheObs class

        Parameters:
        ----------
        datatype (str): data type of observation values.
                         Please consider using the reduced names.
                         Example : heads data -> 'h'
                                   flows data -> 'q'
        obsfile (str): observation filename to read value.
                       Note: if locnme is not provided,
                       the locnme is set as obsfile without file extension.
        iloc (int): i-number of observation location
        locnme (str, optional) : observation location name (ex. BSS id)
        nodata (list/None) : no data values to remove reading observation data.
                             If None, all values are considered.
                             Default is None.
                             NOTE  Careful, an create issues with incomplete
                             series sim/obs mismatch.
        obgnme (str, kwargs): group of observation related.
                              Default is locnme.
        obnme (list, kwargs): custom observation names
                               Default build as 'loc{locnme_id}n{obs_id}'
        weight (list, kwargs): weight per each observations


        Examples
        --------
        mobs = MartheObs(datatype = 'head', obsfile = 'p31.dat')

        """

        # ---- Store arguments as attribute
        self.nodata = nodata
        self.datatype = datatype
        self.obsfile = obsfile
        self.iloc = iloc
        self.obs_df = pd.DataFrame()
        obs_dir, obs_filename = os.path.split(obsfile)
        self.locnme = obs_filename.split('.')[0] if locnme is None else locnme
        self.obgnme = kwargs.get('obgnme', self.locnme)
        self.weight = kwargs.get('weight', 1)

        # ---- Read obsfile and fetch values
        df = self.read_obsfile(self.obsfile, nodata = self.nodata)

        # ---- Get number of observation values
        ndigit = len(str(len(df)-1))

        # ---- Build default observation names
        # Note: maximum number of locnmes is set to 1000 (more than enough)
        if not 'obsnme' in kwargs:
            self.obsnmes = ['loc{}n{}'.format(str(self.iloc).zfill(3),
                            str(i).zfill(ndigit))
                            for i in range(len(df))]

        # ---- Fill observations DataFrame with input data
        self.obs_df = self.obs_df.assign(obsval = df['value'],
                                         date = df.index,
                                         obsnme = self.obsnmes)
        self.obs_df[['datatype', 'locnme', 'obsfile']] = [self.datatype,
                                                          self.locnme,
                                                          self.obsfile]
        # ---- Add kwargs if required
        self.obs_df['weight'] = self.weight
        self.obs_df['obgnme'] = self.obgnme

        # ---- Set observation names as index
        self.obs_df.set_index('obsnme', drop=False, inplace=True)




    def write_obsfile(self, filename=None):
        """
        Write a standard obsfile for a existing locnme from mobs_df.
        Use marthe_utils.write_obsfile()

        Parameters:
        ----------
        self (object): MartheObs instance

        Returns:
        --------
        Write observation values.
        Format : date          value
                 1996-05-09    0.12
                 1996-05-10    0.88

        Examples:
        --------
        mobs = MartheObs( datatype = 'head'
                          obsfile = 'obs/p31.dat')
        mobs.write_obsfile(filename='newobsfolder/p31.dat')

        """
        # ---- Manage filename
        if filename is None:
            filename = self.obsfile

        # ---- Write observation value in basic file
        marthe_utils.write_obsfile(self.df['date'],
                                   self.df['obsval'],
                                   filename)
            



    def read_obsfile(self, obsfile, nodata = None):
        """
        Simple wrapper to read observation file.
        Use marthe_utils.read_obsfile()
        Format : header0        header1
                 09/05/1996     0.12
                 10/05/1996     0.88
        Note: separator is anywhite space (tabulation is prefered)

        Parameters:
        ----------
        obsfile (str): observation filename to read value.
                       Note: if locnme is not provided,
                       the locnme is set as obsfile without file extension.
        nodata (list/None) : no data values to remove.
                             Default is None.

        Returns:
        --------
        df (DataFrame) : observation table
                         Format : date          value
                                  1996-05-09    0.12
                                  1996-05-10    0.88

        Examples:
        --------
        obs_df = mobs.read_obsfile(datatype = 'head', obsfile = 'myobs.dat')

        """
        return marthe_utils.read_obsfile(obsfile, nodata = nodata)



    def write_ins(self, ins_dir = '.'):
        """
        Write formatted instruction file (pest).
        Wrapper of pest_utils.write_insfile().

        Parameters:
        ----------
        ins_dir (str, optional) : directory to write instruction file(s)
                                  Default is '.'.

        Returns:
        --------
        Write insfile file in ins_dir

        Examples:
        --------
        mmobs.write_ins('ins')
        """
        insfile = os.path.join(ins_dir, f'{self.locnme}.ins')
        pest_utils.write_insfile(self.obsnmes, insfile)



    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheObs'


    



