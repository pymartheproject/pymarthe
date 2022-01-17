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
    def __init__(self, iloc, locnme, date, value, obsfile = None,
                 datatype= 'head', nodata = None, **kwargs):
        """
        Instance generator of MartheObs class

        Parameters:
        ----------

        iloc (int): i-number of observation location

        locnme (str) : observation location name (ex. BSS id)

        date (pd.DatatimeIndex) : dates index of observation values.

        value (iterable) : observation values.

        obsfile (str, optional): observation filename where values and dates are stored.
                                 Default is None.

        datatype (str, optional): data type of observation values.
                                  Default is 'head'.

        nodata (list/None, optional) : no data values to remove reading observation data.
                                       If None, all values are considered.
                                       Default is None.
                                       NOTE  Careful, an create issues with incomplete
                                       series sim/obs mismatch.

        obgnme (str, kwargs): group of observation related.
                              Default is locnme.

        obnme (list, kwargs): custom observation names
                               Default build as 'loc{locnme_id}n{obs_id}'
                               
        weight (list, kwargs): weight per each observations

        transform (str/func, kwargs) : keyword/function to use for transforming 
                                       observation values.
                                       Can be:
                                        - function (np.log10, np.sqrt, ...)
                                        - string function name ('log10', 'sqrt')

        Examples
        --------
        dt = pd.date_range('1996-05-09', '2003-06-10', freq='D')
        mobs = MartheObs(0, 'p31', dt, value, weigth = 0.5, transform = 'log10')

        """

        # ---- Store arguments as attribute
        self.nodata = nodata
        self.datatype = datatype
        self.iloc = iloc
        self.locnme = locnme
        self.date = date
        self.value = value
        self.obsfile = obsfile
        self.weight = kwargs.get('weight', 1)
        self.obgnme = kwargs.get('obgnme', locnme)
        self.transform = kwargs.get('transform', 'none')
        self.obs_df = pd.DataFrame()

        # ---- Get number of observation values
        ndigit = len(str(len(self.value)-1))

        # ---- Build default observation names
        # Note: maximum number of locnmes is set to 1000 (more than enough)
        if not 'obsnme' in kwargs:
            self.obsnmes = ['loc{}n{}'.format(str(self.iloc).zfill(3),
                            str(i).zfill(ndigit))
                            for i in range(len(self.value))]

        # ---- Fill observations DataFrame with input data
        self.obs_df = self.obs_df.assign(obsval = self.value,
                                         date = self.date,
                                         obsnme = self.obsnmes)
        self.obs_df[['datatype', 'locnme', 'obsfile']] = [self.datatype,
                                                          self.locnme,
                                                          self.obsfile]
        # ---- Add kwargs if required
        self.obs_df['weight'] = self.weight
        self.obs_df['obgnme'] = self.obgnme
        self.obs_df['transform'] = self.transform

        # ---- Set observation names as index
        self.obs_df.set_index('obsnme', drop=False, inplace=True)



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


    



