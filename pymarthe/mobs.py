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


base_obs = ['obsnme', 'date', 'obsval',
            'datatype', 'locnme', 'obsfile',
            'weight', 'obgnme', 'trans' ]



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

        trans (str/func, kwargs) : keyword/function to use for transforming 
                                       observation values.
                                       Can be:
                                        - function (np.log10, np.sqrt, ...)
                                        - string function name ('log10', 'sqrt')

        insfile (str, kwargs) : path to the instruction file 
                                (for writing purpose).
                                Default build as 'locnme.ins'.

        simfile (str, kwargs) : path to the simulated file
                                (for writing purpose).
                                Default build as 'locnme.dat'.

        fluc_dic (dict, kwargs) : fluctuation information from fluctuation process.
                                  Usefull only for fluctuation information.

        interp_method (str, kwargs): Interpolation method to use 
                                     to match simulated data with observations
                                     Can be 'linear', 'time', 'index', 'values', 'pad', 'nearest',
                                     'zero', 'slinear', 'quadratic', 'cubic', 'spline', 'barycentric',
                                     'polynomial', 'krogh', 'piecewise_polynomial', 'spline', 'pchip',
                                     'akima', 'from_derivatives'.
                                     Default is 'index' (linear interpolation on index).

        Examples
        --------
        dt = pd.date_range('1996-05-09', '2003-06-10', freq='D')
        mobs = MartheObs(0, 'p31', dt, value, weigth = 0.5, trans = 'log10')

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

        # ---- Check transformation validity
        self.trans = kwargs.get('trans', 'none')
        pest_utils.check_trans(self.trans)

        # ---- Manage io files
        self.insfile = kwargs.get('insfile', f'{self.locnme}.ins')
        self.simfile = kwargs.get('simfile', f'{self.locnme}.dat')

        # ---- Get number of observation values
        ndigit = len(str(len(self.value)-1))

        # ---- Build default observation names
        # Note: maximum number of locnmes is set to 1000 (more than enough)
        if not 'obsnme' in kwargs:
            self.obsnmes = ['loc{}n{}'.format(str(self.iloc).zfill(3),
                            str(i).zfill(ndigit))
                            for i in range(len(self.value))]

        # ---- Fill observations DataFrame with input data
        self.obs_df = pd.DataFrame(index=self.obsnmes)
        self.obs_df[base_obs] = [ self.obsnmes, self.date, self.value,
                                  self.datatype, self.locnme, self.obsfile,
                                   self.weight, self.obgnme, self.trans ]

        # ---- Store fluctuation arguments
        self.fluc_dic = kwargs.get('fluc_dic', dict())
        # ---- Store simulated data interpolation
        self.interp_method = kwargs.get('interp_method', 'index')



    def get_obs_df(self, transformed=False):
        """
        Extract a copy of observation data.

        Parameters:
        ----------
        transformed (bool, optional) : whatever apply transformation on output DataFrame.
                                       Default is False.

        Returns:
        --------
        obs_df (DataFrame) : standard observation DataFrame.

        Examples:
        --------
        mobs.get_obs_df(transformed=True)
        
        """
        # ---- Make a copy of observations ddata
        obs_df = self.obs_df.copy(deep=True)
        # ---- Transform values if required
        if transformed:
            obs_df['obsval'] = pest_utils.transform(obs_df['obsval'], self.trans)
        # ---- Return observation DataFrame
        return obs_df




    def write_insfile(self):
        """
        Write formatted instruction file (pest).
        Wrapper of pest_utils.write_insfile().

        Parameters:
        ----------

        Returns:
        --------
        Write insfile file.

        Examples:
        --------
        mobs.write_insfile()
        """
        pest_utils.write_insfile(self.obsnmes, self.insfile)
    

    def write_simfile(self, prn='historiq.prn'):
        """
        Write formatted simulated file (pest).
        Wrapper of pest_utils.extract_prn().
        Note: to get the related simulated values don't 
              forget to (re)run the Marthe model before.

        Parameters:
        ----------
        prn (str, optinal) : path to the simulated values file.
                             Default is 'historiq.prn'.

        Returns:
        --------
        Write simfile file.

        Examples:
        --------
        mobs.write_simfile()
        """
        # ---- Manage prn
        if isinstance(prn, str):
            prn_df = marthe_utils.read_prn(prn)
        elif isinstance(prn, pd.DataFrame):
            prn_df = prn
        else:
            raise ValueError(f"ERROR : could not write simulated file for locnme = {self.locnme}.")

        # ---- Extract and write simulated value(s)
        pest_utils.extract_prn( prn_df,
                                name= self.locnme,
                                dates_out= self.date,
                                fluc_dic= self.fluc_dic,
                                sim_dir= os.path.split(self.simfile)[0] )




    def to_config(self):
        """
        Convert MartheObs main informations to string.
        """
        lines = ['[START_OBS]']
        data = [
            'locnme= {}'.format(self.locnme),
            'datatype= {}'.format(self.datatype),
            'trans= {}'.format(self.trans),
            'fluc_dic= {}'.format(str(self.fluc_dic)),
            'interp_method= {}'.format(self.interp_method),
            'dates_out= {}'.format('|'.join([str(v) for v in self.date]))
              ]
        lines.extend(data)
        lines.append('[END_OBS]')

        return '\n'.join(lines)



    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheObs'