
"""
Contains the MartheOptim class
Designed for structured grid
"""

import os, sys
import numpy as np
import pandas as pd 
import pyemu
from pymarthe.marthe import MartheModel
from pymarthe.mobs import MartheObs
# from pymarthe.mparam import MartheParam
from .utils import marthe_utils, pest_utils


# ---- Set no data customs values
NO_DATA_VALUES = [-9999.,-8888.]

# ----- Set columns informations for observations
base_obs = ['datatype', 'locnme', 'obsval',
            'date', 'obsfile', 'obgnme','obsnme','weight']

# ----- Set columns informations for parameters
base_param = []

# ---- Set encoding 
encoding = 'latin-1'


class MartheOptim():
    """
    Wrapper Python --> PEST.
    Interface to prepare PEST usefull files such as
    instruction, template, control from a Marthe model
    and other external data.
    """
    def __init__(self, mm, name = None):
        """
        Parameters
        ----------
        mm (MartheModel): parent MartheModel instance
                          to interface with PEST utilities.
        name (str) : name of optimisation process.

        Examples
        --------
        mm = MartheModel('/Users/john/models/model/mymodel.rma')
        moptim = MartheOptim(mm)

        """
        # ---- Verify that mm is a MartheModel instance
        err_msg = ' ERROR : mm must be a MartheModel instance.' \
                  f'Given : {mm}'
        assert isinstance(mm, MartheModel), err_msg

        # ---- Set arguments as atributes
        self.mm = mm
        self.name = f'{mm.mlname}_optim' if name is None else name
        # ---- Initialize observation and parameters
        self.obs, self.param = {}, {}
        self._base_obs_df, self.obs_df = [pd.DataFrame(columns = base_obs)]*2
        self._base_param_df, self.param_df = [pd.DataFrame(columns = base_param)]*2
        #self.props = ['permh', 'emmca', 'emmli','kepon', 'aqpump', 'rivpump']
        # ---- Fetch available observation localisation names
        self.available_locnmes = marthe_utils.read_histo_file(mm.mlfiles['histo']).index
        # ---- Set commun no data values
        self.nodata = NO_DATA_VALUES



    def get_nobs(self, locnme = None, null_weight = True):
        """
        Function to fetch number of observation
        stored in MartheOptim instance.

        Parameters:
        ----------
        self : MartheObs instance
        locnme (str, optinonal) : name of a set of observation
        null_weight (bool, optional) : consider observation with null weight.
                                       Default is True

        Returns:
        --------
        nobs (int) : number of observation

        Examples:
        --------
        print(f"There are {moptim.get_nobs()} observations.")
        """
        # ---- Subset by locnme if required
        if locnme is None:
            df = self.obs_df
        else:
            df = self.obs_df.query(f"locnme == '{locnme}'")
        # ---- Ignore null weight observation if required
        if not null_weight:
            nobs = len(df.query("weight != 0"))
        else:
            nobs = len(df)
        # ---- Return
        return nobs


    def get_nlocs(self, datatype = None):
        """
        Function to fetch number of set of 
        observation stored in MartheOptim instance.

        Parameters:
        ----------
        datatype (str, optional) : required data type

        Returns:
        --------
        nlocs (int) : number of set of observation

        Examples:
        --------
        print(f"There are {moptim.get_nlocs()} observations.")
        """
        if datatype is None:
            df = self.obs_df
        else:
            df = self.obs_df.query(f"datatype == '{datatype}'")
        # ---- Return
        return len(df['locnme'].unique())



    def get_ndatatypes(self):
        """
        Function to fetch number of observation data types stored in MartheObs instance

        Parameters:
        ----------
        self : MartheObs instance

        Returns:
        --------
        ndatatypes (int) : number of observation data types

        Examples:
        --------
        ndt = moptim.get_ndatatypes()
        print(f"There are {ndt} observation data types")
        """
        return len(self.obs_df['datatype'].unique())




    def check_loc(self, locnme, error = 'raise'):
        """
        Check existence and uniqueness of a given locnme.

        Parameters:
        ----------
        locnme (str) : observation localisation
                       name to test
        error (str) : error type to handle.
                      Can be :
                      - 'raise': assertions.
                      - 'silent':  boolean return.
                      - 'off': inactive.
                      Default is 'raise'.

        Returns:
        --------
        exi (bool) : locnme validity/existence
        uni (bool) : locnme unicity

        Examples:
        --------
        moptim = MartheOptim(mm, 'opti')
        locnme = 'mylocname'
        exi, uni = moptim.check_loc(locnme)
        """
        # ---- Get existence  if required
        exi = locnme in self.available_locnmes
        # ---- Get unicity
        mask = self.available_locnmes.str.contains(locnme).tolist()
        uni = mask.count(True) <= 1
        # ---- Error handling
        if error == 'raise':
            hf = self.mm.mlfiles['histo']
            assert exi, f"ERROR: '{locnme}' not in {hf}."
            assert uni, f"ERROR: '{locnme}' set multiple times in" \
                           f"{hf}: each locnme must be unique."
        # ---- Return boolean for silent check
        elif error == 'silent':
            return exi, uni
        # ---- Empty return for inactive check
        elif error == 'off':
            return 



    def add_obs(self, datatype, obsfile, locnme = None, check_loc = True, nodata = None, **kwargs):
        """
        Add and set observations 

        Parameters:
        ----------
        datatype (str): data type of observation values.
        obsfile (str): observation filename to read value.
                       Note: if loc_name is not provided,
                       the loc_name is set as obsfile without file extension.
        locnme (str, optional) : observation location name (ex. BSS id)
        check_loc (bool, optional) : check loc_name existence and unicity
                                     Default is True.
        nodata (list/None, optional) : no data values to remove reading observation data.
                                       If None, all values are considered.
                                       Default is None.
                                       NOTE: can create issues with incomplete
                                       series sim/obs mismatch.
        
        obgnme (str, kwargs): name of the group of related observation.
                              Default is locnme.
        obnme (list, kwargs): custom observation names
                               Default build as 'loc{loc_name_id}n{obs_id}'
        weight (list, kwargs): weight per each observations

        Returns:
        --------
        Add set of observation inplace.

        Examples:
        --------
        moptim.add_obs(data_type = 'head', obsfile = 'myobs.dat')

        """
        # ---- Manage locnme
        obs_dir, obs_filename = os.path.split(obsfile)
        if locnme is None:
            locnme = obs_filename.split('.')[0]

        # ---- Check validity and uncity of locnme
        if check_loc:
            self.check_loc(locnme)

        # ---- Build MartheObs instance
        mobs = MartheObs(datatype, obsfile, self.get_nlocs(), locnme, nodata, **kwargs)
        self.obs[locnme] = mobs

        # ---- Update main observation DataFrame
        obs_dfs = [self.obs_df, mobs.obs_df]
        self.obs_df = pd.concat(obs_dfs).drop_duplicates()

        # # ---- Verbose
        # print(f"Observation '{locnme}' had been added successfully.")




    def remove_obs(self, locnme=None):
        """
        Delete observation(s) provided observations

        Parameters:
        ----------
        locnme (str, optional) : observation location name (ex. BSS id).
                                 If None, all locnmes will be removed.
                                 Default is None.

        Returns:
        --------
        Delete observation(s) by locnme in
        observation dictionary and DataFrame.

        Examples:
        --------
        moptim.delete_obs('p31.dat')
        """
        if locnme is None:
            # ---- Load (again) empty DataFrame
            self.obs_df = self._base_obs_df
            self.obs = {}
            print('All provided observations had been removed successfully.')
        else:
            del self.obs[locnme]
            self.obs_df = self.obs_df.query(f"locnme != '{locnme}'")
            print(f"Observation '{locnme}' had been removed successfully.")



    def write_ins(self, locnme=None, ins_dir = '.'):
        """
        Write formatted instruction file (pest).
        Wrapper of pest_utils.write_insfile().

        Parameters:
        ----------
        locnme (str, optional) : observation location name (ex. BSS id)
                                 If None all locnmes are considered.
                                 Default is None
        ins_dir (str, optional) : directory to write instruction file(s)
                                  Default is '.'

        Returns:
        --------
        Write insfile file in ins_dir

        Examples:
        --------
        mm.mobs.write_insfile(locnme = 'myobs', ins_dir = 'ins')
        """
        # ---- Manage single locnme writing
        locnmes = list(self.obs.keys()) if locnme is None else [locnme]
        # ---- Iterate over locnmes
        for locnme in locnmes:
            self.obs[locnme].write_ins(ins_dir=ins_dir)





    def add_fluc(self, locnme=None, on = 'mean', nodata= None):
        """
        Add fluctuations to a existing observation set.

        Parameters:
        ----------
        locnme (str/list, optional) : observation location name(s) (ex. BSS id)
                                        If locnme is None, all locnmes are considered
                                        Default is None
        on (str/numeric/fun, optional) : function, function name or real number to substract
                                         to the existing observation values.
                                         Function names can be 'min', 'max', 'mean', 'std', etc. 
                                         See pandas.core.groupby.GroupBy documentation for more.
        nodata (list/None, optional) : no data values to remove.
                                       Default is None.
                                       NOTE : Can create issues with incomplete
                                       series sim/obs mismatch.

        Returns:
        --------
        Write and add a new set of observation as
        a fluctuation of a existing one.

        Examples:
        --------
        moptim.add_fluc(locnme = ['obs1', 'obs2'], on = 'median')
        """
        # ---- Manage locnme(s)
        if locnme is None:
            locnmes = list(self.obs.keys())
        else:
            locnmes = locnme if marthe_utils.isiterable(locnme) else [locnme]

        # ---- Avoid multiple fluctuation calculation
        locnmes = [ln for ln in locnmes if not ln.endswith('_fluc')]

        # ----- Iterate over locnmes
        for ln in locnmes:
            # ---- Fetch original observation info
            source_file = self.obs[ln].obsfile
            source_datatype = self.obs[ln].datatype
            # ---- Read original obsfile
            obs_df = marthe_utils.read_obsfile(source_file, nodata = nodata)
            # ---- Infer fluctuation manipulation to perform
            s = obs_df['value'].replace(self.nodata, pd.NA)  # replace nodata values by NaN
            sub_val = s.agg(on) if isinstance(on, str) else on      # value to subtract    
            # ---- Get fluctuation by substraction
            fluc_vals = [x - sub_val if not x in self.nodata else x for x in obs_df['value']]
            # ---- Set fluctuation obsfile, datatype
            fluc_obsfile = source_file.replace(ln, f'{ln}_fluc')
            fluc_datatype = f'{source_datatype}_fluc'
            # ---- Write fluctuation data
            marthe_utils.write_obsfile(date = obs_df.index, value = fluc_vals, obsfile = fluc_obsfile)
            # ---- Add fluctuation observation
            self.add_obs(datatype = fluc_datatype, obsfile = fluc_obsfile, check_loc = False)




    def compute_weights(self, lambda_dic=None, sigma_dic=None):
        """
        Compute weigths for all observations.

        -----------
        Parameters
        -----------
        - lambda_dic (dict) : tuning factor dictionary
                              Format: {datatype0 : lambda_0, ... datatypeN : lambda_N}
                              If None, all tuning factors are set to 1.
                              Default is None.
        - sigma_dic (dict) : expected variance (dictionary) between obs/sim data
                             Format: {datatype0 : sigma_0, ... datatypeN : sigma_N}
                             If None, all variaces are set to 1.
                             Default is None
        -----------
        Returns
        -----------
        Set weights in self.mobs_df['weight'] column.
        -----------
        Examples
        -----------
        mm.mobs.compute_weights(lambda_dic)
        """
        # ----- Verify at least 1 locnme exist
        msg = f'ERROR : no observations provided yet. Use .add_obs() function.'
        assert len(self.obs_df) > 0, msg
        # ---- Build default dictionary
        default_dic = {dt: 1 for dt in self.obs_df['datatype'].unique()}
        # ---- Set tuning factor dictionary
        if lambda_dic is None:
            lambda_dic = default_dic
        lambda_n = sum(lambda_dic.values())
        # ---- Set variance dictionary
        if sigma_dic is None:
            sigma_dic = default_dic
        # ---- Iterate over data types
        for datatype in default_dic.keys():
            # -- Get number of observation sets for a given data type
            dt_df = self.obs_df.query(f"datatype == '{datatype}'")
            m = self.get_nlocs(datatype)
            # ---- Iterate over locnmes
            for locnme in dt_df['locnme'].unique():
                # -- Get number of observations for a given set of observation
                n = self.get_nobs(locnme)
                # -- Compute weights
                w = pest_utils.compute_weight(lambda_dic[datatype], lambda_n, m, n, sigma_dic[datatype])
                # -- Set computed weights
                mask = (self.obs_df.datatype == datatype) & (self.obs_df.locnme == locnme)
                self.obs_df.loc[mask, 'weight'] = w
                self.obs[locnme].obs_df['weight'] = w
                self.obs[locnme].weight = w





    def add_param(self):
        """
        *** UNDER DEVELOPMENT ***
        """
        return


    def build_pst(self):
        """
        *** UNDER DEVELOPMENT ***
        """
        return


    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheOptim'