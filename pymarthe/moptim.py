
"""
Contains the MartheOptim class
Designed for structured grid
"""

import os, sys
import numpy as np
import pandas as pd 
import pyemu
import warnings
from datetime import datetime

from pymarthe.marthe import MartheModel
from pymarthe.mobs import MartheObs
from pymarthe.mparam import MartheListParam, MartheGridParam
from .utils import marthe_utils, pest_utils


# ---- Set no data customs values
NO_DATA_VALUES = [-9999.,-8888.]


base_obs = ['obsnme', 'date', 'obsval',
            'datatype', 'locnme', 'obsfile',
            'weight', 'obgnme', 'trans' ]

base_param = ['parnme', 'trans', 'btrans', 'parchglim',
              'defaultvalue', 'parlbnd', 'parubnd',
              'pargp', 'scale', 'offset', 'dercom']


# ---- Set encoding 
encoding = 'latin-1'


class MartheOptim():
    """
    Wrapper Python --> PEST.
    Interface to prepare PEST usefull files such as
    instruction, template, control from a Marthe model
    and other external data.
    """
    def __init__(self, mm, name = None, **kwargs):
        """
        Parameters
        ----------
        mm (MartheModel): parent MartheModel instance
                          to interface with PEST utilities.

        name (str, optional) : name of optimisation process.

        kwargs : some paths to usefull organized folder such as:
                    - `par_dir` : parameter folder
                    - `tpl_dir` : template folder
                    - `ins_dir` : instruction folder
                    - `sim_dir` : simulation folder
                    - `obs_dir` : observation folder

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
        # ---- Get model time window bounds from .mart file
        self.tw_min, self.tw_max = self.mm.get_time_window()
        # ---- Initialize observation and parameters
        self.obs, self.param = {}, {}
        # ---- Fetch available observation localisation names
        self.available_locnmes = marthe_utils.read_histo_file(mm.mlfiles['histo']).index
        # ---- Set commun no data values
        self.nodata = NO_DATA_VALUES
        # ---- Set parameter and observation folder
        self.par_dir = kwargs.get('par_dir', '.')
        self.tpl_dir = kwargs.get('tpl_dir', '.')
        self.ins_dir = kwargs.get('ins_dir', '.')
        self.sim_dir = kwargs.get('sim_dir', '.')
        self.obs_dir = kwargs.get('obs_dir', '.')



    def get_obs_df(self, transformed=False):
        """
        Get all observations information in a large DataFrame.

        Parameters:
        ----------
        transformed (bool, optional) : whatever apply transformation on output DataFrame.
                                       Default is False.

        Returns:
        --------
        obs_df (DataFrame) : merged provided observations

        Examples:
        --------
        moptim.get_obs_df(transformed=True)
        """
        if len(self.obs) > 0:
            return pd.concat([mo.get_obs_df(transformed)for mo in self.obs.values()])
        else:
            return pd.DataFrame(columns = base_obs)




    def get_param_df(self, transformed=False):
        """
        Get all parameters informations in a large DataFrame.

        Parameters:
        ----------
        transformed (bool, optional) : whatever apply transformation(s) on
                                       parameter data sets.
                                       Default is False.

        Returns:
        --------
        param_df (DataFrame) : merged provided parameters

        Examples:
        --------
        moptim.get_param_df(transformed=True)
        """
        if len(self.param) > 0:
            return pd.concat([mp.get_param_df(transformed) for mp in self.param.values()])
        else:
            return pd.DataFrame(columns = base_param)




    def get_nobs(self, locnme = None, null_weight = True):
        """
        Function to fetch number of observation
        stored in MartheOptim instance.

        Parameters:
        ----------
        locnme (str/it, optinonal) : name of a set of observation
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
        _ln = list(self.obs.keys()) if locnme is None else marthe_utils.make_iterable(locnme)
        # ---- Start counting locnames
        nobs = 0
        # ---- Iterate over all MartheObs instance
        for mo in self.obs.values():
            nw_cond = True if null_weight else mo.weight != 0
            if (mo.locnme in _ln) & (nw_cond):
                nobs += len(mo.obs_df)
        # ---- Return
        return nobs



    def get_nlocs(self, datatype = None):
        """
        Function to fetch number of set of 
        observation stored in MartheOptim instance.

        Parameters:
        ----------
        datatype (str/it, optional) : required data type

        Returns:
        --------
        nlocs (int) : number of set of observation

        Examples:
        --------
        print(f"There are {moptim.get_nlocs()} observations.")
        """
        # ---- Subset by locnme if required
        _dt = [mo.datatype for mo in self.obs.values()] if datatype is None else marthe_utils.make_iterable(datatype)
        # ---- Start counting locnames
        nlocs = 0
        # ---- Iterate over all MartheObs instance
        for mo in self.obs.values():
            if mo.datatype in _dt:
                nlocs += 1
        # ---- Return number of locnames with required datatype
        return nlocs



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
        dt = [mo.datatype for mo in self.obs.values()]
        return len(set(dt))




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
        # ---- Get existence if required
        exi = locnme in self.available_locnmes
        # ---- Get unicity
        mask = self.available_locnmes.str.contains(locnme).tolist()
        uni = mask.count(True) <= 1
        # ---- Error handling
        if error == 'raise':
            hf = self.mm.mlfiles['histo']
            assert exi, f"ERROR: '{locnme}' not in {hf}."
            assert uni, f"ERROR: '{locnme}' set multiple times in " \
                           f"{hf}: each locnme must be unique."
        # ---- Return boolean for silent check
        elif error == 'silent':
            return exi, uni
        # ---- Empty return for inactive check
        elif error == 'off':
            return 



    def add_obs(self, data, locnme = None, datatype = 'head',
                      check_loc = True, nodata = None, **kwargs):
        """
        Add and set observations 

        Parameters:
        ----------
        
        data (object): observation data.
                       Can be a path to a observation file to read
                       (wrapper to marthe_utils.read_obsfile()) or a 
                       pandas DataFrame with a column `value` and a 
                       pd.DatatimeIndex as index.
                       Note: if loc_name is not provided,
                       the loc_name is set as obsfile without file extension.

        locnme (str, optional) : observation location name (ex. BSS id)

        datatype (str, optional): data type of observation values.
                                  Default is 'head'.

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

        trans (str/func, kwargs) : keyword/function to use for transforming 
                                       observation values.
                                       Can be:
                                        - function (np.log10, np.sqrt, ...)
                                        - string function name ('log10', 'sqrt')

        Returns:
        --------
        Add set of observation inplace.

        Examples:
        --------
        moptim.add_obs(data = 'obs/07065X0002.dat')

        """
        # ---- Manage external observation file as data input
        if isinstance(data, str):
            # -- Set observation filename
            obsfile = data
            obs_dir, obs_filename = os.path.split(obsfile)
            # -- Get locnme if not provided
            if locnme is None:
                locnme = obs_filename.split('.')[0]
            # -- Read observation file as DataFrame
            df = marthe_utils.read_obsfile(obsfile, nodata = nodata)

        # ---- Manage internal DataFrame as data input
        elif isinstance(data, pd.DataFrame):
            # -- Set observation filename
            obsfile = None
            # -- Get DataFrame
            df = data
            # -- Perform some checks (date and value)
            err_msg = 'ERROR: `data` must contain a `value` column and a DatetimeIndex index.'
            assert ('value' in data.columns) & isinstance(df.index, pd.DatetimeIndex), err_msg
            # -- Get locnme if not provided
            locnme = f'loc{str(self.get_nlocs()).zfill(3)}' if locnme is None else locnme
        
        # ---- Avoid adding same locnme multiple times
        if locnme in self.obs.keys():
            # -- Raise warning message
            warn_msg = f'WARNING : locnme `{locnme}` already added. ' \
                       'It will be overwrited.'
            warnings.warn(warn_msg, Warning)
            # -- Remove existing set of observation
            self.remove_obs(locnme)

        # ---- Check validity and uncity of locnme
        if check_loc:
            self.check_loc(locnme)

        # ---- Avoid observations out of model actual time window
        df_tw = df.loc[self.tw_min:self.tw_max]

        if df_tw.empty:
            # -- Raise warning message
            warn_msg = 'WARNING : no observation data within the model time window was found. ' \
                       f'{locnme} observation set will not be added.'
            warnings.warn(warn_msg)

        if len(df) > len(df_tw):
            # -- Raise warning message
            warn_msg = f'WARNING : some observation data of `locnme` = {locnme} ' \
                       f'are out of actual model time window ({self.tw_min} - {self.tw_max}). ' \
                       'They will not be considered.'
            warnings.warn(warn_msg)

        # ---- Build MartheObs instance from data input
        insfile = kwargs.pop('insfile', os.path.join(self.ins_dir, f'{locnme}.ins'))
        simfile = kwargs.pop('simfile', os.path.join(self.sim_dir, f'{locnme}.dat'))
        mobs = MartheObs(iloc = self.get_nlocs(),
                         locnme = locnme,
                         date = df_tw.index,
                         value = df_tw['value'].values,
                         obsfile = obsfile,
                         insfile = insfile,
                         simfile = simfile,
                         datatype = datatype,
                         **kwargs)

        # ---- Add MartheObs to main observation dictionary
        self.obs[locnme] = mobs




    def remove_obs(self, locnme=None, verbose=False):
        """
        Delete provided observation(s) 

        Parameters:
        ----------
        locnme (str/it, optional) : observation location name (ex. BSS id).
                                 If None, all locnmes will be removed.
                                 Default is None.
        verbose (bool) : print message about deleted observation(s)

        Returns:
        --------
        Delete observation(s) by locnme in
        observation dictionary.

        Examples:
        --------
        moptim.delete_obs('p31')
        """

        if locnme is None:
            self.obs = {}
            if verbose:
                print('All provided observations had been removed successfully.')
        else:
            for ln in marthe_utils.make_iterable(locnme):
                del self.obs[ln]
                if verbose:
                    print(f"Observation `{ln}` had been removed successfully.")


    def remove_param(self, parname=None, verbose=False):
        """
        Delete provided parameter(s).

        Parameters:
        ----------
        parname (str/it, optional) : parameter name(s) (ex. 'p31').
                                 If None, all locnmes will be removed.
                                 Default is None.
        verbose (bool) : print message about deleted parameter(s)

        Returns:
        --------
        Delete parameters(s) by parname in
        parameter dictionary.

        Examples:
        --------
        moptim.delete_param('p31')
        """

        if parname is None:
            self.param = {}
            if verbose:
                print('All provided parameters had been removed successfully.')
        else:
            for par in marthe_utils.make_iterable(parname):
                del self.param[par]
                if verbose:
                    print(f"Parameter `{par}` had been removed successfully.")




    def set_obs_trans(self, trans, datatype=None, locnme=None):
        """
        Set transformation keyword to observations values.

        Parameters:
        ----------
        trans (str/func) : keyword/function to use for transforming 
                                       observation values.
                                       Can be:
                                        - function (np.log10, np.sqrt, ...)
                                        - string function name ('log10', 'sqrt')

        datatype (str, optional): data type of observation values.
                        Default is 'head'.

        locnme (str, optional) : observation location name (ex. BSS id)

        Returns:
        --------
        Set `trans` argument for required observations.

        Examples:
        --------
        moptim.set_trans('log10', datatype=['head', 'flow'])

        """
        # -- Check transformations validity
        pest_utils.check_trans(trans)

        # ---- Get datatype, locnme required as iterable
        _dt = set([mo.datatype for mo in self.obs.values()]) if datatype is None else marthe_utils.make_iterable(datatype)
        _ln = self.obs.keys() if locnme is None else marthe_utils.make_iterable(locnme)

        # ---- Set transformation to all required data
        for mo in self.obs.values():
            if (mo.datatype in _dt) & (mo.locnme in _ln):
                self.obs[mo.locnme].trans = trans
                self.obs[mo.locnme].obs_df['trans'] = trans




    def set_param_trans(self, trans, btrans, parname=None, pargp=None):
        """
        Set transformation keyword to parameters values.

        Parameters:
        ----------
        trans (str) : keyword/function to use for transforming 
                            parameter values.
        btrans (str) : keyword/function to use for back transforming 
                            parameter values.

        parname (str/it, optional): parameter name.
                                 If None, all parameters will be consider.
                                 Default is None.

        pargp (str/it, optional) : parameter group.
                                   If None, all parameter groups will be consider.
                                   Default is None.

        Returns:
        --------
        Set `trans` and `btrans` argument for required parameter values.

        Examples:
        --------
        moptim.set_param_trans('log', 'np.exp', pargp = 'pump')

        """
        # -- Check transformations validity
        pest_utils.check_trans(trans)
        pest_utils.check_trans(btrans)

        # ---- Get parameter names and groups as iterable
        _pn = self.param.keys() if parname is None else marthe_utils.make_iterable(parname)
        _pg = set([mp.pargp for mp in self.param.values()]) if pargp is None else marthe_utils.make_iterable(pargp)

        # ---- Set transformation to all required data
        for mp in self.param.values():
            if (mp.parname in _pn) & (mp.pargp in _pg):
                self.param[mp.parname].trans  = trans
                self.param[mp.parname].btrans = btrans
                if mp.type == 'list':
                    self.param[mp.parname].param_df[['trans', 'btrans']] = [trans, btrans]




    def write_insfile(self, locnme=None):
        """
        Write formatted instruction file in instruction directory (`.ins_dir`).
        Wrapper of pest_utils.write_insfile().

        Parameters:
        ----------
        locnme (str, optional) : observation location name (ex. BSS id)
                                 If None all locnmes are considered.
                                 Default is None

        Returns:
        --------
        Write insfile file in ins_dir

        Examples:
        --------
        moptim.write_insfile(locnme = 'myobs')
        """
        # ---- Manage multiple locnme input
        if locnme is None:
            locnmes = self.obs.keys()
        else:
            locnmes = marthe_utils.make_iterable(locnme)
            # -- Check locnames validity
            not_found = [ln for ln in locnmes if ln not in self.obs.keys()]
            err_msg = "ERROR : Some provided `locnme` not added yet: {}.".format(', '.join(not_found))
            assert len(not_found) == 0, err_msg

        # ---- Iterate over locnmes
        for ln in locnmes:
            self.obs[ln].write_insfile()

            

    def write_simfile(self, locnme=None, prnfile=None):
        """
        Write formatted simulated file in simulate directory (`.sim_dir`).
        Wrapper of pest_utils.extract_prn().
        Note: to write the related simulated values don't 
              forget to (re)run the Marthe model before.

        Parameters:
        ----------
        locnme (str, optional) : observation location name (ex. BSS id)
                                 If None all locnmes are considered.
                                 Default is None.

        prnfile (str, optional) : path to the simulated value file.
                                  If None, prnfile = model_path + 'historiq.prn'.
                                  Default is None.

        Returns:
        --------
        Write simfile file in `.ins_dir`

        Examples:
        --------
        moptim.write_simfile(locnme = 'myobs')
        """
        # ---- Manage multiple locnme input
        if locnme is None:
            locnmes = self.obs.keys()
        else:
            locnmes = marthe_utils.make_iterable(locnme)
            # -- Check locnames validity
            not_found = [ln for ln in locnmes if ln not in self.obs.keys()]
            err_msg = "ERROR : Some provided `locnme` not added yet: {}.".format(', '.join(not_found))
            assert len(not_found) == 0, err_msg

        # ---- Get simalated value as DataFrame
        prnfile = os.path.join(self.mm.mldir, 'historiq.prn') if prnfile is None else prnfile
        prn_df = marthe_utils.read_prn(prnfile)

        # ---- Iterate over locnmes
        for ln in locnmes:
            self.obs[ln].write_simfile(prn_df)




    def add_fluc(self, locnme=None, tag= '', on = 'mean'):
        """
        Add fluctuations to a existing observation set.

        Parameters:
        ----------
        locnme (str/list, optional) : observation location name(s) (ex. BSS id)
                                        If locnme is None, all locnmes are considered
                                        Default is None

        tag (str, optional) : additional string to precise the type of fluctuation.
                              locnme build as locnme + tag + 'fluc'.
                              Default is ''.

        on (str/numeric/fun, optional) : function, function name or real number to substract
                                         to the existing observation values.
                                         Function names can be 'min', 'max', 'mean', 'std', etc. 
                                         See pandas.core.groupby.GroupBy documentation for more.

        Returns:
        --------
        Write and add a new set of observation as
        a fluctuation of a existing one.

        Examples:
        --------
        moptim.add_fluc(locnme = ['obs1', 'obs2'], tag = 'md', on = 'median')
        """
        # ---- Manage locnme(s)
        if locnme is None:
            locnmes = list(self.obs.keys())
        else:
            locnmes = locnme if marthe_utils.isiterable(locnme) else [locnme]

        # ---- Avoid multiple fluctuation calculation
        locnmes = [ln for ln in locnmes if not ln.endswith('fluc')]

        # ----- Iterate over locnmes
        for ln in locnmes:
            # -- Avoid non existing locnmes
            if ln in self.obs.keys():
                # ---- Define new locnme
                new_locnme = ln + tag + 'fluc'
                # ---- Get DataFrame of the source observation
                df = self.obs[ln].obs_df.set_index('date').rename({'obsval':'value'}, axis=1)
                # ---- Infer fluctuation manipulation to perform
                s = df['value'].replace(self.nodata, pd.NA)  # replace nodata values by NaN
                sub_val = s.agg(on) if isinstance(on, str) else on
                # ---- Get fluctuation by substraction
                df['value'] = [x - sub_val if not x in self.nodata else x for x in df['value']]
                # ---- Add fluctuation observation
                new_dt = self.obs[ln].datatype + tag + 'fluc'
                self.add_obs(data = df, locnme = new_locnme,
                             datatype = new_dt, check_loc = False,
                             fluc_dic = {'tag':tag,'on':on})
            else:
                warn_msg = f"WARNING : could not found observation with `locnme` = {ln}. " \
                           "Fluctuation observation set will not be added."
                warnings.warn(warn_msg)




    def compute_weights(self, lambda_dic=None, sigma_dic=None):
        """
        Compute weigths for all observations.

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

        Returns
        -----------
        Set weights in self.mobs_df['weight'] column.

        Examples
        -----------
        mm.mobs.compute_weights(lambda_dic)
        """
        # ----- Verify at least 1 locnme exist
        msg = f'ERROR : no observations provided yet. Use .add_obs() function.'
        assert len(self.obs) > 0, msg
        # ---- Extract actual observation DataFrame
        obs_df = self.get_obs_df()
        # ---- Build default dictionary
        default_dic = {dt: 1 for dt in obs_df['datatype'].unique()}
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
            dt_df = obs_df.query(f"datatype == '{datatype}'")
            m = self.get_nlocs(datatype=datatype)
            # ---- Iterate over locnmes
            for locnme in dt_df['locnme'].unique():
                # -- Get number of observations for a given set of observation
                n = self.get_nobs(locnme=locnme)
                # -- Compute weights
                w = pest_utils.compute_weight(lambda_dic[datatype], lambda_n, m, n, sigma_dic[datatype])
                # -- Set computed weights
                self.obs[locnme].weight = w
                self.obs[locnme].obs_df['weight'] = w





    def add_param(self, parname, mobj, **kwargs):
        """
        Add a parameter a set of parameters.

        Parameters can be either:

            - distributed ('grid'): based on `izone` object which represent a pointer
                                    to a Marthefield instance where field values 
                                    are gathered in zones like:
                                    - izone < 0  ==> zone of piecewise constancy
                                    - izone > 0  ==> zone with pilot points
                                    - izone = -9999, 0, 9999  ==> inactive zone

            - globals ('list'): based on `kmi` (KeysMultiIndex) which represent the 
                                referenced DataFrame columns names to be included in
                                the parametrization.
                                Note: the built-in fonction pymarthe.utils.pest_utils.get_kmi()
                                      make the `kmi` generation easier.

        Parameters
        -----------
        - parname (str) : name of the set of parameters.

        - mobj (str) : Marthe object to parametrize.
                       Can be a:
                            - MartheField instance (grid)
                            - MartheSoil instance  (list)
                            - MarthePump instance  (list)


        Examples
        -----------
        # -- Example on pumping data
        mm.load_prop('soil')
        mp = mm.prop['aqpump']
        kmi_p31 = pest_utils.get_kmi(mp, keys = ['boundname', 'layer', 'istep'], boundname = 'p31')
        mgp.add_param(parname='p31', mobj=mp,
                      kmi=kmi_p31, defaultvalue=-1e-3)

        # -- Example on soil data
        mm.load_prop('soil')
        ms = mm.prop['soil']
        kmi_soil = pest_utils.get_kmi(ms, keys=['soilprop', 'zone'], istep=0)
        mopt.add_param(parname='soil', mobj=ms, kmi=kmi_soil)

        # -- Example on distributed parameter
        mm.load_prop('permh')
        mf = mm.prop['permh']
        pp_data = {1:{1:'gis/pp_l01.shp'}, 2:{1:'gis/pp_l02.shp'}}
        mopt.add_param(parname='hk', mobj=mf,
                       izone='mymodel.ipermh',
                       pp_data=pp_data,
                       trans='log10',
                       btrans='lambda x: 10**x')

        """
        # ---- Avoid adding same parameter multiple times
        if parname in self.param.keys():
            # -- Raise warning message
            warn_msg = f'WARNING : parname `{parname}` already added. ' \
                       'It will be overwrited.'
            warnings.warn(warn_msg, Warning)
            # -- Remove existing parameter
            self.remove_param(parname)

        # ---- Set paths to template and parameter files 
        tplpath = kwargs.pop('tplpath', self.tpl_dir)
        parpath = kwargs.pop('parpath', self.par_dir)

        # ---- Manage grid parameters
        if mobj._proptype == 'grid':

            # -- Build a MartheGridParam instance
            par = MartheGridParam(parname= parname,
                                  mobj= mobj,
                                  tplpath= tplpath,
                                  parpath= parpath,
                                  **kwargs)

        # ---- Manage list parameters
        elif mobj._proptype == 'list':

            # -- Get KeysMultiIndex and value column name
            kmi = kwargs.pop('kmi', None)
            value_col = kwargs.pop('value_col', 'value')

            # -- Bunch of assertion to avoid invalid inputs
            err_kmi = 'ERROR : list-like parameter require a KeysMultiIndex (`kmi`) argument. ' \
                       'See pymarthe.utils.pest_utils.get_kmi() function.'
            err_value_col = f'ERROR : invalid `value_col` argument. '\
                          f'Must be a column name of {str(mobj)} object. ' \
                          f'Given : {value_col}.'
            assert kmi is not None, err_kmi
            assert value_col in mobj.data.columns, err_value_col

            # -- Build a MartheListParam instance
            par = MartheListParam(parname= parname,
                                  mobj= mobj,
                                  kmi=kmi,
                                  value_col=value_col,
                                  parpath = parpath,
                                  tplpath = tplpath,
                                  **kwargs)

        # ---- Store parameter in dictionary
        self.param[parname] = par




    def write_kriging_factors(self, vgm_range, parname=None, krig_transform= 'none', save_cov = False ):
        """
        Compute and write kriging factor files (PEST-like) from exponential variogram
        ranges for given distributed parameters.
        Wrapper to MartheGridParam.write_kfac().

        Note: the ranges must be in the same distance unit as the model fields.


        Parameters
        -----------
        vgm_range (float/int/dict/nested dict) : exponential variagram(s) range(s).
                                                 Can be :
                                                    - numeric
                                                    - dictionary 
                                                        format: {layer_0 : range_0,
                                                                 ...,
                                                                 layer_i : range_i }
                                                    - nested dictionary
                                                        format: {layer_0 : {zone_0: range_0_0, ..., zone_i: range_0_i},
                                                                 ...,
                                                                 layer_i : {zone_0: range_i_0, ..., zone_i: range_i_i} }

        parname (str, optional) : parameter name already added.
                                  If None, all distributed parameters will be considered.
                                  Default is None.

        krig_transform (str, optional): transformation to apply to the 
                                        pyemu.utils.geostats.GeoStruct.
                                        Can be:
                                            - 'none'
                                            - 'log'
                                        Default is 'none'.

        save_cov (bool, optional) : whatever write the covariance matrices as binary files.
                                    Default is False.
                                    Note: the covariance matrices files will take the same
                                          names as kriging factor files with the '.jcb' extension.

        Returns
        -------
        Write kriging factor file in parameter path (with '.fac' extension).
        If save_cov is True, the covariance matrix will be written in
        parameter path too (with extension '.jcb').

        Examples
        --------
        vgm_range= {2: {1:100}, 3: {1:200,2:150}}
        mopt.write_kriging_factors(vgm_range, parname='hk', vgm_transform= 'log',  save_cov=True)

        """
        # ---- Manage parname inputs
        if parname is None:
            parnames = [pn for pn in self.param.keys() if self.param[pn].type == 'grid']
        else:
            parnames = marthe_utils.make_iterable(parname)
            # ---- Bunch of assertions to avoid bad inputs
            not_found = [pn for pn in parnames if pn not in self.param.keys()]
            err_msg = " ERROR : Some provided parameter names not added yet : " \
                      "{}.".format(', '.join([f"'{pn}'" for pn in not_found]))
            assert len(not_found) == 0, err_msg
            not_grid = [pn for pn in parnames if not self.param[pn].type == 'grid']
            err_msg = " ERROR : Some provided parameter are not distributed (grid type) : " \
                      "{}.".format(', '.join([f"'{pn}'" for pn in not_grid]))

        # ---- Iterate over parmeters to use the internal method .write_kfac()
        for pn in parnames:
            self.param[pn].write_kfac(vgm_range, krig_transform = krig_transform, save_cov = False)





    def write_config(self, filename=None):
        """
        Write a standard text (.config) with the essential informations 
        about current parametrisation. The file is organised in 3 parts:

            - Headers : general paths to Marthe model, spatial index, statistics
                        about observations and parameters sets, ...

            - Observation sections : blocks of informations about observation data. 
                                     Each block (=section) is delimited by a:
                                        - start marker : '[START_OBS]'
                                        - end marker : '[END_OBS]'

            - Parameter sections : blocks of informations about parameter data.
                                   Each block (=section) is delimited by a:
                                        - start marker : '[START_PARAM]'
                                        - end marker : '[END_PARAM]'
        
        Parameters
        -----------
        filename (str) : name of the configuration file to write.
                         If None, filename will be construct from the current 
                         MartheOptim parametrisation name with the extension
                         '.config'.
                         Default is None.


        Examples
        --------
        mopt.write_tplfile()
        mopt.write_parfile()
        mopt.write_kriging_factors(vgm_range=9)
        mopt.write_config(filename='parametrisation.config')
        
        """
        # -- Set configuration file name
        dfilename = os.path.join(self.mm.mldir, f'{self.name}.config')
        configfile = dfilename if filename is None else filename

        # -- Generate header informations
        title = 'MARTHE OPTIMIZATION CONFIGURATION FILE'
        title += ' ({})'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        headers = ['***']
        headers.append('Model name: {}'.format(self.mm.mlname))
        headers.append('Model full path: {}'.format(os.path.join(self.mm.rma_path)))
        headers.append('Model spatial index: {}'.format(self.mm.sifile))
        headers.append('Number of parameters: {}'.format(len(self.get_param_df())))
        headers.append('Number of parameters blocks: {}'.format(len(self.param)))
        headers.append('Number of observation data types: {}'.format(self.get_ndatatypes()))
        headers.append('Number of observation blocks: {}'.format(self.get_nlocs()))
        headers.append('Number of observations: {}'.format(self.get_nobs()))
        headers.append('Number of not null observations: {}'.format(self.get_nobs(null_weight=False)))
        headers.append('Parameter files directory: {}'.format(self.par_dir))
        headers.append('Parameter templates directory: {}'.format(self.tpl_dir))
        headers.append('Observation files directory: {}'.format(self.obs_dir))
        headers.append('Simulation files directory: {}'.format(self.sim_dir))
        headers.append('***')

        # -- Write config file
        with open(configfile, 'w', encoding=encoding) as f:
            # -- write headers
            f.write(title)
            f.write('\n'*2)
            f.write('\n'.join(headers))
            f.write('\n'*2)
            # -- Write parameter configuration blocks
            for mp in self.param.values():
                f.write(mp.to_config())
                f.write('\n'*2)
            # -- Write parameter configuration block
            for mo in self.obs.values():
                f.write(mo.to_config())
                f.write('\n'*2)




    def write_parfile(self, parname= None):
        """
        Write parameter file(s) in parameter directory (`par_dir`).

        Parameters
        ----------
        parname (str) : required parameter names.

        Examples
        --------
        mopt.write_parfile(['soil', 'hk'])

        """
        # -- Manage parameter name to write
        if parname is None:
            pnmes = self.param.keys()
        else:
            pnmes = marthe_utils.make_iterable(parname)
            # -- Check parameter names validity
            not_found = [n for n in pnmes if n not in self.param.keys()]
            err_msg = "ERROR : Some provided parameters not added yet: {}.".format(', '.join(not_found))
            assert len(not_found) == 0, err_msg
        # -- Write parameter file for each provided parameter
        for pnme in pnmes:
            self.param[pnme].write_parfile()



    def write_tplfile(self, parname= None):
        """
        Write template file(s) in template directory (`tpl_dir`).

        Parameters
        ----------
        parname (str) : required parameter names.

        Examples
        --------
        mopt.write_tplfile(['soil', 'hk'])

        """
        # -- Manage parameter name to write
        if parname is None:
            pnmes = self.param.keys()
        else:
            pnmes = marthe_utils.make_iterable(parname)
            # -- Check parameter names validity
            not_found = [n for n in pnmes if n not in self.param.keys()]
            err_msg = "ERROR : Some provided parameters not added yet: {}.".format(', '.join(not_found))
            assert len(not_found) == 0, err_msg

        # -- Write parameter file for each provided parameter
        for pnme in pnmes:
            self.param[pnme].write_tplfile()





    def collect_pest_files(self, ftypes=['tpl', 'par', 'ins', 'sim']):
        """
        Collect pest io files as list(s).
        
        Parameters
        -----------
        ftypes (str/iterable, optional) : pest file type aliases or list of pest file type aliases.
                                          The file aliases can be:
                                            - Template files: 'template', 'tpl_file' , 'tpl'
                                            - Input files: 'parameter', 'in_file' , 'in' , 'par'
                                            - Instruction files: 'instruction', 'ins_file', 'ins'
                                            - Output files: 'simulated', 'out_file', 'out', 'sim'
                                          If not provided, `ftypes` will collect all pest io files
                                          required for pyemu.Pst.from_io_files() function.
                                          Default is ['tpl', 'par', 'ins', 'sim'].

        Returns
        -------
        pest_files (list) : required ordered collected pest files.

        Examples
        --------
        mopt.collect_pest_files('instruction')
        mopt.collect_pest_files(['tpl_file', 'in_file'])

        """
        # ---- Make the required ftypes file iterable
        fts = marthe_utils.make_iterable(ftypes)
        # ---- Collect pest files for each required file types
        pest_files = []
        for ft in fts:
            # -- Template file(s)
            if any(tag == ft for tag in ['template', 'tpl_file' , 'tpl']):
                files = [os.path.join(self.tpl_dir, f) for f in os.listdir(self.tpl_dir)
                         if any(k in f for k in self.param.keys()) & f.endswith('.tpl')]

            # -- Input file(s)
            if any(tag == ft for tag in ['parameter', 'in_file' , 'in' , 'par']):
                files = [os.path.join(self.par_dir, f) for f in os.listdir(self.par_dir)
                         if any(k in f for k in self.param.keys()) & f.endswith('.dat')]

            # -- Instruction file(s)
            if any(tag == ft for tag in ['instruction', 'ins_file', 'ins']):
                files = [mo.insfile for mo in self.obs.values()]

            # -- Output file(s)
            if any(tag == ft for tag in ['simulated', 'out_file', 'out', 'sim']):
                files = [os.path.join(self.sim_dir, f'{mo.locnme}.dat') for mo in self.obs.values()]

            # -- Store collected files
            pest_files.append(files)

        # ---- Return required pest files
        if len(fts) == 1:
            return pest_files[0]
        else:
            return pest_files





    def build_pst(self, add_reg0= False, write= False, model_command='model.bat', **kwargs):
        """
        Generate Pest Control File from the current observation
        and parameters sets added to MartheOptim instance.
        
        Parameters
        -----------
        add_reg0 (bool, optional) : whatever adding a 0-order Tikhonov regularization.
                                    Wrapper to pyemu.helpers.zero_order_tikhonov()
                                    Default is False.

        write (bool/str, optional) : .pst file writing management.
                                     If True, the Pest Control File will be written with
                                     a generic name: 'name_of_MartheOptim_instance.pst'.
                                     If string, the Pest Control File will be written with
                                     the user provided name.
                                     if False, the Pest Control File will not be written.
                                     Default is False.

        model_command (str/list, optional) : command(s) to launch forward run.
                                             Default is 'model.bat'.

        **kwargs, additional internal arguments that refer to the pyemu.Pst:
                - `control_data` section:
                    * noptmax
                    * jcosaveitn
                    * reisaveitn
                    * parsaveitn
                    * ... (see pst.control_data.__dict__['_df'].index for more)

                - `reg_data` section:
                    * phimlim
                    * phimaccept
                    * fracphim
                    * iregadj
                    * ... (see pst.reg_data.__dict__.keys() for more)


        Returns
        -------
        pst (pyemu.pst.Pst) : pyemu Pst instance.

        Examples
        --------
        pst = mopt.build_pst(add_reg0=True, write='mycalibration.pst',
                             noptmax=0, phimlim=1)

        """
        # -- Generate basic pst from io files
        pst = pyemu.Pst.from_io_files(*self.collect_pest_files())

        # -- Get clean DataFrame of all parameters
        param_df = self.get_param_df(
                        transformed=True
                            ).rename(
                                {'trans':'partrans',
                                 'defaultvalue':'parval1'},
                                axis=1)
        param_df['parnme'] = param_df['parnme'].str.replace('__','_')
        param_df.set_index('parnme', drop = False, inplace = True)

        # -- Disable parameter transformation (already done by pyMarthe)
        param_df['partrans'] = 'none'

        # -- Push to Pst 'parameter_data' section
        pst.parameter_data.loc[param_df.index] = param_df[pst.par_fieldnames]

        # -- Get clean DataFrame of all parameters
        obs_df = self.get_obs_df(transformed=True)

        # -- Push to Pst 'observation_data' section
        pst.observation_data.loc[obs_df.index] = obs_df[pst.obs_fieldnames]

        # -- Add regularization if required
        if add_reg0:
            pyemu.helpers.zero_order_tikhonov(pst)

        # -- Set kwargs (internal pest parameters)
        for k,v in kwargs.items():
            if k in pst.control_data.__dict__['_df'].index:
                pst.control_data.__dict__['_df'].loc[k,'value'] = v
            elif k in pst.reg_data.__dict__.keys():
                if add_reg0:
                    pst.reg_data.__dict__[k] = v
                else:
                    msg = f"WARNING : a provided kwarg argument (`{k}`) does refer to a " \
                           "pyEMU key of `pst.reg_data` whereas `add_reg0` = False. " \
                           "It will not be considered."
                    warnings.warn(msg)

            else:
                msg = f"WARNING : a provided kwarg argument (`{k}`) does not refer to " \
                       "a existing pyEMU key in `pst.control_data` or `pst.reg_data`. " \
                       "It will not be considered."
                warnings.warn(msg)

        # -- Manage forward run command
        pst.model_command = marthe_utils.make_iterable(model_command)

        # -- Return and write pst if required
        if write == True:
            pst.write(os.path.join(self.mm.mldir, f'{self.name}.pst'))
        elif isinstance(write, str):
            pst.write(write)
        
        return pst



    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheOptim'

