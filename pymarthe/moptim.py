
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
from pymarthe.mparam import MartheListParam
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



    def get_obs_df(self):
        """
        Get all observations information in a large DataFrame.

        Parameters:
        ----------

        Returns:
        --------
        obs_df (DataFrame) : merged provided observations

        Examples:
        --------
        moptim.get_obs_df()
        """
        if len(self.obs) > 0:
            return pd.concat([mo.obs_df for mo in self.obs.values()])
        else:
            return pd.DataFrame(columns = base_obs)




    def get_param_df(self):
        """
        Get all parameters informations in a large DataFrame.

        Parameters:
        ----------

        Returns:
        --------
        param_df (DataFrame) : merged provided parameters

        Examples:
        --------
        moptim.get_param_df()
        """
        if len(self.param) > 0:
            return pd.concat([mp.param_df for mp in self.param.values()])
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
        dt = [mo.datatype for mo in mopt.obs.values()]
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



    def add_obs(self, data, locnme = None, datatype = 'head', check_loc = True, nodata = None, **kwargs):
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
            warn_msg = f'Warning : locnme `{locnme}` already added. ' \
                       'It will be overwrited.'
            warnings.warn(warn_msg, Warning)
            # -- Remove existing set of observation
            self.remove_obs(locnme)

        # ---- Check validity and uncity of locnme
        if check_loc:
            self.check_loc(locnme)

        # ---- Build MartheObs instance from data input
        insfile = kwargs.pop('insfile', os.path.join(self.ins_dir, f'{locnme}.ins'))
        mobs = MartheObs(iloc = self.get_nlocs(),
                         locnme = locnme,
                         date = df.index,
                         value = df['value'].values,
                         obsfile = obsfile,
                         insfile = insfile,
                         datatype = datatype,
                         **kwargs)

        # ---- Build MartheObs instance
        self.obs[locnme] = mobs

        # # ---- Verbose
        # print(f"Observation '{locnme}' had been added successfully.")



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
                self.param[mp.parname].param_df[['trans', 'btrans']] = [trans, btrans]




    def write_ins(self, locnme=None):
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
        moptim.write_insfile(locnme = 'myobs', ins_dir = 'ins')
        """
        # ---- Manage multiple locnme writing
        locnmes = list(self.obs.keys()) if locnme is None else marthe_utils.make_iterable(locnme)
        # ---- Iterate over locnmes
        for locnme in locnmes:
            self.obs[locnme].write_insfile()





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
            # ---- Alerte if a same fluctuation had already been added
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
        *** UNDER DEVELOPMENT ***
        """
        if mobj._proptype == 'array':
            izone = kwargs.pop('izone', None)
            self._add_map(parname, mobj, izone, spacing, **kwargs)
            pass
        elif mobj._proptype == 'list':
            kmi = kwargs.pop('kmi', None)
            self._add_mlp(parname, mobj, kmi, **kwargs)



    def _add_map(self, parname, mobj, izone, **kwargs):
        """
        """
        return



    def _add_mlp(self, parname, mobj, kmi, **kwargs):
        """
        """
        # -- Bunch of assertion to avoid invalid inputs
        value_col = kwargs.get('value_col', 'value')
        err_kmi = 'ERROR : list-like parameter require a KeysMultiIndex (`kmi`) argument. ' \
                   'See pymarthe.utils.pest_utils.get_kmi() function.'
        err_value_col = f'ERROR : invalid `value_col` argument. '\
                      f'Must be a column name of {str(mobj)} object. ' \
                      f'Given : {value_col}.'
        assert kmi is not None, err_kmi
        assert value_col in mobj.data.columns, err_value_col
        # -- Build MartheListParam instance (mlp)
        tplfile = kwargs.pop('tplfile', os.path.join(self.tpl_dir, f'{parname}.tpl'))
        parfile = kwargs.pop('parfile', os.path.join(self.par_dir, f'{parname}.dat'))
        mlp = MartheListParam(parname= parname,
                              mobj= mobj,
                              kmi=kmi,
                              parfile = parfile,
                              tplfile = tplfile,
                              **kwargs)

        self.param[parname] = mlp





    def write_config(self, filename=None):
        """
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
        headers.append('Number of observation sets: {}'.format(self.get_nlocs()))
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
        """
        # -- Manage parameter name to write
        pnmes = self.param.keys() if parname is None else marthe_utils.make_iterable(parname)
        # -- Write parameter file for each provided parameter
        for pnme in pnmes:
            self.param[pnme].write_parfile()



    def write_tpl(self, parname= None):
        """
        """
        # -- Manage parameter name to write
        pnmes = self.param.keys() if parname is None else marthe_utils.make_iterable(parname)
        # -- Write parameter file for each provided parameter
        for pnme in pnmes:
            self.param[pnme].write_tplfile()




    def build_pst(self, add_reg0 = False, write=False, **kwargs):
        """
        """
        # -- Collect io files
        tpl = [mp.tplfile for mp in self.param.values()]
        par = [mp.parfile for mp in self.param.values()]
        ins = [mo.insfile for mo in self.obs.values()]
        sim = [os.path.join(self.sim_dir, f'{mo.locnme}.dat') for mo in self.obs.values()]

        # -- Generate basic pst from io files
        pst = pyemu.Pst.from_io_files(tpl,par,ins,sim)

        # -- Set param data 
        param_df = self.get_param_df().rename({'trans':'partrans',
                                               'defaultvalue':'parval1'}, axis=1)
        param_df['parnme'] = param_df['parnme'].str.replace('__','_')
        param_df.set_index('parnme', drop = False, inplace = True)
        param_df['partrans'] = 'none'
        pst.parameter_data.loc[param_df.index] = param_df[pst.par_fieldnames]

        # -- Set observation data
        obs_df = self.get_obs_df()
        pst.observation_data.loc[obs_df.index] = obs_df[pst.obs_fieldnames]

        # -- Add regularization if required
        if add_reg0:
            pyemu.helpers.zero_order_tikhonov(pst)

        # -- Set kwargs
        for k,v in kwargs.items():
            if k in pst.control_data.__dict__['_df'].index:
                pst.control_data.__dict__['_df'].loc[k,'value'] = v
            elif k in pst.reg_data.__dict__.keys():
                pst.reg_data.__dict__[k] = v

        # -- Return or write pst if required
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

