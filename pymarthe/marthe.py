"""
Contains the MartheModel class
Designed for structured grid and
layered parameterization

"""
import os, sys
import pickle
import subprocess as sp
from shutil import which
import queue 
import threading
from datetime import datetime
import re

import numpy as np
import pandas as pd 
import pyemu

from .utils import marthe_utils
from .mparam import MartheParam
from .mobs import MartheObs
from .mpump import MarthePump

encoding = 'latin-1'

class MartheModel():
    """
    Python wrapper for Marthe
    """
    def __init__(self,rma_path):
        """
        Parameters
        ----------
        rma_path : string
            The path to the Marthe rma file from which the model name 
            and working directory will be identified.

        Examples
        --------

        mm = MartheModel('/Users/john/models/model/mymodel.rma')

        """
        # initialize grid dics to None
        #self.grid_keys = ['permh','kepon','emmli','emmca']
        self.grid_keys = ['permh']
        self.grids = { key : None for key in self.grid_keys }

        # initialize parameter and observation dics
        self.param = {}
        self.obs = {}

        # get model working directory and rma file 
        self.mldir, self.rma_file = os.path.split(rma_path)

        # get model name 
        self.mlname = self.rma_file.split('.')[0]

        # set number of timestep provided
        self.nstep, _ = marthe_utils.read_pastp(os.path.join(self.mldir, self.mlname + '.pastp'))

        # initialize MarthePump class as attribute
        self.aqpump, self.rivpump = None, None

        # read permh data
        # NOTE : permh data also provides data on active/inactive cells
        self.x_vals, self.y_vals, self.grids['permh'] = self.read_grid('permh')

        # get cell size (only valid for regular grid of square cells
        self.cell_size = abs(self.x_vals[1] - self.x_vals[0])

        # get nlay nrow, ncol
        self.nlay, self.nrow, self.ncol = self.grids['permh'].shape

        # set up mask of active/inactive cells from permh data
        # value : 1 for active cells. 0 for inactive cells
        # imask is a 3D array (nlay,nrow,ncol)
        self.imask = (self.grids['permh'] != 0).astype(int)

        # set spatial reference (used for compatibility with pyemu geostat utils)
        self.spatial_reference = SpatialReference(self)
        
        # set up pilot point data frame 
        self.pp_df = pd.DataFrame({'name': pd.Series(None, dtype=str),
            'x': pd.Series(None, dtype=np.float32),
            'y': pd.Series(None, dtype=np.float32),
            'lay': pd.Series(None, dtype=np.float32),
            'zone' : pd.Series(None, dtype=np.float32),
            'parval': pd.Series(None, dtype=np.float32)
            })

        # init number of observation locations
        self.nobs_loc = 0



    def add_pump(self, pastp_file = None, mode = 'aquifer'):
        """
        -----------
        Description:
        -----------
        Extract pumping data from .pastp file and store it in 
        MartheModel.pump attribut
        Note: 2 types of pumping condition can be read:
                    - LIST_MAIL (regional model)
                    - MAILLE (local model) single value or record
    
        Parameters: 
        -----------
        pastp_file (str) : path to .pastp marthe file
        mode (str) : type of withdraw pumping
                     Can be 'aquifer' or 'river'
                     Default is 'aquifer'
        Returns:
        -----------
        MarthePump (class) : set pumping class as attribut (inplace)

        Example
        -----------
        rma = 'mymarthemodel.rma'
        mm = MartheModel(rma)
        mm.add_pump()
        """
        # ---- Build aquifer MarthePump instance
        if mode == 'aquifer':
            if pastp_file is None:
                self.aqpump = MarthePump(self, mode)
            else:
                self.aqpump = MarthePump(self, pastp_file, mode)
                
        # ---- Build aquifer MarthePump instance
        if mode == 'river':
            if pastp_file is None:
                self.rivpump = MarthePump(self, mode)
            else:
                self.rivpump = MarthePump(self, pastp_file, mode)





    def add_param(self, name, default_value = np.nan, izone = None, array = None, log_transform = False) :
        '''
        Parameters
        ----------
        name : string
            Parameter name (ex : 'permh', 'kepon', 'emmli', 'emmca')

        default_value  : float
            
        izone : data 

        Examples
        --------
        mm = MartheModel('/Users/john/models/model/mymodel.rma')
        '''

        # case an array is provided
        if isinstance(array, np.ndarray) :
            assert array.shape == (self.nlay, self.nrow, self.ncol)
            self.grids[name] = array
        else : 
            # fetch mask from mm
            self.grids[name] = np.array(self.imask, dtype=np.float)
            # fill array with default value within mask
            self.grids[name][ self.grids[name] != 0 ] = default_value

        # create new instance of MartheParam
        self.param[name] = MartheParam(self, name, default_value, izone, array, log_transform)       



    def add_obs(self, obs_file, loc_name = None) :
        
        self.nobs_loc += 1
        # prefix will be used to set individual obs name
        prefix = 'loc{0:03d}'.format(self.nobs_loc)

        # infer loc_name from file name if loc_name not provided
        if loc_name is None : 
            obs_dir, obs_filename = os.path.split(obs_file)
            loc_name = obs_filename.split('.')[0]

        # create new MartheObs object
        obs  = MartheObs(self, prefix, obs_file, loc_name)
        
        # remove NAs
        # NOTE currently -9999 are not considered as NAs
        # but weights are zeros for these observations
        obs.df.dropna(inplace=True)

        # Check number of records
        if obs.df.shape[0] > 0 : 
            self.obs[loc_name] = obs
        else :
            print('No records in current obs ' + str(loc_name))



    def load_grid(self,key) : 
        """
        Simple wrapper for read_grid.
        The grid is directly stored to grids dic,
        this function returns nothing. 

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
        Returns x_values, y_values and grid.

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



    def get_outcrop(self):
        """
        Function to get outcropping layer from permh

        Parameters:
        ----------
        self : MartheModel instance

        Returns:
        --------
        outcrop (array) : 2D-array (shape : (nrow,ncol))

        Examples:
        --------
        mm = MartheModel('mymodel.rma')
        outcrop_arr = mm.get_outcrop()
        """
        # ---- Set list of arrays with layer number on active cell
        layers = [ilay * imask for ilay, imask in enumerate(self.imask, start=1)]
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
        return outcrop



    def write_grid(self,key):
        """
        Simple wrapper for write_grid file

        Parameters
        ----------
        key : str
            parameter key (ex. 'kepon')
            
        Examples
        --------
        mm.write_grid('permh')

        """
        # get path to grid file 
        grid_path = os.path.join(self.mldir,self.mlname + '.' + key)

        # load grid file 
        marthe_utils.write_grid_file(grid_path, self.x_vals, self.y_vals, grid = self.grids[key])

        return

    def write_grids(self) : 
        """
        write all available grids
        """
        for key in list(self.grids.keys()) : 
            self.write_grid(key)

        return

    def data_to_shp(self, key, lay, filepath ) :
        data = self.grids[key][lay,:,:]
        marthe_utils.grid_data_to_shp(self.x_vals, self.y_vals, data, file_path, field_name_list=[key])

    def extract_prn(self,prn_file = None, fluct = False, out_dir = None , obs_dir = None):
        """ 
        Simple wrapper to marthe_utils_extract_prn()
        """
        if prn_file == None : 
            prn_file = os.path.join(self.mldir,'historiq.prn')
        if out_dir == None : 
            out_dir = os.path.join(self.mldir,'sim','')
        if fluct == True:
            marthe_utils.extract_prn(prn_file,True, out_dir, obs_dir)
        else :
            marthe_utils.extract_prn(prn_file,False, out_dir, obs_dir)

    def extract_variable(self,path_file,pastsp,variable,dti_present,dti_future,period, out_dir):
        """ 
        Simple wrapper to marthe_utils_extract_prn()
        """
        if path_file == None : 
            prn_file = os.path.join(self.mldir,'histobil_debit.prn')
        if out_dir == None : 
            out_dir = os.path.join(self.mldir,'sim','')
        marthe_utils.extract_variable(path_file,pastsp,variable,dti_present,dti_future,period,out_dir)


    def remove_autocal(self, mart_file=None):
        """
        Function to make marthe auto calibration silent

        Parameters:
        ----------
        self : MartheModel instance
        mart_file (str) : .mart file path
                          If None mart_file = mldir/mlname.mart
                          Default is None

        Returns:
        --------
        Write in .mart inplace

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.remove_autocal()
        """
        # ---- Get mart_file 
        file =  os.path.join(self.mldir, f'{self.mlname}.mart') if mart_file is None else mart_file
        # ---- Fetch .mart file content
        with open(file, 'r', encoding=encoding) as f:
            lines = f.readlines()

        # ---- Define pattern to search
        re_cal = r"^\s*1=Optimisation"

        for line in lines:
            # ---- Search patterns
            cal_match = re.search(re_cal, line)
            # ---- Make calibration/optimisation silent 
            if cal_match is not None:
                wrong = cal_match.group()
                right = re.sub('1','0', wrong)
                new_line  = re.sub(wrong, right, line)
                marthe_utils.replace_text_in_file(file, line, new_line)



    def make_silent(self, mart_file = None):
        """
        Function to make marthe run silent

        Parameters:
        ----------
        self : MartheModel instance
        mart_file (str) : .mart file path
                          If None mart_file = mldir/mlname.mart
                          Default is None

        Returns:
        --------
        Write in .mart inplace

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.make_silent()
        """
        # ---- Get mart_file 
        file = os.path.join(self.mldir, f'{self.mlname}.mart') if mart_file is None else mart_file

        # ---- Fetch .mart file content
        with open(file, 'r', encoding=encoding) as f:
            lines = f.readlines()

        # ---- Define pattern to search
        re_exe = r"^\s*(\s|\w)=Type d'exécution"

        for line in lines:
            # ---- Search patterns
            exe_match = re.search(re_exe, line)
            # ---- Make run silent 
            if exe_match is not None:
                wrong = exe_match.group()
                right = re.sub(r'(\s|\w)=','M=', wrong)
                new_line  = re.sub(wrong, right, line)
                marthe_utils.replace_text_in_file(file, line, new_line)

  

 
    def run_model(self,exe_name = 'marthe', rma_file = None, 
                  silent = True, verbose=False, pause=False,
                  report=False, cargs=None):
        """
        This function will run the model using subprocess.Popen.  It
        communicates with the model's stdout asynchronously and reports
        progress to the screen with timestamps
        Parameters
        ----------
        exe_name : str
            Executable name (with path, if necessary) to run.
        rma_file : str
            rma file of model to run. The rma_file must be the
            filename of the namefile without the path. Namefile can be None
            if it follows the syntax model_name.rma
        silent : bool
            Run marthe model as silent 
        verbose : boolean
            Echo run information to screen (default is True).
        pause : boolean, optional
            Pause upon completion (default is False).
        report : boolean, optional
            Save stdout lines to a list (buff) which is returned
            by the method . (default is False).
        cargs : str or list of strings
            additional command line arguments to pass to the executable.
            Default is None
        Returns
        -------
        (success, buff)
        success : boolean
        buff : list of lines of stdout
        """
        # initialize variable
        success = False
        buff = []
        normal_msg='normal termination'

        # force model to run as silent if required
        if silent:
            self.make_silent()

        # Check to make sure that program and namefile exist
        exe = which(exe_name)
        if exe is None:
            import platform
            if platform.system() in 'Windows':
                    exe = which(exe_name + '.exe')
        if exe is None:
            s = 'The program {} does not exist or is not executable.'.format(
                exe_name)
            raise Exception(s)
        

        # Marthe rma file 
        if rma_file is None : 
            rma_file = os.path.join(self.mldir,self.mlname + '.rma')

        # simple little function for the thread to target
        def q_output(output, q):
            for line in iter(output.readline, b''):
                q.put(line)

        # create a list of arguments to pass to Popen
        argv = [exe_name]
        if rma_file is not None:
            argv.append(rma_file)

        # add additional arguments to Popen arguments
        if cargs is not None:
            if isinstance(cargs, str):
                cargs = [cargs]
            for t in cargs:
                argv.append(t)

        # run the model with Popen
        proc = sp.Popen(argv,
                        stdout=sp.PIPE, stderr=sp.STDOUT)

        # some tricks for the async stdout reading
        q = queue.Queue()
        thread = threading.Thread(target=q_output, args=(proc.stdout, q))
        thread.daemon = True
        thread.start()

        failed_words = ["fail", "error"]
        last = datetime.now()
        lastsec = 0.
        while True:
            try:
                line = q.get_nowait()
            except queue.Empty:
                pass
            else:
                if line == '':
                    break
                line = line.decode('latin-1').lower().strip()
                if line != '':
                    now = datetime.now()
                    dt = now - last
                    tsecs = dt.total_seconds() - lastsec
                    line = "elapsed:{0}-->{1}".format(tsecs, line)
                    lastsec = tsecs + lastsec
                    buff.append(line)
                    if not verbose:
                        print(line)
                    for fword in failed_words:
                        if fword in line:
                            success = False
                            break
            if proc.poll() is not None:
                break
        proc.wait()
        thread.join(timeout=1)
        buff.extend(proc.stdout.readlines())
        proc.stdout.close()

        for line in buff:
            if normal_msg in line:
                print("success")
                success = True
                break

        if pause:
            input('Press Enter to continue...')
        return success, buff

    def read_sim(self, prn_file = None):
        """
        Description
        ----------
        Reads Marthe prn file and append a "sim" columns to
        observation dataframe. 

        Parameters 
        ----------
        prn_file (optional) : str
            path to prn file 

        Example 
        --------
        mm = MartheModel('./model.rma')
        mm.add_obs(obs_file = 'piezo1.dat')
        mm.read_sim()
        mm.obs['piezo1'].df

                          value    weight       sim
        date                                       
        1972-12-31          32.5  1.000000  33.41174
        ...

        """
        if prn_file == None : 
            prn_file = os.path.join(self.mldir,'historiq.prn')

        # load simulated values 
        df_sim = marthe_utils.read_prn(prn_file)

        for loc in self.obs.keys() :
            # build up dataframe with a single column containing 
            # simulated data at loc
            # case : one single aquifer
            if isinstance(df_sim[loc],pd.core.series.Series) :
                df_sim_loc = pd.DataFrame({'sim':df_sim[loc]},index=df_sim.index)
            # case : several aquifers intercepted 
            else : 
                # compute mean over several columns when multiple aquifers are considered
                df_sim_loc = pd.DataFrame({'sim':df_sim[loc].mean(axis=1)},index=df_sim.index)
            # perform outer join with observation 
            self.obs[loc].df = self.obs[loc].df.merge(df_sim_loc,how='outer',left_index=True, right_index=True)

    def compute_phi(self,type='sum_weighted_squared_res') : 
        """
        Description
        ----------
        Returns weighted objective function
        Parameters 
        ----------
        prn_file (optional) : str
            path to prn file
        """
        
        self.read_sim()

        df_comp = df = pd.DataFrame(columns=['obs', 'sim', 'weight'])

        for loc in self.obs.keys() :
            # build up dataframe for current loc 
            df_loc = pd.DataFrame( {'obs':self.obs[loc].df.value,
                'sim':self.obs[loc].df.sim,
                'weight':self.obs[loc].df.value})
            # append current loc to maindataframe. 
            df_comp.append(df_loc)

        if type == 'sum_weighted_squared_res': 
            phi = pest_utils.sum_weighted_squared_res(df_comp.sim,df_comp.obs,df_comp.weight)
        else : 
            print('Phi type currently not implemented')
            return

        return(phi)

    def setup_tpl(self,izone_dic = None, log_transform = None, pp_ncells = None,
            refine_crit = None, refine_crit_type = None, refine_value = None,
            refine_level = 1, refine_layers = None,save_settings = None, reload_settings = None):
        """
        Description
        ----------
        Setup PEST templates files from dictionary of izone data
        
        For pilot points parameters :
             - pp_ncell defines the density, default value is 10
             - a buffer is considered to allow pilot points to lie at the border of the active parameter zone
             - an exponential variogram with 3 times the pilot point spacing is considered
             - a refinement criteria may be defined 

        Parameters
        ----------
        izone_dic : dic, keys parameter type name 
        dictionary of izone 3d arrays

        log_transform :  boolean or dic with keys parameter type name 
        dictonary of boolean 

        pp_ncells : int, dic, or nested dic 
            values with number of cells for given parameter,
            ex : {'permh': 12, kepon: 18}
            or a dic with keys as layers,
            ex : {'permh':{1:12, 2:18}}
            The value associated with the -1 key will be considered as the default value

        refine_crit (str or dic of str) : refinement criteria for pilot points

        refine_crit_type (str or dic of str) : type of criteria (absolute, quantile)

        refine_value (float or dic of float) : threshold value for refinement criteria.
                       if refine_crit_type is absolute, regular grid will be refined
                       where refine_crit > refine_value (e.g. parameter identifiability > 0.9)
                       if refine_crit_type is quantile, regular grid will be refined
                       for the upper quantile defined by refine_value (e.g. 0.25)

        refine_level : refinement level (see pp_refine)
	
	refine_layers (int, list) : layer or list of layers where pilot points refinement will be implemented

        save_settings : None, or name of the file where settings should be saved (e.g. 'case.settings')

        reload_settings : None, or name of the file where settings should be read. 
                          When reload_settings is not None, parameter values are read from files
                          and the grid of pilot points is kept, or refined (not generated).
                          This can be used to recover a parameter set estimated by PEST and
                          refine a series of pilot points. 
                
        Example 
        --------
        izone = np.ones(nlay,nrow,ncol)
        izone_dic = {'permh':izone}
        log_transform = True
        pp_ncells_dic = 12

        setup_tpl(izone_dic, log_transform_dic, pp_ncells_dic)

        # alternative with dictionaries (equivalent in this case)
        log_transform = {'permh':True}
        pp_ncells_dic = {'permh':12}

        setup_tpl(izone_dic, log_transform_dic, pp_ncells_dic)
        # save settings 
        setup_tpl(izone_dic, log_transform_dic, pp_ncells_dic, save_settings = 'case.settings')

        # load settings, read parameter values from former PEST parameter estimation and refine 
        setup_tpl(load_settings = 'case.settings', 
                    refine_crit ='ident', refine_crit_type = 'quantile', refine_value = 0.3 )

        """

        # If reload_settings is not None, load settings 
        if isinstance(reload_settings,str) :
            print('Reloading izone_dic, log_transform_dic and pp_ncells from {}'.format(reload_settings))
            try :
                with open(reload_settings,'rb') as handle : 
                    izone_dic, log_transform_dic, pp_ncells_dic = pickle.load(
                        handle)
            except : 
                print('Could not find {}.settings in current directory'.format(self.mlname))

            # infer adjustable parameter list from izone dictionary
            params = izone_dic.keys()

        # If reload is None, check and initialize provided settings
        else  :
            # izone dictionary
            assert isinstance(izone_dic,dict), 'izone_dic should be a dictionary'

            # infer adjustable parameter list from izone dictionary
            params = izone_dic.keys()

            # log_transform dictionary
            if log_transform is None :
                # default value 
                log_transform_dic = {par:'none' for par in params}
            elif isinstance(log_transform, bool) :
                # propagate value to all parameters
                log_transform_dic = {par:log_transform for par in params}
            elif isinstance(log_transform, dict):
                log_transform_dic = log_transform
            else : 
                print('Error processing log_transform parameter \
                        Provide Boolean or dictionary of Boolean')
                return

            # initialize pp_ncells dictionary
            if pp_ncells is None :
                # default value
                default_value = 10
                # setup nested dictionary
                pp_ncells_dic = { par: {lay:default_value for lay in range(self.nlay)} for par in izone_dic.keys }
            elif isinstance(pp_ncells,int):
                # setup nested dictionary from provided integer value
                pp_ncells_dic = { par: {lay:pp_ncells for lay in range(self.nlay)} for par in izone_dic.keys }
            elif isinstance(pp_ncells,dict) :
                pp_ncells_dic = {}
                # check nested dictionary 
                for par in params :
                    assert par in pp_ncells.keys(), 'Key {0} not found in dic pp_ncells'.format(par)
                    if isinstance(pp_ncells[par],int) :
                        # propagate provided integer value to all layers
                        pp_ncells_dic[par] = {lay:pp_ncells[par] for lay in range(self.nlay)}
                    elif isinstance(pp_ncells[par],dict):
                        pp_ncells_dic[par] = pp_ncells[par]
                    else : 
                        print('Error processing pp_ncells for parameter {0}'.format(par))
                        return
            else : 
                print('Error processing pp_ncells argument, check type and content')
                return

            
            # save settings
            if save_settings is not None : 

                settings_tup = izone_dic, log_transform_dic, pp_ncells_dic

                try : 
                    with open(save_settings,'wb') as handle:
                        pickle.dump(settings_tup,handle)
                except :
                    print('I/O error, cannot save settings to file {}'.format(save_settings))

	# check consistency of refinement settings and convert to dictionaries
        if not isinstance(refine_crit, dict) :
            refine_crit_dic = {par:refine_crit for par in params}
        else : refine_crit_dic = refine_crit

        if not isinstance(refine_crit_type, dict) :
            refine_crit_type_dic = {par:refine_crit_type for par in params}
        else : refine_crit_type_dic = refine_crit_type

        if not isinstance(refine_value, dict) :
            refine_value_dic = {par:refine_value for par in params}
        else : refine_value_dic = refine_value
        if refine_layers is None:
            refine_layers = range(self.nlay)
        # iterate over parameters with izone data  
        for par in params:
            print('Processing parameter {0}...'.format(par))
            # add parameter 
            self.add_param(par, izone = izone_dic[par], default_value = 1e-5, log_transform=log_transform_dic[par])
            # write izone to disk for future use by model_run
            marthe_utils.write_grid_file('{0}.i{1}'.format(self.mlname,par),self.x_vals,self.y_vals,izone_dic[par])
            # parameters with pilot points (izone with positive values)
            if np.max(np.unique(izone_dic[par])) > 0  :
                print('Setting up pilot points for parameter {0}'.format(par))
                if isinstance(reload_settings, str) :
                    # load existing pp_df
                    self.param[par].read_pp_df()
                # iterate over layers 
                for lay in range(self.nlay):
                    # layers with pilot points (izone with positive values)
                    if np.max(np.unique(izone_dic[par][lay,:,:])) > 0 :
                        if reload_settings is None :
                            print('Generation of a regular grid of pilot points for parameter {0}, layer {1} '.format(
                                par,lay+1))
                            # generate pilot point grid
                            self.param[par].pp_from_rgrid(lay, n_cell=pp_ncells_dic[par][lay], n_cell_buffer = True)
                            npp = len(self.param[par].pp_dic[lay])
                            print('{0} pilot points seeded for parameter {1}, layer {2}'.format(npp,par,lay+1))
                        else : 
                            # get base spacing (model coordinates units) from ncell
                            self.param[par].base_spacing[lay] = pp_ncells_dic[par][lay]*self.cell_size
                        # pointer to current pilot point dataframe (reloaded or just generated)
                        pp_df  = self.param[par].pp_dic[lay]
                        # refinement of pilot point grid
                        if refine_crit_dic[par] is not None and lay in refine_layers:
                            # if criteria is nobs, get it with get_pp_nobs()
                            if refine_crit_dic[par]=='nobs':
                                loc_df = marthe_utils.read_histo_file(self.mlname + '.histo')
                                df_crit = self.param[par].get_pp_nobs(lay, loc_df)
                            # if criteria inferred from sensitivities, get it from from file
                            else : 
                                # get dataframe with refinement criteria
                                #df_crit_file = os.path.join('crit','{0}_pp_l{1:02d}_crit.dat'.format(par,lay+1))
                                df_crit_file = os.path.join('crit','df_crit.dat')
                                df_crit = pd.read_csv(df_crit_file, delim_whitespace=True, index_col='param')
                            # refinement based on quantile (highest values selected)
                            if refine_crit_type_dic[par] == 'quantile' :
                                # compute number of pilot points that will be refined 
                                n_pp_refined = int(refine_value_dic[par]*pp_df.shape[0])
                                df_crit = df_crit.loc[df_crit.index.str.contains('l{0:02d}'.format(lay+1)),:]
                                # sort dataframe according to refine_crit column in decreasing order
                                df_crit.sort_values(by=[refine_crit_dic[par]], inplace=True, ascending=False)
                                # initialize refine column (boolean)
                                df_crit['refine'] = False
                                # select points that will be refined
                                df_crit.loc[:n_pp_refined,'refine'] = True
                            # refinement based on absolute criteria value (threshold)
                            else : 
                                df_crit['refine'] = df_crit[refine_crit_dic[par]] > refine_value_dic[par]
                            # append refine column into pp_df (inner join with merge)
                            pp_df_crit = pd.merge(pp_df,df_crit['refine'], left_index=True, right_index=True)
                            # perform refinement if number of points to refine > 0
                            if pp_df_crit['refine'].sum() > 0 : 
                                self.param[par].pp_refine(lay, pp_df_crit, n_cell = pp_ncells_dic[par][lay], level = refine_level )
                            # update pointer to pp_df (yes, this is necessary!)
                            pp_df = self.param[par].pp_dic[lay]
                        # set variogram range (2 times base pilot point spacing)
                        vario_range = 2*self.cell_size*pp_ncells_dic[par][lay]
                        # variogram setup :
                        #   - parameter a is considered as a proxy for range
                        #   - the contribution has no effect without nugget
                        v = pyemu.utils.geostats.ExpVario(contribution=1, a=vario_range)
                        # build up GeoStruct
                        transform = 'log' if self.param[par].log_transform == True else 'none'
                        gs = pyemu.utils.geostats.GeoStruct(variograms=v,transform=transform)
                        # attach geostruct and covariance matrix 
                        self.param[par].gs_dic[lay] = gs
                        self.param[par].ppcov_dic[lay] = gs.covariance_matrix(pp_df.x,pp_df.y,pp_df.name)
                        # set up kriging
                        ok = pyemu.utils.geostats.OrdinaryKrige(geostruct=gs,point_data=pp_df)
                        # spatial reference (for pyemu compatibility only)
                        ok.spatial_reference = self.spatial_reference 
                        # pandas dataframe of point where interpolation shall be conducted
                        # set up index for current zone and lay
                        x_coords, y_coords = self.param[par].zone_interp_coords(lay,zone=1)
                        # compute kriging factors
                        kfac_df = ok.calc_factors(x_coords, y_coords,pt_zone=1, num_threads=4)
                        # write kriging factors to file
                        kfac_file = os.path.join(self.mldir,'kfac_{0}_l{1:02d}.dat'.format(par,lay+1))
                        ok.to_grid_factors_file(kfac_file)
                # write initial parameter value file (all zones)
                self.param[par].write_pp_df()
                self.param[par].write_pp_tpl()
            # set up ZPCs if present
            # will have not effect for parameters and layers with pilot points
            if reload_settings is None :
                # (over)write parameter values for ZPCs
                # files are not overwritten when reload is True to keep parameter values
                self.param[par].write_zpc_data()
            # write template files 
            self.param[par].write_zpc_tpl()

    def setup_ins(self,obs_dir = None, histo_file = None, obs_layers = None) :
        """
        Description
        ----------
        Setup instruction files from an inner join of observation files 
        and simulated outputs (listed in .histo file)

        Parameters
        ----------
        obs_dir (optional) : directory containing observations files (default './obs/')
        histo_file (optional) : Marthe histo file with simulated outputs (default, 'case.histo')
        obs_layer (optional) : list of model layers where observations should be considered (default, all layers)

        Example 
        --------
        obs_dir = os.path.join(mm.mldir,'obs','')
        mm.setup_ins(obs_dir, 'model.histo', obs_layers = [5])
        """
        sim_obs_loc = []
        obs_files = []
        obs_predict = []
        print('Collecting observed and simulated historical records...')
        # initialize arguments if not provided 
        if obs_dir is None : 
            obs_dir = os.path.join(self.mldir,'obs')
        if histo_file is None : 
            histo_file = '{}.histo'.format(self.mlname)
        if obs_layers is None : 
            obs_layers = list(range(self.nlay))
        # load histo file 
        print('Reading history file {}'.format(histo_file))
        df_histo = marthe_utils.read_histo_file(histo_file)
        # get sorted list of all observation files in obs_dir 
        all_obs_files = [os.path.join(obs_dir, f) for f in sorted(os.listdir( obs_dir )) if f.endswith('.dat')]
        print('Found {0} observation history files in {1}'.format(len(all_obs_files),obs_dir))
        # observation locations (BSS ids) identified in model output file (model.histo)
        # iterate over obs_layers and get simulated observation locations
        for lay in obs_layers:
            # simulated records for lay
            lay_sim_list = df_histo.loc[(df_histo.layer - 1)==lay].index
            sim_obs_loc.extend(lay_sim_list)
            print('{0} simulated locations found for layer {1}'.format(len(lay_sim_list),lay+1))
        print('Found {} simulated history locations over selected model layers'.format(len(sim_obs_loc)))
        #  observation files selected for history matching (with simulated counterparts)
        # iterate over observation files
        for obs_file in all_obs_files:
            # infer observation loc (BSS id) from filename
            obs_filename = os.path.split(obs_file)[-1]
            obs_loc = obs_filename.split('_')[0]
            if obs_loc.startswith('stock'):
                obs_predict.append(obs_file)
            # if obs_loc found in simulated outputs, append obs_loc
            if obs_loc in sim_obs_loc:
                obs_files.append(obs_file) 
        obs_files.extend(obs_predict)
        print('{} simulation locations considered with observed counterparts'.format(len(obs_files)))
        print('Generating instruction files for PEST...')
        # add selected observations
        for obs_file in obs_files :
            self.add_obs(obs_file = obs_file)
        # write instruction files
        for obs_loc in self.obs.keys() :
            self.obs[obs_loc].write_ins()

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
