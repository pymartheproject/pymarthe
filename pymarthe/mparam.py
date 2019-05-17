"""
Contains the MartheModel class
Designed for structured grid and
layered parameterization

"""
import os 
import numpy as np
from matplotlib import pyplot as plt 
from .utils import marthe_utils
import pandas as pd 
import pyemu

class MartheParam() :
    """
    Class for handling Marthe parameters

    Parameters
    ----------
    name : str
        parameter name 
    default_value : int or np.array with shape (nlay,nrow,ncol)
        default values
    izone : (optional) array with shape (nlay,nrow,ncol))

    Examples
    --------
    """
    def __init__(self, name, default_value, izone = None) :
        self.name = name
        self.set_default_value(default_value)
        self.set_izone(izone)
        self.set_in_filenames()
        self.set_tpl_filenames()
        self.set_zpc_df()
        self.set_pp_df()

    def set_default_value(self, value) : 
        if isinstance(value,int) : 
            self.default_value = np.ones()*value
      
    def set_izone(self,key,data = None):
        """
        Load izone array from data. 
        If data is None, a single constant zone is considered. 
        Former izone data for given parameter, will be reset. 

        Parameters
        ----------
        key : str
            parameter to which the array is related
        data : None, int or np array of int, shape (nlay,nrow,ncol)

        Examples
        --------

        """
        # reset izone for current parameter from imask
        self.izone = self.imask
        # index of active cells
        idx_active_cells = self.imask == 1

        if data is None :
            # a single zone is considered
            self.izone[key][idx_active_cells] = -1

        if isinstance(data,np.ndarray) : 
            assert data.shape == (nlay,nrow,ncol) 
            # only update active cells  
            self.izone[key][idx_active_cells] = data[idx_active_cells]

        self.update_dics()


    def update_dics(self) :
        """

        Parameters
        ----------
        key : str
            
        Examples
        --------
        """
        nlay = self.izone.shape[0]
        self.zpc_dic = {lay:[] for lay in range(nlay) }
        self.pp_dic  = {lay:[] for lay in range(nlay) }
      
        for lay in range(self.izone.shape[0]) :
            zones = np.unique(self.izone[lay,:,:])
            for zone in zones :
                if zone < 0 :
                    self.zpc_dic[lay].append(abs(zone))
                elif zone > 0 :
                    self.pp_dic[lay].append(zone)

    def set_in_filenames(self, in_file_zpc = None, in_file_pp = None):
        """

        Parameters
        ----------
        key : str
            
        Examples
        --------
        """
        if in_file_zpc is None : 
            self.in_file_zpc = self.name + '_zpc.dat'
        if in_file_pp is None : 
            self.in_file_pp = self.name. '_pp.dat'


    def set_tpl_filenames(self, tpl_file_zpc = None, tpl_file_pp = None) : 
        """

        Parameters
        ----------
        key : str
            
        Examples
        --------
        """
        if tpl_file_zpc is None : 
            self.tpl_file_zpc = self.name + '_zpc.tpl'
        if tpl_file_pp is None : 
            self.tpl_file_pp = self.name. '_pp.tpl'


    def set_zpc_names(self):
        """
        updates zpc names from self.dic_zpc
        """
        self.zpc_names = []
        # iterate over layers
        for lay in self.zpc_dic.keys() :
            # iterate over zones 
            for zone in self.zpc_dic[lay]:
                self.zpc_names.append('{0}_l{1:02d}_z{2:02d}'.format(name,lay,zone))


    def set_pp_names(self):
        """
        updates pp names from self.dic_pp
        """

    
    def grid(self) : 
        """
        Update parameter from input file and returns data array

        Returns
        -------
        parameter value (array)

        Example
        -------

        >> mm.grids['kepon'] = kepon_par.get()

        """

        # case constant zone

        # case pilot point zone

        return(data)

    def setup_pp(self) :

        



        self.pp_df


    def zone_interp_coords(self, mm, lay, zone_id) :

        # set up index for current zone and lay
        idx = self.izone[lay,:,:] == zone_id

        # point where interpolation shall be conducted for current zone
        xx, yy = np.meshgrid(mm.x_vals,mm.y_vals)
        x_select = xx[idx].ravel()
        y_select = yy[idx].ravel()

        return(x_select,y_select)


    def write_zpc_tpl(self):
        """
        Load izone array from data. 
        If data is None, a single constant zone is considered. 
        Former izone data for given parameter, will be reset. 

        Parameters
        ----------
        key : str
            parameter to which the array is related
        data : None, int or np array of int, shape (nlay,nrow,ncol)

        Examples
        --------
        """
        # -- zones of piecewise constancy
        
        zpc_names = []

        if len(names) > 0 :
            tpl_entries = ["~  {0}  ~".format(name) for name in zpc_names]
            zpc_df = pd.Dataframe({'name' : zpc_names,'tpl' : tpl_entries})
            pest_utils.write_tpl_from_df(self.tpl_file_zpc, zpc_df)

        # -- zones with pilot points

        pp_names = []

    def write_pp_tpl(self) : 
        """ set up template data frame for pilot points
        Parameters
        ----------
        Returns 
        -------
            pp_df : pandas.DataFrame
                a dataframe with pilot point information (name,x,y,zone,parval1)
                 with the parameter information (parnme,tpl_str)
        """


        return pp_df



