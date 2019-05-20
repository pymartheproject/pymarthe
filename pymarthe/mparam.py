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
    def __init__(self, mm, name, default_value, izone = None, array = None) :
        self.mm = mm # pointer to the instance of Marthe Model
        self.name = name # parameter name
        self.array = self.mm.grids[self.name] # pointer to corresponding grid
        self.set_default_value(default_value)
        self.set_izone(izone)
        self.set_in_filenames()
        self.set_tpl_filenames()
        self.init_array(array)

    def set_default_value(self, value) : 
        """
        Function

        Parameters
        ----------
        key : str
            parameter
        data : type

        Examples
        --------

        """
        if isinstance(value,int) : 
            self.default_value = np.ones()*value
      
    def set_izone(self,key,izone = None):
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
        self.izone = self.mm.imask
        # index of active cells
        idx_active_cells = self.mm.imask == 1

        if izone is None :
            # a single zone is considered
            self.izone[idx_active_cells] = -1

        if isinstance(izone,np.ndarray) : 
            assert data.shape == (nlay,nrow,ncol) 
            # only update active cells  
            self.izone[idx_active_cells] = izone[idx_active_cells]

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
      
        for lay in range(nlay) :
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
            self.in_file_pp = self.name + '_pp.dat'


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
            self.tpl_file_pp = self.name + '_pp.tpl'


    def set_zpc_names(self):
        """
        updates zpc names from self.dic_zpc

        Parameters
        ----------
        key : str
            parameter
        data : type

        Examples
        --------

        """
        """
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
        """
        Function

        Parameters
        ----------
        key : str
            parameter
        data : type

        Examples
        --------

        """
    
    def set_array(self, lay, zone, values) : 
        """
        Set parameter array for given lay and zone

        Parameters
        ----------
        lay : int
            layer, zero based 
        zone : int
            zone id (<0 or >0)
        values = int or np.ndarray
            int for zones < 0, ndarray for zone > 0 

        Example
        -------
        mm.param['kepon'].set_array(lay = 1,zone = -2, values = 12.4e-3)

        >> 
        """
        assert lay in range(nlay), 'layer {0} is not valid.'.format(lay)
        assert zone in np.unique(self.izone), 'zone {0} is not valid'.format(zone)
        if zone < 0 :
            assert isinstance(values, int), 'values should be int if zone is < 0'
        elif zone > 0 :
            assert isinstance(values, np.ndarray), 'values should be np.ndarray is zone >0'

        # select zone 
        idx = self.izone[lay,:,:] == zone

        # update values within zone 
        self.array[lay,:,:][idx] = values

        return(data)

    def setup_pp(self) :
        """
        Function

        Parameters
        ----------
        key : str
            parameter
        data : type

        Examples
        --------

        """
        



        self.pp_df


    def zone_interp_coords(self, mm, lay, zone_id) :
        """
        Function

        Parameters
        ----------
        key : str
            parameter
        data : type

        Examples
        --------

        """
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


    def pp_from_rgrid(zone,n_cell,lay):

        '''
        Description
        -----------

        This function defines the coordinates of pilot points 
       
        Parameters
        ----------
        zone (int) : area of a layer where pilot points would be used 
        n_cell (int) : Number of cells between pilot points 
        izone_2d (2d np.array) : layer where poilot points are genrated 
        x_vals (1d np.array) :  grid x coordinates 
        y_vals (1d np.array) :  grid y coordinates 
     
        Returns
        ------
        pp_x : 1d np.array pilot points x coordinates  
        pp_y : 1d np.array pilot points y coordinates 
        
        Example
        -----------
        pp_x, pp_y = get_rgrid_pp(zone,n_cell,izone_2d,x_vals,y_vals)
        
        '''
        izone_2d = self.izone[lay,:,:]

        x_vals = self.mm.x_vals
        y_vals = self.mm.y_vals
        nrow = self.mm.nrow
        ncol = self.mm.ncol
            
        rows = range(0,nrow,n_cell)
        cols = range(0,ncol,n_cell)
        
        srows,scols = np.meshgrid(rows,cols)

        pp_select = np.zeros((nrow,ncol))
        pp_select[srows,scols] = 1

        pp_select[izone_2d != zone] = 0

        xx, yy = np.meshgrid(x_vals,y_vals)
        
        pp_x  = xx[pp_select==1].ravel()
        pp_y  = yy[pp_select==1].ravel()

        return pp_x,pp_y


    def init_array(self, array) : 
        if array is None :
            # initialize array for given parameter 
            self.array = np.array(self.mm.imask,dtype=np.float)
            # fill array with nan within mask
            self.array[ self.array != 0 ] = np.nan
        elif isinstance(array, np.ndarray) : 
            assert array.shape == (self.mm.nlay, self.mm.nrow, self.mm.ncol)
            self.array = array

    def read_zpc_df(self,filename = None) :
        if filename is None:
            filename = self.in_file_zpc
        self.zpc_df = pd.read_csv(filename, delim_whitespace=True,
                header=None,names=['parname','value'],usecols=[0,1])
