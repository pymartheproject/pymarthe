"""
Contains the MartheModel class
Designed for structured grid and
layered parameterization

"""
import os 
import numpy as np
from matplotlib import pyplot as plt 
from .utils import marthe_utils
from .utils import pest_utils
import pandas as pd 
import pyemu

# ---  formatting

# NOTE : layer are written with 1-base format, unsigned zone for zpc
# name format zones of piecewise constancy
# example: kepon_l02_zpc03
ZPCFMT = lambda name, lay, zone: '{0}_l{1:02d}_zpc{2:02d}'.format(name,int(lay+1),int(abs(zone)))

# name format for pilot points  
# example : kepon_l02_z02_pp001
PPFMT = lambda name, lay, zone, ppid: '{0}_l{1:02d}_z{2:02d}_{3:03d}'.format(name,int(lay+1),int(zone),int(ppid)) 

# string format
def SFMT(item):
    try:
        s = "{0:<20s} ".format(item.decode())
    except:
        s = "{0:<20s} ".format(str(item))
    return s

# float format
FFMT = lambda x: "{0:<20.10E} ".format(float(x))

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
        self.mm = mm # pointer to the instance of MartheModel
        self.name = name # parameter name
        self.array = mm.grids[self.name] # pointer to grid in MartheModel instance
        self.default_value = default_value 
        self.set_izone(izone)
      
    def set_izone(self,izone = None):
        """
        Load izone array from data. 
        If data is None, a single constant zone per layer is considered. 
        Former izone data for given parameter will be reset. 


        Parameters
        ----------
        data : None, int or np array of int, shape (nlay,nrow,ncol)

        Examples
        --------

        """
        # reset izone for current parameter from imask
        self.izone = self.mm.imask.copy()
        # index of active cells
        idx_active_cells = self.mm.imask == 1

        # case izone is not provided
        if izone is None :
            # a single zone is considered
            self.izone[idx_active_cells] = -1

        # case an izone array is provided
        elif isinstance(izone,np.ndarray) : 
            assert izone.shape == (nlay,nrow,ncol) 
            # only update active cells  
            self.izone[idx_active_cells] = izone[idx_active_cells]

        # update zpc_df and pp_dic
        self.update_zpc_df()
        self.update_pp_dic()


    def update_pp_dic(self) :
        """
        Parameters
        ----------
        key : str
            
        Examples
        --------
        """
        nlay = self.mm.nlay
        self.pp_dic  = {lay:[] for lay in range(nlay) }
      
        for lay in range(nlay) :
            zones = np.unique(self.izone[lay,:,:])
            for zone in zones :
                if zone > 0 :
                    self.pp_dic[lay].append(zone)

    def update_zpc_df(self) :
        """
        Set up dataframe of zones of piecewise constancy from izone
        """

        nlay = self.mm.nlay

        parnames = []
        parlays = []
        parzones = []

        for lay in range(nlay) :
            zones = np.unique(self.izone[lay,:,:])
            for zone in zones :
                if zone < 0 : 
                    parnames.append( ZPCFMT(self.name, lay, zone))
                    parlays.append(lay)
                    parzones.append(zone)

        self.zpc_df = pd.DataFrame({'parname':parnames, 'lay':parlays, 'zone':parzones, 'value': self.default_value})
        self.zpc_df.set_index('parname',inplace=True)

    def set_zpc_values(self,values) : 
        """
        Update value column inf zpc_df

        Parameters
        ---------
        values : int or dic
            if an int, is provided, the value is set to all zpc
            if a dic is provided, the value are updated accordingly
            ex :
            layer only value assignation : simple dic 
            {0:1e-2,1:2e-3,...}
            layer, zone value assignation : nested dicts
            {0:{1:1e-3,2:1e-2}} to update the values of zones 1 and 2 from the first layer. 

        Examples :
        >> # a single value is provided
        >> mm.set_zpc_values(1e-3)
        >> # layer based value assignement 
        >> mm.set_vpc_values({0:2e-3})
        >> # layer and zone assignement
        >> mm.set_zpc_values({0:{1:1e-3,2:1e-2}}
        """

        # case same value for all zones of all layers
        if isinstance(values,(int, float)) : 
            self.zpc_df['values'] = value
            return
        # if a dictionary is provided
        elif isinstance(values, dict) :
            for lay in list(values.keys()):
                # layer-based parameter assignement
                if isinstance(values[lay],(int,float)):
                    # index true for zones within lay
                    value = values[lay]
                    idx = [ name.startswith('{0}_l{1:02d}'.format(self.name,int(lay+1))) for name in self.zpc_df.index]
                    self.zpc_df.loc[idx,'value'] = value
                # layer, zone parameter assignement
                elif isinstance(values[lay],dict) : 
                    for zones in values[lay].keys() :
                        parname = ZPCFMT(self.name,lay,zone)
                        value = values[lay][zone]
                        self.zpc_df.loc[parname,'value'] = value
        else : 
            print('Invalid input, check the help')
            return

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
        assert lay in range(self.mm.nlay), 'layer {0} is not valid.'.format(lay)
        assert zone in np.unique(self.izone), 'zone {0} is not valid'.format(zone)
        if zone < 0 :
            assert isinstance(values, float), 'A float should be provided for ZPC.'
        elif zone > 0 :
            assert isinstance(values, np.ndarray), 'An array should be provided for PP'

        # select zone 
        idx = self.izone[lay,:,:] == zone

        # update values within zone 
        self.array[lay,:,:][idx] = values

        return

    def set_array_from_zpc_df(self) :
        # check for missing values
        for lay, zone, value in zip(self.zpc_df.lay, self.zpc_df.zone, self.zpc_df.value) :
            if value is None :
                print('Parameter value is NA for ZPC zone {0} in lay {1}').format(abs(zone),int(lay)+1)
            self.set_array(lay,zone,value)

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


    def write_zpc_tpl(self, filename = None):
        """
        Load izone array from data. 
        If data is None, a single constant zone is considered. 
        Former izone data for given parameter, will be reset. 

        Parameters
        ----------
        key : str
            filename, default value is name_zpc.tpl
            ex : permh_zpc.tpl

        """

        if filename is None : 
            filename = self.name + '_zpc.tpl'
        
        zpc_names = self.zpc_df.index

        if len(zpc_names) > 0 :
            tpl_entries = ["~  {0}  ~".format(parname) for parname in zpc_names]
            zpc_df = pd.DataFrame({'parname' : zpc_names,'tpl' : tpl_entries})
            pest_utils.write_tpl_from_df(os.path.join(self.mm.mldir,'tpl',filename), zpc_df)

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

    def write_zpc_data(self, filename = None):
        """

        Parameters
        ----------
        key : str
            filename, default value is name_zpc.tpl
            ex : permh_zpc.tpl

        """

        if filename is None : 
            filename = self.name + '_zpc.dat'
        
        zpc_names = self.zpc_df.index

        if len(zpc_names) > 0 :
            f_param = open(os.path.join('param',filename),'w')
            f_param.write(self.zpc_df.to_string(col_space=0,
                              columns=['value'],
                              formatters={'value':FFMT},
                              justify="left",
                              header=False,
                              index=True,
                              index_names=False))

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

    def read_zpc_df(self,filename = None) :
        """
        Reads dataframe with zones of piecewise constancy
        and sets self.zpc_df accordingly

        The file should be white-space-delimited, without header.
        First column : parameter name (with ZPCFMT format)
        Second column : parameter value
        Any additional column will not be considered
        ex : 
        kepon_l01_z01 0.01
        kepon_l02_z02 0.03
        ...

        Parameters
        ----------
        filename : str (optional)
            path to parameter file, default is permh_zpc.dat
            ex : permh_zpc.dat
        
        """

        if filename is None:
            filename = self.name + '_zpc.dat'

        # read dataframe
        df = pd.read_csv(os.path.join('param',filename), delim_whitespace=True,
                header=None,names=['parname','value'], usecols=[0,1])
        df.set_index('parname',inplace=True)
        
        # parse layer and zone from parameter name 
        parnames = []
        values = []
        not_found_parnames = []

        for parname in df.index :
            if parname in self.zpc_df.index : 
                parnames.append(parname)
                values.append(df.loc[parname,'value'])
            else :
                not_found_parnames.append(parname)

        # case not any parameter values found
        if len(parnames) == 0 :
            print('No parameter values could be found.\n'
            'Check compatibility with izone data.')
            return
        
        # case some parameter not found
        if len(not_found_parnames) >0 :
            print('Following names are not compatible with {0} parameter izone :\n'
            '{1}'.format(self.name, ' '.join(not_found_parnames))
                    )
        # merge new dataframe with existing zpc_df
        df = pd.DataFrame({'parname':parnames, 'value':values})
        df.set_index('parname', inplace=True)
        self.zpc_df = pd.merge(self.zpc_df, df, how='left', left_index=True, right_index=True)

        self.zpc_df['value'] = self.zpc_df.value_y

        self.zpc_df.drop(columns=['value_x','value_y'], inplace=True)

        # check for missing parameters
        missing_parnames = self.zpc_df.index[ self.zpc_df.value.isna() ]
        
        if len(missing_parnames) > 0 : 
            print('Following parameter values are missing in zpc parameter file:\n{0}'.format(' '.join(missing_parnames))
                    )

        return

