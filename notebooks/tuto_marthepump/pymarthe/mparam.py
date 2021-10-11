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
from .utils import pp_utils
import pandas as pd 
import pyemu

# ---  formatting

# NOTE : layer are written with 1-base format, unsigned zone for zpc
# name format zones of piecewise constancy
# example: kepon_l02_zpc03
ZPCFMT = lambda name, lay, zone: '{0}_l{1:02d}_zpc{2:02d}'.format(name,int(lay+1),int(abs(zone)))

# float and integer formats
FFMT = lambda x: "{0:<20.10E} ".format(float(x))
IFMT = lambda x: "{0:<10d} ".format(int(x))

# columns for pilot point df
PP_NAMES = ["name","x","y","zone","value"]

# string format
def SFMT(item):
    try:
        s = "{0:<20s} ".format(item.decode())
    except:
        s = "{0:<20s} ".format(str(item))
    return s

# name format for pilot points files  
PP_FMT = {"name": SFMT, "x": FFMT, "y": FFMT, "zone": IFMT, "tpl": SFMT, "value": FFMT, "log_value": FFMT}

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
    array : parameter value


    Examples
    --------
    """
    def __init__(self, mm, name, default_value, izone = None, array = None, log_transform = False) :
        self.mm = mm # pointer to the instance of MartheModel
        self.name = name # parameter name
        self.array = mm.grids[self.name] # pointer to grid in MartheModel instance
        self.default_value = default_value 
        self.log_transform = log_transform
        self.set_izone(izone)
        self.base_spacing = {} # resolution of the main pilot point grid
      
    def set_izone(self,izone = None):
        """
        Set izone 3D array (nlay, nrow, ncol) of integer
        Former izone data for given parameter will be reset.

        Where mm.imask values are 0, izone data are fixed to 0

        izone value < 0, zone of piecewise constancy
        izone value > 0, zone with pilot points
        izone value = 0 for inactive cells

        If data is None, a single constant zone (-1) per layer is considered.

        Parameters
        ----------
        izone : None, int or np array of int, shape (nlay,nrow,ncol)

        Examples
        --------
        >> mm.param['kepon'].set_izone()

        >> izone = -1*np.ones( (nlay, nrow, ncol) )
        >> izone[0,:,:] = 1
        >> mm.param['kepon'].set_izone(izone)

        """
        # case izone is not provided
        if izone is None :
            # reset izone for current parameter from imask
            self.izone = self.mm.imask.copy()
            # index of active cells
            idx_active_cells = self.mm.imask == 1
            # a single zone is considered
            self.izone[idx_active_cells] = -1

        # case an izone array is provided
        elif isinstance(izone,np.ndarray) : 
            assert izone.shape == (self.mm.nlay, self.mm.nrow, self.mm.ncol) 
            # set izone as provided
            self.izone = izone

        # update zpc_df and pp_dic
        # this resets any modification applied to zpc_df and pp_df
        self.init_zpc_df()
        self.init_pp_dic()


    def init_pp_dic(self) :
        """
        Initialize the nested dic of pp_df
        Example format : 
        pp_dic = { 0:pp_df_0, 3:pp_df_3 }
        """
        nlay = self.mm.nlay
        self.pp_dic  = {} # dict of pandas dataframe 
        self.gs_dic = {} # dict of pyemu covariance matrix 
        self.ppcov_dic = {} # dict of pyemu covariance matrix 
      
        # append lay to pp_dic if it contains zones > 0
        for lay in range(nlay) :
            zones = np.unique(self.izone[lay,:,:])
            # number of zones > 0 
            if len( [zone for zone in zones if zone >0] ) > 0 :
                self.pp_dic[lay] = None
                self.ppcov_dic[lay] = None

    def init_zpc_df(self) :
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

        self.zpc_df = pd.DataFrame({'name':parnames, 'lay':parlays, 'zone':parzones, 'value': self.default_value})
        self.zpc_df.set_index('name',inplace=True)

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
            self.zpc_df['value'] = value
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

    
    def set_array(self, value, lay = None, zone = None) : 
        """
        Set value for given layer(s) and zone(s)
        Value can be a float or array of float.
        Only active cells can be set (zone !=0)

        Parameters
        ----------
        value = int or np.ndarray
            int for zones < 0, ndarray for zone > 0 
        lay : None, int or list of int. 
              layer, zero based. If None, all layers considered
        zone : None, int or list of int
            zone id (<0 or >0). If None, all zones considered

        Example
        -------
        # set value to all layers, all zones
        mm.param['kepon'].set_array(value = 12.4e-3)

        # set value to layer 1, zone -2
        mm.param['kepon'].set_array(value = 12.4e-3, lay = 1, zone = -2, )

        # provide a 2D array
        a = 1e-3*np.ones( (nrow,ncol) )
        mm.param['kepon'].set_array(value = , lay=2 )

        # provide a 3D array
        a = 1e-3*np.ones( (nlay,nrow,ncol) )
        mm.param['kepon'].set_array(value = a
        >> 
        """

        # initialize layers
        if lay is None : 
            # all layers considered
            layers  = range(self.mm.nlay)
        elif isinstance(lay,int):
            layers = [lay]
        elif isinstance(lay,list):
            layers = lay

        # check and initialize zones
        if zone is None :
            # all zones considered
            zones = [z for z in np.unique(self.izone) if z != 0]
        elif isinstance(zone, int) : 
            zones = [zone]
        elif isinstance(zone, list) :
            # keep only valid zone values 
            zones = [z for z in zone if (z in np.unique(self.izone) and z !=0)]

        # set from array
        if isinstance(value, np.ndarray) :
            if value.ndim == 3 :
                for zone in zones : 
                    idx = self.izone == zone
                    self.array[idx] = value[idx]
            if value.ndim == 2 :
                for lay in layers : 
                    for zone in zones :
                        idx = self.izone[lay,:,:] == zone
                        self.array[lay,:,:][idx] = value[idx]
        # set from single value
        else :
            for lay in layers : 
                for zone in zones : 
                    idx = self.izone[lay,:,:] == zone
                    self.array[lay,:,:][idx] = value

        return

    def set_array_from_zpc_df(self) :
        # check for missing values
        for lay, zone, value in zip(self.zpc_df.lay, self.zpc_df.zone, self.zpc_df.value) :
            if value is None :
                print('Parameter value is NA for ZPC zone {0} in lay {1}').format(abs(zone),int(lay)+1)
            self.set_array(value, lay, zone)

    def pp_df_from_shp(self, shp_path, lay, zone = 1, value = None , zone_field = None, value_field = None) :
        """
       Reads input shape file, builds up pilot dataframe, and insert into pp_dic 

        Parameters
        ----------
        path : full path to shp with pilot points
        lay : layer id (0-based)
        zone : zone id (>0)

        Examples
        --------
        mm.param['permh'].pp_df_from_shp('./data/points_pilotes_eponte2.shp', lay, zone)

        """
        if value is None : 
            value = self.default_value

        # init pp name prefix
        prefix = 'pp_{0}_l{1:02d}'.format(self.name,lay)
        # get data from shp and update pp_df for given layer
        # NOTE will be further extended for multi-zones
        # and allow update of current pp_d
        self.pp_dic[lay] = pp_utils.ppoints_from_shp(shp_path, prefix, zone, value, zone_field, value_field)
        
    def zone_interp_coords(self, lay, zone) :
        """
        Returns grid coordinates where interpolation
        should be performed for given lay and zone

        Parameters
        ----------
        lay: model layer (0-based)
            parameter
        zone : zone layer (>0 for pilot points)

        Examples
        --------
        x_coords, y_coords = mm.param['permh'].zone_interp_coords(lay=0,zone=1)
        
        """
        # set up index for current zone and lay

        # point where interpolation shall be conducted for current zone
        idx = self.izone[lay,:,:] == zone
        xx, yy = np.meshgrid(self.mm.x_vals, self.mm.y_vals)
        x_coords = xx[idx].ravel()
        y_coords = yy[idx].ravel()

        return(x_coords, y_coords)

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
            zpc_df = pd.DataFrame({'name' : zpc_names,'tpl' : tpl_entries})
            pest_utils.write_tpl_from_df(os.path.join(self.mm.mldir,'tpl',filename), zpc_df)
        else : 
            print('No ZPC identified for parameter {0} in izone data.'.format(self.name))

    def write_pp_tpl(self) : 
        """ 
        set up and write template files for pilot points
        one file per model layer

        """
        for lay in self.pp_dic.keys():
            pp_df = self.pp_dic[lay]
            tpl_filename = '{0}_pp_l{1:02d}.tpl'.format(self.name,lay+1)
            tpl_file = os.path.join(self.mm.mldir,'tpl',tpl_filename)
            tpl_entries = ["~  {0}  ~".format(parname) for parname in pp_df.index]
            pp_tpl_df = pd.DataFrame({
                'name' : pp_df.index ,
                'x': pp_df.x,
                'y': pp_df.y,
                'zone': pp_df.zone,
                'tpl' : tpl_entries
                })
            pest_utils.write_tpl_from_df(tpl_file, pp_tpl_df, columns = ["name", "x", "y", "zone", "tpl"] )

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
            f_param = open(os.path.join(self.mm.mldir,'param',filename),'w')
            # log-transform values to parameter file 
            if self.log_transform == True :
                self.zpc_df['log_value'] = np.log10(self.zpc_df['value'])
                f_param.write(self.zpc_df.to_string(col_space=0,
                  columns=["log_value"],
                  formatters={'log_value':FFMT},
                  justify="left",
                  header=False,
                  index=True,
                  index_names=False))
            else :
                f_param.write(self.zpc_df.to_string(col_space=0,
                                  columns=["value"],
                                  formatters={'value':FFMT},
                                  justify="left",
                                  header=False,
                                  index=True,
                                  index_names=False))


    def write_pp_df(self):
        """
        write pp_df to files
        """
        for lay in self.pp_dic.keys():
            # pointer to pp_df for current layer
            pp_df = self.pp_dic[lay]
            # set up output file 
            pp_df_filename = '{0}_pp_l{1:02d}.dat'.format(self.name, lay+1)
            pp_df_file = os.path.join(self.mm.mldir,'param',pp_df_filename)
            # write output file 
            f_param = open(pp_df_file,'w')
            if self.log_transform == True :
                pp_df['log_value'] = np.log10(pp_df['value'])
                f_param.write(pp_df.to_string(col_space=0,
                                  columns=["x", "y", "zone", "log_value"],
                                  formatters=PP_FMT,
                                  justify="left",
                                  header=False,
                                  index=True,
                                  index_names=False))
            else : 
                f_param.write(pp_df.to_string(col_space=0,
                                  columns=["x", "y", "zone", "value"],
                                  formatters=PP_FMT,
                                  justify="left",
                                  header=False,
                                  index=True,
                                  index_names=False))

    def pp_from_rgrid(self, lay, n_cell, n_cell_buffer = False):
        '''
        Description
        -----------
        This function sets up a regular grid of pilot points 
        NOTE : current version does not handle zone 
       
        Parameters
        ----------
        lay (int) : layer for which pilot points should be placed
        zone (int) : zone of layer where pilot points should be placed 
        n_cell (int) : Number of cells between pilot points 
        n_cell_buffer (Bool or int) : Buffer around the zone. If True, value = n_cell/2.
                                      This is useful to include pilot points that lie close to the zone
    
        Returns
        ------
        pp_x : 1d np.array pilot points x coordinates  
        pp_y : 1d np.array pilot points y coordinates 
        
        Example
        -----------
        pp_x, pp_y = pp_from_rgrid(lay, zone, n_cell)
        
        '''
        # current version does not handle zones
        zone = 1 

        # set base spacing (model coordinates unit) from n_cell
        self.base_spacing[lay] = n_cell*self.mm.cell_size

        izone_2d = self.izone[lay,:,:]

        x_vals = self.mm.x_vals
        y_vals = self.mm.y_vals
        xx, yy = np.meshgrid(x_vals,y_vals)

        nrow = self.mm.nrow
        ncol = self.mm.ncol
            
        rows = range(0,nrow,n_cell)
        cols = range(0,ncol,n_cell)
        
        srows, scols = np.meshgrid(rows,cols)

        pp_select = np.zeros((nrow,ncol))
        pp_select[srows,scols] = 1

        buffered_zone = izone_2d.copy() 

        if n_cell_buffer is not False :
            # set to default value if value is not provided
            if n_cell_buffer is True : 
                n_cell_buffer = np.int(n_cell/2)
            # iterate over every cells from the zone and apply buffer 
            for i,j in np.argwhere(izone_2d==zone):
                # identify min and max indices of the buffer for current cell
                i_min = max(0,i-n_cell_buffer)
                i_max = min(izone_2d.shape[0] - 1, i+n_cell_buffer+1)
                j_min = max(0,j-n_cell_buffer)
                j_max = min(izone_2d.shape[1] - 1, j+n_cell_buffer+1)
                # get all indices within buffer for current cell
                igrid = np.mgrid[i_min:i_max,j_min:j_max]
                # set values within buffer to zone value for current cell
                buffered_zone[ igrid[0].ravel(), igrid[1].ravel() ] = zone

        # select points within (buffered) zone 
        pp_select[buffered_zone != zone] = 0
        pp_x  = xx[pp_select==1].ravel()
        pp_y  = yy[pp_select==1].ravel()

        # number of selected pilot points
        n_pp = len(pp_x)

        # name pilot points
        prefix = '{0}_l{1:02d}_z{2:02d}'.format(self.name,lay+1,zone)
        pp_names = [ '{0}_{1:03d}'.format(prefix,id) for id in range(n_pp)  ]
        
        # build up pp_df
        pp_df = pd.DataFrame({"name":pp_names,"x":pp_x,"y":pp_y, "zone":zone, "value":self.default_value})
        pp_df.set_index('name',inplace=True)
        pp_df['name'] = pp_df.index

        self.pp_dic[lay] = pp_df

    def get_pp_spacing(self, pp_id, lay):
        '''
        Description
        -----------
        Computes the spacing of the pilot point grid from which pp_id pertains

        Parameters
        ----------
        pp_id (str) : pilot point id at which spacing should be computed
        lay (int) : layer

        Example
        -----------
        spacing = get_pp_spacing(pp_id = pp_id, lay=2)

        '''
        # pilot point dataframe 
        assert self.pp_dic[lay] is not None
        df = self.pp_dic[lay]

        # get base spacing 
        assert self.base_spacing[lay] is not None 
        base_spacing= self.base_spacing[lay]

        # only points sharing the same x or y values are selected
        # to make sure to select pilot points from the same generation
        # base_spacing is required for 1st generation points
        # find nearest neighbors along x 
        pp_same_x = df['x'] == df.loc[pp_id,'x']
        pp_same_x.loc[pp_id] = False #remove pp_id
        try : 
            pp_dist_y = min(abs(df.loc[pp_same_x,'y'] - df.loc[pp_id,'y']))
        except :
            pp_dist_y = base_spacing
        # find nearest neighbor along y 
        pp_same_y = df['y'] == df.loc[pp_id,'y']
        pp_same_y.loc[pp_id] = False #remove pp_id 
        try : 
            pp_dist_x = min(abs(df.loc[pp_same_y,'x'] - df.loc[pp_id,'x']))
        except :
            pp_dist_x = base_spacing
        # get minimum value
        spacing = min(pp_dist_x, pp_dist_y,base_spacing)

        return(spacing)

    def get_pp_nobs(self, lay, loc_df):
        '''
        Description
        -----------
        Returns pp_df of current layer with a new "nobs" column
        corresponding to the number of observations
        lying within the cell centered on each pilot point.
        Useful for pilot point meshing adaptive 
        to observation density.

        Parameters
        ----------
        lay (int) : model layer
        loc_df (pd.DataFrame) : location dataframe from 
                                marthe_utils.read_histo())
        
        Example
        -----------
        mm.param['kepon'].get_pp_nobs(lay, loc_df)

        '''
        # get current pilot point dataframe 
        assert self.pp_dic[lay] is not None
        pp_df = self.pp_dic[lay] 

        # init output list with number of obs.
        nobs_list = []

        # iterate over points
        for pp_id in pp_df.index :
            # get spacing for current pilot point
            spacing = self.get_pp_spacing(pp_id, lay)
            # get coordinates of search cell (square centered on current pp)
            xmin = pp_df.loc[pp_id,'x'] - spacing/2.
            xmax = pp_df.loc[pp_id,'x'] + spacing/2.
            ymin = pp_df.loc[pp_id,'y'] - spacing/2.
            ymax = pp_df.loc[pp_id,'y'] + spacing/2.
            loc_df_lay = loc_df[loc_df.layer ==lay+1]
            # get number of observation locations in the cell
            idx = (loc_df_lay['x'] > xmin) & (loc_df_lay['x'] < xmax) & \
                    (loc_df_lay['y'] > ymin) & (loc_df_lay['y'] < ymax)
            nobs = idx.sum()
            nobs_list.append(nobs)

        # add new column and return dataframe 
        pp_df_nobs = pp_df.copy()
        pp_df_nobs['nobs'] = nobs_list
        
        return(pp_df_nobs)

    def pp_refine(self, lay, df, n_cell, level = 1, interpolate = True,remove_parents = False):
        '''
        Description
        -----------
        Sets an existing set of pilot points for current parameter given a boolean 'refine' column. 
        Pilot points selected for refinements are split into 4 new points when level=1, 16 points when level=2.
        NOTE : current version does not handle zones 

        Parameters
        ----------
        lay (int) : layer for which pilot points should be placed
        df : pandas dataframe with (at least) a 'refine' column and pp names as index
             
        n_cell : spacing in number of model cells of the main grid of pilot point 
        level : (default, 1) refinement level (1 = 1 pp -> 4 pp ; 2 = 1 pp -> 16 pp, ...)
        interpolate (bool, default True) : whether refined pilot points should be interpolated 
                 from current pp values. When False, parent values are inherited without interpolation. 

        Example
        -----------
        mm.param['kepon'].pp_refine(lay=2, df = df, n_cell= 10)

        '''
        # zone currently not handled
        zone = 1

        # base spacing (length) of initial pilot point grid obtained with pp_from_rgrid()
        # inferred from n_cell and cell size
        base_spacing = n_cell * self.mm.cell_size

        # initialize new pilot point df from copy of current pp_df
        pp_df = self.pp_dic[lay].copy()

        # select points to refine given criteria in df
        pp_select = df.loc[df['refine'] == True] 

        print('Refining {0} pilot points ' \
                'for parameter {1}, layer {2}'.format(pp_select.shape[0], self.name,lay+1))
        print('Base spacing of main regular grid is {} [model distance unit]'.format(self.base_spacing[lay]))

        # init output lists with new points data
        new_pp_xvals, new_pp_yvals, new_pp_values = [], [], []

        # iterate over refinement level
        for n in range(level) :

            # iterate over points to refine
            for pp_id in pp_select.index :
                spacing = self.get_pp_spacing(pp_id, lay)
                # compute coordinate increment
                coord_inc = spacing/4.
                # add 4 new points
                pp_id_x = pp_df.loc[pp_id,'x']
                pp_id_y = pp_df.loc[pp_id,'y']
                # counter-clockwise from upper-right
                new_pp_xvals.extend([pp_id_x + coord_inc,pp_id_x - coord_inc,
                        pp_id_x - coord_inc,pp_id_x + coord_inc])
                new_pp_yvals.extend([pp_id_y + coord_inc,pp_id_y + coord_inc,
                        pp_id_y - coord_inc,pp_id_y - coord_inc])
                # set new pp value to parent pp value
                new_pp_values.extend([pp_df.loc[pp_id,'value']]*4)

            # create new dataframe 
            new_pp_df = pd.DataFrame({
                'x':new_pp_xvals,
                'y':new_pp_yvals,
                'zone':zone,
                'value':new_pp_values,
                # new pp get True value in refine column
                'refine': True # necessary when merge > 1
                })

            # interpolate values for new points by ordinary kriging 
            if interpolate == True :
                # variogram range inferred from n_cell (2 times base pilot point spacing)
                vario_range = 2*self.mm.cell_size*n_cell
                # set up  PyEMU OrdinaryKriging instance 
                v = pyemu.utils.geostats.ExpVario(contribution=1, a=vario_range)
                transform = 'log' if self.log_transform == True else 'none'
                gs = pyemu.utils.geostats.GeoStruct(variograms=v,transform="log")
                ok = pyemu.utils.geostats.OrdinaryKrige(geostruct=gs,point_data=pp_df)
                ok.spatial_reference = self.mm.spatial_reference 
                # compute kriging factors
                x_coords, y_coords = new_pp_df.x, new_pp_df.y
                kfac_df = ok.calc_factors(x_coords, y_coords,pt_zone=1)
                # write kriging factors to file
                kfac_file = os.path.join(self.mm.mldir,'kfac_temp_refine_{0}_l{1:02d}_z{2:02d}.dat'.format(self.name,lay+1,zone))
                ok.to_grid_factors_file(kfac_file) 
                # fac2real (requires integer index)
                kriged_values_df = pp_utils.fac2real(pp_file = pp_df.reset_index(drop=True) ,factors_file = kfac_file, kfac_sty='pyemu')
                # fetch interpolated values
                new_pp_df['value'] = kriged_values_df['vals'].values
                # remove factor file 
                os.remove(kfac_file)

            # remove refined pp_id  
            if remove_parents == True:
                pp_df.drop(pp_select.index,inplace=True)
            # extend pp_df with new points
            pp_df = pp_df.append(new_pp_df)
            pp_df = pp_df.drop_duplicates(["x","y"],keep = 'first')

            # rename all pilot point in pp_df and set index
            n_pp = pp_df.shape[0]
            prefix = '{0}_l{1:02d}_z{2:02d}'.format(self.name,lay+1,zone)
            pp_names = [ '{0}_{1:03d}'.format(prefix,id) for id in range(n_pp)  ]
            pp_df['name'] = pp_names
            pp_df.set_index('name', inplace=True, drop=False)
            #  update pp_df in pp_dic
            self.pp_dic[lay] = pp_df[['name', 'x', 'y', 'zone', 'value']]

            # reset pp_df, pp_select, and output lists (necessary when level > 1)
            pp_select = pp_df.loc[pp_df['refine'] == True]
            pp_df = self.pp_dic[lay].copy()
            new_pp_xvals, new_pp_yvals, new_pp_values = [], [], []

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
        df = pd.read_csv(os.path.join(self.mm.mldir,'param',filename), delim_whitespace=True,
                header=None,names=['name','value'], usecols=[0,1])
        df.set_index('name',inplace=True)
        
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
        df = pd.DataFrame({'name':parnames, 'value':values})
        df.set_index('name', inplace=True)
        self.zpc_df = pd.merge(self.zpc_df, df, how='left', left_index=True, right_index=True)

        # back transform to natural values if they are log-transformed in the parameter file 
        if self.log_transform == True :
            self.zpc_df['value'] = 10**np.array(self.zpc_df.value_y)
        else :
            self.zpc_df['value'] = self.zpc_df.value_y

        self.zpc_df.drop(['value_x','value_y'],1, inplace=True)

        # check for missing parameters
        missing_parnames = self.zpc_df.index[ self.zpc_df.value == np.nan ]
        
        if len(missing_parnames) > 0 : 
            print('Following parameter values are missing in zpc parameter file:\n{0}'.format(' '.join(missing_parnames))
                    )

        return

    def read_pp_df(self):
        """
        Read pp_df for all layers and fill self.pp_dic
        """
        for lay in self.pp_dic.keys():
            # read dataframe
            filename = '{0}_pp_l{1:02d}.dat'.format(self.name,lay+1)
            pp_file = os.path.join(self.mm.mldir,'param',filename)
            pp_df = pd.read_csv(pp_file, delim_whitespace=True,
                    header=None,names=PP_NAMES, index_col='name')
            pp_df['name'] = pp_df.index

            # back transform to natural values if they are log-transformed in the parameter file 
            if self.log_transform == True: 
                pp_df['value'] = 10**np.array(pp_df['value'])

            # set pp_df for current layer
            self.pp_dic[lay]=pp_df


    def interp_from_factors(self,kfac_sty='pyemu'):
        """
        Interpolate from pilot points df files with fac2real()
        and update parameter array
        """
        for lay in self.pp_dic.keys():
            # select zones with pilot points
            zones = [zone for zone in np.unique(self.izone[lay,:,:]) if zone >0]
            for zone in zones : 
                # path to factor file
                kfac_filename = 'kfac_{0}_l{1:02d}.dat'.format(self.name,lay+1)
                kfac_file = os.path.join(self.mm.mldir,kfac_filename)
                # fac2real (requires integer index)
                pp_df = self.pp_dic[lay].reset_index(drop=True) # reset index (names)
                kriged_values_df = pp_utils.fac2real(pp_file = pp_df ,factors_file = kfac_file,kfac_sty=kfac_sty)
                #kriged_values_df = pyemu.fac2real(pp_file = pp_df ,factors_file = kfac_file)
                # update parameter array
                idx = self.izone[lay,:,:] == zone
                # NOTE for some (welcome) reasons it seems that the order of values
                # from self.array[lay][idx] and kriged_value_df.vals do match
                self.array[lay][idx] = kriged_values_df.vals

        
    def plproc_kfac(self,lay,zone=1) :
        """
        -------- UNDER DEVELOPMENT  ----------------------

        
        # get grid cell coordinates where interpolation shall be conducted
        x_coords, y_coords = self.zone_interp_coords(lay,zone)
        # write coordinate list for plproc
        clist_df = pd.DataFrame( {'x':x_coords , 'y':y_coords, 'zone':zone })
        clist_filename = open('clist_{0}_l{1:02d}_z{2:02d}.dat'.format(par,lay+1,zone),'w')
        clist_file.write(clist_df.to_string(col_space=0,
            formatters={'x':FFMT,'y':FFMT},
            justify="left",
            header=True,
            index=True)
            )
        clist_file.close()
        # write script file for plproc

        # write plist

        # write plproc script
        # call plproc

        # perform interpolation

        mm.param[par].read_pp_df()
        pp_df = mm.param[par].pp_dic[lay].reset_index(drop=True)

        kriged_values_df = pp_utils.fac2real(pp_file = pp_df ,factors_file = 'fac_permh_pp_l04.dat',kfac_sty='plproc')

        idx = mm.param[par].izone[lay,:,:] == zone
        mm.param[par].array[lay][idx] = kriged_values_df.vals
        """


