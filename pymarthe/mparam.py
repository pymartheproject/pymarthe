
"""
Contains the MartheModel class
Designed for structured grid and
layered parameterization

"""
import os 
import numpy as np
from matplotlib import pyplot as plt 
from .utils import marthe_utils, pest_utils, pp_utils, shp_utils
import pandas as pd 
import pyemu
from pymarthe.mfield import MartheField
import warnings
from copy import deepcopy


# ---- SET UP FORMATTERS ---- #

ZPCFMT = lambda name, lay, zone: '{0}_zpc_l{1:02d}_z{2:02d}'.format(name,int(lay),int(abs(zone)))
FFMT = lambda x: "{0:<20.10E} ".format(float(x))
IFMT = lambda x: "{0:<10d} ".format(int(x))
def SFMT(item):
    try:
        s = "{0:<20s} ".format(item.decode())
    except:
        s = "{0:<20s} ".format(str(item))
    return s

PP_NAMES = ["name","x","y","zone","value"]
PP_FMT = {"name": SFMT, "x": FFMT, "y": FFMT, "zone": IFMT, "tpl": SFMT, "value": FFMT, "log_value": FFMT}



base_param = ['parnme', 'trans', 'btrans', 'parchglim',
                  'defaultvalue', 'parlbnd', 'parubnd',
                  'pargp', 'scale', 'offset', 'dercom']



class MartheListParam():
    """
    Class for handling Marthe list-like properties. 
    """
    def __init__(self, parname, mobj, kmi, value_col = 'value', trans = 'none', 
                       btrans = 'none', defaultvalue=None, **kwargs):
        """
        Generator of list parameter instance base on `kmi` (KeysMultiIndex).

        Parameters
        ----------
        parname (str) : parameter name

        mobj (object) : a parametrable Marthe object property.
                        Can be :
                            - MartheSoil ('aqpump'/'rivpump')
                            - MarthePump

        kmi (pandas.MultiIndex) : KeysMultiIndex that correspond to the column names of required
                                  DataFrame of the `mobj` to be consider as changing parameters
                                  (ex: ''istep', 'layer', 'soilprop', ...). 
                                  Note: `kmi` generation can be done easier with the helper fonction
                                  pymarthe.utils.pest_utils.get_kmi().

        value_col (str, optional) : the column name of required DataFrame of the `mobj` to be
                                    consider as the parameter value(s).
                                    Default is 'value'.

        trans (str, optional) : transformation to apply to the initial parameter value(s).
                                If 'none', parameter values will not be transformed.
                                Default is 'none'.

        btrans (str, optional) : back-transformation to apply to the parameter value(s) from parameter file.
                                 If 'none', parameter values will not be transformed.
                                 Default is 'none'.
                                 Note: if a transformation (`trans`) is already set, a back-transformation
                                       has to be provided.

        defaultvalue (float, optional) : default numeric value of the set of parameters.
                                         If None, the current values in mobj will be taken.
                                         Default is None.


        **kwargs :  - additional arguments based on pyemu parameter data such as:
                        - parchlim (str)
                        - parlbnd (float)
                        - parubnd (float)
                        - pargp (str)
                        - scale (int/float)
                        - offset (int/float)
                        - dercom (int)
                    - additional arguments about parameter and template file paths
                        - parpath (str)
                        - tplpath (str)


        Examples
        --------
        kmi = pymarthe.utils.pest_utils.get_kmi(mm.prop['soil'], ['soilprop', 'zone'])
        mlp = MartheListParam('soil', mobj= mm.prop['soil'], kmi=kmi,
                              pargp= 'msoil', parpath='par', tplpath='tpl')

        """
        self.parname = parname
        self.type = 'list'
        self.mobj = mobj
        self.kmi = kmi
        self.value_col = value_col
        # -- Set parameter default value
        if defaultvalue is None:
            self.defaultvalue = mobj.data.set_index(self.kmi.names).loc[self.kmi, value_col].to_list()
        else:
            self.defaultvalue = defaultvalue
        # -- Transformation validity
        pest_utils.check_trans(trans, btrans,
                               test_on = marthe_utils.make_iterable(self.defaultvalue))
        # -- Atributs
        self.parnmes = self.gen_parnmes()
        self.trans = trans
        self.btrans = btrans
        self.parchglim = kwargs.get('parchglim', 'factor')
        self.parlbnd = kwargs.get('parlbnd', 1e-10) 
        self.parubnd = kwargs.get('parubnd', 1e+10) 
        self.pargp = kwargs.get('pargp', self.parname) 
        self.scale = kwargs.get('scale', 1) 
        self.offset = kwargs.get('offset', 0)
        self.dercom = kwargs.get('dercom', 1) 
        # ---- Build parameter DataFrame
        self.param_df = pd.DataFrame(index = self.parnmes)
        self.param_df[base_param] = [ self.parnmes, self.trans,
                                          self.btrans, self.parchglim,
                                          self.defaultvalue, self.parlbnd,
                                          self.parubnd, self.pargp,
                                          self.scale, self.offset, self.dercom ]
        # ---- Manage files io
        self.parpath = kwargs.get('parpath', '.')
        self.tplpath = kwargs.get('tplpath', '.')


    def gen_parnmes(self):
        """
        Internal method to generate parmaeters names from `kmi`.

        Returns
        -------
        parnmes (list) : parameters names according to keys values of `kmi`.
                         Note : name will be created as 'item0__item1__..._itemN'
                                where items are kmi possible values.

        Examples
        --------
        parnmes = mlp.gen_parnmes()

        """
        return ['__'.join(list(map(str, items))) for items in self.kmi]



    def get_param_df(self, transformed=False):
        """
        Return internal parameter DataFrame.

        Parameters
        ----------
        transformed (bool, optional) : whatever apply transformation on output DataFrame.
                                       Default is False.

        Returns
        -------
        param_df (DataFrame) : parameter data as DataFrame.
                               Format:

                                               parnme     trans           btrans   parchglim   defaultvalue       parlbnd       parubnd     pargp  scale  offset  dercom
                parname
                cap_soil_progr_01   cap_soil_progr_01     log10  lambda x: 10**x      factor             40  1.000000e-10  1.000000e+10       csp      1       0       1
                cap_soil_progr_02   cap_soil_progr_02     log10  lambda x: 10**x      factor             40  1.000000e-10  1.000000e+10       csp      1       0       1
                ...                               ...       ...              ...         ...            ...           ...           ...       ...    ...     ...     ...


        Examples
        --------
        parnmes = mlp.get_param_df()

        """
        # ---- Get copy of parameter data
        par_df = self.param_df.copy(deep=True)
        # ---- Transform values if required
        if transformed:
            par_df['defaultvalue'] = pest_utils.transform(par_df['defaultvalue'], self.trans)
        # ---- Return paramater DataFrame
        return par_df



    def to_config(self):
        """
        Return the essential informations of current set of parameters to be
        written in the configuration file as a new parameter section.

        Returns
        -------
        section (str) : parameter section as string.

        Examples
        --------
        print(mlp.to_config())

        """
        lines = ['[START_PARAM]']
        data = [
            'parname= {}'.format(self.parname),
            'type= {}'.format(self.type),
            'class= {}'.format(str(self.mobj)),
            'property name= {}'.format(self.mobj.prop_name),
            'keys= {}'.format(','.join(self.kmi.names)),
            'value_col= {}'.format(self.value_col),
            'trans= {}'.format(self.trans),
            'btrans= {}'.format(self.btrans),
            'parfile= {}'.format(os.path.join(self.parpath, f'{self.parname}.dat'))
              ]
        lines.extend(data)
        lines.append('[END_PARAM]')
        return '\n'.join(lines)



    def write_parfile(self, parpath=None):
        """
        Write parameter file(s) in parameter folder.
        (wrapper to pymarthe.utils.pest_utils.write_mlp_parfile()) 

        Parameters
        ----------
        parpath (str, optional) : path to the folder where parameter files should be writen.
                                  If None, name taken from .parpath.
                                  Default is None.

        Examples
        --------
        mlp.write_parfile(parpath='par')
        """
        path = self.parpath if parpath is None else parpath
        pf = os.path.join(path, self.parname + '.dat')
        pest_utils.write_mlp_parfile(pf, self.param_df,  self.trans)



    def write_tplfile(self, tplpath=None):
        """
        Write template file(s) in template folder.
        (wrapper to pymarthe.utils.pest_utils.write_mlp_tplfile()) 

        Parameters
        ----------
        tplpath (str, optional) : path to the folder where template files should be writen.
                                  If None, name taken from .tplpath.
                                  Default is None.

        Examples
        --------
        mglp.write_tplfile(tplpath='tpl')
        """
        path = self.tplpath if tplpath is None else tplpath
        tf = os.path.join(path, self.parname + '.tpl')
        pest_utils.write_mlp_tplfile(tf, self.param_df)


    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheListParam'















class MartheGridParam():
    """
    Class for handling Marthe grid-like properties.
    """
    def __init__(self, parname, mobj, izone=None, pp_data=None, trans = 'none', 
                       btrans = 'none', defaultvalue=None, **kwargs):
        """
        Generator of grid parameter instance base on `izone` (field id zones).
        2 kinds of parameters can be set by zone:
            - 'zpc' : zone of piecewise constancy
            - 'pp' : pilot points


        Parameters
        ----------
        parname (str) : parameter name

        mobj (MartheField) : a parametrable field object.

        izone (str/MartheField, optional) : field with required zone ids as values.

                                            Izone values can be:
                                                - izone < 0         : zone of piecewise constancy
                                                - izone > 0         : zone with pilot points
                                                - izone = -9999, 0, 9999 : inactive zone

                                            If None, a generic field will be created
                                            from the input `mobj` with a unique zpc
                                            zone (in active cells) for each layer.
                                            Default is None.

        pp_data (dict, optional) : Nested dictionary that contains pilot point data (`ppobj`) 
                                   for each layer and each zone. 
                                   Format: pp_data ={layer_0 : {zone_1: ppobj_0, zone_2: ppobj_1}, ..., ...}
                                   `ppobj` can be: 
                                        - list/array of pilot point coordinates 
                                            Format : [[ppx_0, ppy_0], ..., [ppx_N, ppy_N]]
                                        - path to a (single) point shapefile
                                            Format : 'gis/pp_layer_0.shp'
                                   If None, a set of pilot points will be seed on each zone according to 
                                   the mean spacing of each cell centroid of the pilot points.
                                   Format : spacing = 2 * mean(distance(zone_centroids))
                                   Default is None.

        trans (str, optional) : transformation to apply to the initial parameter value(s).
                                If 'none', parameter values will not be transformed.
                                Default is 'none'.

        btrans (str, optional) : back-transformation to apply to the parameter value(s) from parameter file.
                                 If 'none', parameter values will not be transformed.
                                 Default is 'none'.
                                 Note: if a transformation (`trans`) is already set, a back-transformation
                                       has to be provided.

        defaultvalue (float, optional) : default numeric value of the set of parameters.
                                         If None, the current values of provided field will be taken.
                                         Default is None.


        **kwargs :  - additional arguments based on pyemu parameter data such as:
                        - parchlim (str)
                        - parlbnd (float)
                        - parubnd (float)
                        - pargp (str)
                        - scale (int/float)
                        - offset (int/float)
                        - dercom (int)
                    - additional arguments about parameter and template file paths
                        - parpath (str)
                        - tplpath (str)


        Examples
        --------
        mgp = MartheGridParam(parname= 'hk', mobj= mm.prop['permh'],
                              izone= 'model.ipermh', pp_data={4:{1:'gis/pp_l4.shp'}})

        """
        self.parname = parname
        self.type = 'grid'
        self.mobj = mobj
        self.nlay = self.mobj.maxlayer
        self.defaultvalue = defaultvalue
        self.pp_data = pp_data
        # -- Transformation validity
        _test = marthe_utils.make_iterable(
                    self.mobj.get_data(
                        masked_values=self.mobj.dmv)['value'])
        pest_utils.check_trans(trans, btrans, test_on = _test)
        self.trans = trans
        self.btrans = btrans
        # -- Manage izone
        self.set_izone(izone)
        # -- Manage PEST/pyEMU parameters
        self.parchglim = kwargs.get('parchglim', 'factor')
        self.parlbnd = kwargs.get('parlbnd', 1e-10) 
        self.parubnd = kwargs.get('parubnd', 1e+10) 
        self.pargp = kwargs.get('pargp', self.parname) 
        self.scale = kwargs.get('scale', 1) 
        self.offset = kwargs.get('offset', 0)
        self.dercom = kwargs.get('dercom', 1)
        # ---- Manage files io
        self.parpath = kwargs.get('parpath', '.')
        self.tplpath = kwargs.get('tplpath', '.')




    def set_izone(self, izone = None):
        """
        Manage izone (MartheField) input.
        It will detect zone ids:
            - izone < 0         : zone of piecewise constancy
            - izone > 0         : zone with pilot points
            - izone = -9999, 0, 9999 : inactive zone

        Parameters
        ----------
        izone (str/MartheField, optional) : field with required zone ids as values.
                                            If None, a generic field will be created
                                            from the input `mobj` with a unique zpc
                                            zone (in active cells) for each layer.
                                            Default is None. 

        Examples
        --------
        mgp = MartheGridParam('permh', mm.prop['permh'])
        active = mm.query_grid(active=1, layer=0, target='node').values
        izone['value'][active] = -2
        mgp.set_izone(izone)

        """
        # ---- Manage not provided izone
        if izone is None :
            # -- Warn about generic izone creation
            msg = "WARNING : no `izone` provided. A generic one will be created "
            msg += f" from the `{self.mobj.field}` field with a unique zpc zone "
            msg += " (in active cells) for each layer."
            warnings.warn(msg)
            # -- Build default MartheField instance from .imask with -1 value (zpc) for all layers
            izone = MartheField(f'i{self.parname}', -1, self.mobj.mm)
            # -- Write on disk
            f = os.path.join(self.mobj.mm.mldir,
                             f'{self.mobj.mm.mlname}.{izone.field}')
            izone.write_data(f)
            # -- Set izone attributs
            self.izone = izone
            self.izone_file = f

        # ---- Manage field izone
        elif isinstance(izone, MartheField):
            # -- Write on disk
            f = os.path.join(self.mobj.mm.mldir,
                             f'{self.mobj.mm.mlname}.{izone.field}')
            izone.write_data(f)
            # -- Set izone attributs
            self.izone = izone
            self.izone_file = f

        # ---- Manage string izone
        elif isinstance(izone, str):
            # -- Set izone attributs
            self.izone = MartheField(f'i{self.parname}', izone, self.mobj.mm)
            self.izone_file = izone

        # ---- Initialize zpc/pp data
        self.init_zpc_df()
        self.init_pp_dic()




    def get_dv_from_lz(self, layer, zone, agg=None):
        """
        Extract 'default value' as field mean for a given layer and zone id.

        Parameters
        ----------
        layer (int) : layer id.
        zone (int) : zone id.
        agg (func/str, optional) : string/function to aggregate value.

        Returns
        --------
        dv (float) : default value(s).

        Examples
        --------
        mgp = MartheGridParam('permh', mm.prop['permh'])
        mgp.get_dv_from_lz(layer=2, zone=1)

        """
        # ---- Get value according to activeness, layer and zone id
        mask = np.logical_and.reduce( 
            [ 
            self.izone.get_data(layer=layer, masked_values=self.izone.dmv, as_mask=True),
            self.izone.data['value'] == zone
                ]
            )

        dv = self.mobj.data['value'][mask]

        # ---- Return default value (with(out) aggregation)
        if agg is None:
            return dv
        else:
            return pd.Series(dv).agg(agg)




    def get_dv_from_xy(self, x, y, layer, agg=None):
        """
        Extract 'default value' as field mean for at given xy-coordinates

        Parameters
        ----------
        x (float/it) : pilot point x-coordinate(s)
        y (float/it) : pilot point y-coordinate(s)
        layer (int)  : layer id.
        agg (func/str, optional) : string/function to aggregate value.

        Returns
        --------
        dv (float) : default value(s).

        Examples
        --------
        ppx,ppy = pest_utils.shp2points('myppoints.shp')
        mgp.get_dv_from_xy(ppx, ppy)

        """
        # ---- Get value according to pilot point xy-coordinates
        mask = self.mobj.sample(x,y,layer, as_mask=True)
        dv = self.mobj.data['value'][mask]

        # ---- Return default value (with(out) aggregation)
        if agg is None:
            return dv
        else:
            return pd.Series(dv).agg(agg)




    def init_zpc_df(self):
        """
        Initialise zone of piecewise constancy DataFrame
        """
        # ---- Initialize zpc data lists
        _names, _zones, _layers, _dvs = [[] for _ in range(4)]

        # ---- Build data ietrating over layers and zones
        for ilay in range(self.nlay):
            # -- Fetch zone ids
            zones = np.unique(self.izone.get_data(layer=ilay)['value'])
            for zone in zones :
                # -- Perform zpc computation only when zone id < 0
                if zone < 0 :
                    # -- Build parname
                    _names.append(ZPCFMT(self.parname, ilay, zone))
                    _layers.append(ilay)
                    _zones.append(int(zone))
                    # -- Manage not provided default value
                    dv = self.get_dv_from_lz(ilay, zone, agg='mean') if self.defaultvalue is None else self.defaultvalue
                    _dvs.append(dv)

        # ---- Set zpc DataFrame from data
        zpc_df = pd.DataFrame({'parname':_names, 'layer':_layers, 'zone':_zones, 'value': _dvs})
        self.zpc_df = zpc_df.set_index('parname', drop=False)




    def set_zpc_value(self, value, layer=None, zone=None):
        """
        Set value of required zone of piecewise constancy.

        Parameters
        ----------
        value (float/int) : numeric parameter value.
        layer (int/it, optional) : layer id to set value.
                                   If None, all layers will be considered.
                                   Default is None.
        zone (int/it, optional) : zone id to set value.
                                  If None, all zones (zpc) will be considered.
                                  Default is None.

        Returns
        --------
        Set values in .zpc_df

        Examples
        --------
        mgp = MartheGridParam('permh', mm.prop['permh'])
        mgp.set_zpc_value(value= 1e-3, layer=[0,1], zone=-1)

        """
        # ---- Check value integrity
        err_msg = f"ERROR : `value` must be numerical. Given {value}."
        assert isinstance(value, (int,float)), err_msg

        # ---- Manage inputs (iterables)
        _zone = self.zpc_df.zone.unique() if zone is None else marthe_utils.make_iterable(zone)
        _layer = list(range(self.nlay)) if zone is None else marthe_utils.make_iterable(layer)

        # ---- Query and set zpc inplace
        mask = np.logical_and.reduce(
                   [ self.zpc_df.layer.isin(_layer),
                     self.zpc_df.zone.isin(_zone) ]
                     )
        self.zpc_df.loc[mask,'value'] = value



    def init_pp_dic(self) :
        """
        Initialize pilot points dictionary.
        Format : {  layer_0 : pp_df_0,
                    layer_1 : pp_df_1,
                    ...,
                    layer_N: pp_df_N}

        """
        # ---- Prepare empty pilot point dictionary
        self.pp_dic = {}

        # ---- Iterate over layer and positive zones to set pilot point DataFrames
        for ilay in range(self.nlay):
            ldata = self.izone.get_data(layer=ilay, masked_values=self.izone.dmv)
            zones = np.unique(ldata['value']).astype(int)
            # -- Search for at least 1 defined zpc 
            if len(zones) > 1:
                pp_dfs = []
                for zone in zones:
                    if zone > 0:
                        # -- Try to get pilot point coordinates if provided
                        try:
                            ppobj = self.pp_data[ilay][zone]
                            # ---- Manage pilot point input object
                            coords = shp_utils.shp2points(ppobj) if isinstance(ppobj, str) else ppobj
                        # -- Get izone cell centers as pilot point coordinates if not provided
                        except:
                            # -- Warn about default behaviour
                            msg = f"WARNING : pilot point coordinates not provided for layer = {ilay}" \
                                  f" and zone = {int(zone)}. Default pilot points will be generated."
                            warnings.warn(msg)
                            coords = self.default_pp_coords(layer=ilay, zone=int(zone))
                        # -- Build DataFrame from pilot point coordinates
                        pp_df = self.build_pp_df(coords, layer=ilay, zone=int(zone))
                        pp_dfs.append(pp_df)
                    else:
                        pp_dfs.append(pd.DataFrame())

                # -- Set zone pilot point data for current layer
                pp_df = pd.concat(pp_dfs)
                if not pp_df.empty:
                    self.pp_dic[ilay] = pp_df




    def zone_interp_coords(self, layer, zone) :
        """
        Fetch centroid coordinates of required zone to perform
        pilot points interpolation.

        Parameters
        ----------
        layer (int) : layer id to extract coordinates.
        zone (int) : zone id to extract coordinates.

        Returns
        --------
        (xc,yc) (tuple [1D-array]) : x,y coordinates of cell centers.

        Examples
        --------
        mgp.zone_interp_coords(layer=0, zone=-1)

        """
        # ---- Build layer, zone, active mask
        mask = np.logical_and.reduce(
                    [   
                        self.izone.get_data(
                                masked_values= self.izone.dmv,
                                layer=layer, as_mask=True),
                        self.izone.data['value'] == zone 
                    ]
                 )
        # ---- Extract centroid coordinates
        xc = self.izone.data[mask]['x']
        yc = self.izone.data[mask]['y']
        # ---- Return coordinates
        return xc, yc



    def default_pp_coords(self, layer, zone):
        """
        Infer pilot point coordinates for a specific zone in a required layer.
        Pilot points will be regularly spaced on each x,y direction with
        adefault spacing of two times the average distance between all 
        centroids of the current active cells.

        Parameters
        ----------
        layer (int) : layer id.
        zone (int) : zone id.

        Returns
        --------
        pp_coords (np.ndarray [(N,2)]) : coordinates of generated pilot points.

        Examples
        --------
        mgp = MartheGridParam('permh', mm.prop['permh'], ipermh)
        pp_l2_z1 = mgp.pp_coords(layer=2, zone=1)

        """
        # ---- Check if required layer contain pilot point zone
        err_msg = "ERROR : the `izone` does not contain pilot " \
                  f"point zones for layer = {layer}."
        assert np.any(self.izone.get_data(layer=layer)['value'] > 0), err_msg

        # ---- Get active data for required layer and zone
        m = self.izone.get_data(layer=layer, masked_values=self.izone.dmv, as_mask=True)
        mask = np.logical_and(m, self.izone.data['value'] == zone)
        rec = self.izone.data[mask]

        # ---- Get extent on current active layer
        xmin, ymin = map(np.min, [rec['x'], rec['y']])
        xmax, ymax = map(np.max, [rec['x'], rec['y']])

        # ---- Fetch centroid coordinates of active layer
        centroids = np.column_stack([rec['x'], rec['y']])

        # ---- Define pilot point spatial spacing (2 * mean distance of each centroids)
        spacing = 2 * (np.linalg.norm(centroids)/len(centroids))

        # ---- Compute number of pilot point to seed in each direction
        nx = int(np.ceil((xmax-xmin)/spacing))
        ny = int(np.ceil((ymax-ymin)/spacing))

        # ---- Generate gridded pilot points
        ppxx, ppyy = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))

        # ---- Remove points that are in inactive cells
        pp_coords = []
        for px, py in zip(ppxx.ravel(), ppyy.ravel()):
            sample_value = self.izone.sample(px,py,layer=layer)['value']
            if not sample_value in self.izone.dmv:
                pp_coords.append([px,py])

        # ---- Return generated pilot points
        return np.array(pp_coords)




    def build_pp_df(self, coords, layer, zone):
        """
        Create pilot point Dataframe from xy-coordinates with generic names.
        Wrapper to pp_utils.pp_df_from_coords().

        Parameters
        ----------
        coords (list/array) : pilot point xy-coordinates.
                              Format : [[ppx_0, ppy_0], ..., [ppx_N, ppy_N]]
        layer (int) : layer id.
        zone (int) : zone id.

        Returns
        --------
        pp_df (Dataframe) : pilot point standard DataFrame.
                            Format:
                                                  parname           x      y  zone         value
                             parname
                             hk_l04_z01_00  hk_l04_z01_00  357.136364  209.0     1  1.780510e-05
                             hk_l04_z01_01  hk_l04_z01_01  373.681818  209.0     1  4.042830e-05
                             ...                      ...         ...    ...     .           ...
                             hk_l04_z01_N    hk_l04_z01_N  357.136364  225.4     1  1.337280e-05

        Examples
        --------
        coords = shp_utils.shp2points('gis/pp_l04.shp', stack=True)
        pp_df = mgp.build_pp_df(coords, layer=4, zone=1)

        """
        # ---- Manage pilot point default value
        if self.defaultvalue is None:
            ppx, ppy = np.column_stack(coords)
            dv = self.get_dv_from_xy(ppx, ppy, layer)
            # -- Check default value(s) validity
            if len(dv) != len(coords):
                msg = "WARNINGS : could not extract field data properly at pilot points coordinates. " \
                      "A generic value of 1e-3 will be set as `default_value instead. The reason can be the:\n" \
                      "\t- Absence of spatial index in main model\n" \
                      "\t- Bad spatial index files creation\n" \
                      "\t- Presence of corrupted spatial index files`\n" \
                      "A generic value of 1e-3 will be set as `default_value instead."
                warnings.warn(msg)
                dv = 1e-3
        else:
            dv = self.defaultvalue

        # ---- Build DataFrame with names
        pp_df = pp_utils.pp_df_from_coords(self.parname, coords, layer, zone, value= dv)

        # ---- Check spatial lies distribution
        # msg = f"WARNING : some pilot points are located outside " \
        #       f"the provided zone id : layer = {layer}, zone = {zone}."
        # rec = self.izone.sample(pp_df.x, pp_df.y, layer=layer)
        # if not np.all(rec['value'] == zone):
        #     warnings.warn(msg)

        # ---- Return DataFrame of pilot point
        return pp_df



    def write_parfile(self, parpath=None, only_zpc=False, only_pp=False):
        """
        Write parameter file(s) in parameter folder.
        (wrapper to pymarthe.utils.pest_utils.write_mgp_parfile()) 

        Parameters
        ----------
        parpath (str, optional) : path to the folder where parameter files should be writen.
                                  If None, name taken from .parpath.
                                  Default is None.

        only_zpc (bool, optional) : write zone of piecewise constancy parameter file(s) only.
                                    Default is False.

        only_pp (bool, optional) : write pilot point parameter file(s) only.
                                   Default is False.

        Examples
        --------
        mgp.write_parfile(parpath='par', only_zpc=True)
        """
        # ---- Get path to the parameter folder
        path = self.parpath if parpath is None else parpath

        # ---- Write parameter files for zpc
        if not only_pp:
            # -- Set parameter type to manage
            ptype = 'zpc'
            # ---- Write zpc parameter only if not empty
            pf = os.path.join(path, f'{self.parname}_{ptype}.dat')
            if not self.zpc_df.empty:
                pest_utils.write_mgp_parfile(pf, self.zpc_df, trans= self.trans, ptype=ptype)
            else : 
                print('No ZPC identified for parameter {0} in izone data.'.format(self.mobj.field))

        # ---- Write parameter files for pilot points
        if not only_zpc:
            # -- Set parameter type to manage
            ptype = 'pp'
            # ---- Write parameter files for pilot points in given layer, zone if exists
            if len(self.pp_dic) > 0:
                # ---- Write 1 parameter file per layer and zone
                for ilay, pp_df in self.pp_dic.items():
                    for zone, zpp_df in pp_df.groupby('zone'):
                        # -- Build parameter filename
                        f = '{0}_{1}_l{2:02d}_z{3:02d}.dat'.format(self.parname, ptype, ilay, zone)
                        pf = os.path.join(path, f)
                        # -- Write parameter file
                        pest_utils.write_mgp_parfile(pf, zpp_df, trans= self.trans, ptype=ptype)
            # else :
            #     print('No Pilot Points identified for parameter {0} in izone data.'.format(self.mobj.field))

        # ---- Manage multiple True only
        if np.all([only_zpc, only_pp]):
            print("Careful: passing both `only_zpc` and `only_pp`" \
                  " to `True` will not generate any parameter files.")





    def write_tplfile(self, tplpath=None, only_zpc=False, only_pp=False):
        """
        Write template file(s) in template folder.
        (wrapper to pymarthe.utils.pest_utils.write_mgp_tplfile()) 

        Parameters
        ----------
        tplpath (str, optional) : path to the folder where template files should be writen.
                                  If None, name taken from .tplpath.
                                  Default is None.

        only_zpc (bool, optional) : write zone of piecewise constancy template file(s) only.
                                    Default is False.

        only_pp (bool, optional) : write pilot point template file(s) only.
                                   Default is False.

        Examples
        --------
        mgp.write_tplfile(tplpath='tpl', only_zpc=True)

        """
        # ---- Get path to the template folder
        path = self.tplpath if tplpath is None else tplpath

        # ---- Write template files for zpc
        if not only_pp:
            # -- Set parameter type to manage
            ptype = 'zpc'
            # ---- Write zpc parameter only if not empty
            pf = os.path.join(path, f'{self.parname}_{ptype}.tpl')
            if not self.zpc_df.empty:
                pest_utils.write_mgp_tplfile(pf, self.zpc_df, ptype=ptype)
            else : 
                print('No ZPC identified for parameter {0} in izone data.'.format(self.mobj.field))

        # ---- Write template files for pilot points
        if not only_zpc:
            # -- Set parameter type to manage
            ptype = 'pp'
            # ---- Write template files for pilot points in given layer, zone if exists
            if len(self.pp_dic) > 0:
                # ---- Write 1 parameter file per layer and zone
                for ilay, pp_df in self.pp_dic.items():
                    for zone, zpp_df in pp_df.groupby('zone'):
                        # -- Build parameter filename
                        f = '{0}_{1}_l{2:02d}_z{3:02d}.tpl'.format(self.parname, ptype, ilay, zone)
                        pf = os.path.join(path, f)
                        # -- Write parameter file
                        pest_utils.write_mgp_tplfile(pf, zpp_df, ptype=ptype)
            # else : 
            #     print('No Pilot Points identified for parameter {0} in izone data.'.format(self.mobj.field))

        # ---- Manage multiple True only
        if np.all([only_zpc, only_pp]):
            print("Careful: passing both `only_zpc` and `only_pp`" \
                  " to `True` will not generate any template files.")




    def write_kfac(self, vgm_range, krig_transform= 'none', parpath=None , save_cov = False):
        """
        Compute and write kriging factor files (PEST-like) from exponential variogram
        ranges for each layer and zone of pilot points.
        Wrapper of some pyemu tools : 
            - pyemu.utils.geostats.ExpVario
            - pyemu.utils.geostats.GeoStruct
            - pyemu.utils.geostats.OrdinaryKrige
        
        Note: the ranges must be in the same distance unit as the model fields.

        Parameters
        ----------
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

        krig_transform (str, optional) : transformation to apply to the 
                                        pyemu.utils.geostats.GeoStruct.
                                        Can be:
                                            - 'none'
                                            - 'log'
                                        Default is 'none'.

        parpath (str, optional) : path to write kriging factor files.
                                   If None, it will take the `.kfacpath' argument.
                                   Default is None.
                                   Note: a kriging factor files will be written for each
                                         pilot point layer and zone. The file name will be
                                         generated as '{parname}_l{layer}_z{zone}.fac'.

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
        mgp.write_kfac(vgm_range, vgm_transform= 'log',  save_cov=True)

        """
        # ---- Manage varigram ranges
        vgmr = {}
        if np.isscalar(vgm_range):
            # -- Same range for all variograms (layers and zones)
            vgmr = {k: {zone: vgm_range} for k,v in self.pp_dic.items() for zone in v.zone.unique()}
        elif isinstance(vgm_range, dict):
            # -- Verify layer keys matching between pilot point and variogram dictionaries
            err_msg =  "ERROR : `vgm_range` must have same layer keys as pilot point " 
            err_msg += "dictionary. Given {}, expected {}.".format(
                            str(list(vgm_range.keys())), 
                            str(list(self.pp_dic.keys()))
                            )
            assert sorted(vgm_range.keys()) == sorted(self.pp_dic.keys()), err_msg
            # -- Start iterating over layers
            for ilay, obj in vgm_range.items():
                # -- Same range for each zone for a given layer
                if np.isscalar(obj):
                    vgmr[ilay] = {zone: obj for zone in self.pp_dic[ilay].zone.unique()}
                # -- Explicit range for all zones for each layer
                if isinstance(obj, dict):
                    # -- Verify zone keys matching between pilot point and variogram dictionaries
                    err_msg =  "ERROR : `vgm_range` must have same zone keys as pilot point " 
                    err_msg += "dictionary for layer {}. Given {}, expected {}.".format(
                                    ilay,
                                    str(list(vgm_range[ilay].keys())), 
                                    str(self.pp_dic[ilay].zone.unique())
                                    )
                    assert sorted(obj.keys()) == sorted(self.pp_dic[ilay].zone.unique()), err_msg
                    vgmr[ilay] = obj

        # ---- Iterate over each layer and zone
        for ilay, pp_df in self.pp_dic.items():
            for zone, zpp_df in pp_df.groupby('zone'):
                # -- Build up Variogram
                #   - parameter `a` is considered as a proxy for range
                #   - the contribution has no effect without nugget
                vgm = pyemu.utils.geostats.ExpVario(contribution=1, a=vgmr[ilay][zone])
                # -- Build up GeoStruct
                gs = pyemu.utils.geostats.GeoStruct(variograms=vgm, transform=krig_transform)
                # -- Set up kriging
                ok = pyemu.utils.geostats.OrdinaryKrige(
                            geostruct = gs,
                            point_data = zpp_df.rename({'parname':'name'}, axis=1)
                            )
                # -- Extract cellcenters coordinates to perform kriging
                x_interp, y_interp = self.zone_interp_coords(ilay, zone=zone)
                # -- Compute kriging factors
                kfac_df = ok.calc_factors(x_interp, y_interp, pt_zone=zone, num_threads=4)
                # -- Write kriging factors to file
                path = self.parpath if parpath is None else parpath
                kfac_file = os.path.join(path, '{0}_pp_l{1:02d}_z{2:02d}.fac'.format(self.parname, ilay, zone))
                ok.to_grid_factors_file(kfac_file, ncol=len(kfac_df)) # ncol needed for unstructured pp
                # -- Write covariance matrices if required
                if save_cov:
                    cov = gs.covariance_matrix(zpp_df.x, zpp_df.y, zpp_df.parname)
                    cov.to_binary(kfac_file.replace('.fac', '.jcb'))




    def get_param_df(self, transformed=False):
        """
        Join all parameter informations in a single DataFrame
        for the current field.

        Parameters
        ----------
        transformed (bool, optional) : whatever apply transformation on output DataFrame.
                                       Default is False.

        Returns
        --------
        param_df (DataFrame) : parameter information of current field.

        Examples
        --------
        mgp.get_param_df()
        
        """
        # ---- Concatenate all zpc and pilot point DataFrames
        dfs = []
        if not self.zpc_df.empty:
            dfs.append(self.zpc_df)
        for pp_df in self.pp_dic.values():
            if pp_df is not None:
                dfs.append(pp_df)
        concat = pd.concat(dfs)
        # ---- Set distinct parameter group names
        concat['pargp'] = concat.parname.apply(
                            lambda s: f"{self.pargp}_zpc" if 'zpc' in s else f"{self.pargp}_pp")
        # ---- Build standard parameter data
        par_df = pd.DataFrame(index = concat.parname)
        par_df[base_param] = [ concat['parname'], self.trans,
                                  self.btrans, self.parchglim,
                                  concat['value'], self.parlbnd,
                                  self.parubnd, concat['pargp'],
                                  self.scale, self.offset, self.dercom ]
        # ---- Transform values if required
        if transformed:
            par_df['defaultvalue'] = pest_utils.transform(par_df['defaultvalue'], self.trans)
        # ---- Return parameter DataFrame
        return par_df




    def to_config(self):
        """
        Return the essential informations of current set of parameters to be
        written in the configuration file as a new parameter section.

        Returns
        -------
        section (str) : parameter section as string.

        Examples
        --------
        print(mgp.to_config())
        """
        # ---- Get all parameter file names
        parfiles = [os.path.join(self.parpath, f) for f in os.listdir(self.parpath)
                    if f.endswith('.dat') and any(s in f for s in ['_zpc','_pp'])]

        lines = ['[START_PARAM]']
        data = [
            'parname= {}'.format(self.parname),
            'type= {}'.format(self.type),
            'class= {}'.format(str(self.mobj)),
            'property name= {}'.format(self.mobj.field),
            'izone= {}'.format(self.izone_file),
            'trans= {}'.format(self.trans),
            'btrans= {}'.format(self.btrans),
            'parfile= {}'.format(','.join(parfiles)),
              ]
        lines.extend(data)
        lines.append('[END_PARAM]')
        return '\n'.join(lines)



    def __str__():
        """
        Internal string method.
        """
        return 'MartheGridParam'

