"""
Contains the MartheModel and Spatial Reference classes.
Designed for structured and nested grid.
"""

import os, sys
import warnings
import subprocess as sp
from shutil import which
from copy import deepcopy
import queue 
import threading
import numpy as np
import pandas as pd
from datetime import datetime

from .mfield import MartheField
from .mpump import MarthePump
from .msoil import MartheSoil
from .utils import marthe_utils, shp_utils, pest_utils

encoding = 'latin-1'


class MartheModel():
    """
    Wrapper MARTHE --> Python
    """
    def __init__(self, rma_path, spatial_index = False, modelgrid= False):
        """
        Parameters
        ----------
        rma_path (str): path to the Marthe .rma file from which
                        the model name and working directory
                        will be identified.

        spatial_index (bool/str/dict) : model spatial index.
                                        Allows sample and intersect process from xy coordinates.
                                        Each cell will be inserted in a rtree Index instance with
                                        related cell data (id, layer, inest, ...). 
                                        Can be :
                                            - True  (bool)    : generate generic spatial index (mlname_si.idx/.dat).
                                            - False (bool)    : disable spatial index creation.
                                            - name  (string)  : path to an existing spatial index to read.
                                            - dic   (dict)    : generate custom spatial index.
                                                                Can contains: 
                                                                    - 'name' (str)
                                                                        custom name to external spatial index files
                                                                    - 'only_active' (bool)
                                                                        disable the insertion of inactive cells in 
                                                                        the spatial index. This can be usefull for
                                                                        fast spatial processings on valid large 
                                                                        models (especially with nested grids).
                                                                        Careful, some processes could be affected 
                                                                        and not working as usal.
                                                                Format : {'name':'mymodelsi', 'only_active' : True}
                                        Default is False.

        modelgrid (bool): cell by cell DataFrame containing model grid informations.
                          Format:

                                        node    layer   inest   i     j     xcc     ycc     dx      dy  vertices  active
                                node
                                0        int      int     int int   int   float   float  float   float      list     int
                                1        int      int     int int   int   float   float  float   float      list     int
                                .        ...      ...     ... ...   ...    ...     ...    ...      ...      ...      ...
                                .        ...      ...     ... ...   ...    ...     ...    ...      ...      ...      ...
                                nnode    int      int     int int   int   float   float  float   float      list     int

                          Can be:
                                - True : Build a `.modelgrid` DataFrame from imask (slow)
                                   or from the spatial index if provided (fast).
                                - False : set modelgrid to None.

                          Default is False.


        Examples
        --------
        mm = MartheModel(rma_path = 'model/mymodel.rma',
                         spatial_index = {'name':'mymodel_si'},
                         modelgrid=True)

        """
        # ---- Get model working directory and rma file
        self.rma_path = rma_path 
        self.mldir, self.rma_file = os.path.split(self.rma_path)

        # ---- Get model name 
        self.mlname = self.rma_file.split('.')[0]

        # ---- Get model files paths
        self.mlfiles = marthe_utils.get_mlfiles(os.path.join(rma_path))

        # ---- Get model units
        self.units = marthe_utils.get_units_dic(self.mlfiles['mart'])
        self.mldates = marthe_utils.get_dates(self.mlfiles['pastp'], self.mlfiles['mart'])

        # ---- Get infos about layers
        self.nnest, self.layers_infos = marthe_utils.get_layers_infos(self.mlfiles['layer'], base = 0)
        self.nlay = self.layers_infos.layer.max() + 1

        # ---- Get refine levels for nested grids
        self.rlevels = self.extract_refine_levels()

        # ---- Store hws (Hangling Wall State)
        self.hws = 'explicit' if len(self.layers_infos.epon_sup.unique()) == 1 else 'implicit'

        # ---- Store model grid infos from permh field
        self.imask = self.build_imask()

        # ---- Set number of cell by layer
        self.ncpl = int(len(self.imask.data)/self.nlay)

        # ---- Set number of simulated timestep
        self.nstep = len(self.mldates)

        # ---- Initialize property dictionary with permeability data
        self.prop = {}
        self.load_prop('permh')

        # ---- Load geometry
        self.geometry = {g : None for g in ['sepon', 'topog', 'hsubs']}

        # ---- Set spatial reference (used for compatibility with pyemu geostat utils)
        self.spatial_reference = SpatialReference(self)

        # ---- Manage spatial index instance
        # From existing external file
        if isinstance(spatial_index, str):
            self.si_state = 1       # activate spatial index
            from rtree.index import Index
            self.spatial_index = Index(spatial_index)
            self.sifile = spatial_index
        # From custom dictionary
        elif isinstance(spatial_index, dict):
            name = spatial_index.get('name', f'{self.mlname}_si' )
            only_active = spatial_index.get('only_active', False)
            self.build_spatial_index(name, only_active)
        # From generic
        elif spatial_index is True:
            self.build_spatial_index()
        # Without 
        elif spatial_index is False:
            self.si_state = 0       # desactivate spatial index
            self.sifile = None
            self.spatial_index = None

        # ---- Manage modelgrid DataFrame
        if modelgrid:
            self.build_modelgrid()
        else:
            self.modelgrid = None




    def __iter__(self, only_active=False):

        """
        Generate cell spatial informations for spatial indexing
        using class generator format.
        Each cell will contain following informations:
            - 'node' : cell unique id
            - 'layer': layer id
            - 'inest': nested grid id
            - 'i'    : row number
            - 'j'    : column number
            - 'xcc'  : x-coordinate of the cell centroid
            - 'ycc'  : y-coordinate of the cell centroid
            - 'dx'   : cell width
            - 'dy'   : cell height
            - 'area' : cell area
            - 'vertices': cell vertices
            - 'ative': cell activity (0=inactive, 1=active)

        Parameters:
        ----------
        only_active (bool) : enable/disable inactive cell consideration

        Returns:
        --------
        it (iterator) : generator of cell spatial data.

        Examples:
        --------
        it = mm.__iter__() 

        """
        # -- Initialize nodes properties
        nnodes = self.ncpl * self.nlay
        node = 0
        # -- Iterate over all model grids
        for mg in self.imask.to_grids():
            # -- Vectorize data as flat 1D-arrays (speed up iteration process)
            records = mg.to_records(fmt='full')
            # -- Disable insertion of inactive cells
            if only_active:
                # -- Iterate over structured grid data
                for r in records:
                    # -- For active cell
                    if r.value == 1:
                        # Store infos of current cell
                        obj = ( node, r.layer, r.inest, r.i, r.j,
                                r.x, r.y, r.dx, r.dy, r.dx*r.dy,
                                r.vertices, int(r.value))
                        # Compute cell bounds ((xmin, ymin, xmax, ymax))
                        if self.si_state:
                            bounds = (*r.vertices[0], *r.vertices[2])
                        # -- Push cell id/data to iterator
                        if bool(self.si_state):
                            yield (node, bounds, obj)
                        else:
                            yield obj
                        # -- Incrementation of cell id
                        node += 1
                    # -- For inactive cell
                    else:
                        node += 1
                    # -- Return user progress bar
                    marthe_utils.progress_bar((node+1)/nnodes)

            # -- Enable insertion of inactive cells 
            else:
                # -- Iterate over structured grid data
                for r in records:
                    # Store infos of current cell
                    obj = ( node, r.layer, r.inest, r.i, r.j,
                            r.x, r.y, r.dx, r.dy, r.dx*r.dy,
                            r.vertices, int(r.value))
                    # -- Compute cell bounds
                    if self.si_state:
                        bounds = (*r.vertices[0], *r.vertices[2])
                    # -- Push cell id/data to iterator
                    if bool(self.si_state):
                        yield (node, bounds, obj)
                    else:
                        yield obj
                    # -- Return user progress bar
                    marthe_utils.progress_bar((node+1)/nnodes)
                    # -- Incrementation of unique cell id
                    node += 1


    def build_spatial_index(self, name=None, only_active=False):
        """
        Function to build a spatial index on field data.

        Parameters:
        ----------
        name (str) : filename of output spatial index files.
                     If None, name is .mlname + '_si'.
                     Default is None.
        only_active (bool) : enable/disable inactive cell consideration.

        Returns:
        --------
        spatial_index (rtree.index.Index)

        Examples:
        --------
        si = mm.modelgrid.build_spatial_idx()

        """
        # -- Activate iterator by setting spatial index to 1
        self.si_state = 1

        # -- Import rtree package
        try:
            import rtree
        except:
            ImportError('Could not import `rtree` package.')

        # -- Manage spatial index properties
        p = rtree.index.Property()
        si_name = self.rma_path.replace('.rma', '_si') if name is None else name
        p.set_filename(si_name)
        # -- Build rtree spatial index
        print('\nBuilding spatial index ...')
        if only_active:
            warnings.warn('Spatial index will be generated on active cells only.' \
                ' This can produce some abnormal behaviours on spatial processes.')
        si = rtree.index.Index(si_name,
                               self.__iter__(only_active=only_active),
                               properties=p)
        si.flush()
        self.sifile = si_name
        self.spatial_index = si




    def build_modelgrid(self):
        """
        Build an large pandas.DataFrame and store it in .modelgrid attribute.
        Each row represent a cell with the following informations (columns):
            - 'node' : cell unique id
            - 'layer': layer id
            - 'inest': nested grid id
            - 'i'    : row number
            - 'j'    : column number
            - 'xcc'  : x-coordinate of the cell centroid
            - 'ycc'  : y-coordinate of the cell centroid
            - 'dx'   : cell width
            - 'dy'   : cell height
            - 'area' : cell area
            - 'vertices': cell vertices
            - 'ative': cell activity (0=inactive, 1=active)
        """
        # ---- Get full data for each MartheGrid
        dfs = []
        for mg in self.imask.to_grids():
            df = pd.DataFrame.from_records(mg.to_records(fmt='full'))
            dfs.append(df)

        # ---- Concatenate grid data
        rename_dic = {'x':'xcc', 'y': 'ycc', 'value':'active'}
        df = pd.concat(dfs, ignore_index=True).rename(columns=rename_dic)

        # ---- Insert node numbers at column n°0
        df.insert(0, 'node', df.index)

        # -- Set DataFrame to .modelgrid attribute
        self.modelgrid = df.set_index('node', drop=False)





    def load_geometry(self, g=None, **kwargs):
        """
        Load and store geometry grid information of a MartheModel instance.

        Parameters:
        ----------
        g (str, optional) : Marthe geometry grid file.
                            Can be 'sepon', 'topog', 'hsubs', ..
                            If None all geometry grid file will be loaded.
                            Default is None.

        **kwargs : additional arguments of property classes.
                   Can be :
                        - `use_imask` (bool) : Default will be False.

        Returns:
        --------
        mf (MartheField) : store geometry field in .geometry attribut.

        Examples:
        --------
        mm.load_geometry('sepon')

        """
        # ---- Fetch geometry field to load
        _g = list(self.geometry.keys()) if g is None else marthe_utils.make_iterable(g)

        # ---- Assertion to avoid non geometry field input
        err_msg = 'ERROR : `g` must be a geometry grid file such as `sepon`, `topog`, ... ' \
                  'Given : {}.'.format(', '.join(list(_g)))
        assert all(g in self.mlfiles.keys() for g in _g), err_msg

        # ---- Load geometry as MartheField instance
        for g in _g:
            self.geometry[g] = MartheField(g, 
                                          self.mlfiles[g],
                                          self,
                                          use_imask=kwargs.get('use_imask', False)
                                          )



    def build_imask(self):
        """
        Function to build a imask field based on permh
        with binary data : 0 -> inactive cell
                           1 -> active cell

        Parameters:
        ----------
        self (MartheModel) : MartheModel instance

        Returns:
        --------
        imask (MartheField) : imask field

        Examples:
        --------
        imask = mf.build_imask()

        """
        # ---- Load permh field
        imask = MartheField('imask', self.mlfiles['permh'], self)
        # ---- Change data to binary
        imask.data['value'] = (imask.data['value'] != 0).astype(int)
        # ---- Return MartheField instance
        return imask



    def load_prop(self, prop, **kwargs):
        """
        Load MartheModel properties by name.

        Parameters:
        ----------
        prop (str) : supported property name.
                     Can be :
                     - Field (MartheField)
                        - 'permh'
                        - 'emmca'
                        - 'emmli'
                        - 'kepon'
                     - Pumping (MarthePump)
                        - 'aqpump'
                        - 'rivpump'
                     - Zonal Soil properties (MartheSoil)
                        - 'cap_sol_progr'
                        - 'aqui_ruis_perc'
                        - 't_demi_percol'
                        - 'rumax'
                        - ...
        **kwargs : additional arguments of property classes (e.g. use_imask for MartheField)

        Returns:
        --------
        Stock property class in prop dictionary

        Examples:
        --------
        mm = MartheModel('mona.rma')
        mm.load_prop('emmca')
        """
        # ---- Manage fields
        if prop in self.mlfiles.keys():
            self.prop[prop] = MartheField(prop, self.mlfiles[prop], self, **kwargs)

        # ---- Manage pumping
        elif prop == 'aqpump':
            self.prop[prop] = MarthePump(self, mode = 'aquifer', **kwargs)

        elif prop == 'rivpump':
            self.prop[prop] = MarthePump(self, mode = 'river', **kwargs)

        # ---- Manage soil property
        elif prop == 'soil':
            self.prop['soil'] = MartheSoil(self, **kwargs)

        # ---- Not supported property
        else:
            print(f"Property `{prop}` not supported.")




    def write_prop(self, prop=None):
        """
        Write MartheModel required properties by name.

        Parameters:
        ----------
        prop (str) : supported property name.
                     Can be :
                     - Field (MartheField)
                        - 'permh'
                        - 'emmca'
                        - 'emmli'
                        - 'kepon'
                     - Pumping (MarthePump)
                        - 'aqpump'
                        - 'rivpump'
                     - Zonal Soil properties (MartheSoil)
                        - 'cap_sol_progr'
                        - 'aqui_ruis_perc'
                        - 't_demi_percol'
                        - 'rumax'
                        - ...

        Returns:
        --------
        write property data (already loaded)

        Examples:
        --------
        mm = MartheModel('mona.rma')
        mm.write_prop('emmca')

        """
        # -- Manage property(ies) to write 
        props = self.prop.keys() if prop is None else marthe_utils.make_iterable(prop)
        # -- Write required properties
        for p in props:
            self.prop[p].write_data()




    @classmethod
    def from_config(cls, configfile):
        """
        Load an existing Marthe model from a configuration file written from 
        pymarthe.MartheOptim.write_config(). The return MartheModel instance
        contains all parametrizes properties with values from related parameters
        files ('grid' and 'list' prameters).
        Note : for a forward run use, remember to write all the model properties
               on disk using the .write_prop() method.


        Parameters:
        ----------
        configfile (str) : parametrization configuration file.

        Returns:
        --------
        mm (MartheModel) : MartheModel instance with parametrizes properties.

        Examples:
        --------
        mmfrom = MartheModel.from_config('myconfiguration.config')

        """
        # -- Build MartheModel from configuration file
        hdic, pdics, _ = pest_utils.read_config(configfile)
        si = None if hdic['Model spatial index'] == 'None' else hdic['Model spatial index']
        mm = cls(hdic['Model full path'], spatial_index=si)

        # -- Iterate over parameter dictionaries
        for pdic in pdics:
            # -- Load property by name
            prop = pdic['property name']

            # -- Set list-like properties
            if pdic['type'] == 'list':
                #if not prop in mm.prop.keys():
                mm.load_prop(prop)
                mm.prop[prop].set_data_from_parfile(parfile = os.path.normpath(pdic['parfile']),
                                                    keys = pdic['keys'].split(','),
                                                    value_col = pdic['value_col'],
                                                    btrans = pdic['btrans'])
            # -- Set grid-like properties
            elif pdic['type'] == 'grid':
                #if not prop in mm.prop.keys():
                use_imask = pdic['use_imask']=='True'
                mm.load_prop(prop,use_imask=use_imask)
                # -- Get izone as MartheField instance
                izone = MartheField(f'i{prop}', os.path.normpath(pdic['izone']), mm, use_imask=use_imask)
                # -- Set all field values (zpc and pp)
                for pf in  pdic['parfile'].split(','):
                    mm.prop[prop].set_data_from_parfile(parfile= os.path.normpath(pf),
                                                        izone= izone,
                                                        btrans= pdic['btrans'])

        # -- Return MartheModel instance
        return mm


    def get_extent(self):
        """
        Return the model domain extension.

        Returns:
        --------
        extent (list) : model extension
                        Format: [xmin, ymin, xmax, ymax]

        Examples:
        --------
        mm.modelgrid.get_extent()

        """
        # -- Extract extent from imask (main grid)
        if self.spatial_index is None:
            mg0 = self.imask.to_grids(layer=0)[0]
            xmin, ymin = mg0.xl, mg0.yl
            xmax, ymax = mg0.xl+mg0.Lx, mg0.yl+mg0.Ly
            extent = [xmin, ymin, xmax, ymax]
        # -- Extract extent from spatial index
        else:
            extent = self.spatial_index.bounds
        # -- Return
        return extent



    def get_edges(self, closed=False):
        """
        Return the xy-coordinates of model domain edges.

        Parameters:
        ----------
        closed (bool) : whatever adding first point in return list
                        (in order to close polygon).

        Returns:
        --------
        edges (list) : list of xy-coordinates (points).
                       If closed is False:
                            Format: [edge_lower_left, edge_upper_left,
                                     edge_upper_right, edge_lower_right]
                       If closed is False:
                            Format: [edge_lower_left, edge_upper_left,
                                     edge_upper_right, edge_lower_right]

        Examples:
        --------
        mm.modelgridget_edges(closed=False)

        """
        # -- Fetch model extension
        xmin, ymin, xmax, ymax = self.get_extent()
        # -- Build list of edges coordinates
        edges = [[xmin, ymin],
                 [xmin, ymax],
                 [xmax, ymax],
                 [xmax, ymin]]
        # -- Add first point if required (closed polygon)
        if closed:
            edges.append([xmin, ymin])
        # -- Return
        return edges





    def remove_autocal(self):
        """
        Function to make marthe auto calibration silent.
        wrapper to marthe_utils.remove_autocal().

        Parameters:
        ----------
        self : MartheModel instance

        Returns:
        --------
        Write in .mart file inplace

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.remove_autocal()
        """
        marthe_utils.remove_autocal(self.rma_file, self.mlfiles['mart'])    



    def make_silent(self):
        """
        Function to make marthe run silent

        Parameters:
        ----------
        self : MartheModel instance

        Returns:
        --------
        Write in .mart inplace

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.make_silent()
        """
        marthe_utils.make_silent(self.mlfiles['mart']) 



    def get_outcrop(self, as_2darray=False):
        """
        Function to get outcropping layer number
        (integer) as MartheField instance or 2D-array.

        Parameters:
        ----------
        as_2darray (bool) : return 2D-array of outcroping layer.
                            Only available for non nested model.
                            Default is False

        Returns:
        --------
        outcrop (MartheField/array) : outcropping layer numbers.

        Examples:
        --------
        mm = MartheModel('mymodel.rma')
        outcrop_arr = mm.get_outcrop()
        """
        if as_2darray:
            err_msg = "ERROR : cannot return a 2D-array for nested model."
            assert self.nnest == 0, err_msg
            # ---- Set list of arrays with layer number on active cell
            layers = [ilay * imask for ilay, imask in enumerate(self.imask.as_3darray())]
            # ---- Transform 0 to NaN
            nanlayers = []
            for layer in layers:
                arr = layer.astype('float')
                arr[arr == 0] = np.nan
                nanlayers.append(arr)
            # ---- Get minimum layer number excluding NaNs
            outcrop = np.fmin.reduce(nanlayers)
            # # ---- Back transform inactive zone to 0
            outcrop[np.isnan(outcrop)] = -9999
            # ---- Return outcrop layers as 2D-array
            return outcrop.astype(int)

        else:
            # ---- Fetch DataFrame data from imask (as deepcopy)
            rec = deepcopy(self.imask.data)
            df = pd.DataFrame.from_records(rec)
            # ---- Replace default masked values by NaN and mask it
            df['value'].replace(self.imask.dmv, np.nan, inplace=True)
            mask = df['value'].notnull()
            # ---- Raplace imask boolean value by the layer id
            df.loc[mask,'value'] = df.loc[mask,'layer']
            # ---- Set a unique id for each cell
            ncell = len(rec[rec['layer'] == 0])
            df.index = np.tile(np.arange(0, ncell), self.nlay)
            # ---- Group cells by id and perform minimum avoiding NaN values
            df['value'] = df.groupby(df.index)['value'].apply(list).apply(np.fmin.reduce)
            # ---- Set Nan values to -9999 (not 0) and convert back to recarray
            rec = df.fillna(-9999).to_records(index=False)
            # ---- Return outcrop layers as MartheField instance
            return MartheField('outcrop', rec , self)




    def query_grid(self, target=None, **kwargs):
        """
        High level method to perform queries on model grid.

        Parameters:
        ----------
        target (str/it) : requeried grid information to extract.
                          Can be:
                            - 'node' : cell unique id
                            - 'layer': layer id
                            - 'inest': nested grid id
                            - 'i'    : row number
                            - 'j'    : column number
                            - 'xcc'  : x-coordinate of the cell centroid
                            - 'ycc'  : y-coordinate of the cell centroi
                            - 'dx'   : cell width
                            - 'dy'   : cell height
                            - 'ative': cell activity (0=inactive, 1=active)
                          If None, all grid informations will be considered.
                          Default is None.

        **kwargs : required spatial information to subset grid.

        Returns:
        --------
        df (DataFrame) : subset DataFrame.
                         Index : query variable(s).
                         Columns : target(s) variables. 

        Examples:
        --------
        # -- Query grid to extract x-y cell resolutions
        df = mm.query_grid(target=['dx','dy'],
                           i = [23,45,56],
                           j = [45,67,89],
                           layer = [0,5,4],
                           inest = [0,0,0])
        
        """
        # -- Build modelgrid if not exists
        if self.modelgrid is None:
            self.build_modelgrid()

        # -- Make all kwargs values iterable
        d = {k:marthe_utils.make_iterable(v) for k,v in kwargs.items()}

        # -- Check kwargs names validity
        nf = [f"'{kw}'" for kw in d.keys() if kw not in self.modelgrid.columns]
        err_msg = 'ERROR : some query names not found in ' \
                   'modelgrid : {}.'.format(', '.join(nf))
        assert len(nf) == 0, err_msg

        # -- Check kwargs values validity
        err_msg = 'ERROR : all query values must have the same length. ' \
                  'Given : {}.'.format(', '.join([str(len(v)) for v in d.values()]))
        assert marthe_utils.unanimous(d), err_msg

        # -- Build high level indexes
        kidx = list(d.keys())
        vidx = list(zip(*d.values())) if len(d.keys()) > 1 else list(*d.values())

        # -- Manage target information
        if target is None:
            target = [c for c in self.modelgrid.columns if c not in kidx]
        else:
            target = marthe_utils.make_iterable(target)

        # -- Check target validity
        nf = [f"'{t}'" for t in target if t not in self.modelgrid.set_index(kidx).columns]
        err_msg = 'ERROR : some `target` values not found in ' \
                   'modelgrid or already use for grid query: {}.'.format(', '.join(nf))
        assert len(nf) == 0, err_msg

        # -- Query modelgrid DataFrame
        df = self.modelgrid.set_index(kidx).loc[vidx,target]

        # -- Return subset DataFrame
        return df



    def isin_extent(self, x, y):
        """
        Boolean response to check if (a) point(s) is in model extension.

        Parameters:
        ----------
        x, y (float/iterable) : xy-coordinate(s) of the required point(s)

        Returns:
        --------
        res (list) : boolean mask.
                     True : point in model domain
                     False: point out of model domain

        Examples:
        --------
        mm._isin_extent(x=[256.8,278.1], y=[345.2,349.3])

        """
        # -- Get model domain extension as polygon
        ext = self.get_edges(closed=True)
        # -- Make coords iterable
        _x,_y = [np.array(marthe_utils.make_iterable(coord)) for coord in [x,y]]
        res = shp_utils.point_in_polygon(_x, _y, ext)
        # -- Return boolean response
        return res



    def get_node(self, x, y, layer=None, only_active=False):
        """
        Function to fetch node id(s) from xy-coordinates.

        Parameters:
        ----------
        x, y (float/it) : xy-coordinate(s) of the required point(s)
        layer (int/it, optional) : layer(s) to intersect.
                                   Can be an integer (same layer for all
                                   points) or a sequence of integers for
                                   each xy-coordinates.
                                   If None, all layer will be considered.
                                   Default None.
        only_active (bool): whatever set unactive cell intersected to np.nan.
                            Default is None.

        Returns:
        --------
        nodes (list) : intersected nodes. 

        Examples:
        --------
        x, y = [456788.78, 459388.78], [6789567.2, 6789569.89]
        nodes = get_nodes(x, y, layer=2, only_active=True)
        
        """
        # -- Manage xy coordinates
        _x, _y = [marthe_utils.make_iterable(var) for var in [x,y]]

        # -- Check if xy-coordinates are in model extension
        inext = self.isin_extent(_x, _y)
        if any(x is False for x in inext):
            war_msg = 'Some xy-coordinates provided are out of ' \
                      'the model extension. No node(s) will be ' \
                      'return for these points.'
            warnings.warn(war_msg)

        # -- Build spatial index if required
        if self.spatial_index is None:
            self.build_spatial_idx()

        # -- Intercept coordinates with grid to extract node ids
        if layer is None:
            if only_active:
                nodes = []
                for ix, iy in zip(_x, _y):
                    inodes = []
                    # -- Intercept spatial index on objects only (slower)
                    for hit in sorted(self.spatial_index.intersection((ix,iy), objects='raw')):
                        # -- Verify whatever the cell is active
                        if hit[-1] == 1:
                            inodes.append(hit[0])
                        else:
                            inodes.append(np.nan)
                    nodes.append(inodes)
            else:
                # -- Intercept spatial index on node id only (faster)
                nodes = [sorted(self.spatial_index.intersection((ix,iy))) for ix, iy in zip(_x, _y)]

        else:
            # -- Manage layer input
            _layer = marthe_utils.make_iterable(layer)

            # ---- Allowed layer to be a simple integer for all xy-coordinates
            if (len(_layer) == 1) and (len(_x) > 1) :
                _layer = list(_layer) * len(_x)

            # ---- Assertion on variables length
            err_msg = "ERROR : x, y and layer must have the same length. " \
                      f"Given : x = {len(_x)}, y = {len(_y)}, layer ={len(_layer)}."
            assert len(_x) == len(_y) == len(_layer), err_msg

            nodes = []
            for ix, iy, ilay in zip(_x, _y, _layer):
                # -- Intercept spatial index on objects only (slower)
                for hit in self.spatial_index.intersection((ix,iy), objects='raw'):
                    # -- Verify in intersect required layer
                    if hit[1] == ilay:
                        if only_active:
                            # -- Verify whatever the cell is active
                            if hit[-1] == 1:
                                nodes.append(hit[0])
                            else:
                                nodes.append(np.nan)
                        else:
                            nodes.append(hit[0])
        # -- Return nodes
        return nodes



    def all_active(self, node):
        """
        Check whatever a node or a serie of node are all active.

        Parameters:
        ----------
        node (int/it) : node(s) to test

        Returns:
        --------
        res (bool) : boolean response.

        Examples:
        --------
        mm.is_active(6794)
        
        """
        # -- Get node as iterable
        n = marthe_utils.make_iterable(node)
        # -- Test if imask > 0 for each nodes
        res = all(x > 0 for x in self.imask.data['value'][n])
        # -- Return
        return res




    def any_active(self, node):
        """
        Check whatever a node or a serie of node contains at least 1 active.

        Parameters:
        ----------
        node (int/it) : node(s) to test

        Returns:
        --------
        res (bool) : boolean response.

        Examples:
        --------
        mm.is_active(6794)
        
        """
        # -- Get node as iterable
        n = marthe_utils.make_iterable(node)
        # -- Test if imask > 0 for each nodes
        res = any(x > 0 for x in self.imask.data['value'][n])
        # -- Return
        return res



    @marthe_utils.deprecated
    def get_ij(self, x, y, stack=False):
        """
        Function to extract row(s) and column(s) from provided
        xy-coordinates in model extension.
        Simple wrapper to imask.intersects().

        Parameters:
        ----------
        x, y (float/iterable) : xy-coordinate(s) of the required point(s)
        layer (int/iterable) : layer id(s) to intersect data.

        Returns:
        --------
        i, j (float/iterable) : correspondin row(s) and column(s)
        stack (bool) : stack output array
                       Format : np.array([x1, x1],
                                         [x2, y2],
                                            ...    )
                       Default is False.

        Examples:
        --------
        mm = MartheModel('mymodel.rma')
        x, y = [456788.78, 459388.78], [6789567.2, 6789569.89]
        rowcol = mm.get_ij(x,y, stack=True)

        """
        # ---- Make i and j iterable
        _x, _y = [marthe_utils.make_iterable(var) for var in [x,y]]

        # ---- Assert that i and j have the same length
        err_msg = "ERROR: x and y must have the same length." \
                  f"Given: x = {len(_x)}, y = {len(_y)}."
        assert len(_x) == len(_y), err_msg

        # ---- Intersects points from spatial index in imask
        _layer = [0] * len(_x)
        rec = self.imask.sample(_x, _y, _layer)
        i, j = rec['i'], rec['j']

        # ---- Manage output
        if len(_x) == 1:
            out = np.column_stack([i,j]) if stack else (i[0], j[0])
        else:
            out = np.column_stack([i,j]) if stack else (i, j)

        # ---- Return coordinates
        return out



    @marthe_utils.deprecated
    def get_xy(self, i, j, stack=False):
        """
        Function to extract x-y cellcenters from provided
        row(s) and column(s) in model extension.

        Parameters:
        ----------
        i, j(float/iterable) : row(s), column(s)
        stack (bool) : stack output array
                       Format : np.array([i1, j1],
                                         [i2, j2],
                                            ...    )
                       Default is False.

        Returns:
        --------
        x, y (float/iterable) : correspondinf xy-cellcenters

        Examples:
        --------
        mm = MartheModel('mymodel.rma')
        i, j= [23, 56, 89], [78, 123, 134]
        coords = mm.get_ij(i,j, stack=True)
        """
        # ---- Make i and j iterable
        _i, _j = [marthe_utils.make_iterable(var) for var in [i,j]]

        # ---- Assert that i and j have the same length
        err_msg = "ERROR: i and j must have the same length." \
                  f"Given: i = {len(_i)}, j = {len(_j)}."
        assert len(_i) == len(_j), err_msg

        # ---- Subset data by pairs on first layer
        df = pd.DataFrame.from_records(self.imask.get_data(layer=0))
        df['temp'] = df['i'].astype(str) + '_' + df['j'].astype(str)
        df_ss = df.loc[df.temp.isin([f'{ii}_{jj}' for ii,jj in zip(_i,_j)])]

        # ---- Fetch corresponding xcc, ycc
        x, y = [df_ss[c].to_numpy() for c in list('xy')]

        # ---- Manage output
        if len(_i) == 1:
            out = np.column_stack([x, y]) if stack else (x[0], y[0])
        else:
            out = np.column_stack([x, y]) if stack else (x, y)
        
        # ---- Return coordinates
        return out



    def extract_refine_levels(self):
        """
        Function to extract refine levels of each nested grid.
        The main grid (inest=0) must have a refine level 
        equal to 1 (= division for each x-y direction).

        Parameters:
        -----------

        Returns:
        --------
        rlevels : refine levels from grid file (permh).
                  Format : {inest_0 : None,
                            inest_1 : refine_level_1,
                                   ...
                            inest_N : refine_level_N}

        Examples:
        --------
        rl = mm.extract_refine_levels()
        """
        # -- Read 'permh' with adjacent cells (layer 0)
        mgs = marthe_utils.read_grid_file(
                    self.mlfiles['permh'],
                        keep_adj=True)
        mgs0 = [mg for mg in mgs if mg.layer == 0]
        # -- Compute rlevel (dx_main_grid / dx_nested_grid)
        rlevels = {mg.inest : int(mg.dx[0]//mg.dx[1])
                                if mg.inest > 0 else None
                                    for mg in mgs0}
        # -- Return
        return rlevels




    def get_xycellcenters(self, stack=False):
        """
        Function to get xy-cell centers of the model grid.

        Parameters:
        ----------
        stack (bool) : stack output array
                       Format : np.array([x1, y1],
                                         [x2, y2],
                                            ...    )
                       Default is False.

        Returns:
        --------
        if stack is False:
            xcc, ycc (array) : xy-cell centers.
                               shape: (ncpl,) (ncpl,)
        if stack is True:
            xycc (array) : stacked xy-cell centers
                           shape: (ncpl,2)

        Examples:
        --------
        xcc, ycc = mm.get_xycellcenters()
        """
        # -- Get cell centers from imask
        xcc = self.imask.data['x'][:self.ncpl]
        ycc = self.imask.data['y'][:self.ncpl]
        # -- Return
        if stack:
            return np.column_stack([xcc, ycc])
        else:
            return xcc, ycc



    def get_layer_from_depth(self, x, y, depth, as_list=True):
        """
        Function to infer the layer id at a given xyz coordonates.
        Note: still experimental.

        Parameters:
        ----------
        x, y (float/iterable) : xy-coordinate(s) of the required point(s)
        depth (float/iterable) : depth to infer (=z)
        as_list (bool): whatever returning only list of layer ids or 
                        whole Dataframe with x,y,depth,layer,name.
                        Default is True.

        Returns:
        --------
        ilays (list) : 

        Examples:
        --------
        mm = MartheModel('mymodel.rma')
        x, y = [456788.78, 459388.78], [6789567.2, 6789569.89]
        layers = mm.get_layer_from_depth(x,y,depth=[223.1, 568])

        """
        # ---- Set all grid masked value
        mv = [9999,8888,0,-9999]
        # ---- Get topography and substratum altitude as array
        #      with shape (nlay, ncpl)
        geom_arrs = []
        for g in ['topog', 'hsubs']:
            if self.geometry[g] is None:
                self.load_geometry(g)
            # -- Convert to reshaped array
            mf = self.geometry[g]
            arr = mf.data['value'].reshape((self.nlay,self.ncpl))
            # -- Convert masked grid value to nan
            arr[np.isin(arr, mv)] = np.nan
            geom_arrs.append(arr)

        _topog, _hsubs = geom_arrs

        # ---- Make coordinates iterables
        _x, _y, _d = [marthe_utils.make_iterable(var) for var in [x,y,depth]]

        # -- Initialized output lists
        altitudes = []
        target_layers = []

        # -- Iterate over xy points
        for ix, iy, d in zip(_x,_y,_d):
            # -- Get mask of current point by layer (same for all layer)
            cmask = self.imask.sample(ix,iy, layer=0, as_mask=True)[:self.ncpl]
            # -- Detect if topography correspond to the top altitude
            alti = _topog[0][cmask][0] if ~np.isnan(_topog[0][cmask]) else None
            # -- Compute altitude as first not null substratum  - depth 
            ilay = 0
            while alti is None:
                # -- Detect if substratum at this point is defined
                if ~np.isnan(_hsubs[ilay][cmask]):
                    alti = _hsubs[ilay][cmask] - d
                    alti = alti[0]
                # -- Pass to next layer
                ilay += 1
            # -- Store altitude of point
            altitudes.append(alti)
            # -- Extract target id layer
            target = ilay + np.argmin(np.ravel(_hsubs[ilay:,cmask] < alti))
            target_layers.append(target)

        # ---- Return
        if as_list:
            return target_layers
        else:
            dic = {'x': _x,
                   'y': _y,
                   'depth':_d,
                   'altitude': altitudes,
                   'layer': target_layers,
                   'name':  self.layers_infos.loc[target_layers,'name']}
            return pd.DataFrame.from_dict(dic).reset_index(drop=True)





    def run_model(self,exe_name = 'marthe', rma_file = None, 
                      silent = True, verbose=False, pause=False,
                      report=False, cargs=None):
        """
        Run Marthe model using subprocess.Popen. It communicates 
        with the model's stdout asynchronously and reports progress 
        to the screen with timestamps

        Parameters
        ----------
        exe_name (str, optional) : Marthe executable name.
                                   Note: can be the entire path if the
                                   exename is not in environment path
                                   Default is 'marthe'.
        rma_file (str, optional) : .rma file of model to run.
        silent (bool, optional) : run marthe model as silent 
        verbose (bool, optional) : echo run information to screen
                                   Default is False.
        pause (bool, optional) : pause upon completion
                                 Default is False.
        report (bool, optional) : save stdout lines to a list (buff) 
                                  which is returned by the method
                                  Default is True.
        cargs (str/list, optional) : additional command line arguments to pass to the executable.
                                     Default is None.

        Returns
        -------
        (success, buff)
        success (bool) : Binary success of the run 
        buff (list) :  stdout
        """
        # ---- Initialize variable
        success = False
        buff = []
        normal_msg='normal termination'

        # ---- Force model to run as silent if required
        if silent:
            self.make_silent()

        # ---- Check to make sure that program and namefile exist
        exe = which(exe_name)
        if exe is None:
            # -- Try which() function for window user 
            import platform
            if platform.system() in 'Windows':
                    exe = which(exe_name + '.exe')

        if exe is None:
            s = 'The program {} does not exist or is not executable.'.format(
                exe_name)
            raise Exception(s)
        

        # ---- Fetch Marthe .rma file if not provided
        if rma_file is None : 
            rma_file = os.path.join(self.mldir, self.rma_file)

        # ---- Simple function for the thread to target
        def q_output(output, q):
            for line in iter(output.readline, b''):
                q.put(line)

        # ---- Create a list of arguments to pass to Popen
        argv = [exe_name]
        if rma_file is not None:
            argv.append(rma_file)

        # ---- Add additional arguments to Popen arguments
        if cargs is not None:
            cargs = [arg for arg in cargs if isinstance(cargs, str)]
            for t in cargs:
                argv.append(t)

        # ---- Run the model with Popen
        proc = sp.Popen(argv, stdout=sp.PIPE, stderr=sp.STDOUT)

        # ---- Some tricks for the async stdout reading
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
        # -- Examine run buff
        for line in buff:
            if normal_msg in line:
                print("success")
                success = True
                break

        if pause:
            input('Press Enter to continue...')
        return success, buff




    def get_vtk(self, vertical_exageration=0.05, hws = None,
                      smooth=False, binary=True, xml=False,
                      shared_points=False):

        """
        Build vtk unstructured grid from model geometry.
        Wrapper of pymarthe.utils.vtk_utils.Vtk class.
        Required python `vtk` package.

        Parameters:
        -----------
        vertical_exageration (float) : floating point value to scale vertical
                                       exageration of the vtk points.
                                       Default is 0.05.
        hws (str) : hanging wall state, flag to define whatever the superior
                    hanging walls of the model are defined as normal layers
                    (explivitly) or not (implicitly).
                    Can be:
                        - 'implicit'
                        - 'explicit'
                    If None, self.hws will be use.
                    Default is None.
        smooth (bool) : boolean flag to enable interpolating vertex elevations
                        based on shared cell.
                        Default is False.
        binary (bool) : Enable binary writing, otherwise classic ASCII format 
                        will be consider.
                        Default is True.
                        Note : binary is prefered as paraview can produced bug
                               representing NaN values from ASCII (non xml) files.
        xml (bool) : Enable xml based VTK files writing.
                     Default is False.
        shared_points (bool) : Enable sharing points in grid polyhedron construction.
                               Default is False.
                               Note : False is prefered as paraview has a bug (8/4/2021)
                                      where some polyhedron will not properly close when
                                      using shared points.


        Returns:
        --------
        vtk (pymarthe.utils.vtk_utils.Vtk) : Vtk class containg unstructured vtk grid

        Example:
        --------
        mm = MartheModel('mymodel.rma', spatial_index='mymodel_si')
        myvtk = mm.get_vtk(vertical_exageration=0.02,
                          hws= 'implicit', smooth=True)

        """
        # -- Dynamic import of vtk module
        from .utils import vtk_utils

        # -- Initialize Vtk class
        hws = self.hws if hws is None else hws
        vtk = vtk_utils.Vtk(self, vertical_exageration, hws,
                            smooth, binary, xml, shared_points)

        # -- Return vtk instance
        return vtk



    def show_run_times(self, logfile=None, tablefmt='fancy'):
        """
        Print model run times.

        Parameters:
        ----------
        logfile (str) : log filename.
                        If None, logfile = .mldir + 'bilandeb.txt'
                        Default is None.

        Returns:
        --------
        Print message on console.

        Examples:
        --------
        mm.run_model(exe_name = 'Marth_R8')
        mm.show_run_times()

        """
        # -- Manage log file name
        lf = os.path.join(self.mldir, 'bilandeb.txt') if logfile is None else logfile

        # -- Assert log file exists
        err_msg = f"Could not found {lf} log file. " \
                   "Make sure to run the model before " \
                   "calling `.show_run_times()` method."
        assert os.path.exists(lf), err_msg

        # -- Extract run times per process
        df = marthe_utils.get_run_times(lf)

        # -- Print on console
        try:
            # -- Fancier table print
            import tabulate
            print(df.to_markdown(tablefmt=tablefmt, colalign=("left", "right")))
        except:
            # -- Classic table print
            print(df)
        



    def get_time_window(self, tw_type='date'):
        """
        Function to extract model time window in .mart file.
        Wrapper to marthe_utils.get_tw().


        Parameters:
        ----------
        tw_type (str, optional) : time window output type.
                                  Can be :
                                    - 'date' : return pd.timestamp objects
                                    - 'istep': return integers
                                  Default is 'date'.

        Returns:
        --------
        tw_min, tw_max (tuple): time window bounds (start/end)

        Examples:
        --------
        # -- From isteps
        istart, iend = get_time_window(tw_type='istep')
        # -- Get time window dates
        start, end = get_time_window(tw_type='date')

        """
        # ---- Extract time window
        tw_min, tw_max =  marthe_utils.get_tw(martfile= self.mlfiles['mart'],
                                              pastpfile= self.mlfiles['pastp'],
                                              tw_type=tw_type)
        # ---- Return time window as tuple
        return tw_min, tw_max



    def set_time_window(self, start=None, end=None):
        """
        Function to set/change model time window in .mart file.
        Note: the .pastp file will not be modify.
        Wrapper to marthe_utils.set_tw()

        Parameters:
        ----------
        start (str/int, optional) : string date or istep number of required
                                    first timestep to consider.
                                    If None, the first istep (in .pastp file)
                                    will be considered.
                                    Default is None.

        end (str/int, optional) : string date or istep number of required 
                                  last timestep to consider.
                                  If None, the last istep (in .pastp file)
                                  will be considered.
                                  Default is None.

        Returns:
        --------
        Change .mart file inplace with required time window.

        Examples:
        --------
        # -- From isteps
        mm.set_time_window(start=10, end=35)

        # -- From dates
        mm.set_time_window(start='1999/01/28', end=65)
        mm.set_time_window(start='1992/01/01', end='1993/05/02')

        """
        # ---- Wrapper to utils
        marthe_utils.set_tw( start= start,
                             end= end,
                             martfile= self.mlfiles['mart'],
                             pastpfile= self.mlfiles['pastp'] )




    def set_hydrodyn_periodicity(self, istep, external=False, new_pastpfile=None):
        """
        Function to manage hydrodynamic computation periodicity in .pastp file.
        Wrapper to marthe_utils.mm.set_hydrodyn_periodicity().

        Parameters:
        ----------

        istep (str/int/iterable) : required istep to activate hydrodynamic computation.
                                   Can be :
                                        - 'all' : activate for all timesteps
                                        - 'none': desactivate for all timesteps
                                        - 'start:end:step' : string sequence
                                        - [0,1,2,..] : any integer iterables

        external (bool, optional) : whatever create a external file with required
                                    hydrodynamic computation periodicity.
                                    Note: external optional will not be considered
                                          for global `istep` such as 'all', 'none'
                                    Default is False.

        new_pastpfile (str, optional) : path to the new pastp file to write.
                                        If None, will overwrite the input pastp file.
                                        Default is None.

        Returns:
        --------
        (Over-)write pastp file.
        If `external` == True, will also write 'cacul_hydro.txt'.
                          .

        Examples:
        --------
        # -- All timesteps
        mm.set_hydrodyn_periodicity(istep= 'all', external=False)
        # -- Weekly
        mm.set_hydrodyn_periodicity(istep= '::7', external=True)
        # -- Annual
        mm.set_hydrodyn_periodicity(istep= '::365', external=False)
        # -- Specific
        mm.set_hydrodyn_periodicity(istep= [0,5,6,7,9,11], external=True)

        """
        # ---- Wrapper to utils
        marthe_utils.hydrodyn_periodicity(pastpfile= self.mm.mlfiles['pastp'],
                                          istep= istep,
                                          external= external,
                                          new_pastpfile= new_pastpfile)


    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheModel'



class SpatialReference():
    """
    Inspired from FloPy, for compatibility with PyEMU
    """
    def __init__(self, mm):
        """
        Parameters
        ----------
        ml : instance of MartheModel
        """
        mg = mm.imask.to_grids(layer=0, inest=0)[0]
        self.nrow, self.ncol = mg.nrow, mg.ncol


    def __str__(self):
        """
        Internal string method.
        """
        return 'SpatialReference'

