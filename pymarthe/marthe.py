"""
Contains the MartheModel and Spatial Reference classes.
Designed for structured and nested grid.
"""

import os, sys
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
    def __init__(self, rma_path, spatial_index = False):
        """
        Parameters
        ----------
        rma_path (str): path to the Marthe .rma file from which
                        the model name and working directory
                        will be identified.

        spatial_index (bool/str) : model spatial index management.
                                   If True, a spatial index will be created.
                                   If False, spatial index set to None.
                                   If string, spatial index set from external file.
                                   Default is False.


        Examples
        --------
        mm = MartheModel('/Users/john/zone/model/mymodel.rma')

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

        # ---- Store model grid infos from permh field
        self.imask = self.build_imask()

        # ---- Build model spatial index
        if isinstance(spatial_index, str):
            from rtree.index import Index
            self.spatial_index = Index(spatial_index)
            self.sifile = spatial_index
        else:
            if spatial_index:
                self.build_spatial_idx()
            else:
                self.sifile = None
                self.spatial_index = None

        # ---- Set number of simulated timestep
        self.nstep = len(self.mldates)

        # ---- Initialize property dictionary with permeability data
        self.prop = {}
        self.load_prop('permh')

        # ---- Load geometry
        self.geometry = {g : None for g in ['sepon', 'topog', 'hsubs']}

        # ---- Set spatial reference (used for compatibility with pyemu geostat utils)
        self.spatial_reference = SpatialReference(self)




    def load_geometry(self, g=None):
        """
        Load and store geometry grid information of a MartheModel instance.

        Parameters:
        ----------
        g (str, optional) : Marthe geometry grid file.
                            Can be 'sepon', 'topog', 'hsubs', ..
                            If None all geometry grid file will be loaded.
                            Default is None.

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
        assert all(g in self.geometry.keys() for g in _g), err_msg

        # ---- Load geometry as MartheField instance
        for g in _g:
            self.geometry[g] = MartheField(g, self.mlfiles[g] , self)





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



    def build_spatial_idx(self, sifile = None):
        """
        Function to build a spatial index on field data.

        Parameters:
        ----------
        self (MartheField) : MartheField instance

        Returns:
        --------
        spatial_index (rtree.index.Index)

        Examples:
        --------
        si = mf.build_spatial_idx()

        """
        # ---- Import spatial index from Rtree module
        from rtree.index import Index
        # ---- Initialize spatial index
        if sifile is None:
            sifile = os.path.join(self.mldir, f'{self.mlname}_si')
        si = Index(sifile)
        # ---- Fetch model cell as polygons
        polygons = []
        for mg in self.imask.to_grids():
            polygons.extend([p[0] for p in mg.to_pyshp()])
        # ---- Build bounds
        bounds = []
        for polygon in polygons:
            xmin, ymin = map(min,np.column_stack(polygon))
            xmax, ymax = map(max,np.column_stack(polygon))
            bounds.append((xmin, ymin, xmax, ymax))
        # ---- Implement spatial index
        print('Building spatial index ...')
        for i, bd in enumerate(bounds):
            marthe_utils.progress_bar((i+1)/len(bounds))
            si.insert(i, bd)
        # ---- Stock and Store spatial index
        si.flush()
        self.sifile = sifile
        self.spatial_index = si




    def load_prop(self, prop):
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
            self.prop[prop] = MartheField(prop, self.mlfiles[prop], self)

        # ---- Manage pumping
        elif prop == 'aqpump':
            self.prop[prop] = MarthePump(self, mode = 'aquifer')

        elif prop == 'rivpump':
            self.prop[prop] = MarthePump(self, mode = 'river')

        # ---- Manage soil property
        elif prop == 'soil':
            self.prop['soil'] = MartheSoil(self)

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
        """
        # -- Build MartheModel from configuration file
        hdic, pdics, _ = pest_utils.read_config(configfile)
        mm = cls(hdic['Model full path'], eval(hdic['Model spatial index']))

        # -- Iterate over parameter dictionaries
        for pdic in pdics:
            # -- Load property by name
            prop = pdic['property name']
            if not prop in mm.prop.keys():
                mm.load_prop(prop)

            # -- Set list-like properties
            if pdic['type'] == 'list':
                mm.prop[prop].set_data_from_parfile(parfile = os.path.normpath(pdic['parfile']),
                                                    keys = pdic['keys'].split('\t'),
                                                    value_col = pdic['value_col'],
                                                    btrans = pdic['btrans'])
            # -- Set list-like properties
            elif pdic['type'] == 'array':
                pass

        # -- Return MartheModel instance
        return mm





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
        marthe_utils.remove_autocal(self.mlfiles['mart'])      



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




    def get_layer_from_depth(self, x, y, depth, as_list=True):
        """
        Function to infer the layer id at a given xyz coordonates.

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
        # ---- Verify that
        if self.geometry['topog'] is None:
            self.load_geometry('topog')
        # ---- Make coordinates iterables
        _x, _y, _d = [marthe_utils.make_iterable(var) for var in [x,y,depth]]
        # ---- Get topo at x, y points
        _topo = self.geometry['topog'].sample(x=_x, y=_y, layer=0)['value']
        # ---- Get copy of layer data
        df = self.layers_infos.copy(deep=True)
        # ---- Iterate over xy-topography
        ilays, layer_nmes = [], []
        for topo, d in zip(_topo, _d):
            # -- Compute depth from topo
            df['depth'] = df['thickness'].cumsum() - topo
            # -- identify layer id
            ilay = df.loc[df.depth > d, 'layer'].iloc[0]
            lay_nme =  df.loc[df.depth > d, 'name'].iloc[0]
            ilays.append(ilay)
            layer_nmes.append(lay_nme)
        # ---- Return
        if as_list:
            return ilays
        else:
            return pd.DataFrame({'x': _x, 'y': _y, 'depth':_d,
                                 'layer': ilays, 'name': layer_nmes})





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




    def get_vtk(self, vertical_exageration=0.05, hws = 'implicit',
                      smooth=False, binary=True,
                      xml=False, shared_points=False):

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
                    Default is 'implicit'.
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
        vtk = vtk_utils.Vtk(self, vertical_exageration, hws,
                            smooth, binary, xml, shared_points)

        # -- Return vtk instance
        return vtk



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

