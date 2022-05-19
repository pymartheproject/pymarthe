"""
Contains the classes related to field data.
Designed for handling distributed Marthe properties
(structured and unstructured grid)
"""

import os, sys
import numpy as np
import pandas as pd
from copy import deepcopy
import shutil
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection


from .utils import marthe_utils, shp_utils, pest_utils
from .utils.grid_utils import MartheGrid

encoding = 'latin-1'
dmv = [-9999., 0., 9999] # Default field masked values






class MartheField():
    """
    Wrapper Marthe --> python
    """
    def __init__(self, field, data, mm):
        """
        Marthe gridded property instance.

        Parameters
        -----------
        field (str) : property name.
                        Can be :
                            - 'permh'
                            - 'emmca'
                            - 'emmli'
                            - 'kepon'
                            - ...
        data (object): field data to read/set.
                       Can be:
                        - Marthe property file ('mymodel.permh')
                        - Recarray with layer, inest,... informations
                        - 3D-array
                        - List of MartheGrid instance
        mm (MartheModel) : parent Marthe model.
                           Note: consider providing a parent MartheModel
                           instance ss far as possile

        Examples
        -----------
        mm.mobs.compute_weights(lambda_dic)
        """
        self.field = field
        self.mm = mm
        self.dmv = dmv
        self.set_data(data)
        self.maxlayer = len(self.to_grids(inest=0))
        self.maxnest = len(self.to_grids(layer=0)) - 1 # inest = 0 is the main grid

        # ---- Set property style
        self._proptype = 'grid'



    def get_xyvertices(self, stack=False):
        """
        Function to fetch x and y vertices of the modelgrid.

        Parameters:
        ----------
        stack (bool) : stack output array
                       Format : np.array([vx1, vy1],
                                          [vx2, vy2],
                                                ...    )
                       Default is False.

        Returns:
        --------
        if stack is False:
            vx, vy (array) : xy-vertices.
        if stack is True:
            vxy (array) : stacked xy-vertices

        Examples:
        --------
        vx, vy = mf.get_xyvertices()
        
        """
        # ---- Initialize xy vertices list
        xvertices, yvertices = [], []
        # ---- Iterate over MartheGrid on first layer only
        for mg in self.to_grids(layer=0):
            # -- Get vertices for whole structured grid (1D) 
            xvs, yvs = map(np.ravel, np.meshgrid(mg.xvertices, mg.yvertices))
            # -- Store it in main lists
            xvertices.extend(xvs)
            yvertices.extend(yvs)
        # ---- Return
        if stack:
            return np.column_stack([xvertices, yvertices])
        else:
            return np.array(xvertices), np.array(yvertices)




    def sample(self, x, y, layer, masked_values=None, as_mask=False, as_idx=False):
        """
        Sample field data by x, y, layer coordinates.
        It will perform simple a spatial intersection with field data.

        Parameters:
        ----------
        x, y (float/iterable) : xy-coordinate(s) of the required point(s)
        layer (int/iterable) : layer id(s) to intersect data.
        masked_values (None/list, optional) : values to ignore during the sampling process
                                              Default is None.
        as_mask (bool, optional) : return field boolean mask.
                                   Does not allow duplicated value (array <= x,y).
                                   Default is False.
        as_idx (bool, optional) : return field indexes (=cell ids, =node numbers).
                                  Allows duplicated values (array >= x,y)

        Returns:
        --------
        rec (np.recarray) : data intersected on point(s)
        mask (np.array [1D]) : field boolean mask (if as_mask = True)
        idx (np.array [1D]) : cell ids/node number/field indexes (if as_idx = True)

        Examples:
        --------
        rec = mf.sample(x=456.32, y=567.1, layer=0)
        """
        # ---- Make variables as iterables
        _x, _y, _layer = [marthe_utils.make_iterable(var) for var in [x,y,layer]]
        # ---- Allowed layer to be a simple integer for all xy-coordinates
        if (len(_layer) == 1) and (len(_x) > 1) :
            _layer = list(_layer) * len(_x)
        # ---- Assertion on variables length
        err_msg = "ERROR : x, y and layer must have the same length. " \
                  f"Given : x = {len(_x)}, y = {len(_y)}, layer ={len(_layer)}."
        assert len(_x) == len(_y) == len(_layer), err_msg

        # -- Build spatial index if required
        if self.mm.spatial_index is None:
            self.mm.build_spatial_index()

        # -- Manage masked values
        mv = [] if masked_values is None else marthe_utils.make_iterable(masked_values)

        # ---- Perform intersection on spatial index returning cell ids
        idx = []
        for ix,iy,ilay in zip(_x, _y, _layer):
            for hit in self.mm.spatial_index.intersection((ix,iy), objects='raw'):
                if (hit[1] == ilay) & (hit[-1] not in mv):
                    idx.append(hit[0])

        # ---- Convert nodes/indexes to boolean mask
        mask = np.zeros(len(self.data), dtype=np.bool)
        mask[idx] = True

        # ---- Return as required
        if as_idx:
            return idx
        elif as_mask:
            return mask
        else:
            return self.data[mask]



    def get_data(self, layer=None, inest=None,  as_array=False, as_mask=False, masked_values= list()):
        """
        Function to select/subset data.

        Parameters:
        ----------
        layer (int, optional) : number(s) of layer required.
                                If None, all layers are considered.
                                Default is None.
        inest (int, optional) : number(s) of nested grid required.
                                If None, all nested are considered.
                                Default is None.
        as_array (bool, optional) : returning data as 3D-array.
                                    Format : (layer*inest, row, col)
                                    Default is False.
        as_mask (bool, optional) : returning data as boolean index.
                                   Default is False.

        Returns:
        --------
        if as_array is False:
            rec (np.recarray) : selected data as recarray.
        if as_array is True:
            arr (3D-array) : selected data as array.

        Note: if all arguments are set to None,
              all data is returned.

        Examples:
        --------
        rec1 = mf.get_data(layer=[1,5,8], inest = 1)
        arr1 = mf.get_data(inest=3, as_array=True)

        """
        # ---- Manage layer input
        if layer is None:
            layers = np.unique(self.data['layer'])
        else: 
            layers = marthe_utils.make_iterable(layer)
        # ---- Manage inest input
        if inest is None:
            inests = np.unique(self.data['inest'])
        else: 
            inests = marthe_utils.make_iterable(inest)
        # ---- Manage masked_values
        mv = marthe_utils.make_iterable(masked_values)
        # ---- Apply required mask
        mask   = np.logical_and.reduce([np.isin(self.data['layer'], layers),
                                        np.isin(self.data['inest'], inests),
                                        ~np.isin(self.data['value'], mv)])
        # ---- Return as mask if required
        if as_mask:
            return mask
        # ---- Return as array
        if as_array:
            arrays = []
            # -- Subset by layer(s)
            for l in layers:
                ldata = self.data[self.data['layer'] == l]
                # ---- Subset by nested
                for n in inests:
                    ndata = ldata[ldata['inest'] == n]
                    # -- Fetch nrow, ncol of the current grid
                    nrow = np.max(ndata['i']) + 1
                    ncol = np.max(ndata['j']) + 1
                    # -- Rebuild array by reshaping with nrow, ncol
                    arrays.append(ndata['value'].reshape(nrow,ncol))
            # -- Returning array
            return np.array(arrays)
        # -- Returning as recarray
        else:
            return self.data[mask]





    def set_data(self, data, layer=None, inest=None):
        """
        Function to set field data.

        Parameters:
        ----------
        data (object) : field data to read/set.
                        Can be:
                            - Marthe property file ('mymodel.permh')
                            - Recarray with layer, inest,... informations
                            - 3D-array
        layer (int, optional) : number(s) of layer required.
                                If None, all layers are considered.
                                Default is None.
        inest (int, optional) : number(s) of nested grid required.
                                If None, all nested are considered.
                                Default is None.

        Returns:
        --------
        Set field data inplace.

        Examples:
        --------
        mf.set_data(permh_recarray) 
        mf.set_data(permh3darray) # structured grids only
        mf.set_data(2.3e-3) # all layers/inest
        mf.set_data(2.3e-3, layer=2, inest=3) # one nest

        """
        # ---- Set all usefull conditions
        _none = all(x is None for x in [layer, inest])
        _str = isinstance(data, str)
        _num = isinstance(data, (int, float))
        _rec = isinstance(data, np.recarray)
        _arr = isinstance(data, np.ndarray)
        _list = isinstance(data, list)

        # ---- Manage Marthe filename as input
        if np.logical_and.reduce([_str, _none]):
            grids = marthe_utils.read_grid_file(data)
            self.data =  self._grids2rec(grids)

        # ---- Manage list of MartheGrids as input
        if np.logical_and.reduce([_list, _none]):
            self.data =  self._grids2rec(data)

        # ---- Manage numeric input
        if _num:
            # -- If .data already exist
            try:
                # -- Set value on existing data
                mask = self.get_data(layer=layer, inest=inest,
                            masked_values=self.dmv, as_mask=True)
                self.data['value'][mask] = data
            except:
                # -- Copy .imask recarray and change value by provided float/int
                rec = deepcopy(self.mm.imask.data)
                mask = self.mm.imask.get_data(layer=layer, inest=inest,
                            masked_values=self.dmv, as_mask=True)
                rec['value'][mask] = data
                # -- Setting recarray as main .data
                self.data = rec

        # ---- Manage recarray input
        if _rec:
            self.data = data

        # ---- Manage 3D-array input
        if np.logical_and.reduce([_arr, not _rec, _none]):
            # -- Verify data is a 3D-array
            err_msg = f"ERROR: `data` array must be 3D. Given shape: {data.shape}."
            assert len(data.shape) == 3, err_msg
            self.data = self._3d2rec(data)




    def set_data_from_parfile(self, parfile, izone, btrans='none'):
        """
        Set field data from parameter file inplace.

        Parameters:
        ----------
        parfile (str) : path to parameter file to read.
                        Can be either zpc or pp parameter.

        izone (MartheField) : parameter zone(s) field.

        btrans (str, optional) : string function to back-transform
                                 field values.
                                 Default is 'none'.

        Examples:
        --------
        parfile = 'par/hk_zpc.dat'
        mf.set_data_from_parfile(parfile,
                                 izone,
                                 btrans='lambda x: 10**x')

        """
        # ---- Parse grid parameter file
        ptype, rec = pest_utils.parse_mgp_parfile(parfile, btrans)

        # ---- Manage zone of piecewise constancy (zpc) data
        if ptype == 'zpc':
            # -- Iterate over recarray (layer, zone, value)
            for l,z,v in rec:
                # -- Mask and set
                mask = np.logical_and.reduce(
                            [ self.get_data(layer=l, masked_values=dmv, as_mask=True),
                              izone.data['value'] == z]
                              )
                self.data['value'][mask] = v

        # ---- Manage pilot point (pp) data
        if ptype == 'pp':
            # -- Mask and set 
            l,z,v = rec
            mask = np.logical_and.reduce(
                            [ self.get_data(layer=l, masked_values=dmv, as_mask=True),
                              izone.data['value'] == z]
                              )
            self.data['value'][mask] = v




    def as_3darray(self):
        """
        Wrapper to .get_data(inest=0, as_array=True)
        with error handling for nested model.

        Parameters:
        ----------
        self (MartheField) : MartheField instance.

        Returns:
        --------
        arr (3D-array) : array 3D of main grid only.
                         Format : (layer, row, col)

        Examples:
        --------
        arr3d = mf.as_3darray()

        """
        # ---- Assert that gridded field is structured
        err_msg = "ERROR: `MartheField` can not contain nested grid(s)."
        if self.mm is None:
            condition = all(x == 0 for x in self.data['inest'])
        else:
            condition= self.mm.nnest == 0
        assert condition, err_msg

        # ---- Return 3D-array
        return self.get_data(inest=0, as_array=True)



    def _grids2rec(self, grids):
        """
        Read Marthe field property file.
        Wrapper of marthe_utils.read_grid_file().

        Parameters:
        ----------
        filename (str) : path to Marthe field property file.

        Returns:
        --------
        rec (np.recarray) : recarray with all usefull informations
                            for each model grid cell such as layer,
                            inest, value, ...

        Examples:
        --------
        rec = mf._grid2rec('mymodel.emmca')

        """
        # ---- Get recarray data from consecutive MartheGrid instance (mg) extract from field property file
        rec = np.lib.recfunctions.stack_arrays(
                [mg.to_records() for mg in grids],
                        autoconvert=True, usemask=False, asrecarray=True)
        # ---- Stack all recarrays as once
        return rec


    def _3d2rec(self, arr3d):
        """
        Convert 3D-array
        Wrapper of marthe_utils.read_grid_file().

        Parameters:
        ----------
        filename (str) : path to Marthe field property file.

        Returns:
        --------
        rec (np.recarray) : recarray with all usefull informations
                            for each model grid cell such as layer,
                            inest, value, ...

        Examples:
        --------
        rec = mf._grid2rec('mymodel.emmca')
        
        """
        # ---- Assert MartheModel exist
        err_msg = "ERROR: Building a `MartheField` instance from a 3D-array " \
                  "require a referenced `MartheModel` instance. " \
                  "Try MartheField(field, data, mm = MartheModel('mymodel.rma'))."
        assert self.mm.__str__() == 'MartheModel' , err_msg

        # ---- Assert array is 3D
        err_msg = f"ERROR: `data` must be a 3D-array. Given shape: {arr3d.shape}"
        assert len(arr3d.shape) == 3, err_msg

        # ---- Fetch basic model structure
        rec = deepcopy(self.mm.imask.data)
        df = pd.DataFrame.from_records(rec)
        
        # ---- Modify rec inplace
        for layer, arr2d in enumerate(arr3d):
            mask = (df.layer == layer) & (df.inest == 0)
            df.loc[mask, 'value'] = arr2d.ravel()

        # ---- Return recarray
        return df.to_records(index=False)




    def _rec2grid(self, layer, inest):
        """
        Single MartheGrid instance construction
        from given layer and inest.

        Parameters:
        ----------
        layer (int) : number of layer required.
        inest (int) : number of nested grid required.

        Returns:
        --------
        mg (MartheGrid) : Single Marthe grid instance.

        Examples:
        --------
        rec = mf._grid2rec('mymodel.emmca')

        """
        # ---- Get 2D-array of a unique layer/inest
        array = self.get_data(layer, inest, as_array=True)[0]
        nrow, ncol = array.shape
        # ---- Get xcc, ycc
        rec = self.get_data(layer, inest)
        xcc, ycc = np.unique(rec['x']), np.flip(np.unique(rec['y']))
        # ---- Fetch xl, yl, dx, dy
        dx, dy = map(abs,map(np.gradient, [xcc,ycc])) # Using the absolute gradient
        xl, yl = xcc[0] - dx[0]/2, ycc[-1] - dy[-1]/2
        # ---- Stack all MartheGrid arguments in ordered list
        istep = -9999
        args = [istep, layer, inest, nrow, ncol, xl, yl, dx, dy, xcc, ycc, array]
        # ---- Return MartheGrid instance
        return MartheGrid(*args, field = self.field)




    def to_grids(self, layer=None, inest=None):
        """
        Converting internal data (recarray) to a list of
        MartheGrid instance for given layers and inests.

        Parameters:
        ----------
        layer (int, optional) : number(s) of layer required.
                                If None, all layers are considered.
                                Default is None.
        inest (int, optional) : number(s) of nested grid required.
                                If None, all nested are considered.
                                Default is None.

        Returns:
        --------
        mgrids (list) : Required MartheGrid instances.

        Examples:
        --------
        mg = mf.to_grids(layer)
        """
        # ---- Subset data with required layer, inest
        rec = self.get_data(layer, inest)

        # ---- Iterate over layer and inest
        mgrids = []
        for layer in np.unique(rec['layer']):
            for inest in np.unique(rec['inest']):
                # ---- Convert selected recarray to single MartheGrid instance
                mgrids.append(self._rec2grid(layer, inest))
        # ---- Return list of grids
        return mgrids



    def write_data(self, filename=None):
        """
        Write field data in Marthe field property file.

        Parameters:
        ----------
        filename (str) : path to write field data.

        Returns:
        --------
        Write in Marthe field property file inplace.

        Examples:
        --------
        mf.write_data('modified.permh')

        """
        # ---- Manage filename input
        if filename is None:
            path = os.path.join(self.mm.mldir, self.mm.mlname)
            extension = self.field
            f = '{}.{}'.format(path, extension)
        else:
            f = filename

        # ---- Extract refine levels dictionary
        rl = self.mm.extract_refine_levels()

        # ---- Write field data from list of MartheGrid instance
        with open(f, 'w', encoding = marthe_utils.encoding) as f:
            for mg in self.to_grids():
                f.write(
                            mg.to_string(
                                maxlayer = self.maxlayer,
                                maxnest = self.maxnest,
                                rlevel = rl[mg.inest] )
                                                                        )




    def to_shapefile(self,  filename = None, layer=0, inest=None,
                            masked_values = dmv, log = False,
                            epsg=None, prj=None):
        """
        Write field data as shapefile by layer.


        Parameters:
        ----------
        filename (str, optional) : shapefile path to write.
                                   If None, filename = 'field_layer.shp'.
                                   Default is None
        layer (int, optional) : layer numerical id to export.
                                Note : a unique layer id is allowed
                                Default is 0.
        inest (int, optional) : nested grid numerical id to export.
                                If None, all nested grid are considered.
                                Default is None.
        masked_values (list, optional) : field values to ignore.
                                         Default is [-9999., 0., 9999].
        log (bool, optional) : logarithmic transformation of all values.
                               Default is False.
        epsg (int, optional) : Geodetic Parameter Dataset.
                               Default is None.
        prj (str, optional) : cartographic projection and coordinates
                              Default is None.

        Returns:
        --------
        Write shape in filename.

        Examples:
        --------
        filename = os.path.join('gis', 'permh_layer5.shp')
        mf.to_shapefile(filename, layer=5)

        """
        # -- Build filename if not provided
        if filename is None:
            filename = f'{self.field}_{layer}.shp'

        # ---- Perform a bunch of assertions on `layer` and `inest` arguments
        err_msg = f"`layer` must be an integer between 0 and {self.maxlayer -1}."
        assert isinstance(layer, int), err_msg
        assert 0 <= layer < self.maxlayer, err_msg

        # ---- Fetch pyshp parts (polygons)
        parts = []
        for mg in self.to_grids(layer=layer, inest=inest):
            parts.extend(mg.to_pyshp())

        # ---- fetch data
        data = self.get_data(layer=layer, inest=inest)
        df = pd.DataFrame.from_records(data).assign(parts=parts)

        # ---- Apply mask values
        mv = [] if masked_values is None else masked_values
        df = df[~df['value'].isin(mv)]
        # ---- Fetch subset parts (goemetries)
        parts = df.pop('parts')

        # ---- Log transform if required
        values = df.pop('value')
        col = 'val'
        if log:
            col = f'log({col})'
            values = np.log10(values)
        # ---- Set value 
        df[col] = values

        # ---- Convert reccaray to shafile
        shp_utils.recarray2shp(df.to_records(index=False), np.array(parts),
                               shpname=filename, epsg=epsg, prj=prj)

        # ---- Sum up export
        print("\n ---> Shapefile wrote in {} succesfully.".format(filename))



    def plot(self,  ax=None, layer=0, inest=None, vmin=None,
                    vmax=None, log = False, extent = None,
                    masked_values = dmv, **kwargs):
        """
        Plot data by layer

        Parameters:
        ----------
        ax (matplotlib.axes, optional) : matplotlib.axes custom ax.
                             If None basic ax will be create.
                             Default is None.
        layer (int, optional) : layer numerical id to export.
                                Note : a unique layer id is allowed
                                Default is 0.
        inest (int, optional) : nested grid numerical id to export.
                                If None, all nested grid are considered.
                                Default is None.
        vmin, vmax (float, optional) : min/max value(s) to plot.
        masked_values (list, optional) : field values to ignore.
                                         Default is [-9999., 0., 9999].
        log (bool, optional) : logarithmic transformation of all values.
                               Default is False.
        extent (tuple/list, optional): xy-limits of the plot window.
                                       Format: (xmin, ymin, xmax, ymax).
                                       If None, window correspond to the
                                       entire model domain.
                                       Default is None.
        **kwargs (optional) : matplotlib.PathCollection arguments.
                              (ex: cmap, lw, ls, edgecolor, ...)

        Returns:
        --------
        ax (matplotlib.axes) : standard ax with 2 collections:
                                    - 1 for rectangles
                                    - 1 for colorbar

        Examples:
        --------
        ax = mf.plot(layer=6, cmap='jet')
        ax.set_title('Field (layer = 6)', fontsize=14)

        """
        # ---- Perform a bunch of assertions on `layer` and `inest` arguments
        err_msg = f"`layer` must be an integer between 0 and {self.maxlayer -1}."
        assert isinstance(layer, int), err_msg
        assert 0 <= layer < self.maxlayer, err_msg

        # ---- Get patches
        patches, xmin, xmax, ymin, ymax = [[] for _ in range(5)]
        for mg in self.to_grids(layer=layer, inest=inest):
            patches.extend(mg.to_patches())
            xmin.append(mg.xl)
            ymin.append(mg.yl)
            xmax.append(mg.xl + mg.Lx)
            ymax.append(mg.yl + mg.Ly)

        # ---- Subset required data 
        data = self.get_data(layer=layer, inest=inest)
        df = pd.DataFrame.from_records(data).assign(patches=patches)
        
        # ---- Apply mask values
        if masked_values is None:
            rec = df.to_records(index=False)
        else:
            rec = df[~df['value'].isin(masked_values)].to_records(index=False)
        
        # ---- Prepare basic axe if not provided
        if ax is None:
                plt.rc('font', family='serif', size=10)
                fig, ax = plt.subplots(figsize=(10,8))

        # ---- Build a collection from rectangles patches and values
        collection = PathCollection(rec['patches'])

        # ----- Set values of each polygon
        arr = np.log10(rec['value']) if log else rec['value']
        collection.set_array(arr)

        # ---- Set default values limites
        vmin = arr.min() if vmin is None else vmin
        vmax = arr.max() if vmax is None else vmax
        collection.set_clim(vmin=vmin,vmax=vmax)

        # ---- Set default plot extension
        if extent is not None:
            ax.set_xlim(*extent[::2])
            ax.set_ylim(*extent[1::2])
        else:
            ax.set_xlim(min(xmin), max(xmax))
            ax.set_ylim(min(ymin), max(ymax))

        # ---- Add collection kwargs
        collection.set(**kwargs)

        # ---- Add collection object to main axe
        ax.add_collection(collection)

        # ---- Add color bar
        norm = plt.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(cmap=ax.collections[0].get_cmap(),
                                   norm=norm)
        label = f'log({self.field})' if log else self.field
        plt.colorbar(sm, ax=ax, label=label)

        # ---- Return axe
        return ax




    def zonal_stats(self, stats, polygons, layer=None, names = None, trans=None):
        """
        Perform statistics on zonal areas.

        Parameters:
        ----------
        stats (str/list) : statistics functions to perform.
                           Must be recognize by pandas.DataFrame.agg() method.
                           Example: stats = ['mean', 'min','max','median','count'].

        polygons (it) : iterable containing single part polygons.
                        Format: [ [(x0, y0),(x1,y1),(x2,y2)],.., [(x'N,y'N),..], ..]
                           <=>  [        polygon_0          ,..,    polygon_N      ].

        layer (int/it, optional) : layer id(s) on wich perform zonal statistics.
                                   If None, all layers are considered.
                                   Default is None.
        names (str/it, optional) : names of zones in the output.
                                   If None, generic names are created with field
                                   and zone number ('field_z0', 'field_z1',...).
                                   Default is None.
        trans (str/func, optional): function/function name to transform field values
                                    before applying statistics.
                                    If None, field values are not transformed.
                                    Default is None.

        Returns:
        --------
        zstats_df (DataFrame) : DataFrame with rows MultiIndex ('zone', 'layer').

        Examples:
        --------
        stats = ['mean','max','min','median','count']
        polygons = shp_utils.read_shapefile('mypolygons.shp')['coords']
        zdf = mf.zonal_stats(stats, polygons, layer=[4,5,6])

        """
        # ---- Manage multiple inputs
        _stats = marthe_utils.make_iterable(stats)
        if layer is None:
            _layer = np.unique(self.data['layer'])
        else:
            _layer = marthe_utils.make_iterable(layer)
        if names is None:
            _names = [f'{self.field}_z{i}' for i in range(len(polygons))]
        else:
            _names = marthe_utils.make_iterable(names)

        # ---- Check validity of transformation
        pest_utils.check_trans(trans)

        # ---- Get field vertices
        vx, vy = self.get_xyvertices()

        # ---- Iterate over all polygons
        dfs = []
        for p, name in zip(polygons, _names):
            # -- Mask vertices in polygon
            mask = shp_utils.point_in_polygon(vx, vy, p)
            # ---- Map layer to vertices coord
            rec = np.lib.recfunctions.stack_arrays(
                    [self.sample(vx[mask], vy[mask], layer=ilay, masked_values=self.dmv)
                            for ilay in _layer],
                                autoconvert=True, usemask=False)
            # -- Get data as DataFrame
            df = pd.DataFrame.from_records(rec).drop_duplicates()
            df['zone'] = name
            # -- Perform required stats on tranform value
            if trans is not None:
                df['value'] = df['value'].transform(trans)
            stats = df.groupby(['zone', 'layer'])['value'].agg(_stats)
            dfs.append(stats)
        # ---- Build zonal stats DataFrame
        zstats_df = pd.concat(dfs)
        # ---- Return
        return zstats_df





    def to_vtk(self, filename=None, trans='none', masked_values = dmv, **kwargs):
        """
        Build vtk unstructured grid from model geometry and 
        add current field to cell dataset.

        **kwargs correspond to the arguments of the MartheModel.get_vtk() method.
        
        Required python `vtk` package.

        Parameters:
        -----------
        filename (str, optional) : vtk file name to write without extension.
                                   Extension will be inferred.
                                   If None, filename = field.
                                   Default is None.
        trans (str, optional) : transformation to apply to the values.
                                See pymarthe.utils.pest_utils.transform.
                                Default is 'none'.
        masked_values (float/it, optional) : values to mask of the current field data.
                                             Default are [9999, 0, -9999].
        vertical_exageration (float, kwargs) : floating point value to scale vertical
                                               exageration of the vtk points.
                                               Default is 0.05.
        hws (str, kwargs) : hanging wall state, flag to define whatever the superior
                            hanging walls of the model are defined as normal layers
                            (explivitly) or not (implicitly).
                            Can be:
                                - 'implicit'
                                - 'explicit'
                            Default is 'implicit'.
        smooth (bool, kwargs) : boolean flag to enable interpolating vertex elevations
                                based on shared cell.
                                Default is False.
        binary (bool, kwargs) : Enable binary writing, otherwise classic ASCII format 
                                will be consider.
                                Default is True.
                                Note : binary is prefered as paraview can produced bug
                                       representing NaN values from ASCII (non xml) files.
        xml (bool, kwargs) : Enable xml based VTK files writing.
                             Default is False.
        shared_points (bool, kwargs) : Enable sharing points in grid polyhedron construction.
                                       Default is False.
                                       Note : False is prefered as paraview has a bug (8/4/2021)
                                              where some polyhedron will not properly close when
                                              using shared points.


        Returns:
        --------
        vtk (pymarthe.utils.vtk_utils.Vtk) : Vtk class containg unstructured vtk grid

        Example:
        --------
        mf = mm.prop['permh']
        myvtk = mf.to_vtk(filename = mf.field, trans='log10', vertical_exageration=0.02)
        """

        # -- Manage output file name
        f = self.field if filename is None else filename

        # -- Manage masked values
        mv = marthe_utils.make_iterable(masked_values)

        # -- Get Vtk instance
        vtk = self.mm.get_vtk(**kwargs)

        # -- Add field data to vtk
        vtk.add_array(self.data['value'],
                      name=self.field,
                      trans=trans,
                      masked_values=mv)

        # -- Write vtk file
        vtk.write(f)

        # -- Return Vtk instance
        return vtk




    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheField'












class MartheFieldSeries():
    """
    Wrapper Marthe --> python
    Manage series of MartheField instances.
    """
    def __init__(self, mm, field, simfile= None):
        """
        Time series of MartheField instances.

        Parameters
        -----------
        mm (MartheModel) : parent Marthe model.
                           Note: consider providing a parent MartheModel
                           instance ss far as possile.
        field (str) : property name.
                        Can be :
                            - 'charge'
                            - '%saturation'
                            - ...
        simfile (str, optional): simulated field file
                       Default is 'chasim.out'.


        Examples
        -----------
        mm = MartheField('mona.rma')
        mfs = MartheFieldSeries(mm, field = 'charge')
        """
        self.mm = mm
        self.field = field
        self.simfile = os.path.join(self.mm.mldir, 'chamsim.out') if simfile is None else simfile
        self.data = self._extract_fieldseries()




    def _extract_fieldseries(self):
        """
        Extract field data as a series of MartheField instance.

        Parameters:
        ----------

        Returns:
        --------
        mf_dic (dict) : dictionary of MartheField.
                        Format: { istep_0: MartheField_0,
                                  ...,
                                  istep_N: Marthefield_N}

        Examples:
        --------
        mf_dic = mfs._extract_fieldseries()

        """
        # ---- Read simfile
        print('Reading simulated fields ...')
        all_grids = marthe_utils.read_grid_file(self.simfile)

        # ---- Extract all data for given field (vectorized form for better performance)
        print(f'Collecting `{self.field}` field records data ...')
        isteps, arr =  np.column_stack(
                                [ [ np.tile(np.array(mg.istep), len(mg.array.ravel())),
                                    mg.array.ravel()] 
                                        for mg in all_grids
                                            if mg.field.casefold() == self.field.casefold()] )

        # ---- Rebuild MartheField instance for each provided istep
        # (changing value, field from imask then rebuild initial imask for better performance)
        print('Converting to MartheField instance ...')
        unique_isteps = set(isteps.astype(int))
        digits = len(str(len(unique_isteps)))
        mf_dic = {}
        for istep in unique_isteps:
            mf = deepcopy(self.mm.imask)
            mf.field = '{}_{}'.format(self.field, str(istep).zfill(digits))
            mf.data['value'] = arr[isteps==istep]
            mf_dic[istep] = mf

        # ---- Return field series as dictionary (format: {istep: MartheField()})
        return mf_dic





    def get_tseries(self, x, y, layer, names= None, index = 'date', masked_values = dmv[::2]):
        """
        Sample field data by x, y, layer coordinates and stack timeseries in a DataFrame.
        It will perform simple a spatial intersection with field data.

        Parameters:
        ----------
        x, y (float/iterable) : xy-coordinate(s) of the required point(s)

        layer (int/iterable) : layer id(s) to intersect data.

        names (str/list of str, optional) : additional names of each required point coordinates.
                                            If None, standard names are given according to the 
                                            coordinates. Example: '23i_45j_6k'
                                            Default is None.

        masked_values (None/list): values to ignore during the sampling process
                                   Default are [-9999, 0, 9999].

        index (str, optional) : type of index required for the output DataFrame.
                                Can be:
                                    - 'date': index is a pd.DatetimeIndex.
                                    - 'istep': index is a pd.index
                                    - 'both': index is a pd.MultiIndex(pd.index, pd.DatetimeIndex)
                                Default is 'date'.

        Returns:
        --------
        df (DataFrame): Output timeseries stack in DataFrame.


        Examples:
        --------
        x, y = [343., 385.3], [223., 217.2]
        layer = 0
        names = ['rec1', 'rec2']
        df = mfs.get_tseries(x,y,layer,names)
        """

        _x, _y, _layer = [marthe_utils.make_iterable(arg) for arg in [x,y,layer]]
        if (len(_layer) == 1) and (len(_x) > 1) :
            _layer = list(_layer) * len(_x)


        if names is not None:
            _names = marthe_utils.make_iterable(names)
            err_msg = 'ERROR : arguments `x`, `y` and `names` must have the same length. ' \
                      f'Given: len(x) = {len(_x)}, len(y) = {len(_y)}, len(names) = {len(_names)}.'

        # -- Fetch field value at xy-coordinates
        df = pd.DataFrame.from_records(
              [self.data[istep].sample(_x, _y, _layer)['value']
                    for istep in self.data.keys()]
                    )
        
        # -- Add column names
        if names is None:
            df.columns = [f'{ix}i_{iy}j_{il}k'
                            for ix,iy,il
                            in zip( *self.mm.get_ij(_x,_y), _layer)]
        else:
            df.columns = marthe_utils.make_iterable(names)

        # -- Convert basic index to MultiIndex
        df = df.set_index( pd.MultiIndex.from_tuples(
                                [(istep, self.mm.mldates[istep]) for istep in self.data.keys()],
                                names = ['istep', 'date'])
                          ).replace(
                        marthe_utils.make_iterable(masked_values),
                        np.nan       )

        # -- Return Multiindex DataFrame
        if index == 'date':
            return df.droplevel('istep')
        elif index == 'istep':
            return df.droplevel('date')
        else:
            return df





    def save_animation(self, filename,  dpf = 0.25, dpi=200, **kwargs):
        """
        Build a .gif animation from a series of field data.
        /!/ Package `imageio` required /!/

        Parameters:
        ----------
        filename (str) : required output file name.
                         Example: 'myfieldanimation.gif' 

        dpf (float) : duration per frame (in second).
                      Default is 0.25.

        dpi (int) : dots per inch (=image/plot resolution).
                    Default is 200. 

        **kwargs : MartheField.plot arguments.

        Returns:
        --------
        Save .gif animation in filename.

        Examples:
        --------
        mfs.save_animation('chargeout.gif', dpf = 0.2, dpi=200, layer=5, cmap='jet')

        """
        # ---- Try to import imageio package
        try:
            import imageio
        except ImportError:
            print('ERROR : Could not load `imageio` module. ' \
                  'Try `pip install imageio`.')
        # ---- Create temporal folder
        tdir = '_temp_'
        if os.path.exists(tdir): shutil.rmtree(tdir)
        os.mkdir(tdir)
        # -- Get min/max value to fix colorbar (vectorize form for better efficiency)
        mv = kwargs.get('masked_values', dmv[::2])
        values = np.array([ mf.data['value'][ ~np.isin(mf.data['value'], mv) ]
                                                    for mf in self.data.values()] )
        kwargs['vmin'] = kwargs.get('vmin', values.min())
        kwargs['vmax'] = kwargs.get('vmax', values.max())
        # -- Save animation
        print(f'Building animation of simulated `{self.field}`:')
        with imageio.get_writer(filename, mode='I', duration = dpf) as writer:
            # -- iterate over tiem step
            for i, istep in enumerate(self.data.keys()):
                # -- Plot MartheField
                ax = self.data[istep].plot(**kwargs)
                # -- Add time reference (top left)
                ilay = kwargs.get('layer', 0)
                text =  f'layer : {ilay}\n'     \
                        f'istep : {istep}\n'    \
                        f'date : {self.mm.mldates[istep]}'
                plt.text(0.01, 0.94, text,
                         fontsize = 7.5, transform=ax.transAxes)
                # -- Save plot as image
                digits = len(str(len(self.data)))
                png = os.path.join(tdir, '{}.png'.format(str(istep).zfill(digits)))
                ax.get_figure().savefig(png, dpi=dpi)
                # -- Read image
                image = imageio.imread(png)
                writer.append_data(image)
                # -- Plot progress bar
                marthe_utils.progress_bar((i+1)/len(self.data))
        # ---- Delete temporal folder
        shutil.rmtree(tdir)
        # -- Close all plots
        plt.close('all')
        # -- Success message
        print(f'\nAnimation written in {filename}.')





    def to_shapefile(self, filename = None, layer=0, inest=None, masked_values = dmv, log = False, epsg=None, prj=None):
        """
        Save field series in shapefile.

        Parameters:
        ----------
        filename (str, optional) : shapefile name to write.
                                   If None, filename = 'field_layer.shp'.
                                   Default is None.

        layer (int, optional) : layer numerical id to export.
                                Note : a unique layer id is allowed
                                Default is 0.
        inest (int, optional) : nested grid numerical id to export.
                                If None, all nested grid are considered.
                                Default is None.
        masked_values (list, optional) : field values to ignore.
                                         Default is [-9999., 0., 9999].
        log (bool, optional) : logarithmic transformation of all values.
                               Default is False.
        epsg (int, optional) : Geodetic Parameter Dataset.
                               Default is None.
        prj (str, optional) : cartographic projection and coordinates
                              Default is None.

        Returns:
        --------
        Save vectorial geometries and records in filename.

        Examples:
        --------
        mfs.to_shapefile(layer=3)
        """
        
        # -- Build filename if not provided
        if filename is None:
            filename = f'{self.field}_{layer}.shp'

        # -- Get first istep field
        mf0 = list(self.data.values())[0]

        # ---- Perform a bunch of assertions on `layer` and `inest` arguments
        err_msg = f"`layer` must be an integer between 0 and {mf0.maxlayer -1}."
        assert isinstance(layer, int), err_msg
        assert 0 <= layer < mf0.maxlayer, err_msg

        # ---- Fetch pyshp parts (polygons) for istep field
        parts = []
        for mg in mf0.to_grids(layer=layer, inest=inest):
            parts.extend(mg.to_pyshp())

        # ---- Prepare masked_value deletion
        mv = [] if masked_values is None else masked_values

        # ---- Fetch data for all isteps
        dfs = []
        for istep, mf in self.data.items():
            data = mf.get_data(layer=layer, inest=inest)
            df = pd.DataFrame.from_records(data).assign(parts=parts)
            # -- Manage masked vales
            df  = df[~df['value'].isin(mv)]
            # -- Change column name according to istep
            df.rename(columns={'value': mf.field}, inplace=True)
            dfs.append(df)

        # -- Concatenate drop duplicated layer,inest,i,j,x,y columns
        _df = pd.concat(dfs, axis=1)
        df = _df.loc[:,~_df.columns.duplicated()]

        # ---- Fetch subset parts (goemetries)
        parts = df.pop('parts')

        # ---- Log transform if required
        if log:
            mask = df.columns.str.startswith(self.field)
            df.loc[:,mask] = df.loc[:,mask].apply(
                                lambda col: col.transform('log10'))
            df =  df.loc[:,mask].add_prefix('log_')

        # ---- Convert reccaray to shafile
        shp_utils.recarray2shp(df.to_records(index=False), np.array(parts),
                               shpname=filename, epsg=epsg, prj=prj)






    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheFieldSeries'









