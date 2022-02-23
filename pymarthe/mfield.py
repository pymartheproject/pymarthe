"""
Contains the MartheField class
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


from .utils import marthe_utils, shp_utils
from .utils.grid_utils import MartheGrid

encoding = 'latin-1'
dmv = [-9999., 0., 9999] # Default masked values

class MartheField():
    """
    Wrapper Marthe --> python
    """
    def __init__(self, field, data, mm=None):
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
        data (object): field data to read/set.
                       Can be:
                        - Marthe property file ('mymodel.permh')
                        - Recarray with layer, inest,... informations
                        - 3D-array
        mm (MartheModel) : parent Marthe model.
                           Note: consider providing a parent MartheModel
                           instance ss far as possile.

        Examples
        -----------
        mm.mobs.compute_weights(lambda_dic)
        """
        self.field = field
        self.mm = mm
        self.set_data(data)
        self.maxlayer = len(self.to_grids(inest=0))
        self.maxnest = len(self.to_grids(layer=0)) - 1 # inest = 0 is the main grid
        self.dmv = dmv 
        # ---- Set property style
        self._proptype = 'array'




    def sample(self, x, y, layer, masked_values=None):
        """
        Sample field data by x, y, layer coordinates.
        It will perform simple a spatial intersection with field data.

        Parameters:
        ----------
        x, y (float/iterable) : xy-coordinate(s) of the required point(s)
        layer (int/iterable) : layer id(s) to intersect data.
        masked_values (None/list): values to ignore during the sampling process 

        Returns:
        --------
        rec (np.recarray) : data intersected on point(s)

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
            self.mm.build_spatial_idx()

        # -- Manage masked values
        nd = [] if masked_values is None else marthe_utils.make_iterable(masked_values)

        # ---- Perform intersection on spatial index
        dfs = []
        for ix, iy, ilay in zip(_x, _y, _layer):
            # -- Sorted output index   
            idx = sorted(self.mm.spatial_index.intersection((ix,iy)))
            # -- Subset by layer for value != 0 or 9999
            q1 = f'layer=={ilay} & value not in @nd'
            df = pd.DataFrame(self.data[idx]).query(q1)
            dfs.append(df)
        # ---- Return intersection as recarray
        return pd.concat(dfs).to_records(index=False)




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
        # ---- Transform records to Dataframe for query purpose
        df = pd.DataFrame.from_records(self.data)
        # ---- Get mask
        mask = (df['layer'].isin(layers)) & (df['inest'].isin(inests)) & (~df['value'].isin(mv))
        # ---- Return as mask if required
        if as_mask:
            return mask
        # ---- Return as array
        if as_array:
            arrays = []
            # -- Subset by layer(s)
            for l in layers:
                df_s = df.loc[df['layer'] == l]
                # ---- Subset by nested
                for n in inests:
                    df_ss = df_s.loc[df_s['inest'] == n]
                    # -- Fetch nrow, ncol of the current grid
                    nrow = df_ss['i'].max() + 1
                    ncol = df_ss['j'].max() + 1
                    # -- Rebuild array by reshaping with nrow, ncol
                    arrays.append(df_ss['value'].to_numpy().reshape(nrow,ncol))
            # -- Returning array
            return np.array(arrays)
        # -- Returning as recarray
        else:
            return df.loc[mask].to_records(index=False)





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

        # ---- Manage Marthe filename input
        if np.logical_and.reduce([_str, _none]):
            self.data =  self._grids2rec(data)

        # ---- Manage numeric input
        if _num:
            mask = self.get_data(layer=layer, inest=inest, as_mask=True)
            df = pd.DataFrame(self.data)
            df.loc[mask, 'value'] = data
            self.data = df.to_records(index=False)

        # ---- Manage recarray input
        if _rec:
            self.data = data

        # ---- Manage 3D-array input
        if np.logical_and.reduce([_arr, not _rec, _none]):
            # -- Verify data is a 3D-array
            err_msg = f"ERROR: `data` array must be 3D. Given shape: {data.shape}."
            assert len(data.shape) == 3, err_msg
            self.data = self._3d2rec(data)




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



    def _grids2rec(self, filename):
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
        dfs = [pd.DataFrame(mg.to_records()) for mg in marthe_utils.read_grid_file(filename)]
        # ---- Stack all recarrays as once
        return pd.concat(dfs).to_records(index=False)


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
        Wrapper to marthe_utils.write_grid_file().

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
        if filename is None:
            path = os.path.join(self.mm.mldir, self.mm.mlname)
            extension = self.field
            f = '{}.{}'.format(path, extension)
        else:
            f = filename
        # ---- Write field data from list of MartheGrid instance
        marthe_utils.write_grid_file(f, self.to_grids(),
                                        self.maxlayer,
                                        self.maxnest )



    def to_shapefile(self, filename = None, layer=0, inest=None, masked_values = dmv, log = False, epsg=None, prj=None):
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



    def plot(self, ax=None, layer=0, inest=None, vmin=None, vmax=None, log = False, masked_values = dmv, **kwargs):
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
    def __init__(self, mm, field, outfile= None):
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
        outfile (str, optional): simulated field file
                       Default is 'chasim.out'.


        Examples
        -----------
        mm = MartheField('mona.rma')
        mfs = MartheFieldSeries(mm, field = 'charge')
        """
        self.mm = mm
        self.field = field
        self.outfile = os.path.join(self.mm.mldir, 'chamsim.out') if outfile is None else outfile
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
        # ---- Read outfile
        print('Reading simulated fields ...')
        all_grids = marthe_utils.read_grid_file(self.outfile)

        # ---- Extract all data for given field
        print(f'Collecting `{self.field}` field records data ...')
        _isteps, _arrs =  np.column_stack(
                                [[mg.istep, mg.array.ravel()] 
                                    for mg in all_grids
                                        if mg.field.casefold() == self.field.casefold()])
        arr =  np.concatenate(_arrs)
        n = int(len(arr)/len(_isteps))
        isteps = np.concatenate(
                            [np.tile(np.array(istep), n)
                                    for istep in _isteps]
                                        )

        # ---- Rebuild MartheField instance for each provided istep
        print('Converting to MartheField instance ...')
        mf_dic = {}
        for istep in set(_isteps):
            mf = deepcopy(self.mm.imask)
            mf.field = self.field
            mf.data['value'] = arr[isteps==istep]
            mf_dic[istep] = mf

        # ---- Return field series as dictionary (format: {istep: MartheField()})
        return mf_dic





    def get_timeseries(self, x, y, layer, names= None, index = 'date', masked_values = dmv[::2]):
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
        df = mfs.get_timeseries(x,y,layer,names)
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
        x, y = [343., 385.3], [223., 217.2]
        layer = 0
        names = ['rec1', 'rec2']
        df = mfs.get_timeseries(x,y,layer,names)
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
        # -- Save animation 
        with imageio.get_writer(filename, mode='I', duration = dpf) as writer:
            # -- iterate over tiem step
            for istep, mf in self.data.items():
                # -- Plot MartheField
                ax = mf.plot(**kwargs)
                # -- Add time reference (top left)
                plt.text(0.01, 0.95,
                         f'istep : {istep}\ndate : {self.mm.mldates[istep]}',
                         fontsize = 8, transform=ax.transAxes)
                # -- Save plot as image
                digits = len(str(len(self.data)))
                png = os.path.join(tdir, '{}.png'.format(str(istep).zfill(digits)))
                ax.get_figure().savefig(png, dpi=dpi)
                # -- Read image
                image = imageio.imread(png)
                writer.append_data(image)
        # ---- Delete temporal folder
        shutil.rmtree(tdir)
        # -- Close all plots
        plt.close('all')
        # -- Success message
        print(f'{self.field} animation written in {filename}.')





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
            digits = len(str(len(self.data)))
            colname = '{}_{}'.format(self.field, str(istep).zfill(digits))
            df.rename(columns={'value': colname}, inplace=True)
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









