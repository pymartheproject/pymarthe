"""
Contains the MartheField class
Designed for handling distributed Marthe properties
(structured and unstructured grid)
"""

import os, sys
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

from .utils import marthe_utils, shp_utils
from .utils.grid_utils import MartheGrid

encoding = 'latin-1'


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



    def get_data(self, layer=None, inest=None,  as_array=False, as_mask=False):
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
            layers = layer if marthe_utils.isiterable(layer) else [layer]
        # ---- Manage inest input
        if inest is None:
            inests = np.unique(self.data['inest'])
        else: 
            inests = inest if marthe_utils.isiterable(inest) else [inest]
        # ---- Transform records to Dataframe for query purpose
        df = pd.DataFrame(self.data)
        # ---- Get mask
        mask = (df['layer'].isin(layers)) & (df['inest'].isin(inests))
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
            err_msg = f"ERROR: `data` must have a size equal to {self.data.size}. " \
                      f"Given: {data.size}."
            self.data = data

        # ---- Manage 3D-array input
        if np.logical_and.reduce([_arr, not _rec, _none]):
            # -- Verify data is a 3D-array
            err_msg = f"ERROR: `data` array must be 3D. Given shape: {data.shape}."
            assert len(data.shape) == 3, err_msg
            self.data = self._3d2rec(data)




    def as_array(self):
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
        arr3d = mf.as_array()

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
        recs = tuple([mg.to_records() for mg in marthe_utils.read_grid_file(filename)])
        # ---- Stack all recarrays as once
        return np.concatenate(recs)


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
        df = pd.DataFrame(rec)
        
        # ---- Modify rec inplace
        for layer, arr2d in enumerate(arr3d):
            mask = self.get_data(layer=layer, inest=0, as_mask=True)
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
        args = [layer, inest, nrow, ncol, xl, yl, dx, dy, xcc, ycc, array]
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



    # *** UNDER DEVELOPMENT ***

    # def to_pyshp(self, layer=None, inest=None):

    #     # ---- Fetch pyshp parts (polygons)
    #     parts = []
    #     for mg in self.to_grids(layer=layer, inest=inest):
    #         parts.extend(mg.to_polygons())

    #     # ---- fetch data
    #     rec = self.get_data(layer=layer, inest=inest)

    #     # ---- Return parts and records
    #     return parts, rec



    def plot(self, ax=None, layer=0, inest=None, vmin=None, vmax=None, log = True, masked_values = [-9999., 0.], **kwargs):
        """

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
        df = pd.DataFrame(data).assign(patches=patches)
        
        # ---- Apply mask values
        if masked_values is None:
            rec = df.to_records(index=False)
        else:
            rec = df[~df['value'].isin(masked_values)].to_records(index=False)
        
        # ---- Prepare basic axe if not provided
        if ax is None:
                plt.rc('font', family='serif', size=10)
                fig, ax = plt.subplots(figsize=(8,8))

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

