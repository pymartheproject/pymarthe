"""
Contains the MartheSoil class
Designed for Marthe model soil properties management.
"""


import os, sys
import numpy as np
import pandas as pd 


from pymarthe.mfield import MartheField
from .utils import marthe_utils, pest_utils




class MartheSoil():
    """
    Wrapper Marthe --> python.
    Interface to handle soil properties.
    """

    def __init__(self, mm, martfile=None):
        """
        Read soil properties in .mart file.

        Parameters:
        ----------
        mm (MartheModel) : related Marthe model.

        martfile (str, optional): .mart file 
                                  Default is None.

        Returns:
        --------
        ms (MartheSoil) : class to handle soil property

        Examples:
        --------
        mm = MartheModel('mona.rma')
        ms = MartheSoil(mm)

        """
        self.mm = mm
        self.prop_name = 'soil'
        # ---- Read existing soil properties
        file = self.mm.mlfiles['mart'] if martfile is None else martfile
        self.data = marthe_utils.read_zonsoil_prop(file)
        self.soilprops = self.data['soilprop'].unique()
        # ---- Soil zone numbers as a field
        self.zonep = MartheField('zonep', self.mm.mlfiles['zonep'], self.mm)
        self.zones = np.unique(self.zonep.data['value'])
        # ---- Number of soil properties and zones
        self.nzone = len(np.unique(self.zonep.data['value']))
        self.nsoilprop = len(self.soilprops)
        # ---- Set property style
        self._proptype = 'list'




    def get_data(self, soilprop=None, zone=None, as_style = 'list-like', **kwargs):
        """
        Get soil property field as recarray.
        Wrapper to MartheField.get_data()

        Parameters:
        ----------
        soilprop (str/it) : soil property name.
                         Can be cap_sol_progr, equ_ruis_perc,
                                t_demi_percol, ...
                         If None, all soil properties in .mart file
                         are consider.
                         Default is None
        zone (int/it) : soil zone id(s).
                        If None, all soil zone in .zonep file 
                        are condider. 
        as_style (str) : required output type.
                         Can be 'list-like' or 'array-like'.
                         If 'list-like' return a subset of MartheSoil data.
                         If 'array-like' return a subset recarray (cell-by-cell)
                         Default is 'list-like'.

        **kwargs : arguments of MartheField.get_data() method.
                   Can be layer, inest, as_array, as_mask.


        Returns:
        --------
        df (DataFrame) : soil data by soilprop/zone
        rec (recarray) : soil data for each model cell.

        Examples:
        --------
        ms = MartheSoil(mm)
        rec_l0_i2 = ms.get_data('cap_sol_progr', as_style='array_like', layer=0, inest=2)

        """
        # ---- Manage required input
        _sp = self.soilprops if soilprop is None else marthe_utils.make_iterable(soilprop)
        _zon = self.zones if zone is None else marthe_utils.make_iterable(zone)
        # ---- Subset soil data
        sdf = self.data.query("soilprop in @_sp & zone in @_zon")

        # ---- Return according to required style
        if as_style == 'list-like':
            return sdf
    
        elif as_style == 'array-like':
            err_msg = 'ERROR : getter process with `array-like` style ' \
                       'does not support multiple soil properties. ' \
                       'Given : {}.'.format(', '.join(_sp))
            assert len(_sp) <= 1, err_msg
            # -- Fetch array-like data
            rec = self.zonep.get_data(**kwargs)
            # -- Delete not required zones
            df = pd.DataFrame.from_records(rec).query("value in @_zon")
            # ---- Replace zone values by their property value
            repl_dic = dict(sdf[['zone', 'value']].to_numpy())
            rec =df.replace({'value': repl_dic}).to_records(index=False)
            return rec




    def sample(self, soilprop, x, y):
        """
        Get soil property at specific xy-location.
        Wrapper to MartheField.sample().
        Note: the sample data will be performed 
              on first layer only

        Parameters:
        ----------
        soilprop (str) : soil property name.
                         Can be cap_sol_progr, equ_ruis_perc,
                                t_demi_percol, ...

        x, y (float/iterable) : xy-coordinate(s) of the required point(s)

        Returns:
        --------
        rec (np.recarray) : soil property data at xy-point(s)

        Examples:
        --------
        ms = MartheSoil(mm)
        rec = ms.sample('t_demi_percol', x=456.32, y=567.1)

        """
        # ---- Build MartheField instance from recarray
        rec = self.get_data(soilprop, as_style='array-like')
        mf = MartheField(soilprop, rec, self.mm)
        # ---- Sample points
        rec = mf.sample(x, y, layer=0)
        # ---- Return recarray
        return rec



    def plot(self, soilprop, **kwargs):
        """
        Plot soil property.
        Wrapper to MartheField.plot().

        Parameters:
        ----------
        soilprop (str) : soil property name.
                         Can be cap_sol_progr, equ_ruis_perc,
                                t_demi_percol, ...

        **kwargs : arguments of MartheField.plot() method.
                   Can be ax, layer, inest, vmin, vmax, log, 
                   masked_values, matplotlib.PathCollection arguments.

        Returns:
        --------
        ax (matplotlib.axes) : standard ax with 2 collections:
                                    - 1 for rectangles
                                    - 1 for colorbar

        Examples:
        --------
        ms = MartheSoil(mm)
        ax = ms.plot('cal_sol_progr', cmap = 'Paired')
        """
        rec = self.get_data(soilprop, as_style='array-like')
        mf = MartheField(soilprop, rec, self.mm)
        ax = mf.plot(**kwargs)
        return ax



    def to_shapefile(self, soilprop, filename, **kwargs):
        """
        Export soil property to shapefile
        Wrapper to MartheField.to_shapefile().

        Parameters:
        ----------
        soilprop (str) : soil property name.
                         Can be cap_sol_progr, equ_ruis_perc,
                                t_demi_percol, def_sol_progr,
                                 rumax, defic_sol, ... (GARDENIA)

        **kwargs : arguments of MartheField.to_shapefile() method.
                   Can be layer, inest, masked_values, log, epsg, prj

        Returns:
        --------
        Write shape in filename.

        Examples:
        --------
        filename = os.path.join('gis', 'cap_sol_progr.shp')
        ms.to_shapefile('cap_sol_progr', filename)
        """
        rec = self.get_data(soilprop, as_style='array-like')
        mf = MartheField(soilprop, rec, self.mm)
        mf.to_shapefile(filename= filename, **kwargs)



    def set_data_from_parfile(self, parfile, keys, value_col, btrans):
        """
        """
        # -- Get all data
        df = self.data.copy(deep=True)
        # -- Get kmi and transformed values
        kmi, bvalues = pest_utils.parse_mlp_parfile(parfile, keys, value_col, btrans)
        # -- Convert to MultiIndex Dataframe
        mi_df = df.set_index(keys)
        # -- Set values and back to single index
        mi_df.loc[kmi, value_col] = bvalues.values
        data = mi_df.reset_index()
        # -- Set data inplace
        self.data = data




    def set_data(self, soilprop, value, zone=None):
        """
        Set soil property value(s).

        Parameters:
        ----------
        soilprop (str) : soil property name.
                         Can be cap_sol_progr, equ_ruis_perc,
                                t_demi_percol, ...

        value (int/float) : value to set.

        zone (int/iterable, optional) : required zone to set value.
                                        If None, all soil zones are considered.
                                        Default is None.

        Returns:
        --------
        Set value inplace for required soil property/zone.

        Examples:
        --------
        ms.set_data('cap_sol_progr', 34.6, zone = [2, 8, 11])

        """
        # ---- Manage zones to set value
        zones = self.zones if zone is None else marthe_utils.make_iterable(zone)
        # ---- Mask soil Dataframe with required soil property and zone
        mask = (self.data.soilprop.str.contains(soilprop)) & (self.data.zone.isin(zones))
        # ---- Set provided value inplace
        self.data.loc[mask,'value'] = value




    def write_data(self, soilprop=None, martfile=None):
        """
        Write soil properties in .mart file (initialization block).
        Wrapper to marthe_utils.write_zonsoil_prop().

        Parameters:
        ----------
        soilprop (str, optional) : soil property name.
                                   Can be cap_sol_progr, equ_ruis_perc,
                                   t_demi_percol, ...
                                   If None, all soil properties are considered.
                                   Default is None.

        martfile (str, optional) : path to .mart file to write soil properties.
                                   If None, the MartheModel .mart file is considered.
                                   Default is None.

        Returns:
        --------
        Write soil properties in martfile.

        Examples:
        --------
        ms.set_data('cap_sol_progr', 34.6, zone = [2, 8, 11])
        ms.write_data('cap_sol_progr')

        """
        # ---- Manage soil properties to write
        soilprops = self.soilprops if soilprop is None else marthe_utils.make_iterable(soilprop)
        # ---- Output .mart file to write in
        file = self.mm.mlfiles['mart'] if martfile is None else martfile
        # ---- Write required soil properties
        soil_df = self.data.query(f"soilprop in @soilprops")
        marthe_utils.write_zonsoil_prop(soil_df, file)



    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheSoil'

