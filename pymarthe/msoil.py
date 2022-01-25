"""
Contains the MartheSoil class
Designed for Marthe model soil properties management.
"""


import os, sys
import numpy as np
import pandas as pd 


from pymarthe.mfield import MartheField
from .utils import marthe_utils




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
        # ---- Read existing soil properties
        file = self.mm.mlfiles['mart'] if martfile is None else martfile
        self.soil_df = marthe_utils.read_zonsoil_prop(file)
        self.soilprops = self.soil_df['soilprop'].unique()
        # ---- Soil zone numbers as a field
        self.zonep = MartheField('zonep', self.mm.mlfiles['zonep'], mm)
        # ---- Number of soil properties and zones
        self.nzone = len(np.unique(self.zonep.data['value']))
        self.nsoilprop = len(self.soilprops)




    def get_data(self, soilprop, **kwargs):
        """
        Get soil property field as recarray.
        Wrapper to MartheField.get_data()

        Parameters:
        ----------
        soilprop (str) : soil property name.
                         Can be cap_sol_progr, equ_ruis_perc,
                                t_demi_percol, ...

        **kwargs : arguments of MartheField.get_data() method.
                   Can be layer, inest, as_array, as_mask.


        Returns:
        --------
        rec (recarray) : soil data for each model cell.

        Examples:
        --------
        ms = MartheSoil(mm)
        rec_l0_i2 = ms.get_data('cap_sol_progr', layer=0, inest=2)

        """
        # ---- Subset by required soil property
        soil_df_ss = self.soil_df.query(f"soilprop == '{soilprop}'")
        # ---- Subset field with provided kwargs
        zonep_ss = self.zonep.get_data(**kwargs)
        # ---- Replace zone number by soil property value
        repl_dic = dict(soil_df_ss[['zone', 'value']].to_numpy())
        df = pd.DataFrame.from_records(
                            zonep_ss).replace(
                                        {'value': repl_dic})
        # ---- Return data as recarray
        return df.to_records(index=False)



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
        mf = MartheField(soilprop, self.get_data(soilprop), self.mm)
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
        mf = MartheField(soilprop, self.get_data(soilprop), self.mm)
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
        mf = MartheField(soilprop, self.get_data(soilprop), self.mm)
        mf.to_shapefile(filename= filename, **kwargs)




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
        if zone is None:
            zones = self.soil_df.loc[self.soil_df.soilprop == soilprop, 'zone']
        else:
            zones = marthe_utils.make_iterable(zone)
        # ---- Mask soil Dataframe with required soil property and zone
        mask = (self.soil_df.soilprop.str.contains(soilprop)) & (self.soil_df.zone.isin(zones))
        # ---- Set provided value inplace
        self.soil_df.loc[mask,'value'] = value



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
        if soilprop is None:
            soilprops = self.soilprops
        else:
            soilprops = marthe_utils.make_iterable(soilprop)
        # ---- Output .mart file to write in
        file = self.mm.mlfiles['mart'] if martfile is None else martfile
        # ---- Write required soil properties
        soil_df = self.soil_df.query(f"soilprop in @soilprops")
        marthe_utils.write_zonsoil_prop(soil_df, file)


