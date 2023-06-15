"""
Contains the MartheSoil class
Designed for Marthe model soil properties management.
"""


import os, sys
import numpy as np
import pandas as pd
import re
from copy import deepcopy

from pymarthe.mfield import MartheField
from .utils import marthe_utils, pest_utils

encoding = 'latin-1'


class MartheSoil():
    """
    Wrapper Marthe --> python.
    Interface to handle soil properties.
    """

    def __init__(self, mm, martfile=None, pastpfile=None):
        """
        Read soil properties in whatevever the .mart or .patsp file.
        Note that the mode of soil properties implementation will be
        infered and stock in .mode attribut.
        It can be:
            - 'mart-c' : soil properties are constant and stock in .mart
            - 'pastp-c' : soil properties are constant and stock in .pastp
            - 'pastp-t' = soil properties are transient and stock in .mart


        Parameters:
        ----------
        mm (MartheModel) : related Marthe model.

        martfile (str, optional): .mart Marthe file
                                  If None, it will take the .mart file 
                                  related to the model.
                                  Default is None.

        pastpfile (str, optional): .pastp Marthe file
                                   If None, it will take the .pastp file 
                                   related to the model.
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
        self.martfile = self.mm.mlfiles['mart'] if martfile is None else martfile
        self.pastpfile = self.mm.mlfiles['pastp'] if pastpfile is None else pastpfile
        self.mode, self.data = marthe_utils.read_zonsoil_prop(self.martfile, self.pastpfile)
        self.isteps = np.arange(self.mm.nstep)
        # ---- Soil zone numbers as independant field
        self.zonep = MartheField('zonep', self.mm.mlfiles['zonep'], self.mm, use_imask=False)
        # ---- Set property style
        self._proptype = 'list'



    @property
    def soilprops(self):
        """
        Get array of unique soil property names
        """
        return self.data['soilprop'].unique()


    @property
    def zones(self):
        """
        Get array of unique soil property zone ids
        """
        return self.data['zone'].unique()


    @property
    def nsoilprop(self):
        """
        Get number of defined soil properties
        """
        return len(self.soilprops)
    

    @property
    def nzone(self):
        """
        Get number of defined zone ids
        """
        return len(self.zones)



    def get_data(self, soilprop=None, istep = None, zone=None, force=False, as_style = 'list-like', **kwargs):
        """
        Get soil property field as recarray.
        Wrapper to MartheField.get_data()

        Parameters:
        ----------
        soilprop (str/it, optional) : soil property name.
                                      Can be cap_sol_progr, equ_ruis_perc,
                                      t_demi_percol, ...
                                      If None, all soil properties in .mart file
                                      are consider.
                                      Default is None.
        istep (int/it, optional) : required time steps.
                                   If None, all available time steps are considered.
                                   Default is None. 
        zone (int/it, optional) : soil zone id(s).
                                  If None, all soil zone in .zonep file 
                                  are condider.
        force (bool, optional) : force getting soil property data for all required timesteps
                                 even if there are not provided explicitly in Marthe.
                                 For a not provided required time step the nearest previous
                                 istep (npi) containing soil data will be considered.
                                 Note: can be slow if the model contains a lot of timesteps.
                                 Default is False.
        as_style (str, optional) : required output type.
                                   Can be 'list-like' or 'array-like'.
                                   If 'list-like' return a subset of MartheSoil data.
                                   If 'array-like' return a subset recarray (cell-by-cell)
                                   Default is 'list-like'.

        **kwargs : arguments of MartheField.get_data() method.
                   Can be layer, inest.


        Returns:
        --------
        df (DataFrame) : soil data by soilprop/zone
        rec (recarray) : soil data for each model cell.

        Examples:
        --------
        ms = MartheSoil(mm)
        rec_i2 = ms.get_data('cap_sol_progr', as_style='array_like', inest=2)

        """
        # ---- Manage required input
        _sp = self.soilprops if soilprop is None else marthe_utils.make_iterable(soilprop)
        _zon = self.zones if zone is None else marthe_utils.make_iterable(zone)
        _istep = self.isteps if istep is None else marthe_utils.make_iterable(istep)

        # ---- Subset soil data by required soil property name and zone
        q = 'soilprop in @_sp & zone in @_zon'
        sdf = self.data.query(q)
        
        # ---- Return according to required style
        if as_style == 'list-like':

            # ---- Force all provided isteps 
            if force:
                dfs = []
                for istep in _istep: 
                    df = sdf.loc[sdf.istep == istep]
                    # -- If istep not provided in Marthe Model
                    if df.empty:
                        nip = sdf.loc[sdf.istep < istep, 'istep'].max()   # nip = nearest previous istep
                        np_df = sdf[sdf.istep == nip]
                        np_df['istep'] = istep
                        dfs.append(np_df)
                    else:
                        dfs.append(df)

                return pd.concat(dfs, ignore_index=True)

            else:
                # -- Return only provided and required timesteps
                return sdf.query('istep in @_istep')

        elif as_style == 'array-like':

            err_msg = 'ERROR : gettind data with `array-like` style ' \
                       'does not support multiple soil properties. ' \
                       'Given : {}.'.format(', '.join(_sp))
            assert len(_sp) <= 1, err_msg
            # -- Fetch array-like data
            rec = self.zonep.get_data(**kwargs)
            # -- Delete not required zones
            df = pd.DataFrame.from_records(rec).query("value in @_zon")
            # ---- Replace zone values by their property value
            repl_dic = dict(sdf[['zone', 'value']].to_numpy())
            rec = df.replace({'value': repl_dic}).to_records(index=False)
            return rec




    def sample(self, soilprop, x, y, istep=0):
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

        istep (int, optional) : required time step.
                                Default is 0.

        Returns:
        --------
        rec (np.recarray) : soil property data at xy-point(s)

        Examples:
        --------
        ms = MartheSoil(mm)
        rec = ms.sample('t_demi_percol', x=456.32, y=567.1)

        """
        # ---- Support unique istep
        err_msg = "ERROR : `.sample()` method does not support multiple `istep`. " \
                  f"Given : {istep}."
        assert not marthe_utils.isiterable(istep), err_msg
        # ---- Build MartheField instance from recarray
        rec = self.get_data(soilprop, istep=istep, force=True, as_style='array-like')
        mf = MartheField(soilprop, rec, self.mm)
        # ---- Sample points
        rec = mf.sample(x, y, layer=0)
        # ---- Return recarray
        return rec



    def plot(self, soilprop, istep=0, **kwargs):
        """
        Plot soil property.
        Wrapper to MartheField.plot().

        Parameters:
        ----------
        soilprop (str) : soil property name.
                         Can be cap_sol_progr, equ_ruis_perc,
                         t_demi_percol, ...

        istep (int, optional) : required time step.
                                Default is 0.

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
        # ---- Support unique istep
        err_msg = "ERROR : `.plot()` method does not support multiple `istep`. " \
                  f"Given : {istep}."
        assert not marthe_utils.isiterable(istep), err_msg
        # ---- Getting required data
        rec = self.get_data(soilprop, istep=istep, force=True, as_style='array-like')
        mf = MartheField(f'{soilprop}_{istep}', rec, self.mm, use_imask=False)
        ax = mf.plot(**kwargs)
        return ax



    def to_shapefile(self, soilprop, filename, istep=0, **kwargs):
        """
        Export soil property to shapefile
        Wrapper to MartheField.to_shapefile().

        Parameters:
        ----------
        soilprop (str) : soil property name.
                         Can be cap_sol_progr, equ_ruis_perc,
                                t_demi_percol, def_sol_progr,
                                 rumax, defic_sol, ... (GARDENIA)

        istep (int, optional) : required time step.
                                Default is 0.

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
        # ---- Support unique istep
        err_msg = "ERROR : `.to_shapefile()` method does not support multiple `istep`. " \
                  f"Given : {istep}."
        assert not marthe_utils.isiterable(istep), err_msg
        rec = self.get_data(soilprop, istep=istep, force=True, as_style='array-like')
        mf = MartheField(f'{soilprop}_{istep}', rec, self.mm)
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




    def set_data(self, soilprop, value, istep=None, zone=None):
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
        _sp = self.soilprops if soilprop is None else marthe_utils.make_iterable(soilprop)
        _zon = self.zones if zone is None else marthe_utils.make_iterable(zone)
        _istep = self.isteps if istep is None else marthe_utils.make_iterable(istep)
        # ---- Mask soil Dataframe with required soil property and zone
        mask = np.logical_and.reduce(
                   [ self.data.soilprop.isin(_sp),
                     self.data.zone.isin(_zon),
                     self.data.istep.isin(_istep) ]
                     )
        # ---- Set provided value inplace
        self.data.loc[mask,'value'] = value





    def write_data(self, filename=None):
        """
        Write soil current soil property data.


        Parameters:
        ----------
        filename (str, optional) : path to the outfile to write.
                                   By default, the .mart or .pastp
                                   (according to the implementation mode)
                                   will be overwrited.


        Returns:
        --------
        Write soil data inplace.

        Examples:
        --------
        ms.set_data('cap_sol_progr', 34.6, zone = [2, 8, 11])
        fn = f"file.{ms.mode.split('-')[0]}"
        ms.write_data(fn)
        """

        # ---- Set global usefull regex
        re_num = r"\s*[-+]?\d*\.?\d+|\d+"
        from_istep = r"\*{3}\s*Le pas|Début"

        # ---- Write data in .pastp file
        if 'pastp' in self.mode:
            # ---- Fetch .pastp file content by lines
            with open(self.pastpfile, 'r', encoding=encoding) as f:
                pastp_content = f.read()
            # ---- Extract indices when a new time tsep begin
            idx = [m.start(0) for m in re.finditer(from_istep, pastp_content)]
            # ---- Extract soil data by time step
            re_block = r";\s*\*{3}\n(.*?)/\*{5}"
            blocks = re.findall(re_block, pastp_content, re.DOTALL)
            # ---- Iterate for every istep
            for istep, block in enumerate(blocks):
                # ---- Check if there is available soil data for this istep
                if istep in self.data.istep:
                    # ---- Get available value to replace
                    df = self.data.loc[self.data['istep'] == istep]
                    for d in df.itertuples():
                        # ---- Regex to match in current block
                        re_match = r"\/{0}\/ZONE_SOL\s*Z=\s*{1}V=\s*(.+);".format(
                                                                        d.soilprop.upper(),
                                                                        d.zone)
                        # ---- Match regex in block
                        match = re.search(re_match, block)
                        # ---- Get line with new value
                        repl = re.sub(match.group(1), str(d.value), match.group(0))
                        # --- Update block with modified value line
                        block = re.sub(match.group(0), repl, block)
                    # ---- Change block in pastp file by new block with modified value
                    until_istep  = pastp_content[:idx[istep]]
                    from_istep = pastp_content[idx[istep]:]
                    new_from_istep = from_istep.replace(blocks[istep], block) # re.sub(blocks[istep], block, from_istep) re.sub() not working here
                    pastp_content = until_istep + new_from_istep
            # ---- Write new content
            out = self.pastpfile if filename is None else filename
            with open(out, 'w', encoding=encoding) as f:
                f.write(pastp_content)


        # ---- Write data in .mart file
        elif 'mart' in self.mode:

            # ---- Fetch actual .mart file content as text
            with open(self.martfile, 'r', encoding=encoding) as f:
                mart_content = f.read()
            # ---- Iterate over each soil DataFrame line
            for d in self.data.itertuples():
                # ---- Regex to match
                re_match = r"\/{0}\/ZONE_SOL\s*Z=\s*{1}V=*({2});".format(
                                                                d.soilprop.upper(),
                                                                d.zone, 
                                                                re_num)
                # ---- Search pattern
                match = re.search(re_match, mart_content)
                # ---- Replace by new value
                repl = re.sub(match.group(1), "{:10d}".format(int(d.value)), match.group(0))
                # --- Replace inplace in file content
                mart_content = re.sub(match.group(0), repl, mart_content)
            # ---- Write new content
            out = self.martfile if filename is None else filename
            with open(out, 'w', encoding=encoding) as f:
                f.write(mart_content)



    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheSoil'

