"""
Contains the MartheOutput class
Designed for Marthe model postprocessing
"""


import os, sys
import numpy as np
import pandas as pd 


from pymarthe.marthe import MartheModel
from pymarthe.mfield import MartheField
from .utils import marthe_utils




class MartheOutput():
    """
    Wrapper Marthe --> python.
    Interface to perform Marthe model postprocessing.
    """

    def __init__(self, mm):
        """
        """
        self.mm = mm
        self.historiq_df = marthe_utils.read_mi_prn(os.path.join(self.mm.mldir, 'historiq.prn'))
        self.heads = {i:None for i in range(self.mm.nstep)}
        self.no_epon_val = 9999
        self.no_epon_out_val = 8888









    # def load_heads(self, filename = None):
    #     """
    #     """
    #     # ---- Manage filename
    #     if filename is None:
    #         filename = os.path.join(self.mm.mldir,'chasim.out')
    #     # ---- Read heads field records
    #     all_grids = marthe_utils.read_grid_file(filename)
    #     # ---- Split all grids by number of time step
    #     iterstep = np.array_split(all_grids, self.mm.nstep)
    #     n_digits = len(str(self.mm.nstep))
    #     # ---- Build MartheField instance for each time step
    #     print('Loading transient heads fields ...')
    #     heads = {}
    #     for i, grids in enumerate(iterstep):
    #         field = f'head_{str(i).zfill(n_digits)}'
    #         dfs = [pd.DataFrame(mg.to_records()) for mg in grids]
    #         rec = pd.concat(dfs).to_records(index=False)
    #         marthe_utils.progress_bar((i+1)/self.mm.nstep)
    #         self.heads[i] = MartheField(field, rec, self.mm)



    def plot_heads(self, istep=0, **kwargs):
        """
        """
        ax = self.heads[istep].plot(**kwargs)
        return ax



    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheOutput'



"""
def extract_prn(obsfile, prnfile=None, locnme=None, fluc=False, on='mean', interp_method='index', sim_dir='.'):
    '''
    Description
    -----------
    Reads model.prn read_prn() and writes individual file for each (selected) locations. 
    Each file contains two columns : date and its simulation value.
   
    Parameters
    ----------
    - obsfile (str/list) : path to the related observation file.
                           NOTE : can be a string (1 record to extract) or a 
                                  list of strings (multiple records to extract)
    - prnfile (str) : path the the marthe .prn file
    - locnme (str, optional) : observation location name(s) (ex. BSS id)
                                 Default is None
    - fluct (bool) : enable/disable fluctuations extraction.
    - on (str/numeric/fun) : function, function name or real number to 
                             substract to the simulated values.
                             Function names can be 'min', 'max', 'mean', 'median', etc.
                             See pandas.core.groupby.GroupBy documentation for more.
    - interp_method (str) : Interpolation method to use.
                            Can be 'linear', 'time', 'index', 'values', 'pad', 'nearest',
                            'zero', 'slinear', 'quadratic', 'cubic', 'spline', 'barycentric',
                            'polynomial', 'krogh', 'piecewise_polynomial', 'spline', 'pchip',
                            'akima', 'from_derivatives'.
                            Default is 'index' (linear interpolation on index).
    - sim_dir (str) : directory to store simulated files.
                      Default is '.'.

    
    Return
    ------
    Write simulated values inplace.
        
    Example
    -----------
    extract_prn(prnfile = 'historiq.prn, obsfile = 'obs/myobs.dat',
                interp_method='slinear', sim_dir = 'sim')
    '''
    # ---- Define simple iterable converter
    to_iterable = lambda elem: elem if isinstance(elem, list) else [elem]
    # ---- Get obsfiles as list
    obsfiles = to_iterable(obsfile)
    # ---- Get locnmes as list
    if locnme is None:
        locnmes = [os.path.split(f)[1].split('.')[0] for f in obsfiles]
    else:
        locnmes = to_iterable(locnme)
    # ----- Fetch prnfile name and sim directory
    if prnfile == None : 
        prnfile = 'historiq.prn'
    # ---- Read prnfile
    prn_df = read_prn(prnfile)
    # ---- Manage fluctuation if required
    if fluc:
        # ---- Build fluctuation DataFrame based on 'on' arguments
        fluc_df = prn_df.apply(lambda col: col.sub(col.agg(on) if isinstance(on, str) else on))
        # ---- Concatenate prn and fluctuation in a larger DataFrame
        df = pd.concat([prn_df, fluc_df.add_suffix('_fluc')], axis = 1)
    else:
        df = prn_df
    # ---- Iterate over all locnmes to extract
    for locnme, obsfile in zip(locnmes, obsfiles):
        # ---- Interpolate simulated values on observed based on date
        try:
            dates_out = read_obsfile(obsfile).index
        except:
            dates_out = read_obsfile(obsfile.replace('_fluc', '')).index
        ts_interp = ts_utils.interpolate(df[locnme], dates_out, method = interp_method)
        # ---- Write simulated value in sim directory
        simfile = os.path.join(sim_dir, f'{locnme}.dat')
        pest_utils.write_simfile(dates = ts_interp.index, values = ts_interp, simfile = simfile)
"""














