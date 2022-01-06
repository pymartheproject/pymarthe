
"""
Contains some usefull tools to manage TimeSeries data.

"""

import os
import numpy as np
import pandas as pd



def interpolate(ts, dates_out, method = 'index', **kwargs):
    """
    -----------
    Description:
    -----------
    Interpolate values of a TimeSerie (ts) on target dates.
    
    Parameters: 
    -----------
    ts (pandas.Series) : Timeserie
                         The index must be a DateTimeIndex instance
    dates_out (pandas.DateTimeIndex) : Target date index to interpolate on
    method (str) : Interpolation method to use.
                   Can be 'linear', 'time', 'index', 'values', 'pad', 'nearest',
                   'zero', 'slinear', 'quadratic', 'cubic', 'spline', 'barycentric',
                   'polynomial', 'krogh', 'piecewise_polynomial', 'spline', 'pchip',
                   'akima', 'from_derivatives'.
                   Default is 'index' (linear interpolation on index).
    **kwargs (dict) : Additional arguments for interpolation process.
                      See pandas.core.generic module for more informations.

    Returns:
    -----------
    Interpolated Timeserie on target date index.

    Example
    -----------
    ts_obs = pd.read('data.csv')['myobs']
    interp_obs = interpolate(ts_obs, ts_sim, method = 'nearest', limit = 20)
    """
    # ----- Combine DateTimeIndex
    dti_uni = ts.index.union(dates_out).drop_duplicates()
    # ---- Interpolate values
    ts_interp = ts.reindex(dti_uni).interpolate(method=method, **kwargs)
    # ---- Return only interpolated target values
    return ts_interp[dates_out].dropna()

