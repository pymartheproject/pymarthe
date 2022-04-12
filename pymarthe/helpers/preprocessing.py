
"""
Contains some helper functions for Marthe model preprocessing

"""

import os, sys
import numpy as np
import pandas as pd
from copy import deepcopy
from pymarthe.utils import *




def spatial_aggregation(mm, x, y, layer, value, agg = 'sum', trans ='none', only_active = True, base=0):
    """
    Helper function to aggregate (and transform) values that are in same model cell.

    Parameters:
    ----------

    mm (MartheModel) : MartheModel instance.
                       Note: must contain spatial index.
    x, y ([float]) : xy points coordinates in model extension.
                     Note: must have the same length.
    layer (int/iterable) : layer id to search of point localisation (0-based).
                           If `layer` is an integer, same layer id will be considered
                           for all points.
                           Note: for some values such as river pumping, `layer` has to be
                                 be set to 0 (superficial aquifer).
    value ([float]) : related point values to aggregate.
                      Note: must be same length as x and y.
    agg (func/str, optional) : Function or function name to perform aggregation.
                               Can be: 'mean', 'median', 'sum', 'count', np.sum, ...
                               Default is 'sum'.
    trans (str/func, optional) : keyword/function to use for transforming aggregated values.
                                 Can be:
                                    - function (np.log10, np.sqrt, ...)
                                    - string function name ('log10', 'sqrt')
                                 If 'none', values will not be transformed.
                                 Default if 'none'.
    only_active (bool, optional) : whatever considering active cells only.
                                   Default is True.
                                   Note: for some values such as river pumping, `only_active`
                                         has to be set to False.
    base (int, optional) : output layer id n-base.
                           Marthe is 1-based (base=1), PyMarthe is 0-based (base=0).
                           Default is 0.

    Returns:
    --------
    res (DataFrame) : aggregate and transformed values.
                      Format:
                            node          x         y     layer      value
                      0      789   458963.2  698754.1         4     0.0023
                      1      856   458456.5  698702.8         2     0.0046
                      .

    Examples:
    --------
    x = [458963.2, .., 458456.5]
    y = [698754.1, .., 698702.8]
    layer = [4, .., 2]
    value = [0.0023, .., 0.0046]
    df = spatial_aggregation(mm, x, y, layer, value,
                             agg = 'sum', trans ='lambda x: -x',
                             only_active= True, base=1)
    df.to_csv('file.txt', header=False, index = False, sep='\t')

    """
    # -- Sample node ids for each xylayer-points
    nodes = mm.get_node(x,y,layer)

    # -- Store in DataFrame
    df = pd.DataFrame( { 'node': nodes,
                         'x': x,
                         'y': y,
                         'layer' : layer,
                         '_value' : value  } )

    # -- Groupby by node and perform aggregation
    agg_df = df.groupby('node', as_index=False).agg({'_value': agg})

    # -- Subset on active cells
    if only_active:
        agg_df = agg_df.loc[[mm.all_active(n) for n in agg_df.node]]

    # -- Rename value column for merging purpose
    agg_df.rename({'_value': 'value'}, axis = 1, inplace=True)

    # -- Merge aggregated/initial DataFrames 
    res = pd.merge(df, agg_df).drop_duplicates('node').drop('_value',axis=1)

    # -- Convert to n-base
    res['layer'] = res.layer.add(base)

    # -- Transform aggregated values if required
    if trans != 'none':
        # -- Check transformation validity
        pest_utils.check_trans(trans, test_on= res.value)
        # -- Perform transformation on aggregated values
        res['value'] = pest_utils.transform(res['value'], trans)

    # ---- Return aggregated/transformed values in DataFrame
    return res

