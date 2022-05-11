"""
Contains some helper functions for Marthe model preprocessing

"""

import os, sys
import re
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






def hydrodyn_calc(pastpfile, istep, external=False, new_pastpfile=None):
    """
    Helper function to manage hydrodynamic computation periodicity in .pastp file.

    Parameters:
    ----------

    pastpfile (str) : path to the required model .pastp file.

    istep (str/int/iterable) : required istep to activate hydrodynamic computation.
                               Can be :
                                    - 'all' : activate for all timesteps
                                    - 'none': desactivate for all timesteps
                                    - 'start:end:step' : string sequence
                                    - [0,1,2,..] : any integer iterables

    external (bool, optional) : whatever create a external file with required
                                hydrodynamic computation periodicity.
                                Note: external optional will not be considered
                                      for global `istep` such as 'all', 'none'
                                Default is False.

    new_pastpfile (str, optional) : path to the new pastp file to write.
                                    If None, will overwrite the input pastp file.
                                    Default is None.

    Returns:
    --------
    (Over-)write pastp file.
    If `external` == True, will also write 'cacul_hydro.txt'.
                      .

    Examples:
    --------
    f = 'mymodel.pastp'
    # -- All timesteps
    hydrodyn_calc(f, istep= 'all', external=False)
    # -- Weekly
    hydrodyn_calc(f, istep= '::7', external=True)
    # -- Annual
    hydrodyn_calc(f, istep= '::365', external=False)
    # -- Specific
    hydrodyn_calc(f, istep= [0,5,6,7,9,11], external=True)

    """
    # ---- Read .pastp file by lines
    with open(pastpfile, 'r', encoding = marthe_utils.encoding) as f:
        init = f.readlines()

    # ---- Clear existing hydrodynamic action
    lines = [l for l in init if not 'CALCUL_HDYNAM' in l]
    nlines = deepcopy(lines)

    # -- Manage string istep
    if isinstance(istep, str):
        # ---- Activate all timesteps
        if istep == 'all':
            # -- Find index `i` where insert required line
            for i, l in enumerate(lines):
                if 'Fin de ce pas' in l:
                    break
            # -- Insert line to activate all timesteps (default) 
            _l ='  /CALCUL_HDYNAM/ACTION    I= 0;\n'
            nlines.insert(i, _l)

        # ---- Desactivate all timesteps
        if istep == 'none':
            # -- Find index `i` where insert required line
            for i, l in enumerate(lines):
                if 'Fin de ce pas' in l:
                    break
            # -- Insert line to activate all timesteps (default)
            _l ='  /CALCUL_HDYNAM/ACTION    I= -1;\n'
            nlines.insert(i, _l)

        # ---- Manage string sequence to get istep 
        if ':' in istep:
            # -- Extract total number of timesteps
            nstep = len(re.findall(r'Fin de ce pas', ''.join(lines)))
            # -- Extract time bounds (start, end, step) as integers
            re_seq = r'(\d+|\s*):(\d+|\s*):(\d+)'
            s, e, ii = re.findall(re_seq,istep)[0]
            start = eval(s) if s != '' else 0
            end = eval(e) if e != '' else nstep
            step = eval(ii) if ii != '' else 1
            # -- Set numerical istep as array
            if start < end:
                istep = np.arange(start, end, step)
            else:
                istep = np.arange(end, start, step)

    # -- Manage iterable istep
    if not isinstance(istep, str):
        # -- Make istep iterable if is not already
        isteps = marthe_utils.make_iterable(istep)
        # -- Manage external mode
        if external:
            # -- Set external filename (generic) in first timestep
            ext = 'calcul_hydro.txt'
            _l = f'  /CALCUL_HDYNAM/ACTION    I= 0; File= {ext}\n'
            ext_path = os.path.join(os.path.split(pastpfile)[0], ext)
            for i, l in enumerate(lines):
                if 'Fin de ce pas' in l:
                    break
            nlines.insert(i, _l)
            # -- Fetch timesteps calendar dates
            dates = re.findall(r'\d{2}\/\d{2}\/\d{4}', ''.join(lines))
            # -- Set external file header
            s =  "HYDRODYNAMIC COMPUTATION PLANNING\n"
            # -- Iterate over all time steps (2 = active, 0 = inactive)
            for n, date in enumerate(dates):
                # -- Active if in required timesteps desactive otherwise
                if n in isteps:
                    s += f'2\t{date}\n'
                else:
                    s += f'0\t{date}\n'

            # -- Write external file
            with open(ext_path, 'w', encoding =marthe_utils.encoding) as f:
                f.write(s)

        # -- Manage internal mode
        else:
            # -- Set active/desactive line to write
            _l = '  /CALCUL_HDYNAM/ACTION    I= 2;\n'
            __l = '  /CALCUL_HDYNAM/ACTION    I= 0;\n'
            n = -1
            # -- Iterate over pastp lines
            for i, l in enumerate(lines):
                # -- Detect end of time step line index
                if 'Fin de ce pas' in l:
                    n += 1
                    # -- Insert active/desative line
                    if n in isteps:
                        nlines.insert(i+n,_l)
                    else:
                        nlines.insert(i+n,__l)

    # -- (Over-)write pastp file 
    out = pastpfile if new_pastpfile is None else new_pastpfile
    with open(out, 'w', encoding=marthe_utils.encoding) as f:
        f.write(''.join(nlines))

    # -- Print final message
    print("==> Hydrodynamic computation periodicity " \
          f"had been set successfully in '{out}'") 




