
import os 
import numpy as np
import pandas as pd
from pymarthe.utils import marthe_utils, shp_utils


PP_NAMES = ["parname","x","y","zone","value"]
PPFMT = lambda name, lay, zone, ppid, digit: '{0}_l{1:02d}_z{2:02d}_{3}'.format(name,int(lay),int(zone), str(int(ppid)).zfill(digit))


def pp_df_from_coords(parname, coords, layer, zone, value= 1e-3):
    """
    """
    # -- Manage value input
    if len(marthe_utils.make_iterable(value)) == 1:
        value = np.tile(value, len(coords))
    # -- Generate names
    digit = len(str(len(coords)))
    ppn = [PPFMT(parname,layer, zone, i, digit) for i in  range(len(coords))]
    # -- Build pilot point standart DataFrame
    ppx, ppy = np.column_stack(coords)
    pp_df = pd.DataFrame.from_dict(
                {k:v for k,v in zip(PP_NAMES, [ppn,ppx,ppy,zone,value])}
                                ).set_index('parname', drop=False)
    # -- Return
    return pp_df


