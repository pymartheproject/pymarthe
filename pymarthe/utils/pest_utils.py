# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

############################################################
#        Utils for pest preprocessing for Marthe
############################################################

# ----------------------------------------------------------------------------------------------------------
#Modified from PyEMU 
#https://github.com/jtwhite79/pyemu/
# ----------------------------------------------------------------------------------------------------------

# ---- Set formater dictionaries
def SFMT(item):
    try:
        s = "{0:<20s} ".format(item.decode())
    except:
        s = "{0:<20s} ".format(str(item))
    return(s)

FFMT = lambda x: "{0:<20.10E} ".format(float(x))
IFMT = lambda x: "{0:<10d} ".format(int(x))
FMT_DIC = {"obsnme": SFMT, "obsval": FFMT, "ins_line": SFMT, "date": SFMT, "value": FFMT}
PP_FMT = {"name": SFMT, "x": FFMT, "y": FFMT, "zone": IFMT, "tpl": SFMT, "value": FFMT}

# ---- Set observation character start and length
VAL_START, VAL_CHAR_LEN = 12, 19




def compute_weight(lambda_i, lambda_n, m, n, sigma_i):
    """
    -----------
    Description
    -----------
    Compute weigth for a single observation
    -----------
    Parameters
    -----------
    - lambda_i (int) : tuning factor for current observation data type
    - lambda_n (int) : sum of all tuning factors
    - m (int) : number of station for a given data type
    - n (int) : number of records for a given data type at a given station
    - sigma (float) : the expected variance between simulated and observed data
    -----------
    Returns
    -----------
    w (float) : weight of a given observation
    -----------
    Examples
    -----------
    w = compute_weight(lambda_i = 10, lambda_n = 14, m = 22, n = 365, sigma = 0.01)
    """
    w = np.sqrt(lambda_i / (lambda_n  * m * n * (sigma_i**2)))
    return(w)



def write_insfile(obsnmes, insfile):
    """
    -----------
    Description
    -----------
    Write pest instruction file.
    Format:
        pif ~
        l1 (obsnme0)12:21
        l1 (obsnme1)12:21

    Values start at character 12.
    Values is 21 characters long.

    -----------
    Parameters
    -----------
    - obsnmes (list/Series) : observation names
                              ex: [loc004n01, loc004n02, ...]
                              NOTE : all names must be unique.
    - insfile (str) : path to instruction file to write.
    -----------
    Returns
    -----------
    Write instruction file inplace.
    -----------
    Examples
    -----------
    obsnmes = ['loc001n{}'.format(str(i).zfill(3)) for i in range(250)]
    write_insfile(obsnmes, insfile = 'myinsfile.ins')
    """
    # ---- Build instruction lines
    df = pd.DataFrame(dict(obsnme = obsnmes))
    df['ins_line'] = df['obsnme'].apply(lambda s: 'l1 ({}){}:{}'.format(s,VAL_START,VAL_CHAR_LEN))
    # ---- Write formated instruction file
    with open(insfile,'w') as f:
        f.write('pif ~\n')
        f.write(df.to_string(col_space=0, columns=["ins_line"],
                             formatters=FMT_DIC, justify="left",
                             header=False, index=False, index_names=False,
                             max_rows = len(df), min_rows = len(df)))



def write_simfile(dates, values, simfile):
    """
    -----------
    Description
    -----------
    Write simulated values (Can be extract from .prn file)
    Format:
        1972-12-31  12.755
        1973-12-31  12.746
        1974-12-31  12.523

    -----------
    Parameters
    -----------
    - dates (DatetimeIndex) : time index of the record.
    - values (list/Series) : simulated values.
    - simfile (str) : path to simulated file to write.
    -----------
    Returns
    -----------
    Write simulated file inplace.
    -----------
    Examples
    -----------
    sim = marthe_utils.read_prn('historiq.prn')['loc_name']
    write_simfile(dates = sim.index, sim, 'mysimfile.dat')
    """
    # ---- Build instruction lines
    df = pd.DataFrame(dict(value = values), index = dates)
    # ---- Write formated instruction file
    with open(simfile,'w') as f:
        f.write(df.to_string(col_space=0, columns=["value"],
                             formatters=FMT_DIC, justify="left",
                             header=False, index=True, index_names=False,
                             max_rows = len(df), min_rows = len(df)))





def write_tpl_from_df(tpl_file,df, columns = ["name","tpl"] ) :  
    f_tpl = open(tpl_file,'w')
    f_tpl.write("ptf ~\n")
    f_tpl.write(df.to_string(col_space=0,
        columns=columns,
        formatters=PP_FMT,
        justify="left",
        header=False,
        index=False) + '\n')



def sum_weighted_squared_res(sim,obs,weight):
    """
    Returns sum of weighted squared residuals
    """
    return( np.sum( np.pow(weight(sim-obs), 2) ) )
