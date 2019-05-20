# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import dates as mdates
import datetime
import subprocess
from matplotlib.dates import bytespdate2num
import matplotlib.dates as mdates
import pandas as pd

############################################################
#        Utils for pest preprocessing for Marthe
############################################################

# ----------------------------------------------------------------------------------------------------------
#Modified from PyEMU 
#https://github.com/jtwhite79/pyemu/
# ----------------------------------------------------------------------------------------------------------

# ----- from PyEMU ----------------
def SFMT(item):
    try:
        s = "{0:<20s} ".format(item.decode())
    except:
        s = "{0:<20s} ".format(str(item))
    return s

SFMT_LONG = lambda x: "{0:<50s} ".format(str(x))
IFMT = lambda x: "{0:<10d} ".format(int(x))
FFMT = lambda x: "{0:<20.10E} ".format(float(x))


PP_FMT = {"parname": SFMT, "x": FFMT, "y": FFMT, "zone": IFMT, "tpl": SFMT,
"value": FFMT}


def write_tpl_from_df(tpl_file,df) :  
    f_tpl = open(tpl_file,'w')
    f_tpl.write("ptf ~\n")
    f_tpl.write(df.to_string(col_space=0,
        columns=["parname","tpl"],
        formatters=PP_FMT,
        justify="left",
        header=False,
        index=False) + '\n')

