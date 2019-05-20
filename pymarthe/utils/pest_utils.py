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
#https://github.com/jtwhite79/pyemu/blob/develop/pyemu/
# ----------------------------------------------------------------------------------------------------------

def write_tpl_from_df(tpl_file,df) :  
    f_tpl = open(tpl_file,'w')
    f_tpl.write("ptf ~\n")
    f_tpl.write(df.to_string(col_space=0,
        columns=["name","tpl"],
        formatters=PP_FMT,
        justify="left",
        header=False,
        index=False) + '\n')
