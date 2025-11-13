

print('███████╗ ██████╗ ██████╗ ██╗    ██╗ █████╗ ██████╗ ██████╗     ██████╗ ██╗   ██╗███╗   ██╗')
print('██╔════╝██╔═══██╗██╔══██╗██║    ██║██╔══██╗██╔══██╗██╔══██╗    ██╔══██╗██║   ██║████╗  ██║')
print('█████╗  ██║   ██║██████╔╝██║ █╗ ██║███████║██████╔╝██║  ██║    ██████╔╝██║   ██║██╔██╗ ██║')
print('██╔══╝  ██║   ██║██╔══██╗██║███╗██║██╔══██║██╔══██╗██║  ██║    ██╔══██╗██║   ██║██║╚██╗██║')
print('██║     ╚██████╔╝██║  ██║╚███╔███╔╝██║  ██║██║  ██║██████╔╝    ██║  ██║╚██████╔╝██║ ╚████║')
print('╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝')



# -- Import basic python modules
import os
import pyemu
import multiprocessing as mp
import numpy as np
import pandas as pd
import pymarthe
from pymarthe.utils import marthe_utils, pest_utils



def main():
	# -- Run preproc functions
	# -- Run model from .config file
	pymarthe.utils.pest_utils.run_from_config("./data/hallue\configuration.config", exe_name="marthe", fmt_lite=True)
	# -- Run extra functions


# -- Launch forward run
if __name__ == '__main__':
	mp.freeze_support()
	main()

