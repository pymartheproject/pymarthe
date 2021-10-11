"""
The PyMarthe package consists of a set of Python scripts for the BRGM
groundwater model Marthe. It is designed to facilitate the interfacing
with the PEST suite. 

"""

__name__ = 'pymarthe'
__author__ = 'Ryma Aissat, Yohann Cousquer, Alexandre Pryet'

__version__ = 0.1

# imports
from .utils import marthe_utils, pest_utils, pp_utils
from .marthe import MartheModel, SpatialReference
