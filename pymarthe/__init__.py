"""
The PyMarthe package consists of a set of Python scripts for the BRGM
groundwater model Marthe. It is designed to facilitate the interfacing
with the PEST suite. 

"""

__name__ = 'pymarthe'
__author__ = 'Ryma Aissat, Yohann Cousquer, Alexandre Pryet, Pierre Matran'

__version__ = 1.0



# ---- IMPORTS 

from .marthe import MartheModel
from .mfield import MartheField
from .moptim import MartheOptim
from .mobs import MartheObs
from . import utils
