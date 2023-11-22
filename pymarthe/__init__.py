"""
The PyMarthe package consists of a set of Python scripts for the BRGM
groundwater model Marthe. It is designed to facilitate the interfacing
with the PEST suite. 

"""

__name__ = 'PyMarthe'
__author__ = 'Ryma Aissat, Alexandre Pryet, Pierre Matran'
__version__ = 1.0


print(
    ''.join(
        [
            "\33[90m",
            "".join(['_']*48),
            '\33[35m',
            r"""      
  ____        __  __            _   _          
 |  _ \ _   _|  \/  | __ _ _ __| |_| |__   ___ 
 | |_) | | | | |\/| |/ _` | '__| __| '_ \ / _ \
 |  __/| |_| | |  | | (_| | |  | |_| | | |  __/
 |_|    \__, |_|  |_|\__,_|_|   \__|_| |_|\___|
        |___/                                                                             
"""
            "\33[90m",
            "".join(['_']*48),
            "\033[0;0m"
        ]
    )
)


# ---- Imports from main library
from .marthe import MartheModel
from .mfield import MartheField
from .moptim import MartheOptim
from .msoil import MartheSoil
from .mpump import MarthePump
from .mobs import MartheObs
from . import utils



