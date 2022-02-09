
"""
Contains the MartheModel class
Designed for structured grid and
layered parameterization

"""
import os 
import numpy as np
from matplotlib import pyplot as plt 
from .utils import marthe_utils, pest_utils, pp_utils
import pandas as pd 
import pyemu



# float and integer formats
FFMT = lambda x: "{0:<20.10E} ".format(float(x))
IFMT = lambda x: "{0:<10d} ".format(int(x))
# string format
def SFMT(item):
    try:
        s = "{0:<20s} ".format(item.decode())
    except:
        s = "{0:<20s} ".format(str(item))
    return s



base_param = ['parnme', 'trans', 'btrans', 'parchglim',
                  'defaultvalue', 'parlbnd', 'parubnd',
                  'pargp', 'scale', 'offset', 'dercom']




class MartheListParam():
    """
    Class for handling Marthe list-like properties. 
    """
    def __init__(self, parname, mobj, kmi, optname = 'value', trans = 'none', 
                       btrans = 'none', defaultvalue=None, **kwargs):
        """
        """
        # -- Transformation validity
        pest_utils.check_trans(trans)
        pest_utils.check_trans(btrans)
        
        # -- Atributs
        self.parname = parname
        self.type = 'list'
        self.mobj = mobj
        self.kmi = kmi
        self.optname = optname
        self.parnmes = self.gen_parnmes()
        self.trans = trans
        self.btrans = btrans
        if defaultvalue is None:
            self.defaultvalue = mobj.data.set_index(self.kmi.names).loc[self.kmi, optname].to_list()
        else:
            self.defaultvalue = defaultvalue
        self.parchglim = kwargs.get('parchglim', 'factor')
        self.parlbnd = kwargs.get('parlbnd', 1e-10) 
        self.parubnd = kwargs.get('parubnd', 1e+10) 
        self.pargp = kwargs.get('pargp', self.parname) 
        self.scale = kwargs.get('scale', 1) 
        self.offset = kwargs.get('offset', 0)
        self.dercom = kwargs.get('dercom', 1) 
        # ---- Build parameter DataFrame
        self.param_df = pd.DataFrame(index = self.parnmes)
        self.param_df[base_param] = [ self.parnmes, self.trans,
                                          self.btrans, self.parchglim,
                                          self.defaultvalue, self.parlbnd,
                                          self.parubnd, self.pargp,
                                          self.scale, self.offset, self.dercom ]
        # ---- Manage files io
        self.parfile = kwargs.get('parfile', f'{self.parname}.dat')
        self.tplfile = kwargs.get('tplfile', f'{self.parname}.tpl')



    def gen_parnmes(self):
        """
        """
        return ['__'.join(list(map(str, items))) for items in self.kmi]



    def to_config(self):
        """
        """
        lines = ['[START_PARAM]']
        data = [
            'parname= {}'.format(self.parname),
            'type= {}'.format(self.type),
            'class= {}'.format(str(self.mobj)),
            'property name= {}'.format(self.mobj.prop_name),
            'keys= {}'.format('\t'.join(self.kmi.names)),
            'optname= {}'.format(self.optname),
            'trans= {}'.format(self.trans),
            'btrans= {}'.format(self.btrans),
            'parfile= {}'.format(self.parfile),
              ]
        lines.extend(data)
        lines.append('[END_PARAM]')
        return '\n'.join(lines)



    def write_parfile(self, parfile=None):
        """
        """
        pf = self.parfile if parfile is None else parfile
        pest_utils.write_mlp_parfile(pf, self.param_df,  self.trans)



    def write_tplfile(self, tplfile=None):
        """
        """
        tf = self.tplfile if tplfile is None else tplfile
        pest_utils.write_mlp_tplfile(tf, self.param_df)


    # @classmethod
    # def from_config(self, configblock):
    #     """
    #     """
    #     return