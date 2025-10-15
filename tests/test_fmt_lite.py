#----------------------------------------------------------------------
# ---- Tutorial for model parameterization with ZPC and pilot points ---
#----------------------------------------------------------------------

import os, re, shutil
import unittest

import pandas as pd
from pyemu import Pst

from pymarthe import MartheModel, MartheField
from pymarthe.moptim import MartheOptim
from pymarthe.utils import  pp_utils


# clear directory 
def clear_dirs(dlist):
    for d in dlist:
        if os.path.exists(d):
           shutil.rmtree(d)
        os.mkdir(d)


def add_obs(mopt, loc_list, obs_dir):
    for loc in loc_list:
        obsfile = os.path.join(obs_dir, loc+'.dat')
        df = pd.read_csv(
            obsfile, 
            sep=r"\s{2,}", 
            engine='python',
            header=None, 
            skiprows=1,
            index_col=0, 
            parse_dates=True
            )
        df = df.rename(columns = {1 :'value'})
        df.index.name = 'date'
        mopt.add_obs(data = df,locnme=loc, datatype='head')
    return mopt


def create_file(mopt, mm):
    # write pest I/O files 
    mopt.write_parfile() # model parameter file 
    mopt.write_tplfile() # pest template files 
    mopt.write_insfile() # pest instruction files

    # PyMarthe configuration file and forward run script for runtime 
    fr_file = os.path.join(mm.mldir,'forward_run.py')
    configfile = os.path.join(mm.mldir,'configuration.config')
    mopt.write_config(configfile)
    mopt.write_forward_run(fr_file, configfile, exe_name='marthe')

    # generate and write pest control file  
    pst = mopt.build_pst(model_command='python3 forward_run.py') # python3 si linux
    pst_name = os.path.join(mm.mldir,'opt.pst')
    pst.write(pst_name)


def check_outputs(pst_file):
    # load params names from Pst file
    _pst = Pst(pst_file)
    return _pst.adj_par_names


class TestFmtLite(unittest.TestCase):
    def setUp(self):
        
        # os.chdir()
        clear_dirs(['par', 'tpl', 'ins', 'sim'])
        
        self.mldir = './tests/data/hallue'
        self.rma_file = os.path.join(self.mldir,'hallue.rma')
        self.prn_file = os.path.join(self.mldir,'historiq.prn')
        self.obs_dir = os.path.join(self.mldir,'obs')
        self.sim_dir = os.path.join(self.mldir,'sim')
        self.mm = MartheModel(self.rma_file)
        self.dirs = {f'{f}_dir': os.path.join(self.mldir, f) 
                for f in ['par', 'tpl', 'ins', 'sim','obs'] }
        self.loc_list = ['00464X0013','00471X0010','P1','P2','P3','P4']

    def set_opt(self):
        mopt = MartheOptim(self.mm, name='opt', **self.dirs)
        mopt = add_obs(mopt, self.loc_list, self.obs_dir)
        return mopt

    def test_fmt_lite_zpc_true(self):
        permh = self.mm.prop['permh']
        mopt = self.set_opt()
        # default = zpc
        # test True
        mopt.add_param(
            parname='hk',
            mobj=permh,
            fmt_lite=True
        )
        create_file(mopt, self.mm)
        test = check_outputs('{}/opt.pst'.format(self.mldir))
        self.assertTrue(len(test[0]) <= 12, 'Lite fmt failed for ZPC (len>12)')
        self.assertTrue(re.match(r'\w+_zpc_z\d+', test[0]), 'Long fmt failed for ZPC')
    
    def test_fmt_lite_zpc_false(self):
        permh = self.mm.prop['permh']
        mopt = self.set_opt()
        # default = zpc
        # test False
        mopt.add_param(
            parname='hk2',
            mobj=permh,
            fmt_lite=False
        )
        create_file(mopt, self.mm)
        test = check_outputs('{}/opt.pst'.format(self.mldir))
        self.assertTrue(re.match(r'\w+_zpc_l\d{2}_z\d{2}', test[0]), 'Long fmt failed for ZPC')
    
    def test_fmt_lite_pp_true(self):
        permh = self.mm.prop['permh']
        ipermh = MartheField('ipermh',1, self.mm)
        mopt = self.set_opt()
        pp_shpfile = os.path.join(self.mldir,'pp.shp')
        pp_data = {0:{1:pp_shpfile} }
        mopt.add_param(
            parname='hk', mobj=permh,  trans='log10', btrans='lambda x: 10**x', 
            izone=ipermh, pp_data=pp_data, defaultvalue=1e-4,
            fmt_lite=True
        )
        create_file(mopt, self.mm)
        test = check_outputs('{}/opt.pst'.format(self.mldir))
        # TODO
        # self.assertTrue(re.match('', test[0]), 'Lite fmt failed for PP')
    
    def test_fmt_lite_pp_false(self):
        permh = self.mm.prop['permh']
        ipermh = MartheField('ipermh',1, self.mm)
        mopt = self.set_opt()
        pp_shpfile = os.path.join(self.mldir,'pp.shp')
        pp_data = {0:{1:pp_shpfile} }
        mopt.add_param(
            parname='hk', mobj=permh,  trans='log10', btrans='lambda x: 10**x', 
            izone=ipermh, pp_data=pp_data, defaultvalue=1e-4,
            fmt_lite=False
        )
        create_file(mopt, self.mm)
        test = check_outputs('{}/opt.pst'.format(self.mldir))
        # TODO
        # self.assertTrue(re.match('', test[0]), 'Lite fmt failed for PP')


if __name__ == "__main__":
    
    testcase = TestFmtLite()
    testcase.setUp()
    # testcase.test_fmt_lite_zpc_true()
    # testcase.test_fmt_lite_zpc_false()
    testcase.test_fmt_lite_pp_true()
    # testcase.test_fmt_lite_pp_false()