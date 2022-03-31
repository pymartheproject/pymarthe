# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import re, ast
import pyemu

from pymarthe.utils import ts_utils, marthe_utils

############################################################
#        Utils for pest preprocessing for Marthe
############################################################

# ----------------------------------------------------------------------------------------------------------
#Modified from PyEMU 
#https://github.com/jtwhite79/pyemu/
# ----------------------------------------------------------------------------------------------------------


# ---- Set encoding
encoding = 'latin-1'


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
LL_PARAM_DIC = {"parnme": SFMT, "tplnme": SFMT, "defaultvalue": FFMT, 'transformed': FFMT}

# ---- Set observation character start and length
VAL_START, VAL_END = 21, 40



def write_mlp_tplfile(tplfile, param_df):
    """
    """
    df = param_df.copy(deep=True)
    df['tplnme'] = '~' + df['parnme'].str.replace('__', '_')  + '~'
    with open(tplfile, 'w', encoding=encoding) as f:
        f.write('ptf ~\n')
        f.write(df.to_string(col_space=0, columns=['parnme', 'tplnme'],
                                   formatters=LL_PARAM_DIC, justify="left",
                                   header=False, index=False, index_names=False,
                                   max_rows = len(df), min_rows = len(df)))


def write_mlp_parfile(parfile, param_df, trans='none'):
    """
    """
    df = param_df.copy(deep=True)
    df['transformed'] = transform(param_df['defaultvalue'], trans)
    with open(parfile, 'w', encoding=encoding) as f:
        f.write(df.to_string(col_space=0, columns=['parnme', 'transformed'],
                             formatters=LL_PARAM_DIC, justify="left",
                             header=False, index=False, index_names=False,
                             max_rows = len(df), min_rows = len(df)))



def read_mlp_parfile(parfile):
    """
    """
    par_df = pd.read_csv(parfile, header=None,
                         delim_whitespace=True, names = ['parnme','value'])
    return par_df





def parse_mlp_parfile(parfile, keys, value_col, btrans):
    """
    """
    par_df = read_mlp_parfile(parfile)
    items = []
    for ipar in par_df.parnme:
        parsed = ipar.split('__')
        items.append([ast.literal_eval(s) 
                          if s.isnumeric() 
                          else s 
                          for s in parsed])
    kmi = pd.MultiIndex.from_tuples(items, names = keys)
    bvalues = transform(par_df['value'], btrans)
    return kmi, bvalues




def transform(it, trans, fail = 'raise'):
    """
    """
    s = pd.Series(it)
    # -- 
    res = "Invalid transformation."
    if trans == 'none':
        return s.transform(lambda x: x)
    else:
        try:
            res = s.transform(trans)
        except:
            pass
        finally:
            try:
                res = s.transform(eval(trans))
            except:
                pass
        if isinstance(res, str):
            if fail == 'raise':
                raise ValueError(res)
            else:
                return False
        else:
            return res




def is_valid_trans(trans):
    """
    """
    # -- Generate basic serie
    s = pd.Series(np.arange(1,3))
    res = transform(s, trans, fail = 'bool')
    if isinstance(res, pd.Series):
        return True
    else:
        return False




def check_trans(trans, btrans=None, test_on =None):
    """
    """
    err_trans = 'ERROR: Invalid transformation. Must be a pandas ' \
                'string function or a litteral expression understood ' \
                'by the python built-in eval() function. '

    err_test = 'ERROR: transformation and back-transformation not compatibles -> ' \
                f'btrans(trans(value)) != value. Given trans= {trans}, btrans= {btrans}'

    assert is_valid_trans(trans), err_trans + f'Given: {trans}.'

    if btrans is not None:
        assert is_valid_trans(btrans), err_trans + f'Given: {btrans}.'
        it = pd.Series(np.arange(-1,3)) if test_on is None else pd.Series(test_on)
        test = transform(transform(it, trans), btrans)
        assert all(x for x in it.values == test.values), err_test




def read_config(configfile):
    """
    """
    # -- Get content
    with open(configfile, 'r', encoding=encoding) as f:
        content = f.read()

    # -- Set usefull regex
    re_hblock = r'\*{3}(.+?)\*{3}'
    re_pblock = r'\[START_PARAM\](.+?)\[END_PARAM\]'
    re_oblock = r'\[START_OBS\](.+?)\[END_OBS\]'
    re_item_hblock = r'(.+):\s*(.+)\n'
    re_item_block = r'(.+)=\s*(.+)\n'
    

    # -- Get headers as dictionary
    hblock =  re.search(re_hblock, content, re.DOTALL).group(0)
    hdic = dict(re.findall(re_item_hblock, hblock))

    # -- Get parameters info as dictionary
    pblocks =  re.findall(re_pblock, content, re.DOTALL)
    pdics = []
    for pb in  pblocks:
        pdic = dict(re.findall(re_item_block, pb))
        pdics.append(pdic)

    # -- Get observations info as dictionary
    oblocks =  re.findall(re_oblock, content, re.DOTALL)
    odics = []
    for ob in  oblocks:
        odic = dict(re.findall(re_item_block, ob))
        odics.append(odic)


    # --- return
    return hdic, pdics, odics



def get_kmi(mobj, keys, **kwargs):
    """
    Return standard Keys Multi Index from a marthe object data.
    keys >= 2
    """
    # -- Perform kwargs getting process on Marthe object
    df = mobj.get_data(**kwargs)
    # -- Generate KeysMultiIndex
    kmi = pd.MultiIndex.from_frame(df[keys])
    # -- Return 
    return kmi





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
        l1 (obsnme0)21:40
        l1 (obsnme1)21:40

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
    df['ins_line'] = df['obsnme'].apply(lambda s: 'l1 ({}){}:{}'.format(s,VAL_START,VAL_END))
    # ---- Write formated instruction file
    with open(insfile,'w', encoding=encoding) as f:
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
    df = pd.DataFrame(dict(date= dates, value = values))
    # ---- Write formated instruction file
    df.to_csv(simfile, header=False, index=False, sep='\t', float_format = FFMT, date_format='%s')





def extract_prn(prn, name, dates_out=None, trans='none', interp_method = 'index', fluc_dic=dict(), sim_dir='.'):
    """
    """
    # -- Fetch prn as MultiIndex DataFrame
    if isinstance(prn, pd.DataFrame):
        prn_df = prn
    else:
        prn_df = marthe_utils.read_prn(prn)

    # -- Manage if fluctuation
    if len(fluc_dic) == 0:
        suffix = ''
        validity = name in prn_df.columns.get_level_values('name')
    else:
        suffix, on = fluc_dic['tag'] + 'fluc', fluc_dic['on']
        validity = name.replace(suffix,'') in prn_df.columns.get_level_values('name')
    
    # -- Assert that the required name is in prn
    err_msg = f'ERROR : `{name}` not found in simulated data. ' \
              'It must be provided in the Marthe .histo file.'
    assert validity, err_msg

    # -- Get records by name
    df = prn_df.xs(key=name.replace(suffix,''), level='name', axis=1)
    df.columns = ['value']

    # -- Tranform to fluctuation if required
    if len(fluc_dic) > 0:
        df = df.apply(lambda col: col.sub(col.agg(on) if isinstance(on, str) else on))

    # -- Interpolate values on observations if required
    if not dates_out is None:
        df = ts_utils.interpolate(df['value'],
                                  dates_out,
                                  method = interp_method).to_frame()

    # -- Write simulated data in external file
    write_simfile(dates = df.index,
                  values = transform(df['value'], trans).values,
                  simfile = os.path.join(sim_dir, f'{name}.dat'))







def run_from_config(configfile, **kwargs):
    """
    """
    # -- Load MartheModel with parametrized properties
    from pymarthe import MartheModel
    mm = MartheModel.from_config(configfile)
    # -- Overwrite new data from parfiles
    mm.write_prop()
    # -- Run model
    mm.run_model(**kwargs)
    # -- Extract simulated data
    prn = marthe_utils.read_prn(os.path.join(mm.mldir,'historiq.prn'))
    hdic, _, odics = read_config(configfile)
    for odic in odics:
        extract_prn(prn= prn, 
                    name= odic['locnme'],
                    dates_out= pd.DatetimeIndex(odic['dates_out'].split('|')),
                    trans= odic['trans'],
                    interp_method= odic['interp_method'],
                    fluc_dic= eval(odic['fluc_dic']),
                    sim_dir= os.path.normpath(hdic['Simulation files directory']))

