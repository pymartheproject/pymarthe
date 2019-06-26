import numpy as np
import pandas as pd
import geopandas 

def ppoints_from_file(pp_file):
    return(pp_df)


def ppoints_from_shp(shp_file) : 
    pp_df = geopandas.read_file(pp_shp_file)
    pp_df['x'] = pp_df.geometry.x
    pp_df['y'] = pp_df.geometry.y
    #pp_df['name'] =  [ 'pp' + str(id) for id in pp_df['id'] ]
    pp_df['zone'] = 1
    pp_df['parval1'] = pp_df['y']
    return(pp_df)
    
def ppoints_to_file(pp_df,pp_file)
    pp_df.to_csv('./data/ppoints.csv',sep=' ',index = False, header=False, columns=['name','x','y','zone','parval1'])


# ----------------------------------------------------------------------------------------------------------
#Extraction from PyEMU 
#https://github.com/jtwhite79/pyemu/blob/develop/pyemu/
# ----------------------------------------------------------------------------------------------------------

PP_NAMES = ["name","x","y","zone","parval1"]

def fac2real(pp_file=None,factors_file="factors.dat",
             upper_lim=1.0e+30,lower_lim=-1.0e+30,fill_value=1.0e+30):
    """A python replication of the PEST fac2real utility for creating a
    structure grid array from previously calculated kriging factors (weights)
    Parameters
    ----------
    pp_file : (str)
        PEST-type pilot points file
    factors_file : (str)
        PEST-style factors file
    upper_lim : (float)
        maximum interpolated value in the array.  Values greater than
        upper_lim are set to fill_value
    lower_lim : (float)
        minimum interpolated value in the array.  Values less than lower_lim
        are set to fill_value
    fill_value : (float)
        the value to assign array nodes that are not interpolated
    Returns
    -------
    arr : numpy.ndarray
        if out_file is None
    out_file : str
        if out_file it not None
    Example
    -------
    ``>>>import pyemu``
    ``>>>pyemu.utils.geostats.fac2real("hkpp.dat",out_file="hk_layer_1.ref")``
    """

    if pp_file is not None and isinstance(pp_file,str):
        assert os.path.exists(pp_file)
        pp_data = pp_file_to_dataframe(pp_file)
        pp_data.loc[:,"name"] = pp_data.name.apply(lambda x: x.lower())
    elif pp_file is not None and isinstance(pp_file,pd.DataFrame):
        assert "name" in pp_file.columns
        assert "parval1" in pp_file.columns
        pp_data = pp_file
    else:
        raise Exception("unrecognized pp_file arg: must be str or pandas.DataFrame, not {0}"\
                        .format(type(pp_file)))
    assert os.path.exists(factors_file)
    f_fac = open(factors_file,'r')
    fpp_file = f_fac.readline()
    if pp_file is None and pp_data is None:
        pp_data = pp_file_to_dataframe(fpp_file)
        pp_data.loc[:, "name"] = pp_data.name.apply(lambda x: x.lower())

    fzone_file = f_fac.readline()
    ncol,nrow = [int(i) for i in f_fac.readline().strip().split()]
    npp = int(f_fac.readline().strip())
    pp_names = [f_fac.readline().strip().lower() for _ in range(npp)]

    # check that pp_names is sync'd with pp_data
    diff = set(list(pp_data.name)).symmetric_difference(set(pp_names))
    if len(diff) > 0:
        raise Exception("the following pilot point names are not common " +\
                        "between the factors file and the pilot points file " +\
                        ','.join(list(diff)))

    pp_dict = {int(name):val for name,val in zip(pp_data.index,pp_data.parval1)}
    try:
        pp_dict_log = {name:np.log10(val) for name,val in zip(pp_data.index,pp_data.parval1)}
    except:
        pp_dict_log = {}

    out_index = []
    out_vals = []
    while True:
        line = f_fac.readline()
        if len(line) == 0:
            break
        try:
            inode,itrans,fac_data = parse_factor_line(line)
        except Exception as e:
            raise Exception("error parsing factor line {0}:{1}".format(line,str(e)))
        if itrans == 0:
            fac_sum = sum([pp_dict[pp] * fac_data[pp] for pp in fac_data])
        else:
            fac_sum = sum([pp_dict_log[pp] * fac_data[pp] for pp in fac_data])
        if itrans != 0:
            fac_sum = 10**fac_sum
        out_vals.append(fac_sum)
        out_index.append(inode)

    df = pd.DataFrame(data={'vals':out_vals},index=out_index)

    return(df)


def parse_factor_line(line):
    """ function to parse a factor file line.  Used by fac2real()
    Parameters
    ----------
    line : (str)
        a factor line from a factor file
    Returns
    -------
    inode : int
        the inode of the grid node
    itrans : int
        flag for transformation of the grid node
    fac_data : dict
        a dictionary of point number, factor
    """

    raw = line.strip().split()
    inode,itrans,nfac = [int(i) for i in raw[:3]]
    fac_data = {int(raw[ifac])-1:float(raw[ifac+1]) for ifac in range(4,4+nfac*2,2)}
    return inode,itrans,fac_data


def pp_file_to_dataframe(pp_filename):

    """ read a pilot point file to a pandas Dataframe
    Parameters
    ----------
    pp_filename : str
        pilot point file
    Returns
    -------
    df : pandas.DataFrame
        a dataframe with pp_utils.PP_NAMES for columns
    """

    df = pd.read_csv(pp_filename, delim_whitespace=True,
                     header=None, names=PP_NAMES,usecols=[0,1,2,3,4])
    df.loc[:,"name"] = df.name.apply(str).apply(str.lower)
    return df


