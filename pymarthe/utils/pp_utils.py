
# import os 
# import numpy as np
# import pandas as pd


# PP_NAMES = ["name","x","y","zone","value"]
# PPFMT = lambda name, lay, zone, ppid: '{0}_l{1:02d}_z{2:02d}_{3:03d}'.format(name,int(lay+1),int(zone),int(ppid)) 


# # def ppoints_from_shp(shp_file, prefix, zone = 1, value = 1e-2, zone_field = None, value_field = None) :
# #     """
# #     Collect pilot points x, y and initial value 
# #     Parameters
# #     ----------

# #     Returns
# #     -------
    
# #     Example
# #     -------
# #     >>> ppoints_from_shp('points_pilotes_eponte2.shp', prefix = 'kepon_l02', zone_field='zone', value_field='value')
# #     """
# #     pp_df = geopandas.read_file(shp_file)
# #     pp_df['x'] = pp_df.geometry.x
# #     pp_df['y'] = pp_df.geometry.y

# #     # context dependent zone definition
# #     if 'zone' not in pp_df.columns :
# #         if zone_field is not None : 
# #             pp_df['zone'] = pp_df[zone_field]
# #         else :
# #             pp_df['zone'] = zone
# #     # context dependent initial value 
# #     if 'value' not in pp_df.columns :
# #         if value_field is not None : 
# #             pp_df['value'] = pp_df[value_field]
# #         else :
# #             pp_df['value'] = value

# #     # name pilot points
# #     pp_df['name'] =  [ '{0}_z{1:02d}_{2:03d}'.format(prefix,zone,id) for id,zone in zip(pp_df['id'],pp_df['zone']) ]
    
# #     # set index
# #     pp_df.set_index('name',inplace=True) 
# #     pp_df['name'] = pp_df.index 

# #     # extract only column selection and return a copy of pp_df
# #     return(pp_df[PP_NAMES].copy())
    
# def ppoints_to_file(pp_df,pp_file):
#     pp_df.to_csv('./data/ppoints.csv',sep=' ',index = False, header=False, columns=['name','x','y','zone','value'])

# # ----------------------------------------------------------------------------------------------------------
# #Extraction from PyEMU 
# #https://github.com/jtwhite79/pyemu/blob/develop/pyemu/
# # ----------------------------------------------------------------------------------------------------------


# def fac2real(pp_file=None,factors_file="factors.dat",
#              upper_lim=1.0e+30,lower_lim=-1.0e+30,fill_value=1.0e+30, kfac_sty='pyemu'):
#     """A python replication of the PEST fac2real utility for creating a
#     structure grid array from previously calculated kriging factors (weights)
#     Parameters
#     ----------
#     pp_file : (str)
#         PEST-type pilot points file
#     factors_file : (str)
#         PEST-style factors file
#     upper_lim : (float)
#         maximum interpolated value in the array.  Values greater than
#         upper_lim are set to fill_value
#     lower_lim : (float)
#         minimum interpolated value in the array.  Values less than lower_lim
#         are set to fill_value
#     fill_value : (float)
#         the value to assign array nodes that are not interpolated
#     Returns
#     -------
#     arr : numpy.ndarray
#         if out_file is None
#     out_file : str
#         if out_file it not None
#     Example
#     -------
#     ``>>>fac2real("hkpp.dat",out_file="hk_layer_1.ref")``
#     """
#     # read pp file if provided as argument
#     if pp_file is not None and isinstance(pp_file,str):
#         assert os.path.exists(pp_file)
#         pp_data = pp_file_to_dataframe(pp_file)
#         pp_data.loc[:,"name"] = pp_data.name.apply(lambda x: x.lower())
#     elif pp_file is not None and isinstance(pp_file,pd.DataFrame):
#         assert "name" in pp_file.columns
#         assert "value" in pp_file.columns
#         pp_data = pp_file
#     else:
#         raise Exception("unrecognized pp_file arg: must be str or pandas.DataFrame, not {0}"\
#                         .format(type(pp_file)))
#     assert os.path.exists(factors_file)
#     # read factor file 
#     f_fac = open(factors_file,'r')
#     fpp_file = f_fac.readline()
#     # get pp file if not provided as argument
#     if pp_file is None and pp_data is None:
#         pp_data = pp_file_to_dataframe(fpp_file)
#         pp_data.loc[:, "name"] = pp_data.name.apply(lambda x: x.lower())
#     # get zone file 
#     fzone_file = f_fac.readline()
#     if kfac_sty == 'pyemu' : 
#         # get ncol, nrow 
#         ncol,nrow = [int(i) for i in f_fac.readline().strip().split()]
#         # number of pilot points
#         npp = int(f_fac.readline().strip())
#         # pilot point names 
#         pp_names = [f_fac.readline().strip().lower() for _ in range(npp)]

#         # check that pp_names is sync'd with pp_data
#         diff = set(list(pp_data.name)).symmetric_difference(set(pp_names))
#         if len(diff) > 0:
#             raise Exception("the following pilot point names are not common " +\
#                             "between the factors file and the pilot points file " +\
#                             ','.join(list(diff)))
#     else :
#         trash_lines = [f_fac.readline() for _ in range(4)]
#     # mind that pp_data must have integer index
#     pp_dict = {int(name):val for name,val in zip(pp_data.index,pp_data.value)}
#     try:
#         pp_dict_log = {name:np.log10(val) for name,val in zip(pp_data.index,pp_data.value)}
#     except:
#         pp_dict_log = {}
#     out_index = []
#     out_vals = []
#     while True:
#         line = f_fac.readline()
#         if len(line) == 0:
#             break
#         try:
#             if kfac_sty =='pyemu':
#                 inode,itrans,fac_data = parse_factor_line(line)
#             else : 
#                 inode,fac_data = parse_factor_line_plproc(line)
#                 itrans = 1
#         except Exception as e:
#             raise Exception("error parsing factor line {0}:{1}".format(line,str(e)))
#         if itrans == 0:
#             fac_sum = sum([pp_dict[pp] * fac_data[pp] for pp in fac_data])
#         else:
#             fac_sum = sum([pp_dict_log[pp] * fac_data[pp] for pp in fac_data])
#         if itrans != 0:
#             fac_sum = 10**fac_sum
#         out_vals.append(fac_sum)
#         out_index.append(inode)

#     df = pd.DataFrame(data={'vals':out_vals},index=out_index)

#     return(df)


# def parse_factor_line(line):
#     """ function to parse a factor file line.  Used by fac2real()
#     Parameters
#     ----------
#     line : (str)
#         a factor line from a factor file
#     Returns
#     -------
#     inode : int
#         the inode of the grid node
#     itrans : int
#         flag for transformation of the grid node
#     fac_data : dict
#         a dictionary of point number, factor
#     """

#     raw = line.strip().split()
#     inode,itrans,nfac = [int(i) for i in raw[:3]]
#     fac_data = {int(raw[ifac])-1:float(raw[ifac+1]) for ifac in range(4,4+nfac*2,2)}
#     return inode,itrans,fac_data


# def parse_factor_line_plproc(line):
#     """ function to parse a factor file line.  Used by fac2real()
#     Parameters
#     ----------
#     line : (str)
#         a factor line from a factor file
#     Returns
#     -------
#     inode : int
#         the inode of the grid node
#     itrans : int
#         flag for transformation of the grid node
#     fac_data : dict
#         a dictionary of point number, factor
#     """

#     raw = line.strip().split()
#     inode,nfac = [int(i) for i in raw[:2]]
#     fac_data = {int(raw[ifac])-1:float(raw[ifac+1]) for ifac in range(3,3+nfac*2,2)}
#     return inode,fac_data

# def pp_file_to_dataframe(pp_filename):

#     """ read a pilot point file to a pandas Dataframe
#     Parameters
#     ----------
#     pp_filename : str
#         pilot point file
#     Returns
#     -------
#     df : pandas.DataFrame
#         a dataframe with pp_utils.PP_NAMES for columns
#     """

#     df = pd.read_csv(pp_filename, delim_whitespace=True,
#                      header=None, names=PP_NAMES,usecols=[0,1,2,3,4])
#     df.loc[:,"name"] = df.name.apply(str).apply(str.lower)
#     return df


