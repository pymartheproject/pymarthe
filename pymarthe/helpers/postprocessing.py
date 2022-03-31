"""
Contains some helper functions for Marthe model postprocessing

"""

import numpy as np
from copy import deepcopy
from pymarthe.mfield import MartheField, MartheFieldSeries




# def get_heads(mm, simfile=None, as_array=False):
#     """
#     """
#     mfs = MartheFieldSeries(mm, field='charge', simfile=simfile)
#     if as_array:
#         heads = np.squeeze( [ mf.data['value'].reshape(
#                                 (mm.nlay, mm.ncpl)) 
#                                     for i,mf in mfs.data.items()]
#                                     )
#         return heads
#     else:
#         return mfs




# def get_gradients(mm, heads=None, simfile=None, istep=None, masked_values = [9999,8888,-9999], as_array=False):
#     """
#     """
#     # -- Fetch head data as 3D-array (steps, layers, cells)
#     heads =  get_heads(mm, simfile, as_array=True) if heads is None else heads

#     # -- Get head data as masked array
#     hds = np.ma.array(heads, ndmin = 3, mask = np.isin(heads, masked_values))

#     # -- Manage istep to perform
#     _istep = list(range(mm.nstep)) if istep is None else marthe_utils.make_iterable(istep)

#     # -- Get top of each layer
#     try:
#         from pymarthe.utils.vtk_utils import get_top_botm
#     except:
#         ImportError('Could not import pymarthe.utils.vtk_utils.get_top_botm().')

#     top, botm = get_top_botm(mm, hws=mm.hws)


#     grad = []
#     for istep in _istep:
#         # -- Get head for istep
#         h = hds[istep]
#         # -- Set head values on unsaturated zone (unsat = z > h)
#         z = np.ma.array(top, mask=h.mask)
#         z[z > h] = h[z > h]

#         # ---- Apply .diff on data and mask components separately
#         diff_mask = np.diff(h.mask, axis=0)
#         dz = np.ma.array(np.diff(z.data, axis=0), mask=diff_mask)
#         dh = np.ma.array(np.diff(h.data, axis=0), mask=diff_mask)
#         # convert to 9999-filled array
#         g = np.concatenate( [ (dh / dz).filled(9999),
#                                np.ones((1, mm.ncpl))*9999 ]
#                                )
#         grad.append(g)

#     if as_array:
#         return np.squeeze(grad)
#     else:
#         ndig = len(str(len(grad)))
#         grad_fields= {}
#         for i, istep in enumerate(_istep):
#             rec = deepcopy(mm.imask.data)
#             rec['value'] = np.ravel(grad[i,:,:])
#             grad_fields[istep] = MartheField(field= 'gradient_' + str(i).zfill(ndig),
#                                              data=rec,
#                                              mm=mm)
#         return grad_fields



# # test
# mfs = get_heads(mm, simfile='chasim_cal_histo.out', as_array=False)
# heads = get_heads(mm, simfile='chasim_cal_histo.out', as_array=True)
# gradients = get_gradients(mm, heads = heads, as_array=True)
# grad_dic = get_gradients(mm, heads = heads, as_array=False)


