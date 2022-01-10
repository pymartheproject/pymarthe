"""
Contains the MartheGrid class
Handle single Marthe grid
"""

import numpy as np
import pandas as pd
import re
from matplotlib.path import Path

from . import shp_utils



class MartheGrid():

    """
    Class to handle Marthe single grid.
    """

    def __init__(self, layer, inest, nrow, ncol, xl, yl, dx, dy, xcc, ycc, array, field = None):

        """

        Single Marthe grid with all spatial caracteristics. 

        Parameters: 
        -----------
        layer (int) : layer number
        inest (int) : nested grid number
                      inest = 0 correspond to main grid.
        nrow (int) : number of grid rows
        ncol (int) : number of grid columns
        dx, dy (float) : x/y-resolution of the grid
        xl, yl (float) : x/y origin point (lower left corner)
        xcc, ycc (1Darray) : x/y cell centers
        array (2Darray) : gridded values
        field (str, optional) : field name 

        Example:
        -----------
        mygrid = MartheGrid(0, 0, 125, 126, 325., 750., dx, dy, xcc, ycc, array,  field = 'PERMEAB')

        """
        self.layer  = int(layer)
        self.inest  = int(inest)
        self.nrow   = int(nrow)
        self.ncol   = int(ncol)
        self.dx     = dx.astype(float)
        self.dy     = dy.astype(float)
        self.Lx     = np.sum(self.dx)
        self.Ly     = np.sum(self.dy)
        self.xl     = float(xl)
        self.yl     = float(yl)
        self.origin = (self.xl, self.yl)
        self.xcc    = xcc.astype(float)
        self.ycc    = ycc.astype(float)
        self.array  = array.astype(float)
        self.field  = '' if field is None else str(field)
        self.isnested = True if inest != 0 else False
        self.isregular = True if all(len(np.unique(a)) == 1 for a in [dx,dy]) else False
        self.isuniform = True if len(np.unique(array)) == 1 else False



    def to_records(self, base = 0):
        """
        Convert grid values to flatten recarray.

        Parameters:
        ----------
        self : MarthePump instance

        Returns:
        --------
        Write data inplace.
        (record files, listm files, pastp file)

        Examples:
        --------
        mp.set_data(value = 3, istep=[3,5])
        mp.write_data()
        """
        # ---- Get 2Darray of layer, inest, row, col, xcc, ycc, 
        rr, cc = np.meshgrid(*[np.arange(0 + base, n + base) for n in [self.nrow,self.ncol]])
        ij = np.dstack([rr.ravel('F'), cc.ravel('F')])[0]
        xx, yy = np.meshgrid(self.xcc, self.ycc)
        xy = np.dstack([xx.ravel('F'), yy.ravel('F')])[0]
        ln = np.dstack([i * np.ones((self.nrow, self.ncol)).ravel() for i in [self.layer, self.inest]])[0]

        # ---- Build recarray from arrays 
        arrays = list( np.column_stack([ln, ij, xy, self.array.ravel()]).T )
        rec = np.rec.fromarrays(arrays, names= 'layer,inest,i,j,x,y,value',
                                        formats=[*[int]*4,*[float]*3])

        # ---- Return recarray
        return rec


    def to_string(self, maxlayer=None, maxnest=None):

        """
        Convert grid to a single string
        with Marthe Grid file format.

        Parameters: 
        -----------
        maxlayer (int) : maximum number of layer.
        maxnest (int) : maximum number of nested grid.

        Return:
        -----------
        lines_str (str) : Marthe Grid string format
                          (ready to write)

        Example
        -----------
        mygrid = MartheGrid(0, 0, 125, 126, 325., 750., dx, dy, xcc, ycc, array,  field = 'PERMEAB')
        with open('mymarthegrid.prop', 'r') as f:
            f.write(mygrid.to_string())

        """
        # ---- Manage nested grid number as str 
        inest = str(self.inest) if self.inest > 0 else ' '
        maxl = '0' if maxlayer is None else str(maxlayer)
        maxn = '0' if maxnest is None else str(maxnest)

        # ---- Set main list with Marthe Grid first line
        lines = ['Marthe_Grid Version=9.0']

        # ---- Append headers
        lines.append('Title=Travail{}{} {}{}{}'.format(' '*62, inest, self.field,' '*12, str(self.layer+1)))
        lines.append('[Infos]')
        lines.append('Field={}'.format(str(self.field)))
        lines.append('Type=')
        lines.append('Elem_Number=0')
        lines.append('Name=')
        lines.append('Time_Step=-9999')
        lines.append('Time=0')
        lines.append('Layer={}'.format(str(self.layer+1)))
        lines.append('Max_Layer={}'.format(maxl))
        lines.append('Nest_grid={}'.format(str(self.inest)))
        lines.append('Max_NestG={}'.format(maxn))
        lines.append('[Structure]')
        lines.append('X_Left_Corner={}'.format(str(self.xl)))
        lines.append('Y_Lower_Corner={}'.format(str(self.yl)))
        lines.append('Ncolumn={}'.format(str(self.ncol)))
        lines.append('Nrows={}'.format(str(self.nrow)))
        lines.append('[Data_Descript]')
        lines.append('! Line 1       :   0   ,     0          , <   1 , 2 , 3 , Ncolumn   >')
        lines.append('! Line 2       :   0   ,     0          , < X_Center_of_all_Columns >')
        lines.append('! Line 2+1     :   1   , Y_of_Row_1     , < Field_Values_of_all_Columns > , Dy_of_Row_1')
        lines.append('! Line 2+2     :   2   , Y_of_Row_2     , < Field_Values_of_all_Columns > , Dy_of_Row_2')
        lines.append('! Line 2+Nrows : Nrows , Y_of_Row_Nrows , < Field_Values_of_all_Columns > , Dy_of_Row_2')
        lines.append('! Line 3+Nrows :   0   ,     0          , <     Dx_of_all_Columns   >')
        # ---- Append uniform data
        if self.isuniform:
            lines.append('[Constant_Data]')
            lines.append('Uniform_Value={}'.format(self.array.mean()))
            lines.append('[Columns_x_and_dx]')
            lines.append('\t'.join([str(i+1) for i in range(self.ncol)]))
            lines.append('\t'.join([str(i) for i in self.xcc]))
            lines.append('\t'.join([str(i) for i in self.dx]))
            lines.append('[Columns_y_and_dy]')
            lines.append('\t'.join([str(i+1) for i in range(self.nrow)]))
            lines.append('\t'.join([str(i) for i in self.ycc]))
            lines.append('\t'.join([str(i) for i in self.yx]))
        # ---- Append non uniform data
        else:
            lines.append('[Data]')
            lines.append('\t'.join(['0','0'] + [str(i+1) for i in range(self.ncol)]))
            lines.append('\t'.join(['0','0'] + [str(i) for i in self.xcc]))
            for i in range(self.nrow):
                line_data = [str(i+1), self.ycc[i], *[str(v) for v in self.array[i,:]], self.dy[i]]
                line_str = list(map(str,line_data))
                lines.append('\t'.join(line_str))
            lines.append('\t'.join(['0','0'] + [str(i) for i in self.dx]))
        # ---- Append end grid tag
        lines.append('[End_Grid]')
        # ---- Return all joined elements
        lines_str = '\n'.join(lines) + '\n'
        return lines_str



    def to_polygons(self):
        """
        Convert grid to list of polygons
        with vertices coordinates.
        Wrapper to shp_utils.get_polygons().

        Parameters: 
        -----------
        self (MartheGrid) : MartheGrid instance

        Return:
        -----------
        polygons (list) = polygons parts with 
                          xy-vertices coordinates.

        Example
        -----------
        polygons = mg.to_polygons()
        """
        # ---- Get grid polygons
        polygons = shp_utils.get_polygons(self.xcc, self.ycc, self.dx, self.dy)
        # ---- Return polygons as list
        return polygons



    def to_patches(self):
        """
        """
        patches = [Path(*p) for p in self.to_polygons()]
        return patches



    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheGrid'
