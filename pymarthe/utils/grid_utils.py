"""
Contains the MartheGrid class
Handle single Marthe grid
"""

import numpy as np
import pandas as pd
import re
from matplotlib.path import Path

from . import shp_utils, marthe_utils



class MartheGrid():

    """
    Class to handle Marthe single grid.
    """

    def __init__(self, istep, layer, inest, nrow, ncol, xl, yl, dx, dy, xcc, ycc, array, field = None):

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
        # -- Store original args
        self.field  = '' if field is None else str(field)
        self.istep  = int(istep)
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
        self.xvertices = np.append(np.array(self.xl), self.xl + np.cumsum(self.dx))
        self.yvertices = np.append(np.array(self.yl), self.yl + np.cumsum(self.dy))
        self.isnested = True if inest != 0 else False
        self.isregular = True if all(len(np.unique(a)) == 1 for a in [dx,dy]) else False
        nvalues = len(np.unique(array[~np.isin(array, [-9999,0,8888,9999])]))
        self.isuniform = False if nvalues > 1 else True


    def get_cell_vertices(self, i, j, closed=False):
        """
        DOES NOT WORK PROPERLY!
        """
        x0, x1 = self.xcc[j] - self.dx[j]/2, self.xcc[j] + self.dx[j]/2
        y0, y1 = self.ycc[i] - self.dy[i]/2, self.ycc[i] + self.dy[i]/2
        vertices = [[x0,y0],[x0,y1],[x1,y1],[x1,y0]]
        if closed:
            vertices.append([x0,y0])
        return vertices



    def to_records(self, fmt='light', base = 0):
        """
        Convert grid values to flatten recarray.

        Parameters:
        ----------
        fmt (str) : output format.
                       Can be:
                        - 'light' : output columns: 'layer,inest,i,j,x,y,value'
                        - 'full'  : output columns: 'node,layer,inest,i,j,x,y,dx,dy,vertices,value'
                        Default is 'light'.
                        Note: the 'full' format is obviously slower.

        base (int) : n-based array format
                     Can be :
                        - 0 (Python)
                        - 1 (Fortran)
                     Default is 0.

        Returns:
        --------
        Write data inplace.
        (record files, listm files, pastp file)

        Examples:
        --------
        mg.to_records(fmt='light')
        """
        rows, cols = [np.arange(0 + base, n + base) for n in [self.nrow,self.ncol]]
        ii, jj = np.meshgrid(rows, cols, indexing='ij')
        xx, yy = np.meshgrid(self.xcc, self.ycc, indexing='xy')
        ll = self.layer * np.ones((self.nrow, self.ncol))
        nn = self.inest * np.ones((self.nrow, self.ncol))
        array = self.array
        dt = [('layer', '<i8'), ('inest', '<i8'),
              ('i', '<i8'), ('j', '<i8'),
              ('x', '<f8'), ('y', '<f8')]

        # ---- Manage 'light' recarray
        if fmt == 'light':
            # -- Collect data infos
            data = [ll,nn,ii,jj,xx,yy,array]
            it = list(map(np.ravel, data))
            dt.append(('value', '<f8'))
            # -- Build recarray from arrays
            rec = np.rec.fromarrays(it, dtype=dt)
            # -- Return rec.array
            return rec

        # ---- Manage 'full' recarray
        elif fmt == 'full':
            # -- Extract cell sizes and area
            dxx, dyy = np.meshgrid(self.dx, self.dy, indexing= 'xy')
            area = dxx*dyy
            # -- Extract vertices
            vxy = list(map(np.ravel, [xx-dxx/2, xx+dxx/2, yy-dyy/2, yy+dyy/2]))
            vertices = [ [ [x0,y0],[x0,y1],[x1,y1],[x1,y0] ]
                                for x0,x1,y0,y1 in zip(*vxy)]
            # -- Collect data infos
            data = [ll,nn,ii,jj,xx,yy,dxx,dyy,area,array]
            it = list(map(np.ravel, data))
            dt.extend([('dx', '<f8'), ('dy', '<f8'),('area', '<f8'), ('value', '<f8')])
            df = pd.DataFrame(np.transpose(it), columns=np.dtype(dt).names).astype(dict(dt))
            df.insert(len(dt) -1, 'vertices', vertices)
            # -- return rec.array
            return df.to_records(index=False)



    def to_string(self, maxlayer=None, maxnest=None, rlevel=None, keep_uniform_fmt=False):

        """
        Convert grid to a single string
        with Marthe Grid file format.
        Parameters: 
        -----------
        maxlayer (int/None, optional) : maximum number of layer.
                                        If None, value will be '0'.
                                        Default is None.
        maxnest (int/None, optional) : maximum number of nested grid.
                                       If None, value will be '0'.
                                       Default is None.
        rlevel (int/None, optional) : refine level to extract adjacent
                                      cells informations.
                                      Default is None.
        keep_uniform_fmt (bool, optional) : whatever to conserve marthe light format
                                            for uniform grid.
                                            If True, light format will be conserved.
                                            If False, all grids will be written explictly.
                                            Default is False.
                                            /!/ CAREFULL /!/ keeping uniform light format
                                            on `permh` field can modify the model geometry.

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
        # ---- Add adjecent cell informations if required
        if rlevel is None:
            nrow, ncol, xl, yl = self.nrow, self.ncol, self.xl, self.yl
            dx, dy, xcc, ycc = self.dx, self.dy, self.xcc, self.ycc
            array = self.array
        else:
            dx = np.r_[ self.dx[0]*rlevel, self.dx, self.dx[-1]*rlevel]
            dy = np.r_[ self.dy[0]*rlevel, self.dy, self.dy[-1]*rlevel]
            xcc = np.r_[ self.xcc[0] - (dx[1] + dx[0])/2, self.xcc, self.ycc[-1] + (dx[-2] + dx[-1])/2]
            ycc = np.r_[ self.ycc[0] + (dy[1] + dy[0])/2, self.ycc, self.ycc[-1] - (dy[-2] + dy[-1])/2] # reversed direction
            xl, yl = xcc.min() - (dx[0]/2), ycc.min() - (dy[0]/2)
            array = marthe_utils.bordered_array(self.array, 0) # set bordered array with value 0
            nrow, ncol = array.shape

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
        lines.append('X_Left_Corner={}'.format(xl))
        lines.append('Y_Lower_Corner={}'.format(yl))
        lines.append('Ncolumn={}'.format(ncol))
        lines.append('Nrows={}'.format(nrow))
        if np.logical_and(self.isuniform, keep_uniform_fmt):
            uv = np.unique(array[~np.isin(array,[-9999,0,8888,9999])])
            uniform_value = 0 if len(uv) == 0 else uv[0]
            # uniform_value = 0 if np.isnan(uniform_value) else uniform_value            
            lines.append('[Constant_Data]')
            lines.append('Uniform_Value={}'.format(uniform_value))
            lines.append('[Columns_x_and_dx]')
            lines.append('\t'.join([str(i+1) for i in range(ncol)]))
            lines.append('\t'.join([str(i) for i in xcc]))
            lines.append('\t'.join([str(i) for i in dx]))
            lines.append('[Columns_y_and_dy]')
            lines.append('\t'.join([str(i+1) for i in range(nrow)]))
            lines.append('\t'.join([str(i) for i in ycc]))
            lines.append('\t'.join([str(i) for i in dy]))
        # ---- Append non uniform data
        else:
            lines.append('[Data_Descript]')
            lines.append('! Line 1       :   0   ,     0          , <   1 , 2 , 3 , Ncolumn   >')
            lines.append('! Line 2       :   0   ,     0          , < X_Center_of_all_Columns >')
            lines.append('! Line 2+1     :   1   , Y_of_Row_1     , < Field_Values_of_all_Columns > , Dy_of_Row_1')
            lines.append('! Line 2+2     :   2   , Y_of_Row_2     , < Field_Values_of_all_Columns > , Dy_of_Row_2')
            lines.append('! Line 2+Nrows : Nrows , Y_of_Row_Nrows , < Field_Values_of_all_Columns > , Dy_of_Row_2')
            lines.append('! Line 3+Nrows :   0   ,     0          , <     Dx_of_all_Columns   >')
        # ---- Append uniform data
            lines.append('[Data]')
            lines.append('\t'.join(['0','0'] + [str(i+1) for i in range(ncol)]))
            lines.append('\t'.join(['0','0'] + [str(i) for i in xcc]))
            for i in range(nrow):
                line_data = [i+1, ycc[i], *array[i,:], dy[i]]
                line_str = list(map(str,line_data))
                lines.append('\t'.join(line_str))
            lines.append('\t'.join(['0','0'] + [str(i) for i in dx]))
        # ---- Append end grid tag
        lines.append('[End_Grid]')
        # ---- Return all joined elements
        lines_str = '\n'.join(lines) + '\n'
        return lines_str




    def to_pyshp(self):
        """
        Convert grid to list of polygons
        with vertices coordinates.

        Parameters: 
        -----------
        self (MartheGrid) : MartheGrid instance

        Return:
        -----------
        pyshp_parts (list) = polygons parts with 
                             xy-vertices coordinates.

        Example
        -----------
        parts = mg.to_pyshp()
        """
        # ---- Get grid polygons
        pyshp_parts = shp_utils.get_parts(self.xcc, self.ycc, self.dx, self.dy)
        # ---- Return polygons as list
        return pyshp_parts



    def to_patches(self):
        """
        Convert grid to list of matplotlib.Path

        Parameters: 
        -----------
        self (MartheGrid) : MartheGrid instance

        Return:
        -----------
        patches (list) = list of Path objects

        Example
        -----------
        patches = mg.to_patches()
        """
        patches = [Path(*p) for p in self.to_pyshp()]
        return patches



    def __str__(self):
        """
        Internal string method.
        """
        return 'MartheGrid'





