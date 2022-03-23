"""
Some usefull tools to handle geometry and vtk export.
This script was higly inspired by the flopy package.
https://github.com/modflowpy/flopy
"""


import os
import numpy as np
import warnings
warnings.simplefilter("always", DeprecationWarning)
from pymarthe.utils import marthe_utils, pest_utils



def gridlist_to_verts(gridlist):
    """
    Convert list of MartheGrid instance into vertices.


    Parameters:
    -----------
    gridlidt (list) : List of structured grid (MartheGrid instance).

    Returns:
    --------
    verts, iverts (nd.array,list) : vertices and list of cells which
                                    vertices comprise the cells

    Examples:
    -----------
    verts, iverts = mm.imask.to_grids(layer=0)

    """
    # -- Initialize vertices dictionary and number of cell
    vertdict = {}
    icell = 0
    # -- Iterate over structured grid (nested support)
    for mg in gridlist:
        # -- Iterate over each cell passing through rows and columns
        for i in range(mg.nrow):
            for j in range(mg.ncol):
                # -- Get cell vertices (closed coordinates)
                cv = mg.get_cell_vertices(i, j, closed=True)
                vertdict[icell] = cv
                icell += 1
    # -- Convert vertice dictionary to verts and iverts
    verts, iverts = to_cvfd(vertdict, verbose=False)
    # -- Return
    return verts, iverts



def area_of_polygon(x, y):
    """
    Calculates the signed area of an arbitrary polygon given its vertices
    http://stackoverflow.com/a/4682656/190597 (Joe Kington)

    Parameters:
    -----------
    x,y (it) : vertices coordinates

    Returns:
    --------
    area (float) : area of given polygon

    Examples:
    -----------
    x = [0,0,2,2]
    y = [0,2,2,0]
    area = area_of_polygon(x,y)
    """
    # -- Compute area (/!\ can be negative for clockwise order)
    area = 0.0
    for i in range(-1, len(x) - 1):
        area += x[i] * (y[i + 1] - y[i - 1])
    # -- Return area absolute value
    return abs(area / 2.0)



def centroid_of_polygon(points):
    """
    Compute the centroid coordinates of a given polygon

    Parameters:
    -----------
    points (it) : sequence of xy points coordinates

    Returns:
    --------
    result_x, result_y : xy-coordinates of centroid

    Examples:
    -----------
    points = [(30,50), (200,10), (250,50),
              (350,100), (200,180),(100,140)]
    centroid = centroid_of_polygon(points)
    """

    # -- Try to import itertools
    try:
        import itertools
    except:
        ImportError('Could not import `itertools` package. Try : `pip install itertools`.')

    # -- Find centroid
    area = area_of_polygon(*zip(*points))
    result_x = 0
    result_y = 0
    N = len(points)
    points = itertools.cycle(points)
    x1, y1 = next(points)
    for i in range(N):
        x0, y0 = x1, y1
        x1, y1 = next(points)
        cross = (x0 * y1) - (x1 * y0)
        result_x += (x0 + x1) * cross
        result_y += (y0 + y1) * cross
    result_x /= area * 6.0
    result_y /= area * 6.0
    return (result_x, result_y)




class Point:
    """
    Quick point definition class
    """
    def __init__(self, x, y):
        """
        Define point by giving cartesian coordinates
        """
        self.x = x
        self.y = y
        return


def is_between(a, b, c, epsilon=0.001):
    """
    Boolean response for point between other 2 points
    """
    # -- Test cross product
    crossproduct = (c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y)
    if abs(crossproduct) > epsilon:
        return False
    # -- Test dot product
    dotproduct = (c.x - a.x) * (b.x - a.x) + (c.y - a.y) * (b.y - a.y)
    if dotproduct < 0:
        return False

    # -- Test squared length
    squaredlengthba = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)
    if dotproduct > squaredlengthba:
        return False

    # -- Return otherwise
    return True



def shared_face(ivlist1, ivlist2):
    """
    Boolean response whatever 2 lists of vertices share face.
    """
    for i in range(len(ivlist1) - 1):
        iv1 = ivlist1[i]
        iv2 = ivlist1[i + 1]
        for i2 in range(len(ivlist2) - 1):
            if ivlist2[i2 : i2 + 1] == [iv2, iv1]:
                return True
    return False



def segment_face(ivert, ivlist1, ivlist2, vertices):
    """
    Check the vertex lists for cell 1 and cell 2. 
    Add a new vertex to cell 1 if necessary.

    Parameters:
    -----------
    iverts (int) : vertex id to check
    ivlist1 (list) : vertices for cell 1
    ivlist2 (list) : vertices for cell 2
    vertices (ndarray) : xy-vertices

    Returns:
    --------
    segmented (bool) : True if a face in cell 1 was split
                       up by adding a new vertex

    """

    # -- Go through ivlist1 and find faces that have ivert
    faces_to_check = []
    for ipos in range(len(ivlist1) - 1):
        face = (ivlist1[ipos], ivlist1[ipos + 1])
        if ivert in face:
            faces_to_check.append(face)

    # -- Go through ivlist2 and find points to check
    points_to_check = []
    for ipos in range(len(ivlist2) - 1):
        if ivlist2[ipos] == ivert:
            points_to_check.append(ivlist2[ipos + 1])
        elif ivlist2[ipos + 1] == ivert:
            points_to_check.append(ivlist2[ipos])

    for face in faces_to_check:
        iva, ivb = face
        x, y = vertices[iva]
        a = Point(x, y)
        x, y = vertices[ivb]
        b = Point(x, y)
        for ivc in points_to_check:
            if ivc not in face:
                x, y = vertices[ivc]
                c = Point(x, y)
                if is_between(a, b, c):
                    ipos = ivlist1.index(ivb)
                    if ipos == 0:
                        ipos = len(ivlist1) - 1
                    ivlist1.insert(ipos, ivc)
                    return True

    return False




def to_cvfd(vertdict, nodestart=None, nodestop=None,
            skip_hanging_node_check=False, verbose=False):
    """
    Convert a vertex dictionary into verts and iverts

    Parameters:
    -----------
    vertdict (dict) : vertices dictionary.
                      Format : {icell: [(x1, y1), (x2, y2), (x3, y3), ...]}
    nodestart (int) : starting node number.
                      Default is 0.
    nodestop (int) : ending node number up to but not including.
                     Default is len(vertdict).
    skip_hanging_node_check (bool) : skip the hanging node check.
                                     Only be necessary for quad-based grid
                                     refinement (nested model).
                                     Default is False.
    verbose (bool) : print messages to the screen.
                     Default is False

    Returns:
    --------

    verts (ndarray) : xy-vertices
    iverts (list) : vertice ids for each cell
    """
    # -- Manage node start/stop
    nodestart = 0 if nodestart is None else nodestart
    nodestop = len(vertdict) if nodestop is None else nodestop
    ncells = nodestop - nodestart

    # -- First create vertexdict {(x1, y1): ivert1, (x2, y2): ivert2, ...} and
    #    vertexlist [[ivert1, ivert2, ...], [ivert9, ivert10, ...], ...]
    #    In the process, filter out any duplicate vertices

    vertexdict = {}
    vertexlist = []
    xcyc = np.empty((ncells, 2), dtype=float)
    iv = 0
    nvertstart = 0
    if verbose:
        print("Converting vertdict to cvfd representation.")
        print(f"Number of cells in vertdict is: {len(vertdict)}")
        print(f"Cell {nodestart} up to {nodestop} will be processed.")

    for icell in range(nodestart, nodestop):
        points = vertdict[icell]
        nvertstart += len(points)
        xc, yc = centroid_of_polygon(points)
        xcyc[icell, 0] = xc
        xcyc[icell, 1] = yc
        ivertlist = []
        for p in points:
            pt = tuple(p)
            if pt in vertexdict:
                ivert = vertexdict[pt]
            else:
                vertexdict[pt] = iv
                ivert = iv
                iv += 1
            ivertlist.append(ivert)
        if ivertlist[0] != ivertlist[-1]:
            raise Exception(f"Cell {icell} not closed")
        vertexlist.append(ivertlist)

    # -- Next create a vertex_cell_dict for each vertex, store list of cells
    #    that use it
    nvert = len(vertexdict)
    if verbose:
        print(f"Started with {nvertstart} vertices.")
        print(f"Ended up with {nvert} vertices.")
        print(f"Reduced total number of vertices by {nvertstart - nvert}")
        print("Creating dict of vertices with their associated cells")
    vertex_cell_dict = {}
    for icell in range(nodestart, nodestop):
        ivertlist = vertexlist[icell]
        for ivert in ivertlist:
            if ivert in vertex_cell_dict:
                if icell not in vertex_cell_dict[ivert]:
                    vertex_cell_dict[ivert].append(icell)
            else:
                vertex_cell_dict[ivert] = [icell]
    if verbose:
        print("Done creating dict of vertices with their associated cells")

    # -- Now, go through each vertex and look at the cells that use the vertex.
    #    For quadtree-like grids, there may be a need to add a new hanging node
    #    vertex to the larger cell.
    if not skip_hanging_node_check:
        if verbose:
            print("Checking for hanging nodes.")
        vertexdict_keys = list(vertexdict.keys())
        finished = False
        while not finished:
            finished = True
            for ivert, cell_list in vertex_cell_dict.items():
                for icell1 in cell_list:
                    for icell2 in cell_list:

                        # skip if same cell
                        if icell1 == icell2:
                            continue

                        # skip if share face already
                        ivertlist1 = vertexlist[icell1]
                        ivertlist2 = vertexlist[icell2]
                        if shared_face(ivertlist1, ivertlist2):
                            continue

                        # don't share a face, so need to segment if necessary
                        segmented = segment_face(
                            ivert, ivertlist1, ivertlist2, vertexdict_keys
                        )
                        if segmented:
                            finished = False
        if verbose:
            print("Done checking for hanging nodes.")

    verts = np.array(vertexdict_keys)
    iverts = vertexlist
    # -- Return
    return verts, iverts




def get_top_botm(mm, hws='implicit'):
    """
    Function to extract altitude of top and bottom
    of each layer of a MartheModel instance
    """
    # -- Infer/set nlay, ncpl, mv
    nlay = mm.nlay
    ncpl = mm.ncpl
    mv = [9999,8888,0,-9999]

    # -- Load basic geometry field if not already
    #    and reshape field value as (nlay, ncpl)
    #    (Marthe masked values will be set to nan)
    geom_arrs = []
    for g, mf in mm.geometry.items():
        if mf is None:
            mm.load_geometry(g)
            mf = mm.geometry[g]
        arr = mf.data['value'].reshape((nlay,ncpl))
        arr[np.isin(arr, mv)] = np.nan
        geom_arrs.append(arr)

    _sepon, _topog, _hsubs = geom_arrs

    # -- Infer altitude of each cell top for explicit
    #    hanging wall : top[i] = hsubs[i-1]
    #                   with top[0] = topog[0]
    if hws == 'explicit':
        _top = np.vstack((_topog[0], _hsubs[1:]))

    # -- Infer altitude of each cell top for implicit
    #    hanging wall : top[i] = altitude minimum above hsubs[i]
    if hws == 'implicit':
        _top = np.empty((nlay,ncpl))    # empty top array
        _outcrop = mm.get_outcrop().data['value'].reshape((nlay,ncpl)) # outcrop array

        for icell in range(ncpl):
            marthe_utils.progress_bar((icell+1)/ncpl)
            for ilay in range(nlay):
                # -- Compute z minimum above substratum
                ztop = np.fmin.reduce(
                            np.concatenate( [ _hsubs[:ilay,icell],
                                              _sepon[:ilay+1,icell],
                                              _topog[:1,icell]       ] )
                                                )
                # -- If current cell's top is outcroping
                if np.isnan(ztop):
                    oc = np.fmax.reduce(_outcrop[:ilay+1,icell])
                    if np.isnan(oc):
                        ztop = np.nan
                    elif ilay <= int(oc):
                        ztop = _hsubs[int(oc), icell]
                # -- Store cell top
                _top[ilay,icell] = ztop

        # -- Rectify first layer by topog
        _top = np.vstack((_topog[0], _top[1:]))

    # -- Return top and botm arrays
    return _top, _hsubs









class Vtk:
    """
    Class to build and manage unstructured vtk grid.
    """

    def __init__(self,
                 mm, vertical_exageration=0.05,
                 hws = 'implicit', smooth=False,
                 binary=True, xml=False,
                 shared_points=False):
        """
        Initialize Vtk class 

        Parameters:
        -----------
        mm (dict) : MartheModel instance (must contain default .imask)
        hws (str) : hanging wall state, flag to define whatever the superior
                    hanging walls of the model are defined as normal layers
                    (explivitly) or not (implicitly).
                    Can be:
                        - 'implicit'
                        - 'explicit'
                    Default is 'implicit'.
        vertical_exageration (float) : floating point value to scale vertical
                                       exageration of the vtk points.
                                       Default is 0.05.
        smooth (bool) : boolean flag to enable interpolating vertex elevations
                        based on shared cell.
                        Default is False.
        binary (bool) : Enable binary writing, otherwise classic ASCII format 
                        will be consider.
                        Default is True.
                        Note : binary is prefered as paraview can produced bug
                               representing NaN values from ASCII (non xml) files.
        xml (bool) : Enable xml based VTK files writing.
                     Default is False.
        shared_points (bool) : Enable sharing points in grid polyhedron construction.
                               Default is False.
                               Note : False is prefered as paraview has a bug (8/4/2021)
                                      where some polyhedron will not properly close when
                                      using shared points.

        Example:
        --------

        myvtk = Vtk(mm, vertical_exageration=0.02, smooth=True)

        """
        # -- Handle vtk package import issues
        try:
            import vtk
        except ImportError as error:
            print("Could not load `vtk` package.")

        # -- Set basic attributs
        self.mm = mm
        self.vertical_exageration = vertical_exageration
        self.hws = hws
        self.smooth = smooth
        self.binary = binary
        self.xml = xml
        self.shared_points = shared_points

        # -- Infered other usefull attributs
        print('Collecting model grid vertices ...')
        self.verts, self.iverts = gridlist_to_verts(self.mm.imask.to_grids(layer=0))
        self.nnodes = len(self.mm.imask.data)

        nvpl = 0
        for iv in self.iverts:
            nvpl += len(iv) - 1
        self.nvpl = nvpl                # Number of vertice per layer
        self.ncpl = len(self.iverts)    # Number of cell per layer
        self.nlay = self.mm.nlay        # Number of layer
        self._mv = [9999,8888,0,-9999]  # Marthe values to mask in grids
        self._active =  [1]*self.nlay   # Set all layers as active
        #self.idomain = self.mm.imask.data['value']

        # -- Advance extraction of top and botm of each layer
        #    according to hanging wall state
        print(f'Extracting model {self.hws} top/bottom for each layer ...')
        self.top, self.botm = get_top_botm(self.mm, hws=self.hws)

        # -- Build grid geometry
        print('\nBuilding 3D geometry ...')
        self.points = []
        self.faces = []
        self._build_grid_geometry()
        # -- Set vtk grid geometry
        print('\nBuilding vtk unstructured grid geometry ...')
        self.vtk_grid = None
        self._vtk_geometry_set = False
        self.__vtk = vtk
        self._set_vtk_grid_geometry()


    def _create_smoothed_elevation_graph(self, adjk):
        """
        Method to create a dictionary of shared point
        mean smoothed elevations

        Parameters:
        -----------
        adjk (int) : confining bed adjusted layer

        Returns:
        ----------
        elevation (dict) : {vertex number: elevation}
        """
        elevations = {}
        for i, iv in enumerate(self.iverts):
            iv = iv[1:]
            for v in iv:
                if v is None:
                    continue
                zv = self.top[adjk][i] * self.vertical_exageration
                if v in elevations:
                    elevations[v].append(zv)
                else:
                    elevations[v] = [zv]
        for key in elevations:
            elevations[key] = np.mean(elevations[key])

        return elevations




    def _build_grid_geometry(self):
        """
        Method that creates lists of vertex points and cell faces
        """
        # -- Initialize variables
        points = []
        faces = []
        v0 = 0
        v1 = 0
        ncb = 0
        shared_points = self.shared_points
        # -- Set share points to False if required
        if len(self._active) != self.nlay:
            shared_points = False
        # -- Iterate over layer
        for k in range(self.nlay):
            marthe_utils.progress_bar((k+1)/self.nlay)
            # -- Update adjk
            adjk = k + ncb
            if k != self.nlay - 1:
                if self._active[adjk + 1] == 0:
                    ncb += 1
            # -- Subset top with current layer
            top = self.top[k]
            # -- Get smooth elevation if required
            if self.smooth:
                elevations = self._create_smoothed_elevation_graph(adjk)
            # -- iterate over all vertices
            for i, iv in enumerate(self.iverts):
                iv = iv[1:]
                # -- Iterate over set of vertices
                for v in iv:
                    if v is None:
                        continue
                    # -- Get xy-coordinate
                    xv = self.verts[v, 0]
                    yv = self.verts[v, 1]
                    # -- Get z coordinate (smoothed if required)
                    if self.smooth:
                        zv = elevations[v]
                    else:
                        zv = top[i]  * self.vertical_exageration
                    # -- Store points
                    points.append([xv, yv, zv])
                    v1 += 1
                # -- Get cell faces 
                cell_faces = [ [v for v in range(v0, v1)],
                               [v + self.nvpl for v in range(v0, v1)] ]

                for v in range(v0, v1):
                    if v != v1 - 1:
                        cell_faces.append([v + 1, v, v + self.nvpl, v + self.nvpl + 1])
                    else:
                        cell_faces.append([v0, v, v + self.nvpl, v0 + self.nvpl])

                v0 = v1
                # -- Store cell faces
                faces.append(cell_faces)

            # -- Manage last layer and not shared point
            if k == self.nlay - 1 or not shared_points:
                if self.smooth:
                    elevations = self._create_smoothed_elevation_graph(adjk)
                for i, iv in enumerate(self.iverts):
                    iv = iv[1:]
                    for v in iv:
                        if v is None:
                            continue
                        xv = self.verts[v, 0]
                        yv = self.verts[v, 1]
                        if self.smooth:
                            zv = elevations[v]
                        else:
                            zv = self.botm[adjk][i] * self.vertical_exageration

                        points.append([xv, yv, zv])
                        v1 += 1

                v0 = v1
        # -- Store all poinst and faces
        self.points = points
        self.faces = faces



    def _set_vtk_grid_geometry(self):
        """
        Method to set vtk's geometry and add it to the vtk grid object
        """
        # -- Verify if vtk geometry already provided
        if self._vtk_geometry_set:
            return

        # -- Set geometry if not already provided
        if not self.faces:
            self._build_grid_geometry()

        # -- Initialize grid as unstructured
        self.vtk_grid = self.__vtk.vtkUnstructuredGrid()

        # -- Insert each point as vtk geometry
        points = self.__vtk.vtkPoints()
        for point in self.points:
            points.InsertNextPoint(point)
        self.vtk_grid.SetPoints(points)

        # -- Insert cell by giving id faces
        for node in range(self.nnodes):
            cell_faces = self.faces[node]
            nface = len(cell_faces)
            fid_list = self.__vtk.vtkIdList()
            fid_list.InsertNextId(nface)
            for face in cell_faces:
                fid_list.InsertNextId(len(face))
                [fid_list.InsertNextId(i) for i in face]
            self.vtk_grid.InsertNextCell(self.__vtk.VTK_POLYHEDRON, fid_list)
        # -- Flag to inform that geometry has been added
        self._vtk_geometry_set = True




    def _mask_values(self, array, masked_values = None):
        """
        Method to mask values in array with nan

        Parameters:
        -----------
        array (ndarray) : values in array
        masked_values (float/it) : values to convert to nan
                                   Default are [9999, 8888, 0, -9999].

        Returns:
        --------
        array (ndarray) : array with nan
        """
        mv = [] if masked_values is None else marthe_utils.make_iterable(masked_values)
        array[np.isin(array, mv)] = np.nan
        return array



    def add_array(self, array, name, trans='none', masked_values=None, dtype=None):
        """
        Method to set an array to the vtk grid.
        It will apply a value for each cell

        Parameters:
        -----------
        array (ndarray) : values in array.
                        Note: must be same same as nnodes.
        name (str) : array name for vtk
        trans (str) : transformation to apply to the values.
                      See pymarthe.utils.pest_utils.transform.
                      Default is 'none'.
        masked_values (float/it) : values to mask by converting to nan
                                   Default are [9999, 8888, 0, -9999].
        dtype (vtk datatype) : method to supply and force a vtk datatype

        Returns:
        --------
        Set array data in self.vtk_grid inplace

        Example:
        --------
        mv = [9999,8888,0,-9999]
        array = mm.prop['permh'].data['value']
        myvtk.add_array(array, 'permh', trans='log10', masked_values=mv)

        """
        # -- Dynamic import of vtk numpy utils
        from vtk.util import numpy_support

        # -- Build vtk geometry if not yet provided

        if not self._vtk_geometry_set:
            self._set_vtk_grid_geometry()

        # -- Get flat array not yet
        array = np.ravel(array)

        # -- Assert good dimension array
        err_msg = f"ERROR: `array` must have size = {self.nnodes}" \
                  f" (same as model cells). Given {array.size}."
        assert array.size == self.nnodes, err_msg

        # -- Convert masked values to NaN and apply transformation
        array = pest_utils.transform(
                        self._mask_values(array, masked_values),
                        trans
                        ).to_numpy()

        # -- Convert back to single masked value
        #    (NaN values not supporting by ASCII)
        if not self.binary and not self.xml:
            array = np.nan_to_num(array, nan=1e30)

        # -- Manage vtk dtype
        if dtype is None:
            # -- Set default as float
            dtype = self.__vtk.VTK_FLOAT
            # -- Set dtype to int if required
            if np.issubdtype(array[0], np.dtype(int)):
                dtype = self.__vtk.VTK_INT

        # -- Convert numpy array to vtk array
        vtk_arr = numpy_support.numpy_to_vtk(num_array=array, array_type=dtype)
        vtk_arr.SetName(name)

        # -- Broadcast array to cells
        self.vtk_grid.GetCellData().AddArray(vtk_arr)



    def _get_writer(self):
        """
        Get adequate writer and file extension according 
        to .xlm and .binary attribut.
        """

        if self.xml:
            ext = '.vtu'
            if self.binary:
                writer = self.__vtk.vtkXMLUnstructuredGridWriter()
                writer.SetDataModeToBinary()
            else:
                writer = self.__vtk.vtkXMLUnstructuredGridWriter()
                writer.SetDataModeToAscii()
        else:
            ext = '.vtk'
            if self.binary:
                writer = self.__vtk.vtkUnstructuredGridWriter()
                writer.SetFileTypeToBinary()
            else:
                writer = self.__vtk.vtkUnstructuredGridWriter()
                writer.SetFileTypeToASCII()

        return writer, ext



    def write(self, filename):
        """
        Method to write a unstructured grid from the VTK object

        Parameters:
        -----------
        filename (str) : vtk file name to write without extension.
                         Extension will be inferred 

        """
        # -- Get required writer
        w, ext = self._get_writer()
        # -- Set ugrid and filename
        w.SetInputData(self.vtk_grid)
        w.SetFileName(filename + ext)
        # -- Finally, update and write
        w.Update()
        w.Write()
        print(f'Unstructured grid successfully written in {filename+ext}.')


