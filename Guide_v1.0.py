'''
PYMARTHE DEV Version 1.0 Guide

Get started with new 1.0 guide.
Give examples of some basic commands.
'''

# ---- Put your dev branch path here
import os, sys
#dev_ws = os.path.normpath(r"E:\EauxSCAR\pymarthe_dev")

# ---- Import usefull modules

import pandas as pd
import numpy as np
#sys.path.append(dev_ws)
from pymarthe import MartheModel
from pymarthe.utils import marthe_utils, shp_utils, pest_utils
from pymarthe.mfield import MartheField, MartheFieldSeries
from pymarthe.mpump import MarthePump
from pymarthe.msoil import MartheSoil
from pymarthe.moptim import MartheOptim
import matplotlib.pyplot as plt

# 1) --->  MONA MODEL <---

# --------------------------------------------------------
# ---- MartheModel instance

# -- Load mona.rma 
mona_ws = os.path.join('monav3_pm', 'mona.rma')
mm = MartheModel(mona_ws)


'''
The MartheModel instance provided the creation of a spatial index based on `rtree` 
package. It will create a unique id for each model cell and store some usefull
related data about the cell such as:
    - 'node' : cell unique id
    - 'layer': layer id
    - 'inest': nested grid id
    - 'i'    : row number
    - 'j'    : column number
    - 'xcc'  : x-coordinate of the cell centroid
    - 'ycc'  : y-coordinate of the cell centroid
    - 'dx'   : cell width
    - 'dy'   : cell height
    - 'area' : cell area
    - 'vertices': cell vertices (closed)
    - 'ative': cell activity (0=inactive, 1=active)
The created spatial index will be stored as 2 external files:
    - .idx : binary file containing cell ids
    - .dat : text file containing related cell data
The management of the index creation pass through the argument `spatial_index`.
It can be 
    - True      : generate generic spatial index (mlname_si.idx/.dat).
    - False     : disable spatial index creation.
    - filename  : path to an existing spatial index to read.
    - dic       : generate spatial index with custom options.
The custom form to implement spatial index can contains keys like 
    - 'name' (str) : 
            custom name to external spatial index files
    - 'only_active' (bool)
            disable the insertion of inactive cells in 
            the spatial index. This can be usefull for
            fast spatial processings on valid large 
            models (especially with nested grids).
            Careful, some processes could be affected 
            and not working as usal.
Build a spatial index can be slow for large Marthe model with several cells,
layer and nested grid, a progress bar will be shown on the terminal to 
appreciate the spatial index creation process.
Let's have a look on the spatial index management from MartheModel class.
'''
# -- Load mona.rma with spatial index
# Default (light process)
mm = MartheModel(mona_ws, spatial_index = False)
# Generic spatial index (filename = modelname_si)
mm = MartheModel(mona_ws, spatial_index = True)
# Custom filename spatial index
custom_si = {'name': mona_ws.replace('.rma', '_custom_si'),
             'only_active' : True}
mm = MartheModel(mona_ws, spatial_index = custom_si)
# From an existing spatial index
sifile = mona_ws.replace('.rma', '_si')
mm = MartheModel(mona_ws, spatial_index = sifile)

'''
The MartheModel can also store grid cell informations in a large DataFrame
passing the `modelgrid` argument to True.
This DataFrame will be stored in `modelgrid` attribute. It can be usefull
to perform some queries on grid data with the high level integrated method
named `.query_grid()`. This method allows the user to pass some 'query'
variables and the target required grid informations to extract (=columns).
The `.query_grid()` perform some checking on query variables and targets to
avoid invalid inputs.
Note: if a spatial index was already provided, the modelgrid DataFrame will
      be created from it (faster), otherwise it use the `.imask` (slower). 
Let's load the model with modelgrid.
'''
sifile = mona_ws.replace('.rma', '_si')
mm = MartheModel(mona_ws, spatial_index = sifile, modelgrid=True)
mm.modelgrid.head()

# -- Invalid query examples
# Invalid target columns names
mm.query_grid(node=[45,678,3578], target=['i','j','DX','verts'])
# Invalid query variable names
mm.query_grid(NoDeS=[45,678,3578], target=['i','j'])
# Invalid length of query variables
mm.query_grid(i = [34,67], j = [45,65], layer=2)

# -- Example of some valid grid queries
# Get cell vertices from cell nodes
mm.query_grid(node=[45,678,3578], target='vertices')
# Get cell xy resolution from row,column,layer 
mm.query_grid(i=34,j=65,layer=4, target=['dx','dy'])
# Get all active nodes of the first 2 layers
mm.query_grid(layer=[0,1], active = [1,1] , target='node')
# Get superficie of layer 4
mm.query_grid(layer=4, active = 1)['area'].sum()


'''
The .imask attribute based on permh property correspond to a simple MartheField 
delimiting aquifer extensions by binary values (0 inactive cell, 1 active cell).
'''
mm.imask


'''
Let's begin with some of additional attributes of this brand new 1.0 version of pymarthe
'''
# -- Model name and path
print(f'Model directory :\t {mm.mldir}')
print(f'Model name :\t {mm.mlname}')

# -- Number of timestep and  nested grids
print(f'Number of timestep :\t {mm.nstep}')
print(f'Number of nested grids :\t {mm.nnest}')

# -- All the Marthe files pointed in the main .rma file are stored in .mlfiles attribute
mm.mlfiles

# -- All the available units converters are stored in .units attribute
mm.units
# For example:
print(f"Model flow unit :\t {str(mm.units['flow'])} m/s")
print(f"Model distance unit :\t {mm.units['modeldist']} m")
print(f"Model time unit :\t {mm.units['modeltime']} (=years)")

'''
All the information stored in the .layer file is now also stored in MartheMode
instance as a simple DataFrame
'''
mm.layers_infos

'''
The calendar dates of timesteps in .pastp file are stored in .mldates attribute
as an DatetimeIndex or TimedeltaIndex
'''
mm.mldates

'''
The MartheModel instance store information about refine levels of nested grids in
the attribute `.rlevels`. This is a basic dictionary with format:
    rlevels = {inest_0: None,
               inest_1: refine_level_1, 
                      ..., 
               inest_N: refine_level_N}.

Obviously, the main grid (inest = 0) does not contain refined informations 
(inest_0 = None). The values of rlevels dictionary correspond to the the number of
division of a parent cell (main grid) for each direction x and y. For example, a
refine level of 5 means that the adjacent cells of the grid (parent cells) will be
divided in 5 cells on x-direction and 5 cells on y-direction (25 child cells).
'''
mm.rlevels

'''
The MartheModel instance provide some usefull methods to perform basic processing
about model extension.
'''
# -- Get model extension
mm.get_extent()
# -- Get edges of model extension
mm.get_edges(closed=False)
# -- Get all cell centroids
mm.get_xycellcenters(stack=True)
# -- Check if points are in model extension
mm.isin_extent(x=[22.4, 328.1], y=[67.8, 269.4])

'''
Another usefull spatial processing is to extract cell id(s) (=nodes) from pairs of
xy-coordinates. This sampling process use the spatial index to intersects the grid.
The high level function `.get_node()` can simplify this process with a battery of 
checks to avoid bad inputs.
'''
# -- Setting some xy-coordinates to extract node id(s)
x = [323.1,333.4,346.7]
y = [277.1,289.3,289.5]
# without layer informations
mm.get_node(x, y)
# without layer but active cell only
mm.get_node(x, y, only_active=True)
# with same layer for all points
mm.get_node(x, y, layer=4)
# with different layer for each points
mm.get_node(x, y, layer=[0,3,2])


'''
The MartheModel instance has a .prop attribute, it is a dictionary of Marthe model 
properties. For now the only supported properties are:
    - gridded properties (MartheField() class):
        - permh
        - kepon
        - emmca
        _ emmli
        - ...
    - pumping properties (MarthePump() class):
        - aqpump
        - rivpump
    - Zonal Soil properties (MartheField)
        - 'cap_sol_progr'
        - 'aqui_ruis_perc'
        - 't_demi_percol'
        - 'ru_max'
        - ...


When initiating a MartheModel instance, the basic 'permh' property is always loaded.
To load another property, use the .load_prop() method.
'''
# -- Initial properties
mm.prop

# -- Load some distributed properties
props = ['emmca', 'emmli', 'kepon', 'aqpump']
for prop in props:
    mm.load_prop(prop)


# -- Loaded properties
mm.prop


'''
The .get_outcrop() method (only available for structured grid) return a 2D-array 
of the number of the outcropping layers.
'''
plt.imshow(mm.get_outcrop(), cmap='tab20')
ticks = np.arange(1, len(np.unique(mm.get_outcrop())) + 1)
plt.colorbar(ticks=ticks)
plt.show()


'''
PyMarthe has a Vtk class (pymarthe.utils.vtk_utils.Vtk) base on a MartheModel input
that can reconstruct model 3D geometry as an unstructured grid. The user can get 
the Vtk instance by using the .get_vtk() method. This operation required the `itertools`
and `vtk` packages.  Several usefull arguments can by provided such as:
    - vertical_exageration: scale of vertical exageration
    - hws : hanging wall state, flag to define whatever the superior
                hanging walls of the model are defined as normal layers
                (explivitly) or not (implicitly).
    - smooth : enable interpolating vertex elevations
                    based on shared cell.
                    Default is False.
    - binary : Enable binary writing
    - xml (bool) : Enable xml based VTK files writing.
    - shared_points :Enable sharing points in grid polyhedron construction.
Note: This operation can take a while for large model. Some informations and
      progress bars will be printed to help the user to identify the state
      of the operation progress.
'''
vtk = mm.get_vtk(vertical_exageration=0.02, hws = 'implicit',
                 smooth=False, binary=True,
                 xml=False, shared_points=False)

'''
In order to facilitate visualisation we will use the `pyvista` package.
'''
try:
    import pyvista as pv
except ImportError as error:
    print("Could not load `pyvista` package.")

'''
The user can access the created unstructured grid by using the .vtk_grid attribut.
Let's plot the actual cell connectivities of the 3D grid.
'''
# -- pyvista unstructured grid
ugrid = pv.UnstructuredGrid(vtk.vtk_grid)
conn = ugrid.connectivity()

# -- pyvista plot
from matplotlib.colors import ListedColormap
cmap = ListedColormap(plt.cm.tab20(np.arange(mm.nlay)))
_ = conn.plot(cmap=cmap, show_edges=True)


'''
The MartheModel instance can run the provided model from python with the method
`.run_model()` with required level of verbosity and silence.
To do so, make sure to provide the correct path to your Marthe executable as 
`exe_name`.
After running the model, the user can easily print on screen the execution times
of all main processes by using the `.show_run_times()` method.

Note : If the Marthe executable path is already in your environement variables
of your computer, there is no need to provided the full path, the executable
name is enough.
'''
# -- Launch model run
mm.run_model(exe_name = 'Marth_R8', silent=True, verbose=False)

# -- Get run times summary
mm.show_run_times()



# --------------------------------------------------------
# ---- MartheField instance

'''
The new MartheFiled instance was created to manage Marthe gridded/field data.
It generally instantialized with a Marthe grid file such as permh, emmca, emmli, kepon, ..
All single Marthe grid data in this file are stored in a numpy recarray with
usefull informations: 'layer', 'inest', 'i', 'j', 'x', 'y', 'value'.

'''
mm = MartheModel(mona_ws, spatial_index = True)

# -- Build MartheField instance externaly
mf = MartheField(field = 'permh', data = mm.mlfiles['permh'], mm=mm)

# -- Fetch MartheField instance from a parent MartheModel instance property
mf = mm.prop['permh']


'''
MartheField instance has a very flexible getters/setters (subseting arguments can
be numeric are iterable) to subset and set data easily. The argument `as_array` 
can be set to True to retrieve 3D-array with shape (layer, row, col). 
For structured model, the .as_array() method can be use. It is a simple wrapper
of getting a 3D-array on main grid only.
Let's try some manipulations.
'''
# --> Getter

# -- Get all data
mf.get_data()

# -- Subset by layer
mf.get_data(layer=0)
mf.get_data(layer=[0,9])
mf.get_data(layer=np.arange(5))
mf.get_data(layer=(1,5,9))

# -- Subset by layer and inest
mf.get_data(layer=[1,5,6,8], inest= 0)   # mona model has no nested grid

# -- Getting data as boolean mask
mf.get_data(layer=[1,5,6,8],  as_mask=True)

# -- Get data as 3D-array
mf.get_data(inest= 0, as_array=True).shape
mf.get_data(layer = 0, as_array=True).shape
mf.get_data(layer=[1,5],  as_array=True).shape

# -- .as_array() method. Simple wrapper of get_data(inest=0, as_array=True)
#    raising an error if the model is nested.
mf.as_3darray()


# --> Setter

'''
For setting data, 3 objects can be implemented as `data` argument:
    - Marthe property file ('mymodel.permh')
    - Whole recarray with layer, inest,... informations
    - 3D-array (only for structured model)
'''

# ---- Set the entire recarray
rec = mf.get_data()
mf.set_data(rec)

# ---- Set data by value on layer
mf.set_data(2.3, layer=9)
mf.get_data(layer=9)['value']

# ---- Set data by value on layer and inest
mf.set_data(88.6, layer=2, inest=0)
mf.get_data(layer=2)['value']

# ---- Set data from a 3D-array
arr3d = mf.as_3darray()
arr3d[:] = 3
mf.set_data(arr3d)
mf.get_data()

'''
To get field data at a specific localisation(s) on model domain, the user can
use the .sample() method. This will initialized a Rtree spatial index in parent
MartheModel instance (could be slow on large model) and request the given
 xy-coordinates spatially.
Be careful, this method only return field data on the given input points even if
there correspond to edges of a more complex geometry like lines or polygons.
If the model is nested, the intersection will be performed on all cell but only
the one with higher inest will be returned.
'''
x = [323.1,333.4,346.7]
y = [277.11,289.3,289.5]
mf.sample(x, y, layer=[1,4,6])
mf.sample(x, y, layer=2)

'''
The MartheField instance can (re)build basic MartheGrid instance from the recarray.
To retrieve those singles grids in a ordered list use the method .to_grids()
'''

# -- Get all single grids
mf.to_grids()

# -- Get subset data as grids
mg = mf.to_grids(layer=2)[0]
print(mg.layer)

# -- Reset changes
mm.load_prop('permh')
mf = mm.prop['permh']

'''
The MartheField instance can perform zonal statistics based on single part polygon areas 
on several layers. To do so, make sure to use the appropriate .zonal_stats()
method. It will return a row MultiIndex DataFrame (based on provided zones and layers)
with all required statistics.
'''

# -- Fetch polygons in shapefile
shpname = os.path.join(mm.mldir, 'gis', 'zones.shp')
shp_df = shp_utils.read_shapefile(shpname)
# -- Set required statistics
stats = ['mean','max','min','median','count']
# -- Perform zonal statistics method
zstats_df = mf.zonal_stats(stats=stats,
                           polygons=shp_df['coords'],
                           names=shp_df['zname'],
                           trans='log10')
print(zstats_df)


'''
The MartheField instance has a internal .plot() method to see gridded property values.
The user can determine wich layer has to be plotted (1 layer for each plot call).
He can proivide an existing matplotlib custom axe, a single nested grid, a log
transformation and other collections arguments to perform a prettier plot.
'''

# -- Basic plot (single layer)
mf.plot(layer=6, log=True)
plt.show()
 

# -- Custom plot (single layer)
plt.rc('font', family='serif', size=8)
fig, ax = plt.subplots(figsize=(10,6))
extent = (300,205,490,380)
ax = mf.plot(ax=ax, layer=6, log=True,
             extent= extent, edgecolor='black', 
             lw=0.3, cmap='jet', zorder=10)
ax.set_title('MONA - Permeability (layer = 6)', fontsize = 12, fontweight="bold")
ax.grid('lightgrey', lw=0.5, zorder=50)
fig.delaxes(fig.axes[1])
cb = fig.colorbar(ax.collections[0], shrink=0.9)
cb.set_label('log(permh) $[m/s]$', size=13)
plt.show()


# -- Complex custom plot (multiple layers)
plt.rc('font', family='serif', size=6)
fig, axs = plt.subplots(figsize=(18,18), nrows = 4, ncols = 4,
                        sharex=True, sharey=True,
                        gridspec_kw={'wspace':0.02, 'hspace':0.05})

for ilay, iax in zip(range(mm.nlay), axs.ravel()[:-1]):
    ax = mf.plot(ax= iax, layer=ilay, log=True, zorder=10)
    ax.set_title(f'layer = {ilay+1}', x=0.85, y=-0.01, fontsize = 10)
    ax.grid('lightgrey', lw=0.3, zorder=50)

axs[-1,-1].set_axis_off()
fig.suptitle('MONA v.3 - Permability field', y=0.93, fontsize=16, fontweight="bold")

plt.show()



# -- Write data as Marthe grid file (Don't do it if you want to keep original data)
# mf.write_data()
# mf.write_data('mynewpermhfile.permh')


'''
MartheField instance can be exported to vtk object by using the .to_vtk() method.
This will use the MartheModel.get_vtk() method to build vtk 3D geometries, add current
field data for each cell and finally exporting it. Some values can be masked thank's to
the `masked_values` argument. The user can also apply a transformation to the field data
before exporting (argument `trans`).
Note: the filename has to be provided without any extension, the adequate extension
('.vtu' or '.vtk') will be infered from the `.xml` argument.
Let's export the based 10 logarithm of 'permh' field to vtk file. 
'''
mm.prop['permh'].to_vtk(filename = os.path.join('monav3_pm', 'vtk_permh'),
                        trans = 'log10',
                        vertical_exageration=0.02,
                        smooth=False, binary=True)


# --------------------------------------------------------
# ---- MartheFieldSeries

'''
While running a MARTHE model, if provided in the .pastp file, some fields will be simulated and saved
(currently in the 'chamsim.out' file). For post-processing purpose, the MartheFieldSeries class can
be called to manipulate these series of simulated fields. The MartheFieldSeries constructor will read
the simulated fields and collect all MartheGrid for a given field. 
Let's try to load simulated heads from the MONA model.
'''
chasim = os.path.join('monav3_pm', 'chasim_cal_histo.out')
mfs = MartheFieldSeries(mm=mm, field = 'charge', simfile=chasim)

'''
The MartheFieldSeries object store data all data in a dictionary with format:
{istep_0 : MartheField_0, ..., istep_N : MartheField_N}
'''
mfs.data

'''
Simulated time series of given point(s) (x-y-layer) can be fetch with the .get_tseries() method.
Additional `names` iterable can be added to reference each point. The `index` argument allows 
to manage the required index of the output DataFrame, can be 'date', 'istep', 'both'.
'''
# -- Read points from a shapefile
shpname = os.path.join('monav3_pm', 'gis', 'sim_points.shp')
shp_df = shp_utils.read_shapefile(shpname)

# -- Extract coordinates
x, y = zip(*shp_df.coords.explode())

# -- Get time series
df = mfs.get_tseries(x,y, layer=5)
print(df)

'''
If `names` arguments are not provided, generic names are created according to points row, column, layer
with format: f'{row}i_{col}j_{layer}k'. Let's try again providing points names and a different index.
'''
df = mfs.get_tseries(x,y, layer=5, names = shp_df['ID'], index = 'istep')
print(df)
df = mfs.get_tseries(x,y, layer=5, names = shp_df['ID'], index = 'date')
print(df)
df = mfs.get_tseries(x,y, layer=5, names = shp_df['ID'], index = 'both')
print(df)

'''
One of the main advantage of the DataFrame output is the pandas plot support to visualize field time series.
Let's make a example
'''
df = mfs.get_tseries(x,y, layer=5, names = shp_df['ID'])
plt.rc('font', family='serif', size=7)
df[['rec_8','rec_9','rec_10']].plot(figsize = (8,4), title = 'Simulated heads',
                                    lw = 0.8, xlabel ='', ylabel ='Hydraulic heads [m]')
plt.show()


'''
Another post-processing tool of MartheFieldSeries named .save_animation() allows to save .gif animation
to see the evolution of field values in a given layer. The user can manage the frame rate and the resolution
of the animation with the following arguments:
- `dpf` : duration per frame 
- `dpi` : dot per inch
The .save_animation() method support of kwargs from MartheField.plot() method (cmap, vmin, vmax, extent ..).
This method have a additional dependency to `imageio` python package.
It may be quite slow for model with large number of cells, layers and time steps.
'''
gif = os.path.join('monav3_pm', 'export', 'heads5_animation.gif')
mfs.save_animation(gif, dpf = 0.2, dpi=200, 
                        layer=5, vmin=-50, vmax=150,
                        extent=(300,210,490,380), cmap = 'jet')


'''
MartheFieldSeries allows the vectorial exports of a simulated fiel in shapefile format for a given layer.
To do so, use the .to_shapefile() method.
It's will save non masked cell of a given layer with basic informations like layer, inest, i, j, 
x, y but also the values of each simulated field (1 per column).
Note: column names can be truncated if field name is too long.
'''

filename = os.path.join('monav3_pm', 'export', 'heads_09.shp')
mfs.to_shapefile(filename, layer=9)



# --------------------------------------------------------
# ---- MartheSoil instance

'''
PyMarthe v1.0 can also manage zonal soil proprerties such as:
    - cap_sol_progr
    - equ_ruis_perc
    - t_demi_percol
    - def_sol_progr
    - rumax
    - defic_sol
    - ...
There are parameters of the GARDENIA (@BRGM) software implemented in the .mart
file in the 'Initialization des calculs' section. These list-like parameters
are stored in a single DataFrame (.data) by they have a spatial application 
(cell-by-cell) that's the reason why the MartheSoil class has functionalities
built as MartheField method wrappers.
Let's have a look on this support class.
'''

'''
Soil properties are read from the .mart file. If the main model does not contain
any zonal soil properties an assertion error is raise.
'''

# -- Trying to build MartheSoil instance
ms = MartheSoil(mm)

'''
The actual mova v.3 model does not contain these properties. So, let's try with
another Marthe Model named 'Lizonne.rma'. The MartheSoil instance can support 3
type of soil property data :
    - 'mart-c'  : constant soil properties in .mart file
    - 'pastp-c' : constant soil property in .pastp file
    - 'pastp-t' : transient soil property in .pastp file
It will recognize the `mode` of implementation automatically.

'''
lizonne_ws = os.path.join('lizonne_v0', 'Lizonne.rma')
lizonne_si = os.path.join('lizonne_v0', 'Lizonne_si')
mm = MartheModel(lizonne_ws, spatial_index=lizonne_si)

# -- Build MartheSoil instance externaly
ms = MartheSoil(mm)

# -- Fetch soil property instance from main model
mm.load_prop('soil')
ms = mm.prop['soil']

'''
The main data correspond to a simple table (DataFrame) with the correspondance 
between the existing soil properties in .mart file, the id of the spatial zone
and the value of the given soil property.
'''
# ---- Print basic data
print(ms.data.to_markdown(tablefmt='github', index=False))
print(f'\nSoil data implementation mode : {ms.mode}')
print(f'Number of zone: {ms.nzone}')
print(f'Number of soil properties: {ms.nsoilprop}')
print(f'Soil properties of the {mm.mlname} model:')
print('\n'.join([f'\t- {p}' for p in ms.soilprops]))

'''
To access soil data the .get_data() method can be use. The data can be subset by
soil property (`soilprop`) and by soil zones (`zone`). The output is an subset DataFrame
of soil data but, as explained above, soil properties have a spatial application.
So, it is possible to fetch complete cell-by-cell soil data as recarray turning the 
`as_style` argument to `array-like` It will replace the zone ids in the .zonep file
(MartheField) with the value of a given soil property.
Note: soil properties are defined only on the first layer, others are set to 0.
'''
ms.get_data(soilprop = 'cap_sol_progr', zone=[1,8])
ms.get_data(soilprop = 'cap_sol_progr', as_style = 'array-like', layer=1)    # Constant value (=0)
ms.get_data(soilprop = 'cap_sol_progr', as_style = 'array-like', layer=0)

'''
Even if it's not explicitly written in the .mart or .patp file and store in `ms.data`,
the user can access the soil data value for specific isteps by use the `force` argument.
This works by searching the nearest previous istep (`npi`) where soil data were defined,
that can be if several timesteps are required. 
Let's extract the property `cap_sol_progr` for the istep 10 to 15 (must be the same as 
the first timestep because soil properties are constant). 
'''
# --- Without forcing
ms.get_data(soilprop = 'cap_sol_progr', istep = np.arange(10,16), force=False)
# --- With forcing
ms.get_data(soilprop = 'cap_sol_progr', istep = np.arange(10,16), force=True)

'''
Moreover, some wrappers of usefull functionalities of MartheField instance can be 
use on soil properties. 
'''
# ---- Sampling (from points shapefile)
shpname = os.path.join(mm.mldir, 'export', 'points.shp')
x,y = shp_utils.shp2points(shpname, stack=False)
ms.sample('cap_sol_progr', x, y)

# ---- Ploting
plt.rc('font', family='serif', size=9)
fig, ax = plt.subplots(figsize=(8,6))
ax.set_title('Zonal progressive soil capacity',
              fontsize=12, fontweight="bold")
ms.plot('cap_sol_progr', ax=ax, cmap='Paired')
plt.show()

# ---- Exporting
filename = os.path.join(mm.mldir, 'export','cap_sol_progr.shp')
ms.to_shapefile('cap_sol_progr', filename=filename, epsg=2154)

'''
To change/set data by soil property and by zone use the .set_data() 
method with the required value.

'''
# ---- Changing data
ms.set_data('cap_sol_progr', value = 125, zone = [1,2])
ms.set_data('equ_ruis_perc', value = 17, zone = 8)
print(ms.data.to_markdown(tablefmt='github', index=False))

'''
The soil property data has to be write in .mart file with the
.write_data() method (can be performed by soil property).
(Don't do it if you want to keep original data)
'''

#ms.write_data(martfile='Lizonnetest.mart')



# --------------------------------------------------------
# ---- MarthePump instance

'''
The new behaviour of Marthe pumping data is similar than the old version.
But the pumping data storage has change to be a DataFrame (list-like properties).
The readers (in .pastp file) had been changed in favor of 
regular expressions (regex) instead of the line-by-line reading.

Reminder:
The MarthePump instance reads pumping data from .pastp.file ordered by time steps.
Two kind of withdraw are provided :
    - aquifer pumping (mode= 'aquifer')
    - river pumping (mode= 'river')
Let's see these changes.
'''

# -- Build MarthePump instance externaly
mm = MartheModel(mona_ws, spatial_index = mona_ws.replace('.rma', '_si'))
mp = MarthePump(mm, mode = 'aquifer')

# -- Fetch pumping property from main model
mm.load_prop('aqpump')
mp = mm.prop['aqpump']

'''
Pumping data are stored in a DataFrame with informations on 'istep', 'node', layer',
'i', 'j', 'value' and 'boundname' in the `.data` attribute. But there is also an 
hidden argument with all metadata `._data`. `boundname` correspond to a generic
boundname created from the node ids of each well (format: 'aqpump_node).
'''

# -- User pumping data
mp.data

# -- Internal pumping metadata
mp._data


'''
Like the MartheField instance, MarthePump has very flexible getters/setters methods.
The user can subset data by istep, node, layer, row (i), column (j) or boundname.
'''

# -- Get all data
mp.get_data()

# -- Subset by timestep
mp.get_data(istep=2)
mp.get_data(istep=[3,6,9,14])
mp.get_data(istep=np.arange(0,mm.nstep,4))

# -- Subset by timestep and layer
mp.get_data(istep=3, layer=3)
mp.get_data(istep=3, layer=[5,8])

# -- Subset by timestep, layer and i, j
mp.get_data(istep=3, layer=3, i=102)
mp.get_data(istep=3, layer=3, j=71)
mp.get_data(istep=3, layer=3, i=102, j=71)


# -- Subset by boundname
mp.get_data(istep=3, boundname = 'aqpump_020720')

# -- Switch boundname by another bound names
switch_dic = {'aqpump_046949' : 'pump1', 'aqpump_020822' : 'pump2' }
mp.switch_boundnames(switch_dic)
mp.get_data(i=101, j=70)

'''
Reminder:
There is 3 types of pumping supported by MarthePump:
    - 'listm'
    - 'record'
    - 'mail'
For writting purpose, data are splitted by there respective qtype in .pastp file.
Use the .split_qtype() function to get a DataFrame of each qtype.
'''

# -- Split by qtype (if qtype is not present, return empty DataFrame)
listm_df = mp.split_qtype('listm')[0]
mail_df, record_df, listm_df = mp.split_qtype()

# -- Write data inplace (Don't do it if you want to keep original data)
# mp.write_data()
# -- Data can be wrote individually for each qtype
# mp._write_listm()


# --------------------------------------------------------
# ---- MartheOptim/MartheObs instance


'''
All the calibration/optimisation process/manipulation had been separated from 
the main MartheModel to a new class called MatheOptim. So, we can create a
MartheOptim instace for each calibration try and store it in a new variable.
When creating a MartheOptim instance, it's necessary to give a existing 
MartheModel to link the optimisation process with the model.
'''

# -- Build a mopt instance
kwargs = {f'{f}_dir': os.path.join(mm.mldir, f) 
          for f in ['par', 'tpl', 'ins', 'sim'] }
mopt = MartheOptim(mm, name = 'opti_mona', **kwargs)

print(f"My optimisation is called: {mopt.name}")

'''
The MartheOptim class is not finish yet. It will allowed the user manage 
observation and parameter data of the parent Marthe model in order to use
some PEST program utilities to estimate, calibrate and optimise the model.
To sum up, it's a python wrapper between Marthe model and PEST. MartheOptim
contains 2 basics (empty) dictionaries (.obs and .param). 
For now, only the observations management are available (MartheObs class).
Let's see how it works.
'''

# -- Quick look of .obs and .param attributes
mopt.obs, mopt.param


'''
To add a set of observation, please use the .add_obs() method. For each observation
set to add it is necessary to provided at the observation data. Can be:
    - A observation file (path to the external file containing data)
    - A DataFrame with `value` column and DatetimeIndex 
A `datatype` of these observations (ex: 'head','flow', 'soil', ...) can be provide too,
the default `datatype` is 'head'.
Some other arguments can be explicite such as `loc_name` (name of the observation point),
`check_loc` (verify if this loc_name exist and is unique) and other kwargs such as 
observation weights, observation group name (obgnme), ...
Let's add a single observation.
'''

# -- Add the 07065X0002 observation well
single_obs = os.path.join(mm.mldir, 'obs', '07065X0002.dat' )
mopt.add_obs(data = single_obs , datatype='head')
mopt.obs # View of the added MartheObs instance in .obs

'''
The MartheOptim instance has a method to take a look of  all added observations
data in a single large table : .get_obs_df(). Even if the user can create his own names 
for each observation (**kwargs) it is strongly recommended to let MartheOptim build generic
observations names easily readable for PEST programs. These observation names are build as follow:
-> 'loc' + localisation name id (3 digits) + 'n' + observation id (adaptative n° of digits)
This automatic names generation avoid too high number of character (not suported by PEST). 
'''


'''
The .add_obs() function provided an warning message if the user try to add a set of 
observations already added. In this case, the observations data will remove the oldest
by overwrite it with the new one. 
'''
mopt.add_obs(data = single_obs, datatype='head')


# -- Remove 1 added observation
mopt.remove_obs(locnme = '07065X0002')

# -- Remove all added observations
mopt.remove_obs()
mopt.obs

# -- Trying to add not valid observation (using `check_loc`)
invalid_obs1 = os.path.join(mm.mldir, 'obs', 'p34.dat')
invalid_obs2 = os.path.join(mm.mldir, 'obs', '07588X0048.dat')
mopt.add_obs(data=invalid_obs1, check_loc=True)
mopt.add_obs(data=invalid_obs2, check_loc=True)

# -- Override checking locnmes
ws = os.path.join(mm.mldir, 'obs')
for dat in os.listdir(ws):
    obsfile = os.path.join(ws, dat)
    mopt.add_obs(data = obsfile,  check_loc = False)

mopt.obs



'''
MartheOptim supports the implementation of fluctuations for each observations set.
A fluctuation is the absolute difference between a serie of data and a numeric value.
The .add_fluc() method allows the user to build a new set of observations from a existing
one (locnme). The argument  called `on` can be provided to specify on which specific value
the fluctuation must be computed. It can be a simple numeric value, a basic function name 
such as 'mean', 'median', 'max', ... or a custom function. Another argument called `tag`
allows the user to add multiple fluctuations set on a single existing one by naming them.
'''

# -- Add fluc on 'mean' and 'median'
mopt.add_fluc(locnme = '07065X0002', tag = 'mn' , on = 'mean')
mopt.add_fluc(locnme = '07065X0002', tag = 'md' , on = 'median')

df = mopt.get_obs_df().query(f"locnme.str.contains('07065X0002')", engine='python')
df[::10]

# -- Add fluctuation from a basic numeric value
critical_head = 85 # meters
mopt.add_fluc(locnme = '07095X0117', tag = 'crit' , on = critical_head)
mopt.obs['07095X0117critfluc'].obs_df.tail(10)


'''
Some methods were added to mopt to fetch the number of locnmes, observations and 
datatype added.
'''
# -- Get Number of data types
mopt.get_ndatatypes()

# -- Get number of locnmes (for 1 or all datatype)
mopt.get_nlocs()
mopt.get_nlocs(datatype='headmnfluc')

# -- Get number of observations (for 1 or all locnames)
mopt.get_nobs()
mopt.get_nobs(locnme = '07065X0002', null_weight=True)

'''
The MartheOptim instance has a integrated method to compute observations weight based on:
    - The tuning factor for each datatype (lambda) 
    - The number of datatypes
    - The number of locnme for each datatype
    - The number of observations for each locnme
    - The absolute acceptable error for each datatype (sigma)
Let's try to compute observation weigths.
'''

# -- Set lambda and sigma values
w_df = pd.DataFrame(data = [[2,5,4,7], [0.1,0.008,0.008,0.008]],
                    index = ['lambda', 'sigma'],
                    columns = mopt.get_obs_df().datatype.unique().T).T

print(w_df.to_markdown(tablefmt='github'))

# -- Convert them to dictionaries
lambda_dic, sigma_dic = [w_df[c].to_dict() for c in w_df.columns]
mopt.compute_weights(lambda_dic, sigma_dic)

# -- View of computed weights
weights = pd.DataFrame({'weight' : [mo.weight for mo in mopt.obs.values()]},
                       index = mopt.obs.keys())
print(weights.tail(10).to_markdown(tablefmt='github'))


'''
While adding a new observation, it's possible to pass a additional argument (kwargs)
to transform data. It is possible (and recommended) to set a transformation after 
importing all observations with the .set_obs_transform() method.
If the transformation is not valid, a assertion error will be raised.
'''
mopt.remove_obs(verbose=True)

# ---- Set transformation in .add_obs() constructor
mopt.add_obs(data = single_obs, datatype='head', trans = 'log10')
mopt.obs['07065X0002'].obs_df


# ---- Set transformation using the .set_obs_transform() method on locnme
mopt.add_fluc()
mopt.set_obs_trans('log10', locnme = '07065X0002fluc')
mopt.obs['07065X0002fluc'].obs_df

# ---- Set transformation using the .set_obs_transform() method on datatype
mopt.set_obs_trans('lambda x: np.log10(abs(x))', datatype = 'headfluc')
mopt.obs['07065X0002fluc'].obs_df

# ---- Try invalid transform
mopt.set_obs_trans('-log(e)')

# ---- Get transformed DataFrame
#mopt._apply_transform()


'''
MartheOptim also has a builtin method to write instruction files from added observation.
Use the .write_ins() method.
'''
# -- Set instruction files folder
mopt.ins_dir = os.path.join(mm.mldir,'ins')

# -- Write instruction file by locnme
mopt.write_ins(locnme = '07065X0002')

# -- Write all instruction files
mopt.write_ins()



# --------------------------------------------------------
# ---- MartheListParam instance

'''
The MartheOptim instance support the parametrisation of some MartheModel 
list-like properties such as:
    - `soil` (MartheSoil)
    - `aqpump` (MarthePump)
    - `rivpump` (MarthePump)
The parametrization of list-like properties are based on a KeysMultiIndex (`kmi`)
argument: this is a pandas MultiIndex object where `keys` are provided by the user
and correspond to a non-unique set of parameters. These `keys` must be part of the 
column names of the data of the entity parametrized. For example, for a pumping 
parametrization, the `keys` must be in `mm.prop['aqpump'].data.columns`. The other 
main argument to provide is the name of the column which values will be parametrized 
(`value_col`). 
Note : To build the `kmi` object, the user can use the pest_utils.get_kmi() facility.
Let's try to perform pumping parametrization on the mona model.
'''

mm = MartheModel(mona_ws, spatial_index = mona_ws.replace('.rma', '_si'))
mm.load_prop('aqpump')
mp = mm.prop['aqpump']



'''
To begin with, let's sample some pumping wells to parametrize
(gon generic names)
'''
pwnames = [f'pump_{ij}' for ij in  ['113_14', '87_15', '105_15']]

'''
Now, we want to parametrize the `value` column (pumping rate) according
to the following keys : `istep`, `layer`, `boundname`.
Of course, we have to specilize only the  required values of each keys
(if it's not specilized, all values are considered).
'''
kmi = pest_utils.get_kmi( mobj = mp,
               keys = ['boundname', 'layer', 'istep',],
               boundname = pwnames, layer = 2)
print(kmi)

'''
To add a list-like set of parameters in the main MartheOptim instance,
use the .add_param() method. This will create a new MartheListParam 
instance in the .param dictionary. 
'''
# -- Get kmi for first pumping well
kmi1 = pest_utils.get_kmi( mobj = mp,
                keys = ['boundname', 'layer', 'istep',],
                layer = 2, boundname = pwnames[0])

mopt.add_param(parname = pwnames[0], mobj = mp,
                 kmi = kmi1, value_col = 'value')

print(mopt.param)
print(mopt.param[pwnames[0]].param_df)


'''
Some others parameters (kwargs) can be implemented while adding parameters
such as `defaultvalue`, `parchglim`, `parlbnd`, `parubnd`, `pargp`, `scale`, ...
The `defaultvalue` can be provided by the user, otherwise, the prior value load
in the property class of the model will be implemented.
This is totaly possible to attach a data transform process to the required 
values to optimized. To do so, the `trans` (transformation) and `btrans`
(back-transformation) must be provided by the user. It has to be an string 
function expression understood by the python built-in eval() function or 
pandas.Series.transform() method. If the user set a non valid transformation,
a error will be raised.

Let's try to apply logaritmic transformation to pumping parameter values.
Note that pumping data are always <= 0 so, for a logaritmic transformation
the user need to apply a correct expression function.
'''

# -- Get kmi for second pumping well
kmi2 = pest_utils.get_kmi( mobj = mp,
                keys = ['boundname', 'layer', 'istep',],
                layer = 2, boundname = pwnames[1])
'''
# -- Unvalid transform
mopt.add_param(parname = '87_15', mobj = mp,
                 kmi = kmi2, value_col = 'value',
                 trans = '-log10',
                 btrans = '-10**x')
'''
# -- Valid transform
mopt.add_param(parname = pwnames[1], mobj = mp,
                 kmi = kmi2, value_col = 'value',
                 trans =  'lambda x : - np.log10(x + 1)',
                 btrans = 'lambda x : -1 * (- 10**-x + 1)')

mopt.param[pwnames[1]].param_df.head()

# -- Add last pumping with default value 
kmi3 = pest_utils.get_kmi( mobj = mp,
                keys = ['boundname', 'layer', 'istep',],
                layer = 2, boundname = pwnames[2])
mopt.add_param(parname = pwnames[2], mobj = mp,
                 kmi = kmi3, value_col = 'value',
                 defaultvalue = 0, scale = 2)

mopt.param[pwnames[2]].param_df.head()

# -- Set transformation after adding parameters
mopt.set_param_trans(trans = 'log', btrans= 'exp', parname = pwnames[0])
mopt.param[pwnames[0]].param_df.head(5)

# -- Remove parameter(s)
mopt.remove_param(parname = pwnames[0], verbose=True)

'''
As the observation process, it is possible to get all parameters informations
in a single large table using the .get_param_df() method.
'''
mopt.get_param_df()


'''
To write pest parameter files and template files from the MartheOptim instance
consider using the appropriate writing functions (Can be perform for a single,
a group or all paramaters). The default value of each parameter will be written
in a distinct parameter file according to the user transformation.
'''
# -- Writing by parname
mopt.write_parfile(parname= pwnames[1:])
mopt.write_tpl(parname = pwnames[2])

# -- Writing all parameters
mopt.write_parfile()
mopt.write_tpl()



# --------------------------------------------------------
# ---- MartheArrayParam instance

'''
UNDER DEVELOPMENT
'''



# --------------------------------------------------------
# ---- Parametrization configuration (.config file)

'''
After peforming all parametrization process, the MartheOptim instance can save the
parametrization configuration in an external text file `.config`. This file can be
used during a forward run process by instantialized a MartheModel instance with 
properties values stored in all the parameters files. 
Let's give an example.
'''

# -- Save and Write parametrization configuration of mopt instance
mopt.write_config()

# -- Quick view of config text file
configfile = os.path.join(mm.mldir,'opti_mona.config')

with open(configfile, 'r', encoding='latin-1') as f:
    print(f.read())



'''
The new MartheModel instance automatically load the parametrize properties
and set data from parameter files. As a test, the user can change some values
in a given `parfile` (pest behaviour) and check the properties values after
loading again the model with .from_config().
'''

# -- Let's retrieve a MartheModel instance from this config file
new_mm = MartheModel.from_config(configfile)

