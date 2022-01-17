'''
PYMARTHE DEV Version 1.0 Guide

Get started with new 1.0 guide.
Give examples of some basic commands.
'''

# ---- Put your dev branch path here
import os, sys
dev_ws = os.path.normpath(r"E:\EauxSCAR\pymarthe_dev")

# ---- Import usefull modules

import pandas as pd
import numpy as np
sys.path.append(dev_ws)
from pymarthe import MartheModel
from pymarthe.utils import marthe_utils
from pymarthe.mfield import MartheField
from pymarthe.mpump import MarthePump
from pymarthe.moptim import MartheOptim
import matplotlib.pyplot as plt

# 1) --->  MONA MODEL <---

# --------------------------------------------------------
# ---- MartheModel instance

# -- Load mona.rma 
mm = MartheModel('mona.rma')

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

# -- All the information stored in the .layer file is now also stored in MartheMode
#    instance as a simple DataFrame
mm.layers_infos

# -- The calendar dates of timesteps in .pastp file are stored in .mldates attribute
#    as an DatetimeIndex or TimedeltaIndex
mm.mldates

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

# -- Load some properties
props = ['emmca', 'emmli', 'kepon', 'aqpump']
for prop in props:
    mm.load_prop(prop)

'''
To load soil property stored in .mart file (in the initialisation section),
use the property name 'soil'. If the actual model does not have any soil 
property (like the mona model), an assertion error is return. 
'''
mm.load_prop('soil')


# -- Loaded properties
mm.prop

'''
The .imask attribute based on permh property correspond to a simple MartheField 
delimiting aquifer extensions by binary values (0 inactive cell, 1 active cell).
'''
mm.imask

'''
The Marthe model instance has a integrated pair of function to fetch xy or ij 
coordinates (.get_xy() and .get_ij()).
'''
# -- Get xy (cellcenters) of a set of cells
mm.get_xy(i = 124, j=87)
mm.get_xy(i=[34, 67, 89], j=[56, 58, 83], stack=True)

# -- Get ij of a set of points (based on imask)
mm.get_ij(x=343.1, y=284.3) # could be slow (initialize imask spatial index)
mm.get_ij(x=[323.1,333.4,346.7], y=[277.11,289.3,289.5], stack=True) # must be faster now


'''
The .get_outcrop() method (only available for structured grid) return a 2D-array 
of the number of the outcropping layers.
'''
plt.imshow(mm.get_outcrop(), cmap='tab20')
ticks = np.arange(1, len(np.unique(mm.get_outcrop())) + 1)
plt.colorbar(ticks=ticks)
plt.show()

# --------------------------------------------------------
# ---- MartheField instance

'''
The new MartheFiled instance was created to manage Marthe gridded/field data.
It generally instantialized with a Marthe grid file such as permh, emmca, emmli, kepon, ..
All single Marthe grid data in this file are stored in a numpy recarray with
usefull informations: 'layer', 'inest', 'i', 'j', 'x', 'y', 'value'.

'''
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

# ---- Set data with a 3D-array
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
The MartheField instance has a internal .plot() method to see gridded property values.
The user can determine wich layer has to be plotted (1 layer for each plot call).
He can proivide an existing matplotlib custom axe, a single nested grid, a log
transformation and other collections arguments to perform a prettier plot.
'''

# -- Basic plot (single layer)
mf.plot(layer=6, log=True)
plt.show()


# -- Custom plot (single layer)
plt.rc('font', family='serif', size=11)
fig, ax = plt.subplots(figsize=(12,8))
ax = mf.plot(ax=ax, layer=6, log=True, 
             edgecolor='black', lw=0.3,
             cmap='jet', zorder=10)
ax.set_title('MONA - Permeability (layer = 6)', fontsize = 16, fontweight="bold")
ax.set_xlim(300,490)
ax.set_ylim(205,380)
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



# --------------------------------------------------------
# ---- MarthePump instance

'''
The new behaviour of Marthe pumping data is similar than the old version.
But the pumping data storage has change to be almost the same as MartheField
with recarray. The readers (in .pastp file) had been changed in favor of 
regular expressions (regex) instead of the line-by-line reading.

Reminder:
The MarthePump instance reads pumping data from .pastp.file ordered by time steps.
Two kind of withdraw are provided :
    - aquifer pumping (mode= 'aquifer')
    - river pumping (mode= 'river')
Let's see these changes.
'''

# -- Build MarthePump instance externaly
mp = MarthePump(mm, mode = 'aquifer')

# -- Fetch pumping property from main model
mm.load_prop('aqpump')
mp = mm.prop['aqpump']

'''
Pumping data are stored in a recarray with informations on 'istep', 'layer',
'i', 'j', 'value' and 'boundname' in the `.data` attribute. But there is an 
hidden argument with all metadata `._data`.
'''

# -- User pumping data
mp.data

# -- Internal pumping metadata
mp._data


'''
Like the MartheField instance, MarthePump has very flexible getters/setters methods.
The user can subset data by istep, layer, row (i), column (j) or boundname.
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
mp.get_data(istep=3, boundname = '102_71')

# -- Switch boundname by another bound names
switch_dic = {'102_71' : 'pump1', '87_15' : 'pump2' }
mp.switch_boundnames(switch_dic)
mp.get_data(i=102, j=71)

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

# -- Build a moptim instance
moptim = MartheOptim(mm=mm, name='cal_mona')
print(f"My optimisation is called: {moptim.name}")

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
moptim.obs, moptim.param


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
moptim.add_obs(data = 'obs/07065X0002.dat', datatype='head')
moptim.obs # View of the added MartheObs instance in .obs

'''
The MartheOptim instance has a main DataFrame (.obs_df) with all added observations
data. This allows the user to see directly all the observation information in a single
table. Even if the user can create his own names for each observation (**kwargs) it is
strongly recommended to let MartheOptim build generic observations names easily readable
for PEST programs. These observation names are build as follow:
-> 'loc' + localisation name id (3 digits) + 'n' + observation id (adaptative n° of digits)
This automatic names generation avoid too high number of character (not suported by PEST). 
'''
moptim.obs_df

'''
The .add_obs() function provided an warning message if the user try to add a set of 
observations already added. In this case, the observations data will remove the oldest
by overwrite it. 
'''

moptim.add_obs(data = 'obs/07065X0002.dat', datatype='head')


# -- Remove 1 added observation
moptim.remove_obs(locnme = '07065X0002')

# -- Remove all added observations
moptim.remove_obs()
moptim.obs_df
moptim.obs

# -- Trying to add not valid observation (using `check_loc`)
moptim.add_obs(data='obs/p34.dat', check_loc=True)
moptim.add_obs(data='obs/07588X0048.dat', check_loc=True)

# -- Override checking locnmes
for dat in os.listdir('obs'):
    obsfile = os.path.join('obs', dat)
    moptim.add_obs(data = obsfile, check_loc = False)

moptim.obs
moptim.obs_df


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
moptim.add_fluc(locnme = '07065X0002', tag = 'mn' , on = 'mean')
moptim.add_fluc(locnme = '07065X0002', tag = 'md' , on = 'median')

df = moptim.obs_df.query(f"locnme.str.contains('07065X0002')", engine='python')
df[::10]

# -- Add fluctuation from a basic numeric value
critical_head = 85 # meters
moptim.add_fluc(locnme = '07095X0117', tag = 'crit' , on = critical_head)
moptim.obs['07095X0117critfluc'].obs_df.tail(10)


'''
Some methods were added to moptim to fetch the number of locnmes, observations and 
datatype added.
'''
# -- Get Number of data types
moptim.get_ndatatypes()

# -- Get number of locnmes (for 1 or all datatype)
moptim.get_nlocs()
moptim.get_nlocs(datatype='headmnfluc')

# -- Get number of observations (for 1 or all locnames)
moptim.get_nobs()
moptim.get_nobs(locnme = '07065X0002', null_weight=True)

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
                    columns = moptim.obs_df.datatype.unique()).T

print(w_df.to_markdown(tablefmt='fancy_grid'))

# -- Convert them to dictionaries
lambda_dic, sigma_dic = [w_df[c].to_dict() for c in w_df.columns]
moptim.compute_weights(lambda_dic, sigma_dic)

moptim.obs_df


'''
While adding a new observation, it's possible to pass a additional argument (kwargs)
to transform data. It is possible (and recommended) to set a transformation after 
importing all observations with the .set_transform() method. Moreover, the .apply_transform()
method allows the user to get the DataFrame of transformed values.
'''
moptim.remove_obs()
moptim.add_obs(data = 'obs/07095X0117.dat', datatype='head')

# ---- Set transformation in .add_obs() constructor
moptim.add_obs(data = 'obs/07065X0002.dat', datatype='head', transform = 'log10')
moptim.obs['07065X0002'].obs_df


# ---- Set transformation using the .set_transform() method on locnme
moptim.add_fluc()
moptim.set_transform('log10', locnme = '07095X0117')
moptim.obs['07095X0117'].obs_df

# ---- Set transformation using the .set_transform() method on datatype
moptim.set_transform(abs, datatype = 'headfluc')

# ---- Get transformed DataFrame
moptim.apply_transform()


'''
MartheOptim also has a builtin method to write instruction files from added observation.
Use the .write_ins() method.
'''

# -- Write instruction file by locnme
moptim.write_ins(locnme = '07065X0002', ins_dir = 'ins') 

# -- Write all instruction files
moptim.write_ins(ins_dir = 'ins')

