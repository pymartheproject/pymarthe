'''
Pilot Points tools

'''


import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
from pymarthe import MartheModel, MartheField
from pymarthe.utils import marthe_utils, shp_utils

'''
Set some usefull fixed elements 
'''

PP_NAMES = ["parname","x","y","zone","value"]
# pilot point name format (layer number is 0-based within Python ; 1-based out of Python)
PPFMT = lambda name, lay, zone, ppid, digit: '{0}_l{1:02d}_z{2:02d}_{3}'.format(name,int(lay)+1,int(zone), str(int(ppid)).zfill(digit))
ZONE_KWARGS = {'color':'black', 'lw':1.5, 'label':'pilot points active zone'}
BUFFER_KWARGS = {'color':'green', 'ls':'--', 'lw':1.2, 'label':'pilot points active zone (buffer)'}
PP_KWARGS = {'s':20, 'marker':'+','lw':0.8 , 'color':'red', 'zorder':50, 'label':'pilot points'}



class PilotPoints():
    """
    Manage pilot points generation from parametrize izone field.
    /!/ Required python `shapely` module /!/

    """

    def __init__(self, izone):
        """
        Parameters
        ----------
        izone (MartheField) : field with required pilot point parmetrize zone ids as values.
                              Izone values can be:
                                    - izone < 0         : zone of piecewise constancy
                                    - izone > 0         : zone with pilot points
                                    - izone = -9999, 0, 9999 : inactive zone

        Examples
        --------
        izone = MartheModel('izone', 'model.izone', mm)
        pp = PilotPoints(izone)

        """
        # -- Import high level modules from `shapely` dynamically
        from importlib import import_module
        try:
            # Specific modules
            self.speedups = import_module('shapely.speedups')
            self.ops = import_module('shapely.ops')
            # Usefull classes
            self.MultiPoint = getattr(import_module('shapely.geometry'), 'MultiPoint')
            self.Polygon = getattr(import_module('shapely.geometry'), 'Polygon')
        except:
            ImportError("Could not import python `shapely` package!")
        # -- Disable speed up on Windows system
        if platform.system() == 'Windows':
            self.speedups.disable()
        # -- Store model and parameter zones informations
        self.mm = izone.mm
        self.izone = izone
        # -- Extract polygons for pilot points seed
        self.polygons = self.extract_active_polygons()
        # -- Initialize pilot points dictionary
        self.data =  { l: dict.fromkeys(
                                self.polygons.index.get_loc_level(
                                            key=l,level='layer')[1]
                                )
                            for l in self.polygons.index.levels[0]
                        }



    def extract_active_polygons(self):
        """
        Convert all model grid cells with same pilot point parametrize ids
        into a shapely.polygon referenced by `layer` and `zone`ids.

        Returns:
        --------
        polygons (pandas.Series) : active pilot point zone as shapely Polygon
                                   indexing with a MultiIndex object ([layer, zone])

        Examples
        --------
        active_polygons = pp.extract_active_polygons()

        """
        # -- Fetch modelgrid from MartheModel
        self.mm.build_modelgrid()
        mg = self.mm.modelgrid.copy(deep=True)
        # -- Add parameter zones in modelgrid
        mg['zone'] = self.izone.data['value'].astype(int)
        # -- Extract active pilot point zone extension (Polygon) for each layer
        polygons  =  mg.query('active == 1 & zone > 0'
                        ).groupby(['layer', 'zone']
                            ).apply(
                                lambda row: 
                                    self.ops.unary_union(
                                        row['vertices'].apply(
                                                self.Polygon
                                                )
                                            )
                                        )
        # -- Add name to polygons pandas Series
        polygons.name = 'geometry'

        # -- Return
        return polygons




    def check_layer_zone(self, layer, zone):
        """
        Raise assertion errors if the provided layer and zone ids 
        not correspond to a pilot point parametrize zone
        """
        # -- Check layer existence
        layer_err =  f"ERROR : No pilot points parametrization for `layer={layer}`."
        assert layer in self.data.keys(), layer_err

        # -- Check zone existence
        zone_err = layer_err.replace('.', f' and `zone={zone}`.')
        assert zone in self.data[layer].keys(), zone_err




    def get_polygon(self, layer, zone):
        """
        Subset pilot point active polygons by layer and zone

        Parameters
        ----------
        layer (int) : required layer id

        zone (int) : required zone id (necessarily >0)

        Returns:
        --------
        polygon (shapely.geometry.Polygon) : required active pilot point
                                             zone as shapely Polygon

        Examples
        --------
        polygon = pp.get_polygon(layer=1, zone=1)
        """
        # -- Check layer zone existence
        self.check_layer_zone(layer, zone)
        # -- Return required polygon
        return self.polygons.loc[(layer,zone)]




    def add_spacing_pp(self, layer, zone, xspacing, yspacing, xoffset=0, yoffset=0, buffer=0):
        """
        Generate pilot points from directional (xy) spacing.

        Parameters
        ----------
        layer (int) : required layer id

        zone (int) : required zone id (necessarily >0)

        xspacing (int/float) : distance between pilot point in x direction

        xspacing (int/float) : distance between pilot point in y direction

        xoffset (int/float, optional) : offset distance from the origin seeding point
                                        in x direction (lower left corner of polgyon extension).
                                        Default is 0.

        yoffset (int/float, optional) : offset distance from the origin seeding point
                                        in y direction (lower left corner of polgyon extension).
                                        Default is 0.

        buffer (int/foat, optional) : exterior buffering distance of the required polygon.
                                      Default is 0.
                                      Note: this argument can be used to exaggerate the polygon 
                                            area while seending pilot points. That can be usefull
                                            with local irregular shaped aquifer domain.
        Returns:
        --------
        Store dictionary of generated pilot points data in main `.data` attribut.
        Format: {
                'layer': 2,                         |
                 'zone': 1,                         |
                 'xspacing': 550,                   |
                 'yspacing': 930,                   |   
                 'xoffset': 0,                      |   METDADATA
                 'yoffset': 0,                      |
                 'buffer': 100,                     |
                 'n': 58                            |
                 ...
                 'pp': shapely.geometry.MultiPoint  |   DATA
                 }

        Examples
        --------
        pp = PilotPoints(izone)
        pp.add_spacing_pp(layer=2, zone=1, xspacing=550, yspacing=930, buffer=100)

        """
        # -- Keep the arguments as metadata dictionary
        metadata = {k:v for k,v in locals().items() if k != 'self'}
        # -- Extract required pilot point active polygon zone
        polygon = self.get_polygon(layer, zone)
        bpolygon = polygon.buffer(buffer)
        # -- Fetch polygon bounds 
        minx, miny, maxx, maxy = bpolygon.bounds 
        # -- Generate regurlarly spaced xy-coordinates
        x = np.arange(minx, maxx, xspacing) + xoffset
        y = np.arange(miny, maxy, yspacing) + yoffset
        xy = np.reshape(np.meshgrid(x,y),(2,-1))
        # -- Mask points outside model extension
        pp_coords = xy.T[self.mm.isin_extent(*xy)]
        # -- Get only points taht lie in buffered polygon
        mp = bpolygon.intersection(self.MultiPoint(pp_coords))
        # -- Add to pilot point data (and metadata)
        if mp.is_empty:
            warnings.warn(f'No pilot points within zone {zone} of layer {layer}\n'\
            'Check spatial index and zonation for this layer' )
        else : 
            self.data[layer][zone] = {**metadata, 'n':len(mp.geoms), 'pp':mp}


    def add_n_pp(self, layer, zone, n, tol= 50, xoffset=0, yoffset=0, buffer=0):
        """
        Generate pilot points from directional (xy) spacing.

        Parameters
        ----------
        layer (int) : required layer id

        zone (int) : required zone id (necessarily >0)

        n (int) : number of pilot points to generate.

        tol (int/float, optional) : tolerance on infered spacing.
                                    Default is 50.
                                    Note: a low value can slow down the points 
                                          generation but offers a better precision.

        xoffset (int/float, optional) : offset distance from the origin seeding point
                                        in x direction (lower left corner of polgyon extension).
                                        Default is 0.

        yoffset (int/float, optional) : offset distance from the origin seeding point
                                        in y direction (lower left corner of polgyon extension).
                                        Default is 0.

        buffer (int/foat, optional) : exterior buffering distance of the required polygon.
                                      Default is 0.
                                      Note: this argument can be used to exaggerate the polygon 
                                            area while seending pilot points. That can be usefull
                                            with local irregular shaped aquifer domain.
        Returns:
        --------
        Store dictionary of generated pilot points data in main `.data` attribut.
        Format: {
                'layer': 1,                         |
                 'zone': 1,                         |
                 'xspacing': 600,                   |
                 'yspacing': 600,                   |   
                 'xoffset': 0,                      |   METDADATA
                 'yoffset': 0,                      |
                 'buffer': 100,                     |
                 'n': 40                            |
                 ...
                 'pp': shapely.geometry.MultiPoint  |   DATA
                 }

        Examples
        --------
        pp = PilotPoints(izone)
        pp.add_spacing_pp(layer=2, zone=1, n=40, tol=250, buffer=100)

        """
        # -- Keep the arguments as metadata dictionary
        metadata = {k:v for k,v in locals().items() if k != 'self'}
        # -- Extract required pilot point active polygon zone
        polygon = self.get_polygon(layer, zone)
        bpolygon = polygon.buffer(buffer)
        # -- Fetch polygon bounds 
        minx, miny, maxx, maxy = bpolygon.bounds 
        # ---- Initialize spacing and point counter
        spacing = min((maxy-miny)/4, (maxx-minx)/4)
        point_counter = 0
        # Start while loop to find the better spacing according to tolerance increment
        while point_counter <= n:
            # --- Generate grid point coordinates
            x = np.arange(minx, maxx, spacing) + xoffset
            y = np.arange(miny, maxy, spacing) + yoffset
            xy = np.reshape(np.meshgrid(x,y),(2,-1))
            # -- Mask points outside model extension
            pp_coords = xy.T[self.mm.isin_extent(*xy)]
            # -- Get only points taht lie in buffered polygon
            mp = bpolygon.intersection(self.MultiPoint(pp_coords))
            # ---- Verify number of point generated
            point_counter = len(mp.geoms)
            spacing -= tol
        else:
            # -- Add to pilot point data (removing excess points)
            self.data[layer][zone] = {**metadata, 'xspacing':spacing, 
                                                  'yspacing':spacing,
                                                  'pp':mp[point_counter-n:]} 





    def plot(self, layer, zone, buffer=0, ax=None, zone_kwargs=ZONE_KWARGS, buffer_kwargs=BUFFER_KWARGS, pp_kwargs=PP_KWARGS):
        """
        Pilot point internal ploting facility.

        Parameters
        ----------
        layer (int) : required layer id

        zone (int) : required zone id (necessarily >0)

        buffer (int/foat, optional) : exterior buffering distance of the required polygon.
                                      Default is 0.
                                      Note: this argument can be used to exaggerate the polygon 
                                            area while seending pilot points. That can be usefull
                                            with local irregular shaped aquifer domain.

        ax (matplotlib.axes, optional) : matplotlib custom AxesSubplot .
                                         If None, basic AxesSubplot will be create.
                                         Default is None.

        zone_kwargs (dict, optional) : matplotlib.pyplot.plot() arguments for pilot
                                       point active polygon ploting (as dictioanry).

        buffer_kwargs (dict, optional) : matplotlib.pyplot.plot() arguments for pilot
                                         point active buffered polygon ploting (as dictioanry).

        pp_kwargs (dict, optional) : matplotlib.pyplot.scatter() arguments for pilot
                                     point ploting (as dictioanry).

        Returns:
        --------
        ax (matplotlib.axes) : plotted pilot points in AxesSubplot.

        Examples
        --------
        pp = PilotPoints(izone)
        pp.add_spacing_pp(layer=2, zone=1, xspacing=550, yspacing=930, buffer=100)
        pp.plot(layer=2, zone=1, buffer=100, 
                zone_kwargs={'color':'red', 'ls':', 'lw':1.5', 'label':'PP ZONE'})
        plt.show()

        """
        # -- Get required active domain as polygon
        polygon = self.get_polygon(layer, zone)

        # -- Prepare basic axe if not provided
        if ax is None:
                plt.rc('font', family='serif', size=10)
                fig, ax = plt.subplots(figsize=(8,6))

        # -- Plot pilot point active zone(s) exterior line
        geoms = [polygon] if isinstance(polygon, self.Polygon) else [g for g in polygon.geoms]
        for g in geoms:
            ax.plot(*g.exterior.xy, **zone_kwargs)
            ax.fill(*g.exterior.xy, facecolor='lightgrey')

        # -- Plot pilot point active zone(s) interior line(s)
        for g in geoms:
            for hole in g.interiors:
                ax.plot(*hole.xy, **{k:v for k,v in zone_kwargs.items() if k != 'label'})
                ax.fill(*hole.xy, facecolor='white')

        # -- Plot buffer zone if required
        if buffer != 0:
            ax.plot(*polygon.buffer(buffer).exterior.xy, **buffer_kwargs)

        # -- Check that pilot points have already been generated
        err_msg = 'ERROR : No pilot points have been added yet ' \
                  f' for `layer`={layer} and `zone`={zone}.'
        assert self.data[layer][zone] is not None, err_msg

        # -- Plot pilot points
        x, y = np.column_stack([p.xy for p in self.data[layer][zone]['pp']])
        ax.scatter(x,y, **pp_kwargs)

        # -- Ajust view
        ax.autoscale_view()

        # -- Add Legend
        plt.legend()

        # -- Add title
        ax.set_title(f'Pilot Points (layer {layer}, zone {zone})', fontsize=12, fontweight='bold')

        # -- Return axe
        return ax




    def to_pp_data(self):
        """
        Convert generated pilot point data into a standard `.pp_data`
        dictionary of MartheGridParam class instance.

        Returns:
        --------
        pp_data (dict) : 

        Examples
        --------
        pp = PilotPoints(izone)
        pp.add_spacing_pp(layer=0, zone=1, xspacing=500, yspacing=500)
        pp_data = pp.to_pp_data()

        """
        pp_data = {layer:{zone:None if mp is None
                                    else [[p.x,p.y] for p in mp['pp'].geoms]}
                                    for layer, d in self.data.items()
                                    for zone, mp in d.items()}
        return pp_data




    @staticmethod
    def pp_df_from_coords(parname, coords, layer, zone, value= 1e-3):
        """
        Create pilot point Dataframe from xy-coordinates with generic names.

        Parameters
        ----------
        parname (str) : parameter name

        coords (list/array) : pilot point xy-coordinates.
                              Format : [[ppx_0, ppy_0], ..., [ppx_N, ppy_N]]

        layer (int) : required layer id

        zone (int) : required zone id (necessarily >0)
    
        value (int/float, optional) : initial pilot points value.
                                      Default is 1e-3.

        Returns
        --------
        pp_df (Dataframe) : pilot point standard DataFrame.
                            Format:
                                                  parname           x      y  zone         value
                             parname
                             hk_l04_z01_00  hk_l04_z01_00  357.136364  209.0     1  1.780510e-05
                             hk_l04_z01_01  hk_l04_z01_01  373.681818  209.0     1  4.042830e-05
                             ...                      ...         ...    ...     .           ...
                             hk_l04_z01_N    hk_l04_z01_N  357.136364  225.4     1  1.337280e-05

        Examples
        --------
        coords = shp_utils.shp2points('gis/pp_l04.shp', stack=True)
        pp_df = PilotPoints.pp_df_from_coords(parname='myfield', coords, layer=4, zone=1)

        """
        # -- Manage value input
        if len(marthe_utils.make_iterable(value)) == 1:
            value = np.tile(value, len(coords))
        # -- Generate names
        digit = len(str(len(coords)))
        ppn = [PPFMT(parname,layer, zone, i, digit) for i in  range(len(coords))]
        # -- Build pilot point standart DataFrame
        ppx, ppy = np.column_stack(coords)
        pp_df = pd.DataFrame.from_dict(
                    {k:v for k,v in zip(PP_NAMES, [ppn,ppx,ppy,zone,value])}
                                    ).set_index('parname', drop=False)
        # -- Return pilot point DataFrame
        return pp_df




    def extract_vgm_range(self, factor=3):
        '''
        Infer standard variogram ranges for generated pilot points as 
        N times the maximum distance between neighbors pilot points.
        Should be used in MartheOptim.write_kriging_factors().

        Parameters
        ----------
        factor (int/float) : multiplier of the maximum distance between
                             neighbors pilot points.
                             Default is 3.

        Returns
        --------
        vgm_range (dict) : infered exponential variagrams ranges 
                           from added pilot points.
                           Format: {layer_0 : {zone_0: range_0_0, ..., zone_i: range_0_i},
                                    ...,
                                    layer_i : {zone_0: range_i_0, ..., zone_i: range_i_i} }

        Examples
        --------
        pp.extract_vgm_range()
        '''
        vgm_range = {layer:{zone: None if d is None
                                       else factor * max(d['xspacing'], d['yspacing'])}
                                       for layer, zdic in self.data.items()
                                       for zone, d in zdic.items()
                                       }
        return vgm_range





    def to_shapefile(self, path='.', onefile=False, epsg=None, prj=None):
        """
        Export generated pilot points to shapefile object(s).

        Parameters
        ----------
        path (str, optional) :  if `onefile` is set to False:
                                    --> path the output folder to export 
                                        pilot points shapefiles
                                if `onefile` is set to True:
                                    --> output file name of exported 
                                        pilot points.
                                Default is '.'.

        onefile (bool, optional) : whatever exporting pilot points in a
                                   unique shapefile.
                                   If False, all pilot point sets will be
                                   export in separated shape files with 
                                   understable generic names.
                                   Format: 'path/pp_l01_z01.shp'.
                                   Default is False.

        epsg (int, optional) :  EPSG code.
                                See https://www.epsg-registry.org/ or spatialreference.org

        prj (str, optional) : Existing projection file to be used with new shapefile.

        Examples
        --------
        pp.to_shapefile('allpp.shp', onefile=True)

        """
        # -- Colect all data and geometry (single part)
        df = pd.concat(
                [pd.DataFrame.from_dict(d)
                    for layer,zdic in self.data.items()
                        for d in zdic.values()
                            if d is not None]
                            )
        if onefile:
            # -- Extract geometry (point)
            geoms =  np.column_stack(df.pop('pp').apply(lambda p: p.xy)).T
            # -- Convert data to recarray
            recarray = df.to_records(index=False)
            # -- Manage shapefile name
            shpname = 'pilot_points.shp' if path == '.' else path
            # -- Export to shapefile
            shp_utils.recarray2shp(recarray, geoms, shpname=shpname,
                                   geomtype='Point', epsg=epsg, prj=prj)
        else:
            # -- Iterate over layer and zone
            for idx, idf in df.groupby(['layer', 'zone']):
                # -- Extract geometry (point)
                geoms =  np.column_stack(idf.pop('pp').apply(lambda p: p.xy)).T
                # -- Convert data to recarray
                recarray = idf.to_records(index=False)
                # -- Build shapefile name
                shpname = os.path.join(path, 'pp_l{0:02d}_z{1:02d}.shp'.format(*idx))
                # -- Export to shapefile
                shp_utils.recarray2shp(recarray, geoms, shpname=shpname,
                                       geomtype='Point', epsg=epsg, prj=prj)


    def __str__(self):
        """
        Internal string method.
        """
        return 'PilotPoints'

