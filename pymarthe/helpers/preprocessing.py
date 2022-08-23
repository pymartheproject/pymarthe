"""
Contains some helper functions for Marthe model preprocessing

"""

import os, sys
import re
import numpy as np
import pandas as pd
from copy import deepcopy
from pymarthe import MartheModel
from pymarthe.utils import *




def spatial_aggregation(mm, x, y, layer, value, agg = 'sum', trans ='none', only_active = True, base=0):
    """
    Helper function to aggregate (and transform) values that are in same model cell.

    Parameters:
    ----------

    mm (MartheModel) : MartheModel instance.
                       Note: must contain spatial index.
    x, y ([float]) : xy points coordinates in model extension.
                     Note: must have the same length.
    layer (int/iterable) : layer id to search of point localisation (0-based).
                           If `layer` is an integer, same layer id will be considered
                           for all points.
                           Note: for some values such as river pumping, `layer` has to be
                                 be set to 0 (superficial aquifer).
    value ([float]) : related point values to aggregate.
                      Note: must be same length as x and y.
    agg (func/str, optional) : Function or function name to perform aggregation.
                               Can be: 'mean', 'median', 'sum', 'count', np.sum, ...
                               Default is 'sum'.
    trans (str/func, optional) : keyword/function to use for transforming aggregated values.
                                 Can be:
                                    - function (np.log10, np.sqrt, ...)
                                    - string function name ('log10', 'sqrt')
                                 If 'none', values will not be transformed.
                                 Default if 'none'.
    only_active (bool, optional) : whatever considering active cells only.
                                   Default is True.
                                   Note: for some values such as river pumping, `only_active`
                                         has to be set to False.
    base (int, optional) : output layer id n-base.
                           Marthe is 1-based (base=1), PyMarthe is 0-based (base=0).
                           Default is 0.

    Returns:
    --------
    res (DataFrame) : aggregated and transformed values.
                      Format:
                            node          x         y     layer      value
                      0      789   458963.2  698754.1         4     0.0023
                      1      856   458456.5  698702.8         2     0.0046
                      .

    Examples:
    --------
    x = [458963.2, .., 458456.5]
    y = [698754.1, .., 698702.8]
    layer = [4, .., 2]
    value = [0.0023, .., 0.0046]
    df = spatial_aggregation(mm, x, y, layer, value,
                             agg = 'sum', trans ='lambda x: -x',
                             only_active= True, base=1)
    df.to_csv('file.txt', header=False, index = False, sep='\t')

    """
    # -- Sample node ids for each xylayer-points
    nodes = mm.get_node(x,y,layer)

    # -- Store in DataFrame
    df = pd.DataFrame( { 'node': nodes,
                         'x': x,
                         'y': y,
                         'layer' : layer,
                         '_value' : value  } )

    # -- Groupby by node and perform aggregation
    agg_df = df.groupby('node', as_index=False).agg({'_value': agg})

    # -- Subset on active cells
    if only_active:
        agg_df = agg_df.loc[[mm.all_active(n) for n in agg_df.node]]

    # -- Rename value column for merging purpose
    agg_df.rename({'_value': 'value'}, axis = 1, inplace=True)

    # -- Merge aggregated/initial DataFrames 
    res = pd.merge(df, agg_df).drop_duplicates('node').drop('_value',axis=1)

    # -- Convert to n-base
    res['layer'] = res.layer.add(base)

    # -- Transform aggregated values if required
    if trans != 'none':
        # -- Check transformation validity
        pest_utils.check_trans(trans, test_on= res.value)
        # -- Perform transformation on aggregated values
        res['value'] = pest_utils.transform(res['value'], trans)

    # ---- Return aggregated/transformed values in DataFrame
    return res





def extract_model_geology(mm, shpout, epsg=2154, on_grid=False):
    """
    Extract geology on a projected Marthe Model extension from BRGM 
    BDCharm50 data base.
    (only available for projected model located in France country).

    Parameters
    ----------
    mm (MartheModel) : MARTHE model to consider.

    shpout (str) : output shapefile name. 

    epsg (int, optional) : Geodetic Parameter Dataset.
                           Default is 2154 (RGF93).

    on_grid (bool, optional) : whatever set geological data on grid cell.
                               If False, real geological polygons on model 
                               extension will be written.
                               If True, a spatial join operation will be 
                               performed between grid and geological data.
                               Default is False.


    Examples
    --------
    mm = MartheModel('mymodel.rma')
    extract_model_geology(mm, 'Lambert_3_model_geol.shp', epsg=27572, on_grid=True)

    """
    # ---- Import all additional required python modules
    try:
        import requests, shutil
        from shapely.geometry import Polygon
        import geopandas as gpd
    except:
        pcks = ['requests','shutil','shapely','geopandas']
        err_msg = 'Could not import all the required python module(s):\n\t- '
        err_msg += '\n\t- '.join(pcks)
        ImportError(err_msg)

    # ---- Query BDCharm50 data base online
    print('Query BRGM `BDCharm50` data base online ...')
    r = requests.get("http://infoterre.brgm.fr/telechargements/BDCharm50/FR_vecteur.zip")

    # ---- Create _temp_ working folder and download data
    if os.path.exists('_temp_'):
        shutil.rmtree('_temp_')
    os.mkdir('_temp_')
    open(os.path.join('_temp_', '_temp_.zip'), 'wb').write(r.content)

    # ---- Read geology shapefile of whole France country 
    #     (since `bbox` and mask attribut doesn't work on 0.11.0)
    print('Read geological data ...')
    data = "zip://_temp_/_temp_.zip!FR_vecteur/GEO001M_CART_FR_S_FGEOL_2154.shp"
    gdf = gpd.read_file(data).to_crs(f'epsg:{epsg}')

    # ---- Manage whatever set geological data on grid
    if on_grid:
        # -- Build `.modelgrid` if not exists
        if mm.modelgrid is None:
            mm.build_modelgrid()
        # -- Convert grid to Polygons GeoDataFrame
        print('Processing geological data on model grid ...')
        geoms = mm.modelgrid['vertices'].apply(Polygon)
        grid = gpd.GeoDataFrame( mm.modelgrid, geometry=geoms, crs=epsg
                               ).drop_duplicates('geometry')
        # -- Perform spatial join between geology and modelgrid
        out = grid.sjoin(gdf).drop(columns='index_right')
    else:
        # ---- Extract model extension as bounding box geometry
        print('Processing geological data on model extension ...')
        bbox = gpd.GeoDataFrame(geometry=[gpd.base.box(*mm.get_extent())], crs=epsg)
        out = gdf.overlay(bbox, how='intersection')

    # ---- Convert columns names to lower case
    out.columns = out.columns.str.lower()

    # ---- Export to shapefile
    out.drop(columns='vertices', errors='ignore').to_file(shpout, index=False)
    print(f'Shapefile {shpout} have been written successfully!')

    # ---- Remove temporary folder
    shutil.rmtree('_temp_')





def extract_Hubeau_stations(mm, epsg, only_active=True):
    """

    Helper function to extract available observation points (stations) in a Marthe model extension.
    Perform a requests to the French Hubeau API.

    Specific python module required:
        - pyproj
        - requests
        - shapely
        - geopandas

    Inspired by Guillaume Attard
    https://guillaumeattard.com/exploring-hydro-geological-data-of-france-with-python/

    Parameters:
    ----------
    mm (MartheModel) : MartheModel instance.
                       Note: model must be (1) projected
                                           (2) located on French territory
    epsg (int) : Geodetic Parameter Dataset.
    only_active (bool) : keep only stations located in model active domain.
                         Default is True.

    Returns:
    --------
    stations (GeoDataFrame) : available stations. 

    Examples:
    --------
    mm = MartheModel('my_projected_model.rma')
    stations = extract_Hubeau_stations(mm, epsg=2154)

    """
    # ---- Import all additional required python modules
    try:
        from pyproj import Transformer
        import requests
        from shapely.geometry import Polygon
        import geopandas as gpd
    except:
        pcks = ['pyproj', 'requests', 'shapely','geopandas']
        err_msg = 'Could not import all the required python module(s):\n\t- '
        err_msg += '\n\t- '.join(pcks)
        ImportError(err_msg)

    # ---- Build modelgrid if required
    if mm.modelgrid is None:
        mm.build_modelgrid()

    # ---- Build long/lat transformer
    tfm = Transformer.from_crs( f'epsg:{epsg}', 'epsg:4326')
    llc = tfm.transform(*mm.get_extent()[:2])   # Lower Left Corner
    urc = tfm.transform(*mm.get_extent()[2:])   # Upper Right Corner
    bbox = np.ravel([np.flip(p) for p in [llc, urc]])   # Bounding Box

    # ---- Build Hubeau url
    url_root = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations?"
    url_bbox = "bbox=" + "%2C".join([str(round(coord, 4)) for coord in bbox])
    url_date = "&date_recherche=" + str(mm.get_time_window()[0].date())
    url_tail = "&format=json&size=20000"
    url = url_root + url_bbox + url_date + url_tail

    # ---- Request web data from Hubeau API
    r = requests.get(url)
    df = pd.DataFrame.from_dict(r.json()["data"])

    # ---- Convert to GeoDataFrame and change 
    stations = gpd.GeoDataFrame(df, 
                        geometry=gpd.points_from_xy(df.x, df.y),
                        crs = 'epsg:4326'
                    ).to_crs(f'epsg:{epsg}')

    # ---- Return Hubeau stations as GeoDataFrame
    if only_active:
        # -- Keep only stations that fall in model active domain
        active_domain = gpd.GeoDataFrame(
                    geometry= mm.query_grid(active=1)['vertices'].apply(Polygon),
                    crs=epsg
                ).drop_duplicates('geometry'
                ).unary_union
        # -- Return subset stations
        return stations[stations.within(active_domain)]

    else:
        # -- Return all stations
        return stations





def extract_Hubeau_head_records(code_bss, start_date, end_date):
    """
    Helper function to extract water table/head records at a observation point (station)
    referenced by it's BSS id.
    Perform a requests to the French Hubeau API.

    Specific python module required:
        - requests

    Inspired by Guillaume Attard
    https://guillaumeattard.com/exploring-hydro-geological-data-of-france-with-python/

    Parameters:
    ----------
    code_bss (str) : normalized station name (ex: '07334X0540/F')
    start_date, end_date (str) : required time window (ex: '2021-03-04')

    Returns:
    --------
    df (DataFrame) : available data between required time window.
                     Note: water table <=> 'profondeur_nappe'
                           head <=> 'niveau_nappe_eau'

    Examples:
    --------
    mm = MartheModel('my_projected_model.rma')
    start_date, end_date = [str(ts.date()) for ts in mm.get_time_window()]
    df = extract_Hubeau_head_records('07334X0579/F', '2003-01-01', '2008-02-01')

    """
    # ---- Import additional required python module
    try:
        import requests
    except:
        err_msg = 'Could not import `requests` python module.'
        ImportError(err_msg)
        
    # ---- Build Hubeau url
    url_root = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques?"
    url_code = "code_bss=" + code_bss.replace("/","%2F")
    url_date = "&date_debut_mesure=" + start_date + "&date_fin_mesure=" + end_date
    url_tail = "&size=20000&sort=asc"
    url = url_root + url_code + url_date + url_tail

    # ---- Request web data from Hubeau API
    r = requests.get(url)
    df = pd.DataFrame.from_dict(r.json()["data"])

    # ---- Set DateTimeIndex
    df['date'] = pd.to_datetime(df['date_mesure'])
    df.set_index('date', inplace=True)

    # ---- Return DataFrame
    return df

