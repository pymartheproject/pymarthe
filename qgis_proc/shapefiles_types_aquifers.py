import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
import fiona
#Import specific function 'from_epsg' from fiona module : allows to pass coord syst for thr geodataframe
from fiona.crs import from_epsg
from geopandas import GeoDataFrame
from shapely.geometry import Point


#**********************************************************************
def read_file (path_file):


    df_sim = pd.read_csv(path_file, sep=';')  # Dataframe
    id_columns = df_sim.columns[:]

    return  id_columns, df_sim

#*************************************************************************

columns, geom_model =  read_file('./Geom_MONA_V3.csv')
top_layers = ['']
sub_layer = ['Sub_PlioQuat','Sub_Helv','Sub_Aqui','Sub_Oligo',
'Sub_EocSup','Sub_EocMoy','Sub_EocInf','Sub_Camp','Sub_CoSt','Sub_Turo',
'Sub_Ceno','Sub_Thitho','Sub_Kim','Sub_BaCX','Sub_Bajo']
geom_model = geom_model.drop(sub_layer, 1)
top_layers_list = geom_model.columns[6:]


comp_htop_list = []

nlayer = 15 # number of layers
for i in range(1,nlayer+1):

    id_columns,df_sim = read_file('./chasim'+str(i)+'.csv') # reading chasim file of each layer
    df_sim['hmax'] = df_sim[id_columns[6:]].max(axis=1) # Adding column with hmax
    df_sim['hmin'] = df_sim[id_columns[6:]].min(axis=1) # Adding column with hmin
    df_chasim_geom = pd.merge(df_sim,geom_model,on = 'N__maille') # merge the tow dataframes
    df_chasim_geom = df_chasim_geom.replace(8888, np.nan) # Replace 8888 values with nan
    hmax_top = df_chasim_geom['hmax'] - df_chasim_geom[top_layers_list[i-1]] 
    hmin_top = df_chasim_geom['hmin'] - df_chasim_geom[top_layers_list[i-1]]
    h_top = pd.concat([df_chasim_geom['N__maille'],df_chasim_geom['Colonne_x'],df_chasim_geom['Ligne_x'],df_chasim_geom['Coord_X_x']
        ,df_chasim_geom['Coord_Y_x'],df_chasim_geom['Coord_X_y'],df_chasim_geom['Coord_Y_y'],hmax_top, hmin_top], axis=1)
    h_top = h_top.rename(columns={h_top.columns[-2]: "hmax_top",h_top.columns[-1]: "hmin_top"}) # rename columns
    h_top.loc[((h_top['hmin_top']) > 0)  , 'Aquifere'] = 'Captif' # Test a condition and when it is true, create a column "Aquifere"
    h_top.loc[((h_top['hmax_top']) <= 0) , 'Aquifere'] = 'Libre'  
    h_top.loc[((h_top['hmax_top']) > 0) & ((h_top['hmin_top']) <= 0) , 'Aquifere'] = 'Cap/libre'  
    comp_htop_list.append(h_top)



crs = {'a': 6378249.145,
 'b': 6356514.96582849,
 'lat_0': 44.1,
 'lat_1': 43.20317004,
 'lat_2': 44.99682996,
 'lon_0': 2.337229104484,
 'no_defs': True,
 'proj': 'lcc',
 'units': 'km',
 'x_0': 600000,
 'y_0': 200000}

for i in range(1,nlayer+1):
    #geometry  = [Point(xy) for xy in zip(comp_htop_list[i-1].Coord_X_y, comp_htop_list[i-1].Coord_Y_y)]
    geometry  = [Point(xy) for xy in zip(comp_htop_list[i-1].Coord_X_x, comp_htop_list[i-1].Coord_Y_x)]
    gdf = GeoDataFrame(comp_htop_list[i-1], crs=crs, geometry=geometry)
    partie_captive = gdf[gdf.Aquifere == 'Captif'] 
    partie_libre   = gdf[gdf.Aquifere == 'Libre']
    #partie_mixte  = gdf[gdf.Aquifere == 'Cap/libre'] 
    partie_captive = partie_captive.append(partie_captive)
    partie_libre   = partie_libre.append(partie_libre) 
    #outfp_cap = r"sC:/Documents these/SIG/decoupage_cap_libre_seulement/"+str(i)+"/points_partie_cap"+str(i)+".shp"
    outfp_cap = r"sC:/Documents these/SIG/decoupage_type_aq_projection_km/"+str(i)+"/points_partie_cap"+str(i)+".shp"
    #outfp_libre = r"C:/Documents these/SIG/decoupage_cap_libre_seulement/"+str(i)+"/points_partie_libre"+str(i)+".shp"
    outfp_libre = r"C:/Documents these/SIG/decoupage_type_aq_projection_km/"+str(i)+"/points_partie_libre"+str(i)+".shp"
    #outfp_mixte = r"C:/Documents these/SIG/zone_par_couche/"+str(i)+"/partie_mixte_couche"+str(i)+".shp"
    #outfp = r"C:/Documents these/SIG/Zone_captive_libre_monaV3/"+str(i)+".shp"
    #gdf.to_file(outfp)
    if len(partie_captive) != 0:
        partie_captive.to_file(outfp_cap)
    if len(partie_libre) != 0:
        partie_libre.to_file(outfp_libre)
    #if len(partie_mixte) != 0:
        #partie_mixte.to_file(outfp_mixte)


