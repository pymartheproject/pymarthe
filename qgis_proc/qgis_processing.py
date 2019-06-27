# --- Load a shapefile from Qgis console ---
import processing

nlayer = 15 # number of layers
type_list = ['cap','libre']
names_layers = ['plio','helv','aqui','olnp','EOCS','EOCM','EOCI','camp','cost','turo','ceno','tith','kimm','BACX','bajo',]
path_points_file = "C:/Documents these/SIG/decoupage_type_aq_projection_km/"
path_limits_extension =  "C:/Documents these/SIG/limV3_polygone_proj_marc/" 
alpha_cap = 0.05
alpha_libre = 0.1
for i in range(1,nlayer+1):
    for id in type_list:
        layer_path = path_points_file+str(i)+"/points_partie_"+id+str(i)+".shp"
        layer_name = "points_partie_"+id+str(i)+""
        # load layer 
        vlayer = QgsVectorLayer(layer_path, layer_name, "ogr")
        dic_feat = {feat.id():feat for feat in vlayer.getFeatures()}
        npoints = len(dic_feat.keys())
        print(npoints)
        if npoints > 4 :
        # alpha threshold in [0;1] where 1 is eq. to convex hull
            if id == "cap" :
                alpha_threshold = alpha_cap
                print(alpha_threshold)
            elif id == "libre" : 
                alpha_threshold = alpha_libre
                print(alpha_threshold)
            else:
                print("ERROR")
            # whether to allow holes 
            allow_holes = True
            # output path, here a memory layer named "output_hull"
            # note that it can be a file and the name can be set to whatever convenient 
            output_layer_path = path_points_file+str(i)+"/extension_concave_partie_"+id+""+str(i)+".shp"
            # build param dic
            dic_param_concave_hull = {'INPUT': layer_path ,
                    'ALPHA': alpha_threshold ,
                    'HOLES': allow_holes,
                    'OUTPUT': output_layer_path}
            # call processing algorithm
            concave_hull_result = processing.run('qgis:concavehull',dic_param_concave_hull )
            # ---- Processing intersection ---
            overlay_layer_name  = path_points_file+str(i)+"/extension_concave_partie_"+id+""+str(i)+".shp"
            input_layer_name    = path_limits_extension+"lim_"+names_layers[(i-1)]+".shp"
            output_layer_path   = path_points_file+str(i)+"/extension_rognee_partie_"+id+""+str(i)+".shp"
            # build param dic
            dic_param_intersection = {'INPUT': input_layer_name ,
                    'OVERLAY': overlay_layer_name,
                    'OUTPUT': output_layer_path}
            # call processing algorithm
            intersection_result = processing.run('native:intersection',dic_param_intersection )
            # ---- Processing random points in layer bounds ---
            input_layer_name = path_points_file+str(i)+"/extension_rognee_partie_"+id+""+str(i)+".shp"
            points_number = 500
            min_distance  = 40 
            output_layer_name = path_points_file+str(i)+"/points_pilotes_"+id+""+str(i)+".shp"
            # build param dic
            dic_param_random_points = {'INPUT': input_layer_name ,
                    'POINTS_NUMBER': points_number,
                    'MIN_DISTANCE': min_distance,
                    'OUTPUT':output_layer_name}
            # call processing algorithm
            random_points_result = processing.run('qgis:randompointsinlayerbounds',dic_param_random_points)
        else:
            pass


#random points in Eponte 
path_lim_eponte_file = "C:/Documents these/SIG/LIM_EPONTE/"
for i in range (14): 
    input_layer_name = path_lim_eponte_file+"/lim_eponte_"+str(i+2)+".shp"
    points_number = 500
    min_distance  = 40 
    output_layer_name = path_lim_eponte_file+"pp_eponte"+"/points_pilotes_"+str(i+2)+".shp"
    dic_param_random_points = {'INPUT': input_layer_name ,
                        'POINTS_NUMBER': points_number,
                        'MIN_DISTANCE': min_distance,
                        'OUTPUT':output_layer_name}
    random_points_result = processing.run('qgis:randompointsinlayerbounds',dic_param_random_points)

